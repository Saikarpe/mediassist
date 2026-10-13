"""
Train and evaluate the healthcare assistant on the MedQuAD dataset.

  1. Question-type classifier  : 80/20 stratified split, 3 models compared
  2. Retrieval (RAG) QA        : Hit@k, MRR and answer token-F1 on paraphrased test questions
  3. Agent task planner        : intent-plan accuracy on a hand-written request set
  4. Final models trained on the full data and saved to models/

Run:  python train_evaluate.py
"""
import json
import os
import random
import re
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from src.knowledge_base import (MedicalRetriever, QuestionTypeClassifier, clean_text, load_medquad,
                                save_models)

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)
random.seed(42)
np.random.seed(42)

df = load_medquad()
print(f"MedQuAD loaded: {len(df)} unique Q&A pairs, {df.qtype.nunique()} question types")
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["qtype"], random_state=42)
metrics = {"dataset": {"pairs": len(df), "train": len(train_df), "test": len(test_df),
                       "question_types": int(df.qtype.nunique()),
                       "class_distribution": df.qtype.value_counts().to_dict()}}

# ---------------------------------------------------------------- 1. Question-type classification
print("\n[1] Question-type classifier")
Xtr, Xte = [clean_text(q) for q in train_df.Question], [clean_text(q) for q in test_df.Question]
ytr, yte = train_df.qtype.values, test_df.qtype.values
candidates = {
    "Naive Bayes": make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB(alpha=0.1)),
    "Linear SVM": make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LinearSVC(C=1.0)),
    "Logistic Regression": None,  # our model (QuestionTypeClassifier)
}
clf_results = {}
for name, model in candidates.items():
    t0 = time.time()
    if model is None:
        model = QuestionTypeClassifier().fit(train_df.Question, ytr)
        pred = model.predict(test_df.Question)
        best_model = model
    else:
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
    clf_results[name] = {"accuracy": round(accuracy_score(yte, pred), 4),
                         "macro_f1": round(f1_score(yte, pred, average="macro"), 4),
                         "weighted_f1": round(f1_score(yte, pred, average="weighted"), 4),
                         "train_time_s": round(time.time() - t0, 2)}
    print(f"  {name:20s} acc={clf_results[name]['accuracy']:.4f}  macroF1={clf_results[name]['macro_f1']:.4f}")
metrics["qtype_classification"] = clf_results

pred = best_model.predict(test_df.Question)
report = classification_report(yte, pred, output_dict=True, zero_division=0)
metrics["qtype_per_class_f1"] = {k: round(v["f1-score"], 3) for k, v in report.items()
                                 if k not in ("accuracy", "macro avg", "weighted avg")}
labels = sorted(df.qtype.unique())
cm = confusion_matrix(yte, pred, labels=labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False)
plt.title("Question-type classifier - confusion matrix (test set)")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.xticks(rotation=45, ha="right"); plt.tight_layout()
plt.savefig(f"{RESULTS}/confusion_matrix.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 2. Retrieval QA
print("\n[2] Retrieval-augmented QA")
PARAPHRASES = [
    (r"^what (is|are) the symptoms of", "what signs are seen in"),
    (r"^what (is|are) the treatments? for", "how can doctors treat"),
    (r"^what causes", "what is the reason for"),
    (r"^who is at risk for", "risk factors of"),
    (r"^how many people are affected by", "how common is"),
    (r"^how to diagnose", "which tests detect"),
    (r"^how to prevent", "ways to avoid"),
    (r"^what (is|are) the outlook for", "prognosis of"),
    (r"^what (is|are) \(are\)", "explain"),
    (r"^what (is|are)", "tell me about"),
    (r"^is (.*) inherited", r"can \1 be passed on in families"),
    (r"^what are the genetic changes related to", "which genes cause"),
]
SYN = {"symptoms": "signs", "treatment": "therapy", "treatments": "therapies", "disease": "illness",
       "people": "persons", "children": "kids"}


def paraphrase(q: str) -> str:
    """Simulate how a patient might re-word a question (template change + synonyms + word drop)."""
    q = clean_text(q).rstrip("?").strip()
    for pat, rep in PARAPHRASES:
        if re.search(pat, q):
            q = re.sub(pat, rep, q, count=1)
            break
    words = [SYN.get(w, w) for w in q.split()]
    words = [w for w in words if not (len(w) <= 3 and random.random() < 0.3)]  # drop some short words
    return " ".join(words) + "?"


test_sample = test_df.sample(n=min(2000, len(test_df)), random_state=42)
queries = [paraphrase(q) for q in test_sample.Question]
metrics["retrieval_examples"] = [{"original": o, "paraphrased": p}
                                 for o, p in zip(test_sample.Question[:5], queries[:5])]


def token_f1(pred, gold):
    p, g = clean_text(pred).split(), clean_text(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


TEMPLATES = r"^(what (is|are) are|what (is|are) the (symptoms|treatments?|outlook|stages) (of|for)|what causes|" \
            r"who is at risk for|how many people are affected by|how to (diagnose|prevent)|is|" \
            r"what are the genetic changes related to|what (is|are) the complications of|" \
            r"what to do for|what research or clinical trials is being done for|what (is|are))\s+"


def focus(q: str) -> str:
    """Disease/topic the question is about (question template removed)."""
    q = re.sub(TEMPLATES, "", clean_text(q).rstrip("?").strip())
    return re.sub(r"\s+(inherited|\?)$", "", q).strip()


retr_results = {}
for name, use_char in (("TF-IDF (word)", False), ("Hybrid TF-IDF (word + char)", True)):
    retriever = MedicalRetriever(use_char_ngrams=use_char).fit(df)  # knowledge base = all MedQuAD
    hits = {1: 0, 3: 0, 5: 0}
    rr, f1s, topic_hits = [], [], 0
    t0 = time.time()
    for q, gold_q, gold_a in zip(queries, test_sample.Question, test_sample.Answer):
        res = retriever.search(q, top_k=5)
        rank = next((i + 1 for i, r in enumerate(res) if r["question"] == gold_q), None)
        for k in hits:
            hits[k] += int(rank is not None and rank <= k)
        rr.append(1 / rank if rank else 0)
        f1s.append(token_f1(res[0]["answer"], gold_a))
        topic_hits += int(focus(res[0]["question"]) == focus(gold_q))
    n = len(queries)
    retr_results[name] = {"hit@1": round(hits[1] / n, 4), "hit@3": round(hits[3] / n, 4),
                          "hit@5": round(hits[5] / n, 4), "mrr": round(float(np.mean(rr)), 4),
                          "topic_hit@1": round(topic_hits / n, 4),
                          "answer_token_f1": round(float(np.mean(f1s)), 4),
                          "avg_latency_ms": round((time.time() - t0) / n * 1000, 2)}
    print(f"  {name:28s} " + "  ".join(f"{k}={v}" for k, v in retr_results[name].items()))
metrics["retrieval_qa"] = retr_results

# generalisation check: classifier on re-worded questions (MedQuAD questions are template-based)
para_acc = accuracy_score(test_sample.qtype, best_model.predict(queries))
metrics["qtype_on_paraphrased_accuracy"] = round(para_acc, 4)
print(f"  Question-type accuracy on paraphrased questions: {para_acc:.4f}")

# ---------------------------------------------------------------- 3. Agent planner
print("\n[3] Agent task planner")
from src.agent import HealthcareAgent  # noqa: E402

PLAN_TESTS = [
    ("What are the symptoms of diabetes?", {"medical_qa"}),
    ("What causes high blood pressure?", {"medical_qa"}),
    ("How is asthma treated?", {"medical_qa"}),
    ("Is Huntington disease inherited?", {"medical_qa"}),
    ("Book an appointment with a cardiologist tomorrow at 10am", {"book_appointment"}),
    ("I want to schedule a visit to the skin doctor on friday", {"book_appointment"}),
    ("Please book me a consultation for my back pain", {"book_appointment"}),
    ("Show my appointments", {"list_appointments"}),
    ("Cancel appointment A1B2C3", {"cancel_appointment"}),
    ("Summarize my blood test report", {"summarize_report"}),
    ("Can you explain my lab results?", {"summarize_report"}),
    ("How can I prevent heart disease?", {"preventive"}),
    ("Give me tips to stay healthy", {"preventive"}),
    ("How do I reduce my risk of diabetes?", {"preventive"}),
    ("Hello", {"greeting"}),
    ("Summarize my report and then book an appointment with the right doctor",
     {"summarize_report", "book_appointment"}),
    ("What is glaucoma? Also book an eye check up tomorrow", {"medical_qa", "book_appointment"}),
    ("What are the symptoms of flu and how to prevent it?", {"medical_qa", "preventive"}),
    ("Explain my test results. Give me lifestyle tips too.", {"summarize_report", "preventive"}),
    ("What is anemia? Then schedule an appointment with general medicine on monday",
     {"medical_qa", "book_appointment"}),
]
agent = HealthcareAgent(api_key="")  # offline planner
correct = 0
for text, expected in PLAN_TESTS:
    got = {s["intent"] for s in agent.plan(text)}
    correct += int(got == expected)
    if got != expected:
        print(f"  MISS: {text!r} expected={expected} got={got}")
metrics["agent_planner"] = {"requests": len(PLAN_TESTS), "exact_plan_accuracy": round(correct / len(PLAN_TESTS), 4)}
print(f"  plan accuracy = {correct}/{len(PLAN_TESTS)}")

# ---------------------------------------------------------------- charts
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
names = list(clf_results)
axes[0].bar(names, [clf_results[n]["accuracy"] for n in names], color=["#94a3b8", "#60a5fa", "#0d9488"])
axes[0].set_ylim(0.8, 1.0); axes[0].set_title("Question-type classifier accuracy")
for i, n in enumerate(names):
    axes[0].text(i, clf_results[n]["accuracy"] + 0.003, f"{clf_results[n]['accuracy']:.3f}", ha="center")
keys = ["hit@1", "hit@3", "hit@5", "mrr", "topic_hit@1"]
x = np.arange(len(keys))
for j, (name, color) in enumerate(zip(retr_results, ["#94a3b8", "#0d9488"])):
    vals = [retr_results[name][k] for k in keys]
    axes[1].bar(x + (j - 0.5) * 0.38, vals, 0.38, label=name, color=color)
axes[1].set_xticks(x, keys); axes[1].set_ylim(0, 1); axes[1].legend(loc="lower right")
axes[1].set_title("Retrieval QA on paraphrased questions")
plt.tight_layout(); plt.savefig(f"{RESULTS}/model_comparison.png", dpi=150); plt.close()

plt.figure(figsize=(8, 4.5))
dist = df.qtype.value_counts()
plt.barh(dist.index[::-1], dist.values[::-1], color="#0d9488")
plt.title("MedQuAD - questions per type"); plt.tight_layout()
plt.savefig(f"{RESULTS}/class_distribution.png", dpi=150); plt.close()

# ---------------------------------------------------------------- final models on full data
print("\n[4] Training final models on full dataset and saving to models/")
save_models(MedicalRetriever().fit(df), QuestionTypeClassifier().fit(df.Question, df.qtype))
with open(f"{RESULTS}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Done. Metrics -> results/metrics.json, charts -> results/*.png")
