"""
MediAssist - AI Healthcare Conversational Assistant (Streamlit UI)

Run:  streamlit run app.py
"""
import json
import os

import pandas as pd
import streamlit as st

from src import tools
from src.agent import HealthcareAgent

ROOT = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="MediAssist - AI Healthcare Assistant", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
.hero {background: linear-gradient(90deg,#0f766e,#14b8a6); color:white; padding:18px 24px;
       border-radius:14px; margin-bottom:12px;}
.hero h1 {color:white; margin:0; font-size:1.8rem;}
.hero p {margin:4px 0 0 0; opacity:.9;}
.badge {display:inline-block; padding:3px 10px; border-radius:999px; font-size:.8rem;
        background:#ccfbf1; color:#115e59; font-weight:600;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading MedQuAD knowledge base...")
def get_agent(api_key: str, model: str):
    return HealthcareAgent(api_key=api_key, model=model)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🩺 MediAssist")
    st.caption("LLM + Agentic AI healthcare assistant")
    patient = st.text_input("Patient name", value="Rahul Sharma")
    age = st.number_input("Age", min_value=0, max_value=120, value=48)
    api_key = st.text_input("Gemini API key (optional)", type="password",
                            value=os.environ.get("GEMINI_API_KEY", ""),
                            help="Get a free key at aistudio.google.com. Without a key the assistant "
                                 "runs fully offline using the models trained on MedQuAD.")
    model = st.text_input("Gemini model", value=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"))
    agent = get_agent(api_key.strip(), model.strip())
    st.markdown(f"**Mode:** <span class='badge'>{agent.mode}</span>", unsafe_allow_html=True)
    attach = st.checkbox("Attach report (from Report tab) to chat", value=False)
    st.divider()
    st.markdown("**Try asking:**")
    examples = [
        "What are the symptoms of diabetes?",
        "How is high blood pressure treated?",
        "Book an appointment with a cardiologist tomorrow at 11am",
        "Summarize my report and book the right doctor",
        "How can I prevent heart disease?",
        "Show my appointments",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
            st.session_state.page = "💬 Chat"
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []

st.markdown("""<div class="hero"><h1>🩺 MediAssist</h1>
<p>Ask medical questions · Book appointments · Summarize health reports · Get preventive tips</p></div>""",
            unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "report_text" not in st.session_state:
    st.session_state.report_text = ""

PAGES = ["💬 Chat", "📅 Appointments", "📄 Report Summary", "📊 Model Evaluation", "ℹ️ About"]
page = st.segmented_control("Navigation", PAGES, default=PAGES[0], key="page",
                            label_visibility="collapsed") or PAGES[0]

# ------------------------------------------------------------------ chat
if page == PAGES[0]:
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🩺"):
            st.markdown(f"Hello **{patient}**! I'm MediAssist. I can **answer medical questions**, "
                        "**book appointments**, **summarize your health reports** and suggest "
                        "**preventive care**. You can combine requests, e.g. *\"Summarize my report and "
                        "book the right doctor\"*.")
    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🩺"):
            st.markdown(m["content"], unsafe_allow_html=True)
            if m.get("trace"):
                with st.expander("🧠 Agent reasoning trace"):
                    for i, step in enumerate(m["trace"], 1):
                        st.markdown(f"**{i}. {step['step']}** {step.get('detail', '')}")
                        if "input" in step or "output" in step:
                            st.json({k: step[k] for k in ("input", "output") if k in step}, expanded=False)

    prompt = st.chat_input("Type your health question or request...")
    if st.session_state.get("pending"):
        prompt = st.session_state.pop("pending")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🩺"):
            with st.spinner("Planning and executing..."):
                history = [{"role": m["role"], "content": m["content"]}
                           for m in st.session_state.messages[:-1]]
                reply, trace = agent.chat(prompt, history=history, patient_name=patient, age=age,
                                          report_text=st.session_state.report_text if attach else "")
        st.session_state.messages.append({"role": "assistant", "content": reply, "trace": trace})
        st.rerun()

# ------------------------------------------------------------------ appointments
if page == PAGES[1]:
    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.subheader("Book an appointment")
        with st.form("book"):
            spec = st.selectbox("Specialty", list(tools.DOCTORS))
            day = st.date_input("Date")
            avail = tools.check_availability(spec, day.isoformat())["available_slots"].get(day.isoformat(), [])
            time_slot = st.selectbox("Time slot", avail or ["No slots (Sunday)"])
            reason = st.text_input("Reason for visit")
            if st.form_submit_button("Book", type="primary") and avail:
                res = tools.book_appointment(patient, spec, day.isoformat(), time_slot, reason)
                st.success(f"Booked! ID {res['appointment']['id']} with {res['appointment']['doctor']} "
                           f"on {res['appointment']['date']} at {res['appointment']['time']}")
    with col2:
        st.subheader("Upcoming appointments")
        appts = tools.list_appointments()["appointments"]
        if appts:
            st.dataframe(pd.DataFrame(appts)[["id", "patient", "doctor", "specialty", "date", "time", "reason"]],
                         use_container_width=True, hide_index=True)
            cancel_id = st.selectbox("Cancel appointment", [a["id"] for a in appts])
            if st.button("Cancel selected"):
                tools.cancel_appointment(cancel_id)
                st.rerun()
        else:
            st.info("No appointments yet. Book one here or ask the assistant in the chat.")

# ------------------------------------------------------------------ report summary
if page == PAGES[2]:
    st.subheader("Health report summarizer")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        up = st.file_uploader("Upload report (.txt)", type=["txt"])
        if up:
            st.session_state.report_text = up.read().decode("utf-8", errors="ignore")
        if st.button("Load sample report"):
            with open(os.path.join(ROOT, "data", "sample_report.txt"), encoding="utf-8") as f:
                st.session_state.report_text = f.read()
        st.session_state.report_text = st.text_area("...or paste report text", st.session_state.report_text,
                                                    height=330)
    with c2:
        if st.session_state.report_text.strip():
            summary = tools.summarize_health_report(st.session_state.report_text)
            st.markdown(f"#### Overview\n{summary['overview']}")
            if summary["results"]:
                table = pd.DataFrame(summary["results"])[["test", "value", "unit", "normal_range", "status"]]
                table["value"] = table["value"].map(lambda v: f"{v:g}")
                st.dataframe(table.style.apply(
                    lambda r: ["background-color:#fee2e2" if r.status != "Normal" else "" for _ in r], axis=1),
                    use_container_width=True, hide_index=True)
            if summary["recommendations"]:
                st.markdown("#### Recommendations")
                for rec in summary["recommendations"]:
                    st.markdown(f"- {rec}")
            if summary["suggested_specialty"]:
                st.info(f"Suggested follow-up: **{summary['suggested_specialty']}** "
                        f"({tools.DOCTORS[summary['suggested_specialty']]})")
                if st.button(f"Book {summary['suggested_specialty']} appointment"):
                    res = tools.book_appointment(patient, summary["suggested_specialty"], "tomorrow", "10am",
                                                 "Follow-up on lab report")
                    a = res["appointment"]
                    st.success(f"Booked {a['doctor']} on {a['date']} at {a['time']} (ID {a['id']})")
            if agent.client and st.button("✨ Explain in simple words (LLM)"):
                with st.spinner("Gemini is writing a patient-friendly summary..."):
                    reply, _ = agent.chat("Explain my attached health report in simple words.",
                                          patient_name=patient, age=age,
                                          report_text=st.session_state.report_text)
                st.markdown(reply)
        else:
            st.info("Upload, paste, or load the sample report to see the summary.")

# ------------------------------------------------------------------ evaluation
if page == PAGES[3]:
    path = os.path.join(ROOT, "results", "metrics.json")
    if not os.path.exists(path):
        st.warning("Run `python train_evaluate.py` first to generate evaluation results.")
    else:
        m = json.load(open(path))
        d = m["dataset"]
        st.subheader("Dataset: MedQuAD (Medical Question Answering Dataset)")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Q&A pairs", f"{d['pairs']:,}")
        k2.metric("Train / Test", f"{d['train']:,} / {d['test']:,}")
        k3.metric("Question types", d["question_types"])
        k4.metric("Agent plan accuracy", f"{m['agent_planner']['exact_plan_accuracy']:.0%}")

        st.subheader("1 · Question-type classification")
        st.dataframe(pd.DataFrame(m["qtype_classification"]).T, use_container_width=True)
        if "qtype_on_paraphrased_accuracy" in m:
            st.caption(f"MedQuAD questions follow fixed templates, so accuracy is near-perfect on the test split. "
                       f"On re-worded (paraphrased) questions it drops to "
                       f"**{m['qtype_on_paraphrased_accuracy']:.1%}**, a known generalisation limit.")
        st.subheader("2 · Retrieval-augmented QA (paraphrased test questions)")
        st.dataframe(pd.DataFrame(m["retrieval_qa"]).T, use_container_width=True)
        img1, img2 = st.columns(2)
        for col, name in ((img1, "model_comparison.png"), (img2, "confusion_matrix.png")):
            p = os.path.join(ROOT, "results", name)
            if os.path.exists(p):
                col.image(p, use_container_width=True)
        p = os.path.join(ROOT, "results", "class_distribution.png")
        if os.path.exists(p):
            st.image(p, width=650)

# ------------------------------------------------------------------ about
if page == PAGES[4]:
    st.markdown("""
### How it works
1. **Understand & plan** – the agent splits the request into sub-tasks (QA, booking, report, prevention).
2. **Act** – it calls tools autonomously: MedQuAD retrieval, scheduler, report analyzer, prevention engine.
3. **Observe & reason** – tool outputs feed later steps (e.g. abnormal sugar → book *Endocrinology*).
4. **Respond** – the LLM (Google Gemini) or the offline composer writes a grounded, safe answer.

**Safety:** emergency keyword detection, no diagnosis/prescriptions, medical disclaimer on every answer.

**Tech:** Python · Streamlit · scikit-learn (TF-IDF, Logistic Regression) · Google Gemini API (function calling).
""")
