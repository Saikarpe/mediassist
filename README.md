# 🩺 MediAssist — LLM + Agentic AI Healthcare Assistant

A virtual healthcare assistant that can

1. **Answer medical questions**: retrieval-augmented answers grounded in the **MedQuAD** dataset (NIH sources)
2. **Schedule appointments**: check free slots, book, list and cancel
3. **Summarize health reports**: parse lab values, flag abnormal results, suggest a specialist
4. **Recommend preventive measures**: topic tips plus age-based screenings

An **agent** plans each request, calls the right tools on its own (in any order, several per message), and combines the results.
For example: *"Summarize my report and book the right doctor"* → summarize → reason (high sugar → Endocrinology) → book.

## Architecture

```
User ─► Streamlit UI ─► HealthcareAgent ──► Safety check (emergency keywords)
                              │
              ┌───────────────┴────────────────┐
        LLM mode (Gemini function calling)   Offline mode (rule planner)
              └───────────────┬────────────────┘
                     Tools (src/tools.py)
   search_medical_knowledge │ check_availability │ book/list/cancel_appointment
   summarize_health_report  │ get_preventive_recommendations
                              │
          MedQuAD models (src/knowledge_base.py)
   Hybrid TF-IDF retriever  +  Question-type classifier (LogReg)
```

* **LLM mode**: set `GEMINI_API_KEY` (or paste the key in the sidebar). Google Gemini (`gemini-3.5-flash` by default; change it with `GEMINI_MODEL` or in the sidebar) plans the steps, calls the tools through function calling, and writes the final answer from the MedQuAD passages.
* **Offline mode**: no key needed. A rule-based planner and the trained MedQuAD models handle everything, so the demo always works.

## Run it

```bash
pip install -r requirements.txt
set GEMINI_API_KEY=your-key-here     # optional (Windows; use `export` on Mac/Linux)
python train_evaluate.py      # trains + evaluates on MedQuAD, saves models/ and results/
streamlit run app.py
```

## Project structure

| Path | Purpose |
|---|---|
| `app.py` | Streamlit UI with Chat, Appointments, Report Summary, Model Evaluation and About tabs |
| `src/agent.py` | Agent: planner, function-calling loop (Gemini), offline executor, trace |
| `src/tools.py` | Appointment scheduler, report analyzer, preventive engine, safety |
| `src/knowledge_base.py` | MedQuAD loading, retriever, question-type classifier |
| `train_evaluate.py` | Training and evaluation; writes `results/metrics.json` and charts |
| `data/medquad.csv` | MedQuAD dataset (16,407 Q&A pairs, 14,978 after de-duplication) |
| `data/sample_report.txt` | Sample lab report for the demo |
| `presentation/MediAssist_Presentation.pptx` | Project presentation |

## Demo script

1. Chat: *"What are the symptoms of glaucoma?"* Open **Agent reasoning trace** to show the retrieval step.
2. Chat: *"Book an appointment with a cardiologist tomorrow at 11am"*, then open the Appointments tab.
3. Report tab: **Load sample report**. Abnormal values are highlighted and a specialist is suggested.
4. Tick **Attach report** in the sidebar, then chat: *"Summarize my report and book the right doctor"* (a multi-step plan).
5. Chat: *"How can I prevent heart disease?"*
6. Chat: *"I have chest pain"* to show the emergency safety guard.
7. Model Evaluation tab: walk through the metrics and charts.

> ⚠️ Educational project only. Not a substitute for professional medical advice.
