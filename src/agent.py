"""
Agentic healthcare assistant.

The agent follows a  PLAN -> ACT (tools) -> OBSERVE -> RESPOND  loop.

Two interchangeable "brains":
  * LLM mode     - Google Gemini (function calling) decides which tools to call and writes
                   the final answer, grounded on MedQuAD passages (Retrieval-Augmented Generation).
  * Offline mode - a rule-based planner + the models trained on MedQuAD. Used when no
                   GEMINI_API_KEY is set, so the project can always be demonstrated.

Every step is recorded in `trace` so the UI can show the agent's reasoning.
"""
import json
import os
import re
from datetime import date

from . import tools
from .knowledge_base import load_models

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
DISCLAIMER = ("_I am an AI assistant, not a doctor. This information is educational - "
              "please consult a healthcare professional for diagnosis or treatment._")

SYSTEM_PROMPT = """You are MediAssist, a friendly virtual healthcare assistant for a hospital.
You help patients with four things: answering medical questions, scheduling appointments,
summarizing health/lab reports, and recommending basic preventive measures.

How to work:
- First think about which steps the request needs, then call the tools to carry them out.
  A single message may need several tools (e.g. summarize a report, then book a doctor).
- For any medical question ALWAYS call search_medical_knowledge first and base your answer on
  the returned MedQuAD passages. If the passages are not relevant, say so honestly.
- To book, you need the patient name, specialty and preferred day/time. The patient name is
  given in the context; if day/time are missing, pick the earliest free slot.
- If a report summary suggests a specialty, offer to book (or book if the user asked).
- Never diagnose or prescribe medication doses. For emergencies tell the user to call
  emergency services immediately.
- Answer in simple language, use short paragraphs or bullet points, and keep it under 250 words."""

TOOL_SCHEMAS = [
    {
        "name": "search_medical_knowledge",
        "description": "Search the MedQuAD medical question-answering knowledge base (NIH sources). "
                       "Returns the most similar questions with their expert answers.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The medical question"},
                           "top_k": {"type": "integer", "description": "Number of passages (1-5)"}},
            "required": ["query"],
        },
    },
    {
        "name": "check_availability",
        "description": "List free appointment slots for a medical specialty, optionally on a given day "
                       "('tomorrow', 'monday', '2026-10-10').",
        "input_schema": {
            "type": "object",
            "properties": {"specialty": {"type": "string"}, "day": {"type": "string"}},
            "required": ["specialty"],
        },
    },
    {
        "name": "book_appointment",
        "description": "Book an appointment. Specialty can be a specialty name or a symptom/organ "
                       "(e.g. 'heart', 'skin'). If the requested time is taken, the nearest free slot is used.",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_name": {"type": "string"},
                "specialty": {"type": "string"},
                "day": {"type": "string", "description": "e.g. 'tomorrow', 'friday', '2026-10-10'"},
                "time": {"type": "string", "description": "e.g. '10am', '15:30'"},
                "reason": {"type": "string"},
            },
            "required": ["patient_name", "specialty"],
        },
    },
    {
        "name": "list_appointments",
        "description": "List the confirmed appointments of a patient.",
        "input_schema": {"type": "object", "properties": {"patient_name": {"type": "string"}}},
    },
    {
        "name": "cancel_appointment",
        "description": "Cancel an appointment by its ID.",
        "input_schema": {"type": "object", "properties": {"appointment_id": {"type": "string"}},
                         "required": ["appointment_id"]},
    },
    {
        "name": "summarize_health_report",
        "description": "Parse a lab/health report, compare each value with normal reference ranges "
                       "and flag abnormal results.",
        "input_schema": {"type": "object", "properties": {"report_text": {"type": "string"}},
                         "required": ["report_text"]},
    },
    {
        "name": "get_preventive_recommendations",
        "description": "Preventive health tips for a topic (diabetes, heart, hypertension, obesity, "
                       "cancer, infection, mental health, general) plus age-based screenings.",
        "input_schema": {
            "type": "object",
            "properties": {"topic": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["topic"],
        },
    },
]


def shorten(text: str, max_sentences: int = 5) -> str:
    text = re.sub(r"^\s*Espa\S*ol\s+", "", str(text))  # stray language-link label in some answers
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out = " ".join(sentences[:max_sentences])
    out += " ..." if len(sentences) > max_sentences else ""
    # MedQuAD writes lists inline ("are: - thirst  - fatigue"); turn them into markdown bullets
    return re.sub(r"(:|\s{2})\s*-\s+", lambda m: (":" if m.group(1) == ":" else "") + "\n- ", out)


class HealthcareAgent:
    def __init__(self, api_key: str = None, model: str = None):
        self.retriever, self.qtype_clf = load_models()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model = model or DEFAULT_MODEL
        self.client = None
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None

    @property
    def mode(self):
        return f"LLM (Gemini: {self.model}) + Agent" if self.client else "Offline (MedQuAD models + rule planner)"

    # ------------------------------------------------------------------ tools
    def run_tool(self, name: str, args: dict):
        if name == "search_medical_knowledge":
            hits = self.retriever.search(args["query"], top_k=min(int(args.get("top_k", 3)), 5))
            qtype, conf = self.qtype_clf.predict_one(args["query"])
            for h in hits:
                h["answer"] = shorten(h["answer"], 8)
            return {"predicted_question_type": qtype, "type_confidence": round(conf, 2), "passages": hits}
        fn = {
            "check_availability": tools.check_availability,
            "book_appointment": tools.book_appointment,
            "list_appointments": tools.list_appointments,
            "cancel_appointment": tools.cancel_appointment,
            "summarize_health_report": tools.summarize_health_report,
            "get_preventive_recommendations": tools.get_preventive_recommendations,
        }[name]
        return fn(**args)

    # ------------------------------------------------------------------ entry point
    def chat(self, message: str, history=None, patient_name: str = "Patient", age: int = 0,
             report_text: str = ""):
        """Returns (reply_markdown, trace). `trace` is a list of agent steps."""
        if tools.detect_emergency(message):
            return tools.EMERGENCY_MESSAGE, [{"step": "Safety check", "detail": "Emergency keywords detected - escalated."}]
        if self.client:
            try:
                return self._chat_llm(message, history or [], patient_name, age, report_text)
            except Exception as e:  # network / auth problems -> graceful fallback
                reply, trace = self._chat_offline(message, patient_name, age, report_text)
                trace.insert(0, {"step": "LLM unavailable", "detail": f"{type(e).__name__}: {e}. Used offline mode."})
                return reply, trace
        return self._chat_offline(message, patient_name, age, report_text)

    # ------------------------------------------------------------------ LLM mode
    def _chat_llm(self, message, history, patient_name, age, report_text):
        from google.genai import types

        context = f"[Patient name: {patient_name}; age: {age or 'unknown'}; today: {date.today()}]"
        text = f"{context}\n{message}"
        if report_text:
            text += f"\n\n--- Attached health report ---\n{report_text}"
        contents = [types.Content(role="user" if m["role"] == "user" else "model",
                                  parts=[types.Part.from_text(text=m["content"])])
                    for m in history[-10:]]
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t["description"],
                                          parameters_json_schema=t["input_schema"])
                for t in TOOL_SCHEMAS])],
            # we run the tool loop ourselves so every step can be shown in the trace
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0.3,
        )
        trace = [{"step": "Plan", "detail": f"Gemini ({self.model}) reads the request and decides which tools to call."}]

        for _ in range(8):  # safety cap on agent iterations
            response = self.client.models.generate_content(model=self.model, contents=contents, config=config)
            calls = response.function_calls or []
            if not calls:
                answer = (response.text or "").strip() or "Sorry, I could not generate an answer."
                trace.append({"step": "Respond", "detail": "Gemini composed the final answer from tool results."})
                return f"{answer}\n\n{DISCLAIMER}", trace

            contents.append(response.candidates[0].content)  # the model's function-call turn
            parts = []
            for call in calls:
                args = dict(call.args or {})
                try:
                    out = self.run_tool(call.name, args)
                except Exception as e:
                    out = {"error": str(e)}
                trace.append({"step": f"Act: {call.name}", "input": args, "output": out})
                parts.append(types.Part.from_function_response(
                    name=call.name, response={"result": json.loads(json.dumps(out, default=str))}))
            contents.append(types.Content(role="user", parts=parts))
        return "Sorry, I could not complete the request in time.", trace

    # ------------------------------------------------------------------ Offline mode
    INTENT_RULES = [
        ("cancel_appointment", r"\bcancel\b"),
        ("list_appointments", r"(my appointments|show .*appointment|list .*appointment|upcoming appointment|view .*appointment)"),
        ("book_appointment", r"\b(book|schedule|appointment|consult(ation)?|see a doctor|meet a doctor)\b"),
        ("summarize_report", r"\b(report|lab results?|blood test|test results?)\b"),
        ("preventive", r"\b(prevent\w*|avoid|tips|stay healthy|lifestyle|reduce (my )?risk|healthy habits)\b"),
        ("greeting", r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you)\b[\s!.]*$"),
    ]

    def plan(self, message: str, report_text: str = ""):
        """Split the request into clauses and map each clause to an intent -> ordered plan."""
        clauses = [c.strip() for c in re.split(r"(?<=[.?!;])\s+|,?\s+(?:and then|then|and also|also)\s+",
                                               message) if c and c.strip()]
        steps = []
        for clause in clauses:
            low = clause.lower()
            matched = [intent for intent, pat in self.INTENT_RULES if re.search(pat, low)]
            if "cancel_appointment" in matched or "list_appointments" in matched:
                matched = [m for m in matched if m != "book_appointment"]
            if matched == ["preventive"] and re.search(
                    r"\b(what (is|are)|symptoms?|causes?|treat\w*|diagnos\w*)\b", low):
                matched.append("medical_qa")
            if not matched:
                matched = ["medical_qa"]
            for m in matched:
                steps.append({"intent": m, "text": clause})
        if report_text and not any(s["intent"] == "summarize_report" for s in steps):
            steps.insert(0, {"intent": "summarize_report", "text": "attached report"})
        # execution order: report first (its output can drive booking), booking last
        order = {"greeting": 0, "summarize_report": 1, "medical_qa": 2, "preventive": 3,
                 "list_appointments": 4, "cancel_appointment": 5, "book_appointment": 6}
        return sorted(steps, key=lambda s: order[s["intent"]])

    def _chat_offline(self, message, patient_name, age, report_text):
        plan = self.plan(message, report_text)
        trace = [{"step": "Plan", "detail": " -> ".join(s["intent"] for s in plan)}]
        parts, memory = [], {}

        for s in plan:
            intent, text = s["intent"], s["text"]
            if intent == "greeting":
                parts.append(f"Hello {patient_name}! I can answer medical questions, book appointments, "
                             "summarize your health reports and share preventive tips. How can I help?")

            elif intent == "summarize_report":
                if not report_text:
                    parts.append("Please paste or upload your report in the **Report Summary** tab "
                                 "(or attach it in the sidebar) so I can summarize it.")
                    continue
                out = self.run_tool("summarize_health_report", {"report_text": report_text})
                trace.append({"step": "Act: summarize_health_report", "output": out})
                memory["report"] = out
                lines = [f"**Report summary:** {out['overview']}"]
                for r in out["results"]:
                    if r["status"] != "Normal":
                        lines.append(f"- **{r['test']}**: {r['value']} {r['unit']} ({r['status']}, normal "
                                     f"{r['normal_range']}) - {r['advice']}")
                if out["suggested_specialty"]:
                    lines.append(f"\nSuggested follow-up: **{out['suggested_specialty']}**.")
                parts.append("\n".join(lines))

            elif intent == "medical_qa":
                out = self.run_tool("search_medical_knowledge", {"query": text, "top_k": 3})
                trace.append({"step": "Act: search_medical_knowledge", "input": {"query": text},
                              "output": {"predicted_question_type": out["predicted_question_type"],
                                         "top_matches": [(p["question"], p["score"]) for p in out["passages"]]}})
                best = out["passages"][0]
                if best["score"] < 0.25:
                    parts.append("I couldn't find a reliable answer for that in my medical knowledge base. "
                                 "Please rephrase or consult a doctor.")
                else:
                    parts.append(f"**{best['question']}**\n\n{shorten(best['answer'], 5)}\n\n"
                                 f"<sub>Source: MedQuAD (NIH) | question type: "
                                 f"{out['predicted_question_type']} | match score: {best['score']}</sub>")

            elif intent == "preventive":
                topic = text
                if topic.strip().lower() in ("tips", "") and memory.get("report"):
                    topic = " ".join(r["test"] for r in memory["report"]["results"] if r["status"] != "Normal")
                out = self.run_tool("get_preventive_recommendations", {"topic": topic, "age": age})
                trace.append({"step": "Act: get_preventive_recommendations", "input": {"topic": topic, "age": age},
                              "output": out})
                lines = [f"**Preventive tips ({', '.join(out['topics'])}):**"] + [f"- {t}" for t in out["tips"]]
                if out["age_based_screenings"]:
                    lines += ["", f"**Recommended screenings for age {age}:**"] + \
                             [f"- {t}" for t in out["age_based_screenings"]]
                parts.append("\n".join(lines))

            elif intent == "book_appointment":
                specialty = tools.resolve_specialty(text)
                if specialty == "General Medicine" and memory.get("report", {}).get("suggested_specialty"):
                    specialty = memory["report"]["suggested_specialty"]  # autonomous decision from report
                    trace.append({"step": "Reason", "detail": f"No specialty given; using {specialty} suggested by the report."})
                args = {"patient_name": patient_name, "specialty": specialty, "day": text,
                        "time": tools.parse_time(text) or "", "reason": text}
                out = self.run_tool("book_appointment", args)
                trace.append({"step": "Act: book_appointment", "input": args, "output": out})
                if out["status"] == "confirmed":
                    a = out["appointment"]
                    parts.append(f"**Appointment booked** (ID `{a['id']}`): {a['doctor']} ({a['specialty']}) "
                                 f"on **{a['date']} at {a['time']}**.")
                else:
                    parts.append(out["message"])

            elif intent == "list_appointments":
                out = self.run_tool("list_appointments", {"patient_name": patient_name})
                trace.append({"step": "Act: list_appointments", "output": out})
                appts = out["appointments"]
                parts.append("You have no upcoming appointments." if not appts else
                             "**Your appointments:**\n" + "\n".join(
                                 f"- `{a['id']}` {a['date']} {a['time']} - {a['doctor']} ({a['specialty']})"
                                 for a in appts))

            elif intent == "cancel_appointment":
                m = re.search(r"\b([A-F0-9]{6})\b", text.upper())
                if not m:
                    parts.append("Please give the 6-character appointment ID to cancel (see the Appointments tab).")
                    continue
                out = self.run_tool("cancel_appointment", {"appointment_id": m.group(1)})
                trace.append({"step": "Act: cancel_appointment", "output": out})
                parts.append(f"Appointment `{m.group(1)}` has been cancelled." if out["status"] == "cancelled"
                             else out["message"])

        # proactive suggestion: abnormal report but no booking requested
        if memory.get("report", {}).get("abnormal_count") and not any(s["intent"] == "book_appointment" for s in plan):
            parts.append(f"Would you like me to book an appointment with **{memory['report']['suggested_specialty']}**?")
        trace.append({"step": "Respond", "detail": f"Combined {len(parts)} result(s) into one reply."})
        return "\n\n---\n\n".join(parts) + f"\n\n{DISCLAIMER}", trace
