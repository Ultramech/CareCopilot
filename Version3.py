
import os
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import json
from typing import Optional,Dict,Any,List
from llama_index.core import Settings
from config import apply_settings
apply_settings()
import io
from llama_parse import LlamaParse
# from PyPDF2 import PdfReader
import requests
import os
from dotenv import load_dotenv
load_dotenv()

pinecone_api_key=os.getenv("pinecone_api_key")
import fitz  # PyMuPDF (pip install pymupdf)

def fallback_extract_text(pdf_bytes: bytes) -> str:
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception:
        return ""
    
def download_file_from_supabase(url: str) -> bytes:
    """
    Download file from a public Supabase URL and return raw bytes.
    """
    response = requests.get(url)
    response.raise_for_status()
    return response.content

def parse_uploaded_report(file_url: str, age=None, sex=None, stated_condition=None, symptoms_list=None, pregnant=None, meds_allergies=None):
    """
    Uses LlamaParse (preferred) or fallback PyPDF2 to extract report text and structured info.
    Returns dict: {"summary": ..., "structured_data": {...}}
    """
    # try:
    #     parser = LlamaParse(
    #         api_key=llamaCloud_api,
    #         result_type="text",      # or "markdown" if you prefer cleaner text
    #         max_timeout=600          # wait up to 10 minutes for cloud parse job
    #     )
    #     parsed_docs = parser.load_data(file_url)  # <-- IMPORTANT
    #     full_text = " ".join(d.text for d in parsed_docs if getattr(d, "text", None))
    # except Exception as e:
    #     full_text = ""
    #     print(f"[warn] LlamaParse failed: {e}")

    # if not full_text:

    file_name = file_url.split("/")[-1] or "report.pdf"
    try:
        pdf_bytes = download_file_from_supabase(file_url)
        parser = LlamaParse(api_key=llamaCloud_api)
        parsed_docs = parser.load_data(io.BytesIO(pdf_bytes), extra_info={"file_name": file_name})  # Pass file-like object
        # full_text = _fallback_extract_text(pdf_bytes)
        full_text = " ".join(d.text for d in parsed_docs if getattr(d, "text", None))
    except Exception as e:
        print(f"[warn] Fallback PDF extract failed: {e}")
        full_text = ""
    # if not full_text:
    # return {"summary": None, "structured_data": {}}

    # 🧠 Summarize the report using your LLM
    llm = Settings.llm
    prompt = f"""
You are a careful clinical information assistant. You give definitive diagnoses or prescribe medications. You summarize findings, highlight patterns, assess urgency, and suggest questions/tests to discuss with a clinician.

CONTEXT:
- age: {age}
- sex: {sex}
- pregnant: {pregnant}
- stated_condition: "{stated_condition}"
- symptoms: {symptoms_list}
- meds/allergies: {meds_allergies}

Report text (first 8000 characters):
{full_text[:8000]}

### ✅ Your tasks:
1. **Most Probable Diagnosis (Not Confirmed)**:
   - Provide the most likely condition(s) based on lab reports + symptoms.
   - Explain why this diagnosis fits.

2. **Differential Diagnosis**:
   - List 2–4 other possible conditions ranked by likelihood.

3. **How Lab Results + Symptoms Correlate**:
   - Interpret each abnormal lab value.
   - Connect it to the suspected condition.

4. **Recommended Treatment Approach**:
   -Be defininte , don't be general.
   - Both medical (if commonly prescribed) and non-medical (diet, lifestyle, hydration, rest, etc.)
   - If medication is typically used, mention it as: 
     *"Doctors commonly prescribe X class of medicine (e.g., NSAIDs, antibiotics, etc.), but exact prescription depends on a physician’s judgment."*

5. **When to Seek Urgent Medical Attention**:
   - Warn clearly if symptoms indicate emergency conditions.

6. **Suggested Follow-Up Tests** (if needed)
"""

    try:
        response = llm.complete(prompt)
        text = str(response)

        # print(response)

        import re, json
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            parsed = json.loads(m.group(0))
            return parsed
    except Exception as e:
        print(f"⚠️ LLM summarization failed: {e}")

    return {"summary": full_text[:600], "structured_data": {}}


from llama_index.core.llms import ChatMessage
from llama_index.core import Settings
from llama_index.core.callbacks import TokenCountingHandler

# Track token usage across chats
token_handler = TokenCountingHandler()
Settings.llm.callback_manager.add_handler(token_handler)

MAX_CONTEXT_TOKENS = 3200  # Your model context limit ≈ 3900. Keeping safe margin.

def get_total_tokens():
    """
    Returns total tokens used so far in prompt + output for the session.
    """
    prompt_tokens = token_handler.prompt_llm_token_count
    completion_tokens = token_handler.completion_llm_token_count
    return (prompt_tokens or 0) + (completion_tokens or 0)

# def summarize_history_if_needed(session_id: str, history: List[ChatMessage], chat_engine):
#     """
#     Summarizes old history into a compressed version when token size exceeds safe limit.
#     Returns updated history (NOT string).
#     """
#     # Convert history to text
#     history_text = "\n".join([f"User: {m.content}" if m.role == "user" else f"Assistant: {m.content}" for m in history])

#     if len(history_text) < MAX_CONTEXT_TOKENS:
#         return history  # ✅ Do nothing if inside limit

#     summary_prompt = """
#     Summarize the entire conversation between patient and AI in 8 bullet points.
#     Include:
#     - Symptoms, duration, severity
#     - Lab findings or medical reports discussed
#     - Age, gender, key clinical context
#     - Any suggestions or follow-up needed
#     Keep it short, medical, factual.
#     """

#     summary_response = chat_engine.chat(summary_prompt)
#     summary_text = str(summary_response)

#     # ✅ Replace entire history with compressed summary
#     new_history = [ChatMessage(role="assistant", content=f"Conversation summary: {summary_text}")]
#     chat_store.clear_history(session_id)
#     chat_store.add_message(session_id, "SYSTEM", summary_text)

#     return new_history

def summarize_history_if_needed(session_id: str, history):
    """
    Summarize chat history ONLY if it becomes too long.
    This does NOT use chat_engine (no retriever, no memory, no RAG).
    """
    if not history:
        return history

    # Convert to plain text format
    history_text = "\n".join(
        f"User: {m.content}" if m.role == "user" else f"Assistant: {m.content}"
        for m in history
    )

    # If within safe token length → return unchanged
    if len(history_text) < 3000:
        return history

    # ✅ Use only LLM, not ContextChatEngine
    llm = Settings.llm
    summary_prompt = """
    Summarize this medical conversation in 8 concise bullet points.
    Include symptoms, duration, concerns, previous advice, report discussions.
    Do NOT add assumptions or diagnoses.

    Conversation:
    """ + history_text[:8000]  # (Hard cap so we never overload)

    try:
        response = llm.complete(summary_prompt)
        summary = str(response)

        # Save summary to DB
        chat_store.add_message(
            session_id=session_id,
            user_msg="[SUMMARY GENERATED]",
            bot_msg=summary,
            user_id=None
        )

        # ✅ Reset history to just summary
        return [ChatMessage(role="assistant", content=summary)]

    except Exception as e:
        print("⚠️ Summarization failed:", e)
        return history  # fallback


def process_and_store_uploaded_reports(session_id: str, user_id: str):
    res = supabase.table("patient_reports").select("*").eq("session_id", session_id).eq("user_id", user_id).execute()
    if not res.data:
        return

    # ✅ Fetch latest saved symptoms/condition from patient_medical_data
    med_data = supabase.table("patient_medical_data") \
        .select("*") \
        .eq("session_id", session_id) \
        .order("created_at") \
        .execute()

    stated_condition = None
    symptoms_list = None
    if med_data.data:
        stated_condition = med_data.data[-1].get("symptoms")  # using symptoms as condition
        symptoms_list = med_data.data[-1].get("symptoms")

    # ✅ Get age & sex info
    user_info = get_user_details(user_id) if user_id else {}
    age = user_info.get("age")
    sex = user_info.get("gender")

    for report in res.data:
        if report.get("summary"):
            continue

        file_url = report["file_path"]
        parsed = parse_uploaded_report(
            file_url=file_url,
            age=age,
            sex=sex,
            stated_condition=stated_condition,
            symptoms_list=symptoms_list
        )
        # print("PROCESS AND STORE UPLOADED_REPORTS :",parsed)

        supabase.table("patient_reports").update({
            "summary": parsed.get("summary"),
            "structured_data": json.dumps(parsed.get("structured_data", {}))
        }).eq("id", report["id"]).execute()


# ---------- 1. LOAD DOCUMENTS ----------
# with open("all_docs.pkl", "rb") as f:
#     all_docs = pickle.load(f)

# ---------- 2. PINECONE + INDEX ----------
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext, VectorStoreIndex

pc = Pinecone(api_key=pinecone_api_key)
index_name = "carecopilot"

if index_name not in [i.name for i in pc.list_indexes()]:
    print("⚙️ Creating new Pinecone index...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

pinecone_index = pc.Index(index_name)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

from llama_index.readers.file import PDFReader
from llama_index.core import VectorStoreIndex

def extract_medical_info(user_input: str) -> Optional[Dict[str, Any]]:
    llm = Settings.llm  # ✅ Correct way to access the LLM
    if llm is None:
        return None
    
    prompt = f"""
Extract structured medical information from the user's message. 
Return ONLY JSON.

Fields:
- stated_condition: (disease or concern user believes they have)
- symptoms: (list or string)
- duration: (how long it has been)
- severity: (0-10 or mild/moderate/severe)
- medical_history: (diabetes, thyroid, heart disease, etc.)
- medications: (if mentioned)
- allergies: (if mentioned)
- lifestyle: (smoking, alcohol, exercise)

If unknown → set value to null.

User input:
\"\"\"{user_input}\"\"\"

Return JSON only:
"""

    try:
        resp = llm.complete(prompt)
        text = str(resp)
        # Try direct JSON parse
        try:
            return json.loads(text)
        except Exception:
            # Try to extract from code fences if present
            import re
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                return json.loads(m.group(0))
    except Exception:
        return None
    return None

def save_patient_medical_data(session_id: str, user_id: Optional[str], data: Dict[str, Any]) -> None:
    """
    Saves extracted medical info to patient_medical_data table if it exists.
    """
    payload = {
        "session_id": session_id,
        "user_id": user_id,
        "symptoms": data.get("symptoms"),
        "duration": data.get("duration"),
        "medical_history": data.get("medical_history"),
        "medications": data.get("medications"),
        "allergies": data.get("allergies"),
        "lifestyle": data.get("lifestyle"),
    }
    try:
        supabase.table("patient_medical_data").insert(payload).execute()
    except Exception as e:
        # Table might not exist yet; continue without crashing
        print(f"[warn] Could not insert into patient_medical_data: {e}")


# Check if first run
if pc.describe_index(index_name)["total_vector_count"] == 0:
    print("📥 Inserting documents into Pinecone...")
    # index = VectorStoreIndex.from_documents(all_docs, storage_context=storage_context)
else:
    print("✅ Pinecone index already has data – using it.")
    index = VectorStoreIndex.from_vector_store(vector_store)

retriever = index.as_retriever(similarity_top_k=5)

# ---------- 3. SUPABASE ----------
from supabase import create_client

llamaCloud_api=os.getenv("llamaCloud_api")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY =os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------- 4. CHAT MEMORY ----------
from llama_index.core.llms import ChatMessage
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import Memory

system_prompt = """
You are CareCopilot — a cautious, ethical, patient-focused medical assistant for informational support only.

CONTEXT YOU MAY HAVE
- Prior conversation turns in this session.
- Patient profile if provided (age, gender, key history). If age/gender are known, tailor wording appropriately (e.g., “Since you are a 55-year-old male…”). If unknown and relevant, ask briefly.

CORE BEHAVIOR
1) Ask before answering: If the user reports symptoms or a condition, first ask focused follow-up questions to close critical gaps, instead of jumping to conclusions.
2) Be concise, empathetic, and plain-language. Avoid jargon unless asked.
3) Safety first: If red-flag features are present, clearly advise urgent in-person evaluation.
4) Retrieval-augmented: When answering factual questions (not personal triage), use the retrieved context; if not enough evidence, say so.
5) Boundaries: Decline non-medical or unsafe requests and redirect to health topics.
6) No hidden reasoning disclosure: Do your reasoning internally and share only conclusions, questions, and guidance.

WHAT TO COLLECT BEFORE ANALYSIS (when symptoms are reported)
- Onset & duration
- Severity (0–10), pattern (constant/intermittent), triggers/relievers
- Key associated symptoms
- Age & gender (if relevant and unknown)
- Past medical history; surgeries; relevant family history
- Current medications & allergies
- Recent exposures/travel/injuries; pregnancy status if relevant

RED-FLAG SCREEN (examples; not exhaustive)
- Chest pain with shortness of breath, radiation to arm/jaw/back, sweating, nausea, fainting
- Sudden weakness/numbness on one side, facial droop, slurred speech, worst headache of life, confusion
- Severe shortness of breath, blue lips, oxygen device failure
- High fever with stiff neck or rash; severe dehydration; persistent vomiting
When such features are reported or strongly suspected: state the concern and advise urgent medical care.

ANSWER STYLE (pick the minimal necessary sections)
- If information is insufficient: 
  - “I need a bit more info to help.” Then list 3–6 targeted questions (bulleted).
- If enough info to give general guidance (not diagnosis):
  1) Summary (1–2 lines personalized; reference age/gender if known)
  2) What it could relate to (non-diagnostic possibilities, hedged)
  3) What you can do now (self-care / monitoring steps if appropriate)
  4) When to seek care (clear thresholds; include emergency signs if applicable)
  5) Sources (brief cites if retrieved context was used)
- Keep responses under ~12 concise sentences unless the user asks for more depth.

EXAMPLES
User: “I have chest tightness and left arm discomfort.”
Assistant (follow-up first):
- “Thanks for sharing—I'd like a bit more detail:
  • When did this start and how long does each episode last?
  • On a 0–10 scale, how intense is it? Does it feel pressure, sharp, or burning?
  • Does it worsen with exertion or deep breaths?
  • Any sweating, nausea, shortness of breath, or dizziness?
  • Do you have a history of heart disease, diabetes, high blood pressure, or high cholesterol?”
(If red flags confirmed, advise urgent care clearly.)

User: “What is COPD? (general info)”
Assistant:
- Provide a short definition using retrieved context; list typical symptoms, risk factors, and standard evaluation steps. Include brief, plain-language citations from retrieved documents.

OUT-OF-SCOPE & SAFETY
- Non-medical requests → politely decline and redirect to health topics.
- Self-harm, illegal drug advice, or dangerous instructions → refuse and provide crisis resources or safe alternatives where appropriate.

ALWAYS
- Be kind and non-judgmental.
- Prefer short bullet lists over long paragraphs.
- If you’re uncertain, say what you would need to be more confident.
"""

def get_user_details(user_id: str):
    res = supabase.table("users").select("age, gender").eq("user_id", user_id).execute()
    if res.data:
        return res.data[0]
    return None

class SupabaseChatStore:
    def add_message(self, session_id: str, user_msg: str, bot_msg: str, user_id: str = None):
        supabase.table("chat_history").insert({
            "session_id": session_id,
            "user_id": user_id,
            "user_message": user_msg,
            "bot_message": bot_msg
        }).execute()

    def get_messages(self, session_id: str):
        response = supabase.table("chat_history") \
            .select("user_message, bot_message") \
            .eq("session_id", session_id) \
            .order("created_at") \
            .execute()
        messages = []
        for row in response.data:
            if row["user_message"]:
                messages.append(ChatMessage(role="user", content=row["user_message"]))
            if row["bot_message"]:
                messages.append(ChatMessage(role="assistant", content=row["bot_message"]))
        return messages

chat_store = SupabaseChatStore()

from typing import List, Optional
from fastapi import UploadFile

def latest_report_summary(session_id: str, user_id: Optional[str]) -> Optional[str]:
    try:
        res = supabase.table("patient_reports") \
            .select("summary") \
            .eq("session_id", session_id) \
            .eq("user_id", user_id) \
            .neq("summary", None) \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()
        print(res)
        if res.data and res.data[0].get("summary"):
            return res.data[0]["summary"]
    except Exception as e:
        print(f"[warn] fetching latest summary failed: {e}")
    return None

# def chat_with_persistence(prompt: str, session_id: str = "default_session", user_id: str = None):
#     history = chat_store.get_messages(session_id) or []

#     user_info = get_user_details(user_id) if user_id else None

#     # if user_id:
#     #     process_and_store_uploaded_reports(session_id, user_id)
    
#     memory = Memory.from_defaults(
#         session_id=session_id,
#         token_limit=100000,
#         chat_history=history
#     )
#     personalized_prompt = system_prompt
#     if user_info:
#         age = user_info.get("age")
#         gender = user_info.get("gender")
#         if age or gender:
#             personalized_prompt += f"\nPatient Info: Age = {age}, Gender = {gender}."

#     latest_summary = latest_report_summary(session_id, user_id)
#     print("LATEST SUMMARY :",latest_summary)

#     effective_prompt = prompt
#     if latest_summary:
#         effective_prompt = f"""

#     {personalized_prompt}
#     Patient's question:
#     {prompt}

#     The patient's lab report has already been analyzed. Here is the summary:
#     ---
#     {latest_summary}
#     ---

#     1. Interpret the lab report in context of symptoms, age, and gender.
#     2. Identify abnormal values, Diagnose them in context with the user provided symptoms.
#     3. Explain if the symptoms and lab abnormalities are possibly related.
#     4. Suggest safe next steps:
#     - Lifestyle advice (if relevant)
#     - Follow-up blood tests or imaging (if needed)
#     - When the patient should see a doctor or seek urgent care
#     Also, tell the user which disease it can be , be specific , don't give general answers.
#     5. If findings are unclear or insufficient: clearly say so and mention which tests could help.
#     6. Do *not* ignore the lab report or patient context. If unsure, say “I’m not fully sure; further evaluation is recommended.”

#     Keep your response clear, respectful, and within informational boundaries (no prescribing medication).
#     """
#     else:
#         effective_prompt = prompt
#     chat_engine = ContextChatEngine.from_defaults(
#         retriever=retriever,
#         memory=memory,
#         system_prompt=personalized_prompt,
#         streaming=False
#     )
#     history = summarize_history_if_needed(session_id, history, chat_engine)

#     extracted = extract_medical_info(effective_prompt)
#     print("EXTRACTED : " ,extracted)
#     if extracted and extracted.get("symptoms"):
#         save_patient_medical_data(session_id, user_id, extracted)

#     print("Effective_prompt: ",effective_prompt)
#     response = chat_engine.chat(effective_prompt)
#     chat_store.add_message(session_id, effective_prompt, str(response), user_id)
#     return str(response)

def chat_with_persistence(prompt: str, session_id="default_session", user_id=None):
    # 1️⃣ Load full history from DB
    all_messages = chat_store.get_messages(session_id) or []

    # 2️⃣ Fetch user details (age/gender)
    user_info = get_user_details(user_id) if user_id else None

    # 3️⃣ Build personalized system prompt (ONLY here – don't repeat elsewhere)
    personalized_prompt = system_prompt
    if user_info:
        age = user_info.get("age")
        gender = user_info.get("gender")
        if age or gender:
            personalized_prompt += f"\nPatient Info: Age: {age}, Gender: {gender}."

    # ✅ Summarize if token load exceeds 3000 tokens
    reduced_messages = summarize_history_if_needed(session_id, all_messages)

    # 5️⃣ Now build final memory using summarized/chat-trimmed messages
    memory = Memory.from_defaults(
        session_id=session_id,
        chat_history=reduced_messages,
        token_limit=3900  # matches model capacity
    )

    chat_engine = ContextChatEngine.from_defaults(
        retriever=retriever,
        memory=memory,
        system_prompt=personalized_prompt,
        streaming=False
    )

    # 6️⃣ Add lab report summary if available
    latest_summary = latest_report_summary(session_id, user_id)
    if latest_summary:
        latest_summary = latest_summary[:1500]  # truncate if very long
        effective_prompt = f"""
Patient asks: {prompt}

Recent Lab Report Summary:
--------------------------
{latest_summary}

Using this summary + patient age/gender/symptoms, answer:
1. Key lab abnormalities and meaning.
2. How they relate to symptoms.
3. Possible conditions (NOT a diagnosis).
4. Safe next steps or follow-up tests.
"""
    else:
        effective_prompt = prompt

    # ✅ Extract symptoms/medical info & store
    extracted = extract_medical_info(prompt)
    if extracted and extracted.get("symptoms"):
        save_patient_medical_data(session_id, user_id, extracted)

    # ✅ Final chat response
    response = chat_engine.chat(effective_prompt)

    # ✅ Save message to DB
    chat_store.add_message(session_id, prompt, str(response), user_id)

    return str(response)
