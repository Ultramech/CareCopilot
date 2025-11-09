from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import Optional, List
from supabase import create_client, Client
from Version3 import chat_with_persistence,process_and_store_uploaded_reports
import json
import re
import httpx
from datetime import datetime, timezone
from fastapi import BackgroundTasks
import os
from dotenv import load_dotenv
load_dotenv()

# ✅ FastAPI App
app = FastAPI(title="CareCopilot API", version="1.0")

# ✅ Supabase Credentials (Move to .env later)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = "patient_reports"

# ✅ Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ✅ Custom HTTP client to fix timeout issues
http_client = httpx.Client(
    timeout=httpx.Timeout(180.0, connect=30.0, read=180.0, write=180.0)
)

# ✅ Pydantic Models
class ChatRequest(BaseModel):
    session_id: str = "default_session"
    user_id: Optional[str] = None
    prompt: str

class UserRequest(BaseModel):
    full_name: str
    email: str
    role: str = "patient"
    age: Optional[int] = None
    gender: Optional[str] = None


@app.get("/")
def home():
    return {"message": "✅ CareCopilot API is running!"}


@app.post("/add_user")
def add_user(req: UserRequest):
    res = supabase.table("users").insert({
        "full_name": req.full_name,
        "email": req.email,
        "role": req.role,
        "age": req.age,
        "gender": req.gender
    }).execute()
    return {"status": "success", "data": res.data}

# ✅ Chat + File Upload Endpoint
@app.post("/chat")
async def chat(
    session_id: str = Form("default_session"),
    user_id: Optional[str] = Form(None),
    prompt: str = Form(...),
    upload_files: Optional[List[UploadFile]] = File(None)
):
    uploaded_files = []

    if upload_files:
        for file in upload_files:
            try:
                # ✅ 1. Sanitize filename
                safe_filename = re.sub(r'[^A-Za-z0-9._-]', '_', file.filename)
                file_path = f"{user_id or 'anonymous'}/{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{safe_filename}"
                file_data = await file.read()

                async with httpx.AsyncClient(timeout=180.0) as client:
                    response = await client.put(
                        f"{SUPABASE_URL}/storage/v1/object/{BUCKET_NAME}/{file_path}",
                        headers={
                            "apikey": SUPABASE_KEY,
                            "Authorization": f"Bearer {SUPABASE_KEY}",
                            "x-upsert": "true",
                            "Content-Type": file.content_type or "application/octet-stream",
                        },
                        content=file_data
                    )



                if response.status_code not in [200, 201]:
                    return {"error": f"Upload failed: {response.text}"}

                # ✅ 5. Public File URL (works because bucket is public)
                public_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{file_path}"

                # ✅ 6. Save metadata in DB
                supabase.table("patient_reports").insert({
                    "user_id": user_id,
                    "session_id": session_id,
                    "report_type": "uploaded",
                    "file_path": public_url,
                    "summary": None,
                    "structured_data": json.dumps({})
                }).execute()

                uploaded_files.append({"file_name": safe_filename, "file_url": public_url})

            except Exception as e:
                return {"error": f"File upload failed: {str(e)}"}
            
    if upload_files and user_id:
        process_and_store_uploaded_reports(session_id, user_id)


    # ✅ 7. Continue Chat Logic
    reply = chat_with_persistence(prompt, session_id, user_id)

    return {
        "response": reply,
        "uploaded_files": uploaded_files
    }


@app.get("/history/{session_id}")
def get_history(session_id: str):
    response = supabase.table("chat_history").select("*").eq("session_id", session_id).order("created_at").execute()
    return {"history": response.data}


@app.get("/medical_data/{session_id}")
def get_medical_data(session_id: str):
    response = supabase.table("patient_medical_data").select("*").eq("session_id", session_id).order("created_at").execute()
    return {"patient_data": response.data}