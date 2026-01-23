import base64
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from groq import Groq
from gtts import gTTS
from pydantic import BaseModel

load_dotenv()

api = FastAPI(title="Legal Voice Assistant Backend")

client = Groq()

UPLOAD_DIR = Path("../user_uploaded_pdfs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-AI-Result", "X-Thread-ID"],
)


class GraphRequest(BaseModel):
    query: str
    doc_path: Optional[str] = None
    thread_id: Optional[str] = None


import httpx


async def run_graph_logic(query: str, thread_id: str, doc_path: Optional[str] = None):
    payload = {"query": query, "thread_id": thread_id, "doc_path": doc_path}

    timeout = httpx.Timeout(120.0, connect=60.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            print(f"--- Sending to Wrapper: {query[:50]}... ---")
            response = await client.post(
                "http://localhost:8787/run-legal-graph", json=payload
            )
            print(f"--- Received from Wrapper: {response.status_code} ---")
            response.raise_for_status()
            data = response.json()

            return data.get("result", {}).get(
                "final_response", "No response from graph."
            )
        except httpx.HTTPStatusError as e:
            print(f"Wrapper Server Error: {e.response.text}")
            return f"Error from legal graph: {e.response.status_code}"
        except Exception as e:
            print(f"Connection Error: {e}")
            return "Failed to connect to the legal graph server."


async def summarise_response(query: str, response: str):
    payload = {"query": query, "response": response}

    timeout = httpx.Timeout(120.0, connect=60.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            print(f"--- Sending to Wrapper Summariser: {response[:50]}... ---")
            response = await client.post(
                "http://localhost:8787/summarise", json=payload
            )
            print(f"--- Received from Wrapper: {response.status_code} ---")
            response.raise_for_status()
            data = response.json()

            return data.get("result", "No response from graph.")
        except httpx.HTTPStatusError as e:
            print(f"Wrapper Server Error: {e.response.text}")
            return f"Error from legal graph: {e.response.status_code}"
        except Exception as e:
            print(f"Connection Error: {e}")
            return "Failed to connect to the legal graph server."


# --- API Endpoints ---


@api.post("/api/text")
async def analyze_text(
    user_query: str = Form(...),
    thread_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    print(thread_id)
    current_thread = thread_id or str(uuid.uuid4())
    file_path = None

    if file:
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    try:
        response_text = await run_graph_logic(
            user_query, current_thread, str(file_path) if file_path else None
        )

        return {
            "status": "success",
            "result": response_text,
            "thread_id": current_thread,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/voice")
async def analyze_voice(
    audio_file: UploadFile = File(...),
    thread_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """
    Analyze voice from audio file.

    Parameters:
    audio_file (UploadFile): Audio file to be analyzed.
    thread_id (Optional[str]): Thread ID to be used.
    file (Optional[UploadFile]): File to be used.

    Returns:
    JSONResponse: A JSON response containing the thread ID, AI text, and audio file as base64.
    """
    current_thread = thread_id or str(uuid.uuid4())

    temp_audio_name = f"audio_{uuid.uuid4()}.wav"
    with open("./input_audio/" + temp_audio_name, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    try:
        print("GROQ REQUEST")
        transcription = ""
        with open("./input_audio/" + temp_audio_name, "rb") as audio:
            transcription = client.audio.transcriptions.create(
                file=("./input_audio/" + temp_audio_name, audio.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )
        print("GROQ RESPONSE")
        user_text = transcription.text
        print("user text : ", user_text)

        file_path = None
        if file:
            file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        ai_response_text = await run_graph_logic(
            user_text, current_thread, str(file_path) if file_path else None
        )

        # Summarise and Voice output
        short_response = await summarise_response(user_text, ai_response_text)
        tts = gTTS(text=short_response, lang="en")
        tts.save("./output_audio/" + temp_audio_name)
        print("GTTS SUCCESS")
        with open("./output_audio/" + temp_audio_name, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("ascii")

        return JSONResponse(
            {
                "thread_id": current_thread,
                "ai_text": ai_response_text,
                "audio_base64": audio_b64,
            }
        )
    except Exception as e:
        print(e)
        if os.path.exists("./input_audio/" + temp_audio_name):
            os.remove("./input_audio/" + temp_audio_name)
        if os.path.exists("./output_audio/" + temp_audio_name):
            os.remove("./output_audio/" + temp_audio_name)
        print("Error over groq or llm")
        raise HTTPException(status_code=500, detail=str(e))
