"""
Vortyx Pulse - Render Deployment
Production API with jailbreak guard + Hugging Face Inference API
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import re
import time
import requests

app = FastAPI(title="Vortyx Pulse API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "Anony7b/vortyx-pulse")

print("Loading Vortyx Pulse...")
print(f"   HF Model: {HF_MODEL_ID}")
print(f"   Jailbreak Guard: ACTIVE")

API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

def hf_generate(prompt, max_tokens=1000, temperature=0.7):
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "return_full_text": False
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "")
        return str(result)
    elif response.status_code == 503:
        raise Exception("HF API: Model is loading. Try again in a few seconds.")
    else:
        raise Exception(f"HF API Error {response.status_code}: {response.text}")

# Simple jailbreak guard
import re
class JailbreakGuard:
    def __init__(self):
        self.patterns = [
            r'ignore all previous instructions',
            r'DAN mode', r'jailbreak', r'do anything now',
            r'how to make (a )?bomb', r'how to hack',
            r'how to kill', r'steal (credit card|password)',
        ]
        self.regex = [re.compile(p, re.IGNORECASE) for p in self.patterns]
    
    def scan(self, text):
        for pattern in self.regex:
            if pattern.search(text):
                return False, f"Blocked: matched '{pattern.pattern}'"
        return True, None

guard = JailbreakGuard()

class CodeRequest(BaseModel):
    prompt: str
    language: str = "python"

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

class ReasoningRequest(BaseModel):
    problem: str

class AgentRequest(BaseModel):
    task: str
    context: Optional[str] = None

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Vortyx-Guard"] = "ACTIVE"
    return response

@app.get("/")
async def root():
    return {
        "name": "Vortyx Pulse API",
        "version": "1.0.0",
        "status": "operational",
        "security": "Jailbreak Guard ACTIVE",
        "model": HF_MODEL_ID
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model": HF_MODEL_ID}

@app.post("/code")
async def code(request: CodeRequest):
    is_safe, reason = guard.scan(request.prompt)
    if not is_safe:
        return {"success": False, "blocked": True, "reason": reason}
    system = f"You are Vortyx Pulse. Write clean {request.language} code. Wrap in markdown."
    prompt = f"{system}\n\nUser: {request.prompt}\n\nVortyx Pulse:"
    response = hf_generate(prompt, 1500, 0.3)
    code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', response, re.DOTALL)
    explanation = re.sub(r'```.*?```', '', response, flags=re.DOTALL).strip()
    return {"success": True, "code": code_blocks[0] if code_blocks else response, "explanation": explanation}

@app.post("/chat")
async def chat(request: ChatRequest):
    conversation = ""
    for msg in request.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system": conversation += f"System: {content}\n"
        elif role == "user": conversation += f"User: {content}\n"
        else: conversation += f"Vortyx Pulse: {content}\n"
    conversation += "Vortyx Pulse:"
    return {"success": True, "response": hf_generate(conversation, 1000, 0.7)}

@app.post("/reasoning")
async def reasoning(request: ReasoningRequest):
    is_safe, reason = guard.scan(request.problem)
    if not is_safe:
        return {"success": False, "blocked": True, "reason": reason}
    prompt = f"Solve step by step.\n\nProblem: {request.problem}\n\nSolution:"
    return {"success": True, "response": hf_generate(prompt, 2000, 0.5)}

@app.post("/agent")
async def agent(request: AgentRequest):
    combined = request.task + " " + (request.context or "")
    is_safe, reason = guard.scan(combined)
    if not is_safe:
        return {"success": False, "blocked": True, "reason": reason}
    prompt = f"Break down into steps.\n\nTask: {request.task}\n\nPlan:"
    return {"success": True, "response": hf_generate(prompt, 2000, 0.4)}

@app.post("/generate")
async def generate(request: GenerateRequest):
    is_safe, reason = guard.scan(request.prompt)
    if not is_safe:
        return {"success": False, "blocked": True, "reason": reason}
    return {"success": True, "response": hf_generate(request.prompt, request.max_tokens, request.temperature)}
