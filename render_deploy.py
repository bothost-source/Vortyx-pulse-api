"""
Vortyx Pulse - Render Deployment (Network-Fixed)
Multiple fallback strategies to bypass Render DNS issues
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import re
import socket
import time
import requests
import traceback
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Disable SSL warnings (Render sometimes has cert issues)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = FastAPI(title="Vortyx Pulse API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "Anony7b/vortyx-pulse")

# ========== DNS FIXES ==========
# Strategy 1: Try resolving with Google DNS
# Strategy 2: Use hardcoded IPs if DNS fails
# Strategy 3: Use alternative HF endpoints

# HF Inference API IPs (these change, but we try known ones)
HF_IP_FALLBACKS = [
    "34.120.195.249",   # Known HF inference IP (may change)
]

# Multiple endpoint strategies
ENDPOINT_STRATEGIES = [
    # Strategy 0: Direct DNS (normal)
    f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}",
    # Strategy 1: Google DNS resolution
    None,  # Will resolve dynamically
    # Strategy 2: Router endpoint
    f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_ID}",
    # Strategy 3: EU endpoint
    f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}",
]

print("=" * 60)
print("VORTYX PULSE API v4.0 - Network Fixed")
print("=" * 60)
print(f"HF Model: {HF_MODEL_ID}")
print(f"HF Token set: {bool(HF_API_TOKEN)}")

# Test DNS resolution
try:
    ip = socket.gethostbyname("api-inference.huggingface.co")
    print(f"DNS Status: api-inference.huggingface.co = {ip}")
except socket.gaierror as e:
    print(f"DNS Status: FAILED - {e}")
    print("Will use fallback strategies...")

print("=" * 60)

# Create session with retries
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

def resolve_hf_ip():
    """Try to resolve HF API IP using multiple methods."""
    # Method 1: Standard DNS
    try:
        ip = socket.gethostbyname("api-inference.huggingface.co")
        return ip
    except:
        pass

    # Method 2: Use Google DNS via socket
    try:
        import socket
        addrinfo = socket.getaddrinfo("api-inference.huggingface.co", None, socket.AF_INET)
        if addrinfo:
            return addrinfo[0][4][0]
    except:
        pass

    # Method 3: Fallback IPs
    for ip in HF_IP_FALLBACKS:
        return ip

    return None

def hf_generate(prompt, max_tokens=1000, temperature=0.7):
    """Call HF API with multiple fallback strategies."""
    if not HF_API_TOKEN:
        raise ValueError("HF_API_TOKEN environment variable is not set")

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

    last_error = None

    # Strategy 0: Direct URL
    try:
        print(f"Trying direct URL: {ENDPOINT_STRATEGIES[0]}")
        response = session.post(
            ENDPOINT_STRATEGIES[0], 
            headers=headers, 
            json=payload, 
            timeout=120,
            verify=False  # Skip SSL verification as fallback
        )
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            return str(result)
        elif response.status_code == 503:
            last_error = "Model loading..."
        else:
            last_error = f"Direct URL: HTTP {response.status_code}"
    except Exception as e:
        last_error = f"Direct URL failed: {str(e)}"
        print(last_error)

    # Strategy 1: IP-based (bypass DNS)
    try:
        ip = resolve_hf_ip()
        if ip:
            url = f"https://{ip}/models/{HF_MODEL_ID}"
            print(f"Trying IP-based: {url}")
            # Need to pass Host header for SSL to work
            headers_ip = headers.copy()
            headers_ip["Host"] = "api-inference.huggingface.co"
            response = session.post(
                url, 
                headers=headers_ip, 
                json=payload, 
                timeout=120,
                verify=False
            )
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "")
                return str(result)
            else:
                last_error = f"IP-based: HTTP {response.status_code}"
    except Exception as e:
        last_error = f"IP-based failed: {str(e)}"
        print(last_error)

    # Strategy 2: Router endpoint
    try:
        url = ENDPOINT_STRATEGIES[2]
        print(f"Trying router: {url}")
        response = session.post(
            url, 
            headers=headers, 
            json=payload, 
            timeout=120,
            verify=False
        )
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            return str(result)
        else:
            last_error = f"Router: HTTP {response.status_code}"
    except Exception as e:
        last_error = f"Router failed: {str(e)}"
        print(last_error)

    raise RuntimeError(f"All strategies failed. Last error: {last_error}")

# ========== EXCEPTION HANDLER ==========
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    error_msg = f"{type(exc).__name__}: {str(exc)}"
    print(f"ERROR: {error_msg}")
    print(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": error_msg}
    )

# ========== JAILBREAK GUARD ==========
class JailbreakGuard:
    def __init__(self):
        self.patterns = [
            r'ignore all previous instructions',
            r'DAN mode', r'jailbreak', r'do anything now',
            r'how to make (a )?bomb', r'how to hack',
            r'how to kill', r'steal (credit card|password)',
            r'create (a )?virus', r'how to (build|make) (a )?weapon',
            r'ignore (your )?instructions',
            r'pretend you are', r'roleplay as',
            r'forget (your )?rules', r'override (your )?safety',
        ]
        self.regex = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    def scan(self, text):
        for pattern in self.regex:
            if pattern.search(text):
                return False, f"Blocked: matched '{pattern.pattern}'"
        return True, None

guard = JailbreakGuard()

# ========== REQUEST MODELS ==========
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

# ========== MIDDLEWARE ==========
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Vortyx-Guard"] = "ACTIVE"
    response.headers["X-Vortyx-Version"] = "4.0.0"
    return response

# ========== ROUTES ==========
@app.get("/")
async def root():
    return {
        "name": "Vortyx Pulse API",
        "version": "4.0.0",
        "status": "operational",
        "security": "Jailbreak Guard ACTIVE",
        "model": HF_MODEL_ID,
        "mode": "network-fixed"
    }

@app.get("/health")
async def health():
    # Test DNS
    dns_ok = False
    try:
        socket.gethostbyname("api-inference.huggingface.co")
        dns_ok = True
    except:
        pass

    return {
        "status": "healthy", 
        "model": HF_MODEL_ID,
        "dns_working": dns_ok,
        "token_set": bool(HF_API_TOKEN)
    }

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
