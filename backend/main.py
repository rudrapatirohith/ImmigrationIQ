# main.py — your entire FastAPI app starts here
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from services.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate

from api.classify import router as classify_roter
from api.chat_v2 import router as chat_v2_router

app = FastAPI(
    title="ImmigrationIQ API",
    description="AI-powered immigration guidance",
    version="0.2.0"
)

# CORS — allow your Next.js frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add production URL later
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register the routers
# This tells FastAPI: "all routes defined in classify_router exist in this app"
# With prefix="/classify", the endpoint becomes POST /classify/
# With prefix="/chat", the endpoint becomes POST /chat/v2

app.include_router(classify_roter) # For the classification endpoint
app.include_router(chat_v2_router) # For the multi-turn chat endpoint


# ---- Models (Pydantic) ----
class ChatRequest(BaseModel):
    message: str
    session_id: str
    user_situation: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    sources: list[str] = []
    session_id: str

# ---- Routes ----
@app.get("/health")
async def health():
    """Health check endpoint — always include this"""
    return {"status": "ok", "version": "0.1.0"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are ImmigrationIQ, an AI assistant that provides accurate and up-to-date immigration guidance. Be concise."),
        ("human", "{message}")
    ])

    chain  = prompt | llm
    response = chain.invoke({"message": request.message})

    # LLM response might be a string (Ollama) or AIMessage (Groq)
    # Handle both:
    content  = response.content if hasattr(response, 'content') else str(response)
    return ChatResponse(
        message=content,
        session_id=request.session_id
    )

# Run with: uvicorn main:app --reload
# Docs at: http://localhost:8000/docs