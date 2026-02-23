from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from services.llm import get_llm

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)

# -------------------------------------------------------
# In-Memory Session Store
#
# This is a Python dictionary that stores conversation history.
# Key = session_id (string), Value = InMemoryChatMessageHistory object
#
# LIMITATION: This lives in RAM. If you restart the server, all
# conversations are lost. In Month 3, we move this to Redis.
# For now, it's fine.
# -------------------------------------------------------
session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Returns the chat history for a session.
    Creates a new empty history if the session doesn't exist yet.
    
    This function is called by RunnableWithMessageHistory automatically.
    You never call it directly.
    """
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory() # Create a new history object for this session ID
    return session_store[session_id] # Return the history object for this session ID

# -------------------------------------------------------
# Build the Chain with Memory — done ONCE at module load
#
# Why outside the endpoint function?
# If you built the chain inside the function, it would be
# recreated on EVERY request. That's wasteful.
# Building it once and reusing is more efficient.
# -------------------------------------------------------

# The prompt includes MessagesPlaceholder — this is where
# previous conversation messages get injected automatically

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are ImmigrationIQ, an AI assistant that helps people understand 
    US immigration forms and processes. You are currently in a multi-turn conversation.
     
    IMPORTANT RULES:
    - Answer based on what the user has told you in this conversation
    - Remember all details the user shares (visa type, family situation, goals)
    - Never give legal advice — explain processes and forms only, if the user asks for legal advice and you are confident about the answer, give a general response but always include a disclaimer that they should consult an attorney for legal advice specific to their situation.
    - If you don't know something, say so clearly
    - Always add: "⚠️ This is for educational purposes only. Consult an immigration 
      attorney for legal advice specific to your situation."
    - Always add a citation for each USCIS form you reference, e.g. "I-485 (https://www.uscis.gov/i-485)"
    """),
    
    MessagesPlaceholder(variable_name="chat_history"),  # ← History injected here, before the user message
    
    ("human", "{input}")  # ← The current user message
])

# Build chain: prompt → LLM
# Note: no parser here because we want free-form text for chat
base_chain = CHAT_PROMPT | get_llm()

# Wrap the chain with memory management
# RunnableWithMessageHistory intercepts .invoke() calls and:
# 1. BEFORE running: loads history from session_store, injects into prompt
# 2. AFTER running: saves the new human message + AI response to session_store

chain_with_memory = RunnableWithMessageHistory(
    base_chain,
    get_session_history, # This function is called to load the history for a session ID
    input_messages_key="input",  # The key in the prompt where the human message goes
    history_messages_key="chat_history"  # The key in the prompt where the history goes
)


# -------------------------------------------------------
# Request/Response Models
# -------------------------------------------------------
class ChatV2Request(BaseModel):
    session_id: str # Client generates a unique session ID for each conversation and sends it with every message
    message: str

class MessageData(BaseModel):
    role: str  # "human" or "assistant"
    content: str

class ChatV2Response(BaseModel):
    message: str
    session_id: str
    history_length: int # How many messages are in this session (useful for debugging)


# -------------------------------------------------------
# The Endpoint
# -------------------------------------------------------

@router.post("/v2", response_model=ChatV2Response)
async def chat_v2(request: ChatV2Request):
    """
    Chat with memory — the AI remembers the full conversation.
    
    The frontend must:
    1. Generate a unique session_id (use uuid4) when starting a new conversation
    2. Send the SAME session_id on every subsequent message in that conversation
    3. Start a NEW session_id to start a fresh conversation

    """

   # The config tells RunnableWithMessageHistory WHICH session to load
    config = {"configurable": {"session_id": request.session_id}}

    # .invoke() runs the chain
    # RunnableWithMessageHistory automatically:
    # - Loads history for this session_id
    # - Injects it into the prompt
    # - Saves the new exchange to history
    response = chain_with_memory.invoke(
        {"input": request.message},
        config=config
    )

    # Response is AIMessage (from Groq) or string (from Ollama)
    content = response.content if hasattr(response, 'content') else str(response)

    # Get current history length for the response
    history = get_session_history(request.session_id)

    return ChatV2Response(
        message=content,
        session_id=request.session_id,
        history_length=len(history.messages)
    )



@router.get("/v2/{session_id}/history")
async def get_history(session_id: str):
    """
    Returns the full conversation history for a session.
    Useful for the frontend to display previous messages on page load.
    """
    if session_id not in session_store:
        return {session_id: session_id, "messages": []} # Return empty history if session ID not found
    
    history = session_store[session_id] # Get the history object for this session ID

    messages = []
    for msg in history.messages:
        messages.append({
            "role": "user" if isinstance(msg, HumanMessage) else "assistant",
            "content": msg.content
        })
    
    return {
        "session_id": session_id,
        "messages": messages,
        "total": len(messages)
    }


@router.delete("/v2/{session_id}")
async def clear_session(session_id: str):
    """Clear a session's history — for 'New Conversation' button"""
    if session_id in session_store:
        del session_store[session_id] # Remove the history for this session ID
    return {"session_id": session_id, "cleared": True}