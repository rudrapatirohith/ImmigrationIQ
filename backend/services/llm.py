import os
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from langchain_core.language_models import BaseLanguageModel

def get_llm ( temperature: float = 0.1) -> BaseLanguageModel:
    """
    LLM factory — uses Ollama locally, Groq in production.
    This is THE most important function in your codebase.
    Every AI feature goes through here.
    """

    if os.getenv("GROQ_API_KEY"):
        # In production, use Groq free tier
        from langchain_groq import ChatGroq
        return ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=temperature,
            max_tokens=2048
        )
    else:
        # Local: Ollama
        return Ollama(
            model="llama3.2",
            temperature=temperature,
        )