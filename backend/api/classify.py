from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from tenacity import retry, stop_after_attempt, wait_exponential

from services.llm import get_llm
from models.immigration import ImmigrationAnalysis


# -------------------------------------------------------
# Why APIRouter instead of adding @app.post directly?
#
# In FastAPI, you COULD add all endpoints directly to `app`
# in main.py. But as your project grows, main.py becomes
# 500+ lines. Instead, you use APIRouter — it's like a
# Spring @RestController. You define routes in separate files,
# then "register" them in main.py with app.include_router().
# -------------------------------------------------------

router = APIRouter(
    prefix="/classify", # All routes in this file will start with /classify
    tags=["classification"] # Group routes together in the docs
)

# -------------------------------------------------------
# Request Model: What the frontend sends us
# -------------------------------------------------------
class ClassifyRequest(BaseModel):
    situation: str # "I'm on H1B, married to a US citizen, want a green card"


# -------------------------------------------------------
# Response Model: What we send back to the frontend
# -------------------------------------------------------
class ClassifyResponse(BaseModel):
    analysis: ImmigrationAnalysis # The structured analysis from the LLM
    raw_situation: str # Echo back what was sent (useful for debugging)


# -------------------------------------------------------
# The Prompt
#
# This is the SYSTEM message — it tells the LLM who it is
# and how to behave.
#
# {format_instructions} is a placeholder that gets replaced
# with the JSON schema instructions from our Pydantic parser.
# -------------------------------------------------------
CLASSIFY_SYSTEM_PROMPT = """ You are an expert US immigration attorney and form classifer.
Your job is to analyze a user's immigration situation and identify:
1. What immigration category they fall into (e.g. family-based, employment-based, humanitarian, naturalization, nonimmigrant)
2. What specific USCIS forms are most relevant to their situation (e.g. I-485, I-130, N-400)
3. What concrete next steps they should take (e.g. "File Form I-130 with USCIS, including evidence of marriage and proof of spouse's citizenship")
4. A rough timeline estimate for their case (e.g. "12-24 months for green card approval, depending on USCIS workload")

IMPORTANT RULES:
- Only give legal advice if user explicitly asks for it. Otherwise, focus on explaining the process and forms only.
- Be conservative - if you're not sure about the category or forms, set confidence low and ask clarifying questions to the user rather than guessing.
- Base your analysis on common USCIS processes and forms. not invented information you made up.
- If the situation involves removal/deportation proceedings and you don't have enough information, always set confidence to 0.2 
  and recommend they consult an immigration attorney immediately. If you are confident on the solution you can set confidence to 0.8 or higher and tell the user what to do but also highlight
  that they should seek legal advice from an attorney immediately.
  {format_instructions}
"""


# -------------------------------------------------------
# RETRY WRAPPER FUNCTION
# 
# This function wraps the chain execution and retries
# up to 3 times if parsing fails, with exponential backoff.
# -------------------------------------------------------
@retry(
    stop=stop_after_attempt(3),  # Try up to 3 times
    wait=wait_exponential(multiplier=1, min=2, max=10),  # Wait 2s, 4s, 8s between retries
    reraise=True  # Re-raise the exception if all retries fail
)
def run_classification_chain(chain, situation: str) -> ImmigrationAnalysis:
    """
    Runs the classification chain with automatic retries.
    If the parser fails on attempt 1, it automatically retries
    (the LLM might give a better response the 2nd time).
    """
    return chain.invoke({"situation": situation})     # .invoke() runs the full chain synchronously


@router.post("/", response_model=ClassifyResponse)
async def classify_situation(request: ClassifyRequest):
    """
    Analyzes a user's immigration situation and returns structured guidance.
    
    Includes automatic retry logic — if the LLM's response can't be parsed,
    the whole chain is retried up to 3 times before failing.
    
    Example input:
    {
        "situation": "I'm on H1B visa, married to a US citizen for 2 years. 
                      I want to get a green card."
    }
    
    Example output:
    {
        "analysis": {
            "immigration_category": "family_based",
            "applicable_forms": ["I-485", "I-130", "I-765", "I-131"],
            "priority_steps": [...],
            "estimated_timeline": "12-24 months",
            "confidence": 0.85,
            "needs_more_info": false,
            "clarifying_questions": []
        },
        "raw_situation": "I'm on H1B visa..."
    }
    """

    # STEP 1: Create the output parser
    # This object does two things:
    # a) Generates format instructions for the prompt
    # b) Parses the LLM's JSON response into an ImmigrationAnalysis object
    parser = PydanticOutputParser(pydantic_object=ImmigrationAnalysis)

    # STEP 2: Build the prompt template
    # .partial() pre-fills one variable so we don't have to pass it every time
    # format_instructions is always the same, so we fill it now
    prompt = ChatPromptTemplate.from_messages([
        ("system", CLASSIFY_SYSTEM_PROMPT),
        ("human", "Please analyse this immigration situation and provide accurate guidance: {situation}")
    ]).partial(format_instructions=parser.get_format_instructions())

    
    # STEP 3: Build the chain
    # prompt → LLM → parser
    # Each component's output feeds into the next component's input
    llm = get_llm()
    chain = prompt | llm | parser

    # STEP 4: Run the chain WITH RETRIES
    # If it fails, tenacity automatically retries up to 3 times
    try:
        analysis: ImmigrationAnalysis = run_classification_chain(chain, request.situation)

    except OutputParserException as e:
        # After 3 retries, parsing still failed
        # This happens when the LLM produces text that can't be parsed as JSON.
        # Common with smaller models (llama3.2 3B sometimes does this).
        # We don't crash — we return a useful error state.
        raise HTTPException(
            status_code=422,
            detail=f"Could not parse LLM response after 3 retries. Try rephrasing your situation. Error: {str(e)}"
        )
    
    except Exception as e:
        # Catch-all for other errors (LLM timeout, connection issues, etc.)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while analyzing your situation: {str(e)}"
        )
    
    # STEP 5: Return the response
    return ClassifyResponse(
        analysis=analysis,
        raw_situation=request.situation
    )