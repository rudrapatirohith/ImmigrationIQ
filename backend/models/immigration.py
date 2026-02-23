from typing import List, Dict, Any
from pydantic import BaseModel, Field


class ImmigrationAnalysis(BaseModel):
    """
    The structured output we ask the LLM to produce when analyzing
    a user's immigration situation.
    
    Every field has a description — that description becomes part of
    the prompt that tells the LLM what to put in each field.
    """
    immigration_category: str = Field(
        description="Single category: 'family_based', 'employment_based', 'nonimmigrant', 'humanitarian', 'naturalization', 'unknown'"
    )
    
    applicable_forms: List[str] = Field(
        description="3-5 USCIS forms like 'I-130 (family petition)', 'I-485 (adjustment)'",
        min_items=1, max_items=5
    )
    
    priority_steps: List[str] = Field(
        description="3-5 actionable steps starting with verbs",
        min_items=1, max_items=5
    )
    
    estimated_timeline: str = Field(
        description="Timeline like '6-12 months' or 'Varies'"
    )
    
    confidence: float = Field(ge=0, le=1)
    needs_more_info: bool = Field()
