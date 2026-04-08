from pydantic import BaseModel
from typing import Optional

class EvaluationRequest(BaseModel):
    text: str

class EvaluationResponse(BaseModel):
    is_safe: bool
    reason: Optional[str] = None
    classifier_name: str
    attack_type: Optional[str] = None
    recommended_defense: Optional[str] = None