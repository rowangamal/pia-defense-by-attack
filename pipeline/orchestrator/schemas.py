from typing import Optional
from pydantic import BaseModel

class UserRequest(BaseModel):
    instruction: str
    data: str
    fallback_defense: str = "spotlight"

class ClassifierResponse(BaseModel):
    is_safe: bool
    reason: Optional[str] = None
    classifier_name: str = "Unknown"
    attack_type: Optional[str] = None
    recommended_defense: Optional[str] = None