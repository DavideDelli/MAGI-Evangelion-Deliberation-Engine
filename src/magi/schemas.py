from typing import TypedDict
from pydantic import BaseModel

class MAGIState(TypedDict):
    dilemma: str
    melchior_response: str
    balthasar_response: str
    casper_response: str
    melchior_elapsed: float
    balthasar_elapsed: float
    casper_elapsed: float
    melchior_model_used: str
    balthasar_model_used: str
    casper_model_used: str
    final_decision: str

class DilemmaRequest(BaseModel):
    dilemma: str