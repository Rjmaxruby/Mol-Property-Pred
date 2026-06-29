from typing import Optional

from pydantic import BaseModel


class PredictRequest(BaseModel):
    input: str
    input_type: str = "auto"


class ExplainRequest(BaseModel):
    input: str
    input_type: str = "auto"
    targets: Optional[list[str]] = None
    top_n_atoms: int = 6