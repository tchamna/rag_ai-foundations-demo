from pydantic import BaseModel, Field

class BatteryChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000, pattern=r'\S')
    top_k: int = Field(default=8, ge=1, le=12)
