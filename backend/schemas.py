from pydantic import BaseModel


class PredictionHistory(BaseModel):

    id: int
    news_text: str
    prediction: str
    confidence: float

    class Config:
        from_attributes = True
