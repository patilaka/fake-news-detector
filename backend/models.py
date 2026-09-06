from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class NewsPrediction(Base):

    __tablename__ = "news_predictions"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    news_text = Column(
        String,
        nullable=False
    )

    prediction = Column(
        String,
        nullable=False
    )

    confidence = Column(
        Float,
        nullable=False
    )
