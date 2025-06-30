from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    ForeignKey,
    UniqueConstraint,
    Text,
    JSON,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Product(Base):
    __tablename__ = "products"

    _id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    trust_score = Column(Float, default=0.0)
    check_chain_review_done = Column(Boolean, default=False)
    mining_done = Column(Boolean, default=False)
    rewards_distributed = Column(Boolean, default=False)

    predictions = relationship(
        "MinerPrediction", back_populates="product", cascade="all, delete-orphan"
    )


class MinerPrediction(Base):
    __tablename__ = "miner_predictions"
    __table_args__ = (
        UniqueConstraint("product_id", "miner_id", name="uix_product_miner"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String, ForeignKey("products._id"), nullable=False)
    miner_id = Column(Integer, nullable=False)
    
    # Core prediction data
    prediction = Column(Float)  # Numerical score (0-100)
    review = Column(Text)  # Review text (max 140 chars)
    keywords = Column(JSON)  # List of keywords as JSON
    
    # Analysis metadata
    sentiment = Column(String)  # positive, neutral, negative, unknown
    keyword_verification_score = Column(Float)  # 0-5 score
    coherence_score = Column(Float)  # 0-15 score
    total_reward = Column(Float)  # Final reward score
    
    # Timestamps
    created_at = Column(String)  # ISO timestamp
    updated_at = Column(String)  # ISO timestamp

    product = relationship("Product", back_populates="predictions")
