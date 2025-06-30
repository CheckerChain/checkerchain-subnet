import json
from datetime import datetime
from sqlalchemy import select, delete, update
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert
from sqlalchemy.orm import Session
from sqlalchemy.exc import NoResultFound
from checkerchain.database.model import Product, MinerPrediction
from .utils import with_db_session
import typing as ty


@with_db_session
def get_products(session: Session) -> ty.List[Product]:
    return session.query(Product).all()


@with_db_session
def get_unreviewed_products(session: Session):
    return session.query(Product).filter(Product.check_chain_review_done == False).all()


@with_db_session
def get_product(session: Session, **kwargs):
    return session.query(Product).filter_by(**kwargs).first()


@with_db_session
def add_product(
    session: Session,
    _id,
    name,
    trust_score=0.0,
    check_chain_review_done=False,
    mining_done=False,
    rewards_distributed=False,
):
    product = Product(
        _id=_id,
        name=name,
        trust_score=trust_score,
        check_chain_review_done=check_chain_review_done,
        mining_done=mining_done,
        rewards_distributed=rewards_distributed,
    )
    session.add(product)
    session.commit()


@with_db_session
def remove_product(session: Session, _id):
    session.execute(delete(Product).where(Product._id == _id))
    session.commit()


@with_db_session
def remove_bulk_products(session: Session, product_ids: ty.List[str]):
    """
    Remove multiple products from the database.

    Args:
        product_ids: List of product IDs to remove.
    """
    if not product_ids:
        return

    session.execute(delete(Product).where(Product._id.in_(product_ids)))
    session.commit()


@with_db_session
def add_prediction(
    session: Session, product_id, miner_id, prediction_data, analysis_data=None
):
    """
    Add a complete miner prediction to the database.

    Args:
        prediction_data: Dict containing 'score', 'review', 'keywords'
        analysis_data: Dict containing 'sentiment', 'keyword_verification_score', 'coherence_score', 'total_reward'
    """
    now = datetime.utcnow().isoformat()

    # Extract data from prediction_data
    score = prediction_data.get("score") if isinstance(prediction_data, dict) else None
    review = (
        prediction_data.get("review") if isinstance(prediction_data, dict) else None
    )
    keywords = (
        prediction_data.get("keywords", []) if isinstance(prediction_data, dict) else []
    )

    # Extract analysis data
    sentiment = analysis_data.get("sentiment") if analysis_data else None
    keyword_verification_score = (
        analysis_data.get("keyword_verification_score") if analysis_data else None
    )
    coherence_score = analysis_data.get("coherence_score") if analysis_data else None
    total_reward = analysis_data.get("total_reward") if analysis_data else None

    # Convert keywords to JSON string
    keywords_json = json.dumps(keywords) if keywords else None

    ups_stmt = sqlite_upsert(MinerPrediction).values(
        product_id=product_id,
        miner_id=int(miner_id),
        prediction=float(score) if score is not None else None,
        review=review,
        keywords=keywords_json,
        sentiment=sentiment,
        keyword_verification_score=(
            float(keyword_verification_score)
            if keyword_verification_score is not None
            else None
        ),
        coherence_score=float(coherence_score) if coherence_score is not None else None,
        total_reward=float(total_reward) if total_reward is not None else None,
        created_at=now,
        updated_at=now,
    )
    query = ups_stmt.on_conflict_do_update(
        index_elements=[
            "product_id",
            "miner_id",
        ],
        set_=dict(
            prediction=ups_stmt.excluded.prediction,
            review=ups_stmt.excluded.review,
            keywords=ups_stmt.excluded.keywords,
            sentiment=ups_stmt.excluded.sentiment,
            keyword_verification_score=ups_stmt.excluded.keyword_verification_score,
            coherence_score=ups_stmt.excluded.coherence_score,
            total_reward=ups_stmt.excluded.total_reward,
            updated_at=now,
        ),
    )
    session.execute(query)
    session.commit()


@with_db_session
def add_prediction_legacy(session: Session, product_id, miner_id, prediction):
    """
    Legacy function for backward compatibility - only stores the score.
    """
    now = datetime.now().isoformat()

    ups_stmt = sqlite_upsert(MinerPrediction).values(
        product_id=product_id,
        miner_id=int(miner_id),
        prediction=float(prediction) if prediction is not None else None,
        created_at=now,
        updated_at=now,
    )
    query = ups_stmt.on_conflict_do_update(
        index_elements=[
            "product_id",
            "miner_id",
        ],
        set_=dict(
            miner_id=ups_stmt.excluded.miner_id,
            prediction=ups_stmt.excluded.prediction,
            updated_at=now,
        ),
    )
    session.execute(query)
    session.commit()


@with_db_session
def remove_prediction(session: Session, prediction_id):
    session.execute(delete(MinerPrediction).where(MinerPrediction.id == prediction_id))
    session.commit()


@with_db_session
def update_product_status(session: Session, _id, **kwargs):
    session.execute(update(Product).where(Product._id == _id).values(**kwargs))
    session.commit()


@with_db_session
def get_predictions_for_product(session: Session, product_id):
    return session.query(MinerPrediction).filter_by(product_id=product_id).all()


@with_db_session
def delete_a_product(session: Session, product_id):
    session.execute(
        delete(MinerPrediction).where(MinerPrediction.product_id == product_id)
    )
    session.execute(delete(Product).where(Product._id == product_id))
    session.commit()


@with_db_session
def db_get_unreviewd_products(session: Session):
    return session.query(Product).filter(Product.check_chain_review_done == False).all()
