# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import numpy as np
import bittensor as bt
import asyncio

from checkerchain.types.checker_chain import ReviewedProduct
from neurons.validator import Validator
from checkerchain.miner.llm import (
    analyze_complete_response,
)
from checkerchain.database.model import MinerPrediction


def get_stake_score(self: Validator, miner_uid: int):
    max_stake = 2000
    min_stake = 500
    miner_stake = self.metagraph.S[miner_uid]

    # Handle MockTensor objects by converting to float
    if hasattr(miner_stake, "item"):
        miner_stake = miner_stake.item()

    if miner_stake >= max_stake:
        return 1.0

    miner_stake = float(miner_stake)

    if max_stake == min_stake:
        return 1.0
    return (miner_stake - min_stake) / (max_stake - min_stake)


def get_deviation_percentage(prediction: float, actual: float) -> float:
    """
    Calculate the percentage deviation of a prediction from the actual value.
    Returns 0 if actual is 0 to avoid division by zero.
    """
    if actual == 0:
        return 0.0
    return abs((prediction - actual) / actual) * 100


async def reward(
    self: Validator, prediction: MinerPrediction, actual: float, miner_uid: int
) -> float:
    """
    Enhanced reward function that uses a single OpenAI request to analyze the complete response.
    Returns a comprehensive reward value for the miner.
    """
    if not prediction or not isinstance(prediction, MinerPrediction):
        bt.logging.warning(f"Invalid prediction for miner {miner_uid}: {prediction}")
        return 0.0

    if get_deviation_percentage(prediction.prediction, actual) > 10:
        bt.logging.warning(
            f"Prediction {prediction.prediction} deviates too much from actual {actual} for miner {miner_uid}"
        )
        return 0.0

    # Use single-request analysis
    analysis_result = await analyze_complete_response(prediction, actual)

    # Extract analysis components
    sentiment_score = 20 if analysis_result["sentiment"] != "unknown" else 5
    keyword_score = min(analysis_result["keyword_verification_score"], 5)
    coherence_score = min(analysis_result["coherence_score"], 20)  # Cap at 20
    accuracy_score = min(analysis_result["score_accuracy"], 40)

    # Calculate performance score (0-60 points)
    perf_score = accuracy_score + sentiment_score + keyword_score + coherence_score

    # Stake-based score (0-20 points)
    stake_score = get_stake_score(self, miner_uid=miner_uid)
    final_stake_score = 15 * stake_score

    # Total score (max 100 points)
    total_score = perf_score + final_stake_score

    return total_score


async def get_rewards(
    self: Validator,
    reviewed_product: ReviewedProduct,
    responses: list[MinerPrediction],
    miner_uids: list[int],
) -> np.ndarray:
    """
    Enhanced reward function that processes responses asynchronously and returns
    comprehensive rewards based on score, sentiment, and keyword coherence.
    """
    if reviewed_product.trustScore == 0:
        return np.full(len(responses), 100 / len(responses))

    # Process rewards asynchronously
    reward_tasks = []
    for i, (response, uid) in enumerate(zip(responses, miner_uids)):
        if response.prediction is not None:
            task = reward(self, response, reviewed_product.trustScore, uid)
            reward_tasks.append((i, task))
        else:
            bt.logging.info(
                f"Skipping reward task for miner {uid} at index {i} - response is None"
            )

    # Execute all reward calculations concurrently
    rewards_dict = {}
    if reward_tasks:
        results = await asyncio.gather(*[task for _, task in reward_tasks])
        for (i, _), result in zip(reward_tasks, results):
            rewards_dict[i] = result
    else:
        bt.logging.warning("No reward tasks to execute!")

    bt.logging.info(f"Initial rewards_dict: {rewards_dict}")

    # Keep top 90% of miners
    keep_count = int(np.ceil(0.9 * len(rewards_dict)))

    if len(rewards_dict) > 0:
        top_indices = sorted(
            rewards_dict.keys(), key=lambda k: rewards_dict[k], reverse=True
        )[:keep_count]
        kept_indices = set(top_indices)
    else:
        kept_indices = set()

    final_rewards = [
        rewards_dict[i] if i in kept_indices else 0.0 for i in range(len(responses))
    ]

    return np.array(final_rewards)


def calculate_reward(analysis: dict) -> float:
    """
    Calculate reward based on comprehensive analysis.
    Uses LLM-based quality keyword evaluation instead of hardcoded lists.
    """
    try:
        # Extract scores from analysis
        score_accuracy = analysis.get("score_accuracy", 0.0)
        coherence_score = analysis.get("coherence_score", 0.0)
        keyword_verification_score = analysis.get("keyword_verification_score", 0.0)
        quality_keyword_score = analysis.get("quality_keyword_score", 0.0)

        # Calculate weighted reward
        # Score accuracy: 40% weight (0-40 points)
        # Coherence: 30% weight (0-15 points)
        # Keyword verification: 20% weight (0-5 points)
        # Quality keyword score: 10% weight (0-5 points)

        total_score = (
            score_accuracy * 0.4  # 40% weight
            + coherence_score * 2.0  # Scale 0-15 to 0-30, then 30% weight
            + keyword_verification_score * 4.0  # Scale 0-5 to 0-20, then 20% weight
            + quality_keyword_score * 2.0  # Scale 0-5 to 0-10, then 10% weight
        )

        # Normalize to 0-1 range
        reward = max(0.0, min(1.0, total_score / 100.0))

        return reward

    except Exception as e:
        bt.logging.error(f"Error calculating reward: {e}")
        return 0.0
