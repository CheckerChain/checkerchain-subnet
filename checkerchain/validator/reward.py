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
from typing import List
import bittensor as bt
from typing import List, Dict
import asyncio
import json
import re

from checkerchain.types.checker_chain import ReviewedProduct
from neurons.validator import Validator
from checkerchain.miner.llm import create_llm, create_text_llm
from langchain.schema import SystemMessage, HumanMessage
from checkerchain.database.model import MinerPrediction


# def normalize(value: float, min_val: float, max_val: float) -> float:
#     """Normalize deviation so that 0 deviation gives a score of 1, and larger deviations get lower scores."""
#     if min_val == max_val:
#         return 1.0  # If no deviation, return full score
#     return 1 - ((value - min_val) / (max_val - min_val))


# def compare_and_normalize(
#     predictions: List[Dict[str, float]], actuals: List[Dict[str, float]]
# ) -> Dict[str, float]:
#     """Compare predictions and actual scores, compute variations, sum them up, and return a normalized score."""
#     variations = []

#     for act in actuals:
#         _id = act["_id"]
#         actual_score = act["trustScore"]
#         predicted_score = next(
#             (pred["prediction"] for pred in predictions if pred["_id"] == _id), 0
#         )
#         variations.append(abs(actual_score - predicted_score))
#     total_variation = sum(variations)
#     normalized_variation = normalize(
#         total_variation, min(variations, default=0), max(variations, default=1)
#     )

#     return normalized_variation


# def reward(
#     predictions: List[Dict[str, float]], actuals: List[Dict[str, float]]
# ) -> float:
#     """
#     Reward the miner response to the dummy request. This method returns a reward
#     value for the miner, which is used to update the miner's score.

#     Returns:
#     - float: The reward value for the miner.
#     """
#     if not predictions:
#         score = 0
#     else:
#         score = compare_and_normalize(predictions, actuals)
#     bt.logging.info(f"In rewards,rewards val: {score}")
#     return score


# def get_rewards(
#     self,
#     last_epoch_reviewed_products: List[Dict[str, int | str]],
#     responses: List[List[Dict[str, int | str]]],
# ) -> np.ndarray:
#     """
#     Returns an array of rewards for the given query and responses.

#     Args:
#     - query (int): The query sent to the miner.
#     - responses (List[float]): A list of responses from the miner.

#     Returns:
#     - np.ndarray: An array of rewards for the given query and responses.
#     """
#     # Get all the reward results by iteratively calling your reward() function.

#     return np.array(
#         [reward(response, last_epoch_reviewed_products) for response in responses]
#     )


async def analyze_sentiment(review: str) -> str:
    """
    Use OpenAI to analyze the sentiment of a review. Returns 'positive', 'neutral', or 'negative'.
    """
    if not review or len(review.strip()) < 10:
        return "unknown"
    
    prompt = f"Analyze the sentiment of the following review. Respond with only one word: positive, neutral, or negative.\nReview: {review}"
    try:
        llm = await create_text_llm()
        result = await llm.ainvoke([
            SystemMessage(content="You are a sentiment analysis expert."),
            HumanMessage(content=prompt),
        ])
        
        # Extract the text content from the response
        if hasattr(result, 'content'):
            sentiment = result.content.strip().lower()
        else:
            sentiment = str(result).strip().lower()
        
        # Validate sentiment
        valid_sentiments = ['positive', 'neutral', 'negative']
        if sentiment in valid_sentiments:
            return sentiment
        else:
            return "unknown"
    except Exception as e:
        bt.logging.error(f"Sentiment analysis failed: {e}")
        return "unknown"


async def analyze_keyword_coherence(keywords: List[str], review: str, score: float) -> float:
    """
    Analyze how well quality-descriptive keywords align with the review and score.
    Returns a score between 0-15.
    """
    if not keywords or not review or score is None:
        return 0.0
    
    try:
        llm = await create_text_llm()
        
        # Create a comprehensive prompt for quality keyword analysis
        prompt = f"""
        Analyze the coherence between quality-descriptive keywords, review, and score for a product assessment.
        
        **Quality Keywords:** {keywords}
        **Review:** {review}
        **Score:** {score}/100
        
        **Expected Quality Indicators based on Score:**
        - Score 80-100: excellent, highly-trusted, very-low-risk, leading, established
        - Score 70-79: good, trusted, low-risk, promising, stable
        - Score 60-69: average, moderate, medium-risk, emerging, acceptable
        - Score 40-59: poor, untrusted, high-risk, declining, suspicious
        - Score 0-39: very-poor, suspicious, very-high-risk, failing, scam
        
        **Rate the coherence from 0-15 based on:**
        1. Keywords accurately reflect the quality level implied by the score (0-5 points)
        2. Keywords align with the sentiment and tone of the review (0-4 points)
        3. Keywords are appropriate quality indicators (not just technical terms) (0-3 points)
        4. Keywords are consistent with each other (no contradictions) (0-3 points)
        
        **Quality Keywords Examples:**
        - Good: "excellent", "trusted", "low-risk", "established", "promising"
        - Bad: "blockchain", "crypto", "defi", "web3", "technology" (these are technical, not quality indicators)
        
        Respond with only a number between 0-15.
        """
        
        result = await llm.ainvoke([
            SystemMessage(content="You are an expert at analyzing quality-descriptive keyword coherence in product assessments."),
            HumanMessage(content=prompt),
        ])
        
        # Extract the text content from the response
        if hasattr(result, 'content'):
            response_text = result.content.strip()
        else:
            response_text = str(result).strip()
        
        # Try to extract a number from the response
        try:
            coherence_score = float(response_text)
            return max(0, min(15, coherence_score))  # Clamp between 0-15
        except ValueError:
            # If we can't parse a number, try to extract it from the text
            import re
            numbers = re.findall(r'\d+\.?\d*', response_text)
            if numbers:
                return max(0, min(15, float(numbers[0])))
            return 0.0
            
    except Exception as e:
        bt.logging.error(f"Keyword coherence analysis failed: {e}")
        return 0.0


async def verify_quality_keywords(keywords: List[str], score: float) -> float:
    """
    Verify that keywords are actually quality-descriptive and not just technical terms.
    Returns a score between 0-5.
    """
    if not keywords or score is None:
        return 0.0
    
    try:
        llm = await create_text_llm()
        
        # Define expected quality keywords based on score
        if score >= 80:
            expected_quality = ["excellent", "highly-trusted", "very-low-risk", "leading", "established"]
        elif score >= 70:
            expected_quality = ["good", "trusted", "low-risk", "promising", "stable"]
        elif score >= 60:
            expected_quality = ["average", "moderate", "medium-risk", "emerging", "acceptable"]
        elif score >= 40:
            expected_quality = ["poor", "untrusted", "high-risk", "declining", "suspicious"]
        else:
            expected_quality = ["very-poor", "suspicious", "very-high-risk", "failing", "scam"]
        
        prompt = f"""
        Verify if the provided keywords are quality-descriptive and appropriate for the given score.
        
        **Provided Keywords:** {keywords}
        **Score:** {score}/100
        **Expected Quality Level:** {expected_quality[0]}
        
        **Quality Keywords (Good):** excellent, good, average, poor, trusted, untrusted, low-risk, high-risk, promising, suspicious, established, failing
        
        **Technical Keywords (Bad):** blockchain, crypto, defi, web3, mobile, finance, technology, platform, app, token
        
        **Rate from 0-5:**
        - 5: All keywords are quality-descriptive and appropriate for the score
        - 4: Most keywords are quality-descriptive, 1-2 technical terms
        - 3: Mix of quality and technical keywords
        - 2: Mostly technical keywords, few quality indicators
        - 1: All technical keywords, no quality indicators
        - 0: Completely inappropriate or irrelevant keywords
        
        Respond with only a number between 0-5.
        """
        
        result = await llm.ainvoke([
            SystemMessage(content="You are an expert at verifying quality-descriptive keywords in product assessments."),
            HumanMessage(content=prompt),
        ])
        
        # Extract the text content from the response
        if hasattr(result, 'content'):
            response_text = result.content.strip()
        else:
            response_text = str(result).strip()
        
        # Try to extract a number from the response
        try:
            verification_score = float(response_text)
            return max(0, min(5, verification_score))  # Clamp between 0-5
        except ValueError:
            # If we can't parse a number, try to extract it from the text
            import re
            numbers = re.findall(r'\d+\.?\d*', response_text)
            if numbers:
                return max(0, min(5, float(numbers[0])))
            return 0.0
            
    except Exception as e:
        bt.logging.error(f"Quality keyword verification failed: {e}")
        return 0.0


def get_stake_score(self: Validator, miner_uid: int):
    max_stake = 2000
    min_stake = 500
    miner_stake = self.metagraph.S[miner_uid]

    
    # Handle MockTensor objects by converting to float
    if hasattr(miner_stake, 'item'):
        miner_stake = miner_stake.item()
    
    if miner_stake >= max_stake:
        return 1.0

        
    miner_stake = float(miner_stake)
    
    if max_stake == min_stake:
        return 1.0
    return (miner_stake - min_stake) / (max_stake - min_stake)


async def reward(
    self: Validator, prediction: MinerPrediction, actual: float, miner_uid: int
) -> float:
    """
    Enhanced reward function that uses a single OpenAI request to analyze the complete response.
    Returns a comprehensive reward value for the miner.
    """
    bt.logging.info(f"reward called for miner {miner_uid} with prediction: {prediction.prediction}")
    
    if not prediction or not isinstance(prediction, MinerPrediction):
        bt.logging.warning(f"Invalid prediction for miner {miner_uid}: {prediction}")
        return 0.0
    
    # Use single-request analysis
    bt.logging.info(f"Starting analysis for miner {miner_uid}")
    analysis_result = await analyze_complete_response(prediction, actual)
    bt.logging.info(f"Analysis result for miner {miner_uid}: {analysis_result}")
    
    # Extract analysis components
    sentiment_score = 20 if analysis_result["sentiment"] != "unknown" else 5
    keyword_score = analysis_result["keyword_verification_score"]
    coherence_score = min(analysis_result["coherence_score"], 10)  # Cap at 10
    accuracy_score = analysis_result["score_accuracy"]
    
    # Calculate performance score (0-60 points)
    perf_score = accuracy_score + sentiment_score + keyword_score + coherence_score
    
    # Stake-based score (0-20 points)
    stake_score = get_stake_score(self, miner_uid=miner_uid)
    final_stake_score = 15 * stake_score
    
    # Total score (max 100 points)
    total_score = perf_score + final_stake_score
    
    bt.logging.info(
        f"Miner UID: {miner_uid}, Score: {prediction.prediction}, "
        f"Sentiment: {analysis_result['sentiment']}, Keyword Score: {keyword_score}, "
        f"Coherence Score: {coherence_score}, Accuracy: {accuracy_score}, "
        f"Performance: {perf_score}, Stake Score: {final_stake_score}, Total: {total_score}"
    )
    
    return total_score


async def get_rewards(
    self: Validator,
    reviewed_product: ReviewedProduct,
    responses: list[dict | None],
    miner_uids: list[int],
) -> np.ndarray:
    """
    Enhanced reward function that processes responses asynchronously and returns
    comprehensive rewards based on score, sentiment, and keyword coherence.
    """
    bt.logging.info(f"get_rewards called with {len(responses)} responses and {len(miner_uids)} miner_uids")
    bt.logging.info(f"Responses: {responses}")
    bt.logging.info(f"Miner UIDs: {miner_uids}")
    
    if reviewed_product.trustScore == 0:
        bt.logging.info("Actual score is 0, returning equal rewards")
        return np.full(len(responses), 100 / len(responses))

    # Process rewards asynchronously
    reward_tasks = []
    for i, (response, uid) in enumerate(zip(responses, miner_uids)):
        if response is not None:
            task = reward(self, response, reviewed_product.trustScore, uid)
            reward_tasks.append((i, task))
            bt.logging.info(f"Created reward task for miner {uid} at index {i}")
        else:
            bt.logging.info(f"Skipping reward task for miner {uid} at index {i} - response is None")
    
    # Execute all reward calculations concurrently
    rewards_dict = {}
    if reward_tasks:
        bt.logging.info(f"Executing {len(reward_tasks)} reward tasks")
        bt.logging.info(f"Reward tasks: {reward_tasks}")
        print(f"Reward tasks: {reward_tasks}")
        results = await asyncio.gather(*[task for _, task in reward_tasks])
        for (i, _), result in zip(reward_tasks, results):
            rewards_dict[i] = result
            bt.logging.info(f"Reward for index {i}: {result}")
    else:
        bt.logging.warning("No reward tasks to execute!")

    bt.logging.info(f"Initial rewards_dict: {rewards_dict}")

    # Keep top 90% of miners
    keep_count = int(np.ceil(0.9 * len(rewards_dict)))
    bt.logging.info(f"Keeping top {keep_count} out of {len(rewards_dict)} miners")
    
    if len(rewards_dict) > 0:
        top_indices = sorted(rewards_dict.keys(), key=lambda k: rewards_dict[k], reverse=True)[:keep_count]
        kept_indices = set(top_indices)
        bt.logging.info(f"Top indices: {top_indices}")
        bt.logging.info(f"Kept indices: {kept_indices}")
    else:
        kept_indices = set()
        bt.logging.warning("No rewards to process!")

    final_rewards = [
        rewards_dict[i] if i in kept_indices else 0.0 for i in range(len(responses))
    ]
    
    bt.logging.info(f"Final rewards: {final_rewards}")
    return np.array(final_rewards)


async def analyze_complete_response(prediction: MinerPrediction, actual_score: float) -> dict:
    """
    Analyze a complete miner response (score, review, keywords) in a single OpenAI request.
    Returns comprehensive analysis including sentiment, keyword verification, and coherence.
    Uses LLM for quality keyword evaluation instead of hardcoded lists.
    """
    try:
        score = prediction.prediction
        review = prediction.review
        keywords = prediction.keywords
        
        if not review or not keywords or score is None:
            return {
                "sentiment": "unknown",
                "keyword_verification_score": 0.0,
                "coherence_score": 0.0,
                "score_accuracy": 0.0,
                "total_analysis_score": 0.0,
                "quality_keyword_score": 0.0,
                "quality_keyword_count": 0,
                "quality_keyword_matches": []
            }
        
        prompt = f"""
        Analyze this DeFi/crypto product assessment and provide comprehensive analysis in JSON format.

        **Miner Assessment:**
        - Score: {score}/100
        - Review: {review}
        - Keywords: {keywords}
        - Actual Score: {actual_score}/100

        **Analysis Requirements:**

        1. **Sentiment Analysis:** Analyze the review text
           - "positive": Optimistic, praising, recommending
           - "negative": Critical, warning, discouraging  
           - "neutral": Balanced, factual, objective
           - "unknown": Unclear or mixed sentiment

        2. **Keyword Verification (0-5):** Check if keywords are quality-descriptive
           - 5: All keywords are quality-descriptive (excellent, trusted, low-risk, etc.)
           - 4: Most keywords are quality-descriptive, 1-2 technical terms
           - 3: Mix of quality and technical keywords
           - 2: Mostly technical keywords (blockchain, crypto, defi, etc.)
           - 1: All technical keywords, no quality indicators
           - 0: Completely inappropriate or irrelevant keywords

        3. **Coherence Analysis (0-20):** Check consistency between score, review, and keywords
           - Score-Review Consistency (0-10): Does the review sentiment match the score?
           - Score-Keyword Consistency (0-5): Do keywords match the score level?
           - Review-Keyword Consistency (0-5): Do keywords match the review sentiment?

        4. **Score Accuracy (0-40):** How close is the predicted score to actual?
           - 40: Within 5% of actual score
           - 30: Within 10% of actual score  
           - 20: Within 20% of actual score
           - 10: Within 30% of actual score
           - 0: More than 30% deviation

        5. **Quality Keyword Analysis:** Evaluate which keywords are quality-descriptive
           - Quality keywords describe product quality, trust, risk, or performance
           - Examples: excellent, good, poor, trusted, untrusted, low-risk, high-risk, promising, suspicious, established, failing, innovative, secure, developing, etc.
           - Technical keywords (blockchain, crypto, defi, web3) are NOT quality indicators
           - Count how many keywords are quality-descriptive and rate overall quality (0-5)

        **Response Format (JSON only):**
        {{
            "sentiment": "positive",
            "keyword_verification_score": 4.5,
            "coherence_score": 12.0,
            "score_accuracy": 35.0,
            "total_analysis_score": 51.5,
            "quality_keyword_score": 4.0,
            "quality_keyword_count": 4,
            "quality_keyword_matches": ["excellent", "trusted", "low-risk", "established"]
        }}

        Respond with ONLY the JSON object, no additional text.
        """

        result = await create_text_llm()
        result = await result.ainvoke([
            SystemMessage(content="You are an expert at analyzing DeFi/crypto product assessments. Provide comprehensive analysis in JSON format only."),
            HumanMessage(content=prompt),
        ])
        
        # Extract the text content from the response
        if hasattr(result, 'content'):
            response_text = result.content.strip()
        else:
            response_text = str(result).strip()
        
        # Clean the response - remove any markdown formatting
        response_text = re.sub(r'^```json\s*', '', response_text)
        response_text = re.sub(r'\s*```$', '', response_text)
        
        # Parse the JSON response
        analysis_data = json.loads(response_text)
        
        # Validate and structure the response
        validated_response = {
            "sentiment": str(analysis_data.get("sentiment", "unknown")),
            "keyword_verification_score": float(analysis_data.get("keyword_verification_score", 0.0)),
            "coherence_score": float(analysis_data.get("coherence_score", 0.0)),
            "score_accuracy": float(analysis_data.get("score_accuracy", 0.0)),
            "total_analysis_score": float(analysis_data.get("total_analysis_score", 0.0)),
            "quality_keyword_score": float(analysis_data.get("quality_keyword_score", 0.0)),
            "quality_keyword_count": int(analysis_data.get("quality_keyword_count", 0)),
            "quality_keyword_matches": list(analysis_data.get("quality_keyword_matches", []))
        }
        print(validated_response)
        return validated_response
        
    except Exception as e:
        bt.logging.error(f"Error in complete response analysis: {e}")
        # Return fallback response
        return {
            "sentiment": "unknown",
            "keyword_verification_score": 0.0,
            "coherence_score": 0.0,
            "score_accuracy": 0.0,
            "total_analysis_score": 0.0,
            "quality_keyword_score": 0.0,
            "quality_keyword_count": 0,
            "quality_keyword_matches": []
        }


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
            score_accuracy * 0.4 +  # 40% weight
            coherence_score * 2.0 +  # Scale 0-15 to 0-30, then 30% weight
            keyword_verification_score * 4.0 +  # Scale 0-5 to 0-20, then 20% weight
            quality_keyword_score * 2.0  # Scale 0-5 to 0-10, then 10% weight
        )
        
        # Normalize to 0-1 range
        reward = max(0.0, min(1.0, total_score / 100.0))
        
        bt.logging.info(f"Reward calculation: accuracy={score_accuracy:.1f}, coherence={coherence_score:.1f}, "
                       f"keyword_verification={keyword_verification_score:.1f}, quality_keyword={quality_keyword_score:.1f}, "
                       f"total_score={total_score:.1f}, reward={reward:.3f}")
        
        return reward
        
    except Exception as e:
        bt.logging.error(f"Error calculating reward: {e}")
        return 0.0
