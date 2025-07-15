# OpenAI API Key (ensure this is set in env variables or a secure place)
import bittensor as bt
from pydantic import BaseModel, Field
from checkerchain.types.checker_chain import UnreviewedProduct
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from checkerchain.utils.config import OPENAI_API_KEY
from typing import List
import json
import re
from checkerchain.database.model import MinerPrediction
import time


class ScoreBreakdown(BaseModel):
    """Detailed breakdown of product review scores."""

    project: float = Field(
        ..., ge=0, le=10, description="Project concept and innovation"
    )
    userbase: float = Field(..., ge=0, le=10, description="User adoption and community")
    utility: float = Field(
        ..., ge=0, le=10, description="Practical utility and use cases"
    )
    security: float = Field(
        ..., ge=0, le=10, description="Security measures and audits"
    )
    team: float = Field(..., ge=0, le=10, description="Team experience and credibility")
    tokenomics: float = Field(
        ..., ge=0, le=10, description="Token economics and distribution"
    )
    marketing: float = Field(
        ..., ge=0, le=10, description="Marketing strategy and reach"
    )
    roadmap: float = Field(
        ..., ge=0, le=10, description="Development roadmap and milestones"
    )
    clarity: float = Field(
        ..., ge=0, le=10, description="Project clarity and communication"
    )
    partnerships: float = Field(
        ..., ge=0, le=10, description="Strategic partnerships and collaborations"
    )


class ReviewScoreSchema(BaseModel):
    """Structured output schema for product reviews."""

    breakdown: ScoreBreakdown
    overall_score: float = Field(
        ..., ge=0, le=100, description="Overall trust score (0-100)"
    )
    review: str = Field(
        ..., max_length=140, description="Brief review text (max 140 characters)"
    )
    keywords: List[str] = Field(
        ..., min_items=3, max_items=7, description="Quality-descriptive keywords"
    )


# Create separate LLM instances for different purposes
llm_structured = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=2000)

llm_text = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_tokens=1000)


async def create_llm():
    """
    Create an instance of the LLM with structured output.
    """
    try:
        model = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-4o",
            max_tokens=1000,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n\n"],
        )
        return model.with_structured_output(ReviewScoreSchema)
    except Exception as e:
        raise Exception(f"Failed to create LLM: {str(e)}")


async def create_text_llm():
    """
    Create an instance of the LLM for text generation (no structured output).
    """
    try:
        model = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-4o",
            max_tokens=500,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return model
    except Exception as e:
        raise Exception(f"Failed to create text LLM: {str(e)}")


async def generate_review_score(product: UnreviewedProduct):
    """
    Generate review scores for a product using OpenAI's GPT.
    """
    prompt = f"""
    You are an expert evaluator analyzing products based on multiple key factors. Review the product below and provide a score out of 100 with a breakdown (0-10 for each criterion). Calculate the overall score as the average of the breakdown scores multiplied by 10.

    **Product Details:**
    - Name: {product.name}
    - Description: {product.description}
    - Category: {product.category}
    - URL: {product.url}
    - Location: {product.location}
    - Network: {product.network}
    - Team: {len(product.teams)} members
    - Marketing & Social Presence: {product.twitterProfile}
    - Current Review Cycle: {product.currentReviewCycle}

    **Evaluation Criteria:**
    1. Project (Innovation/Technology)
    2. Userbase/Adoption
    3. Utility Value
    4. Security
    5. Team
    6. Price/Revenue/Tokenomics
    7. Marketing & Social Presence
    8. Roadmap
    9. Clarity & Confidence
    10. Partnerships

    Scores must be integers between 0 and 10.
    """

    try:
        llm = await create_llm()
        result = await llm.ainvoke(
            [
                SystemMessage(content="You are an expert product reviewer."),
                HumanMessage(content=prompt),
            ]
        )
        return result
    except Exception as e:
        raise Exception(f"Failed to generate review score: {str(e)}")


async def generate_review_text(product: UnreviewedProduct):
    """
    Generate a concise review (≤140 chars) for a product using OpenAI's GPT.
    """
    prompt = f"""
    You are an expert evaluator. Write a concise, helpful review (max 130 characters) for the following product, summarizing its strengths or weaknesses:

    **Product Details:**
    - Name: {product.name}
    - Description: {product.description}
    - Category: {product.category}
    - URL: {product.url}
    - Location: {product.location}
    - Network: {product.network}
    - Team: {len(product.teams)} members
    - Marketing & Social Presence: {product.twitterProfile}
    - Current Review Cycle: {product.currentReviewCycle}

    Write only the review text, nothing else. The review must be 130 characters or less. **Make absolutely sure that the review is 130 characters or less.**
    """
    try:
        llm = await create_text_llm()
        result = await llm.ainvoke(
            [
                SystemMessage(content="You are an expert product reviewer."),
                HumanMessage(content=prompt),
            ]
        )
        if hasattr(result, "content"):
            review_text = result.content
        else:
            review_text = str(result)

        review_text = review_text.strip()

        return review_text
    except Exception as e:
        raise Exception(f"Failed to generate review text: {str(e)}")


async def generate_keywords(product: UnreviewedProduct) -> list[str]:
    """
    Generate quality-descriptive keywords for a product using OpenAI's GPT.
    Returns a list of around 5 keywords that describe the QUALITY assessment of the product.
    Keywords should reflect whether the project is good, average, poor, scam, etc.
    """
    prompt = f"""
    Analyze the following product and extract exactly 5 quality-descriptive keywords that reflect your assessment of the project's QUALITY.
    
    **IMPORTANT:** Focus on QUALITY indicators, not just technical features. Keywords should describe:
    - Quality level: "excellent", "good", "average", "poor", "scam", "suspicious"
    - Trust indicators: "trusted", "verified", "reliable", "risky", "untrusted"
    - Performance indicators: "promising", "established", "declining", "growing"
    - Risk level: "low-risk", "medium-risk", "high-risk", "very-risky"
    - Market position: "leading", "emerging", "failing", "stable"
    
    **Product Details:**
    - Name: {product.name}
    - Description: {product.description}
    - Category: {product.category}
    - URL: {product.url}
    - Location: {product.location}
    - Network: {product.network}
    - Team: {len(product.teams)} members
    - Marketing & Social Presence: {product.twitterProfile}
    - Current Review Cycle: {product.currentReviewCycle}

    Return only the quality-descriptive keywords as a comma-separated list, no additional text.
    Example format: good, trusted, promising, low-risk, established
    """

    try:
        llm = await create_text_llm()
        result = await llm.ainvoke(
            [
                SystemMessage(
                    content="You are an expert at assessing product quality and extracting quality-descriptive keywords."
                ),
                HumanMessage(content=prompt),
            ]
        )

        if hasattr(result, "content"):
            response_text = result.content
        else:
            response_text = str(result)

        keywords = [kw.strip() for kw in response_text.split(",")]

        # Clean up keywords (remove quotes, extra spaces, etc.)
        keywords = [kw.strip(" \"'") for kw in keywords if kw.strip()]

        # Ensure we have around 5 keywords, trim if too many, pad if too few
        keywords = keywords[:5]  # Take first 5
        while len(keywords) < 5:
            keywords.append("unknown")

        return keywords
    except Exception as e:
        bt.logging.error(f"Failed to generate quality keywords: {str(e)}")
        return [
            "unknown",
            "unverified",
            "risky",
            "suspicious",
            "poor",
        ]


async def generate_quality_keywords_with_score(
    product: UnreviewedProduct, score: float
) -> list[str]:
    """
    Generate quality-descriptive keywords based on both product analysis and the calculated score.
    This ensures consistency between the score and keywords.
    """
    if score >= 80:
        quality_level = "excellent"
        trust_level = "highly-trusted"
        risk_level = "very-low-risk"
    elif score >= 70:
        quality_level = "good"
        trust_level = "trusted"
        risk_level = "low-risk"
    elif score >= 60:
        quality_level = "average"
        trust_level = "moderate"
        risk_level = "medium-risk"
    elif score >= 40:
        quality_level = "poor"
        trust_level = "untrusted"
        risk_level = "high-risk"
    else:
        quality_level = "very-poor"
        trust_level = "suspicious"
        risk_level = "very-high-risk"

    prompt = f"""
    Based on the product analysis and calculated score of {score}/100, generate exactly 5 quality-descriptive keywords.
    
    **Score Analysis:**
    - Score: {score}/100
    - Quality Level: {quality_level}
    - Trust Level: {trust_level}
    - Risk Level: {risk_level}
    
    **Product Details:**
    - Name: {product.name}
    - Description: {product.description}
    - Category: {product.category}
    
    Generate keywords that are consistent with the score and reflect the quality assessment.
    Include a mix of quality indicators, trust indicators, and risk indicators.
    
    Return only the keywords as a comma-separated list.
    """

    try:
        llm = await create_text_llm()
        result = await llm.ainvoke(
            [
                SystemMessage(
                    content="You are an expert at generating quality-descriptive keywords that align with numerical scores."
                ),
                HumanMessage(content=prompt),
            ]
        )

        # Extract the text content from the response
        if hasattr(result, "content"):
            response_text = result.content
        else:
            response_text = str(result)

        # Parse keywords from comma-separated text
        keywords = [kw.strip() for kw in response_text.split(",")]

        # Clean up keywords (remove quotes, extra spaces, etc.)
        keywords = [kw.strip(" \"'") for kw in keywords if kw.strip()]

        # Ensure we have around 5 keywords
        keywords = keywords[:5]
        while len(keywords) < 5:
            keywords.append(quality_level)

        return keywords
    except Exception as e:
        bt.logging.error(f"Failed to generate score-based keywords: {str(e)}")
        return [quality_level, trust_level, risk_level, "assessed", "evaluated"]


async def generate_complete_assessment(product_data: UnreviewedProduct) -> dict:
    """
    Generate a complete product assessment (score, review, keywords) in a single OpenAI request.
    Returns a structured JSON response.
    """
    try:
        # Prepare product information
        product_name = product_data.name
        product_description = product_data.description
        product_website = product_data.url
        product_category = product_data.category

        prompt = f"""
        Analyze this DeFi/crypto product and provide a complete assessment in JSON format.

        **Product Information:**
        - Name: {product_name}
        - Description: {product_description}
        - Website: {product_website}
        - Category: {product_category}

        **Assessment Requirements:**
        1. **Score Breakdown (0-10 each):**
           - project: Project concept and innovation
           - userbase: User adoption and community
           - utility: Practical utility and use cases
           - security: Security measures and audits
           - team: Team experience and credibility
           - tokenomics: Token economics and distribution
           - marketing: Marketing strategy and reach
           - roadmap: Development roadmap and milestones
           - clarity: Project clarity and communication
           - partnerships: Strategic partnerships and collaborations

        2. **Overall Score (0-100):** Weighted average based on breakdown scores

        3. **Review (max 140 chars):** Brief, professional assessment

        4. **Keywords (3-7 items):** Quality-descriptive keywords like "excellent", "trusted", "low-risk", "suspicious", "scam", etc. (NOT technical terms like "blockchain", "crypto", "defi")

        **Response Format (JSON only):**
        {{
            "breakdown": {{
                "project": 8.5,
                "userbase": 7.0,
                "utility": 8.0,
                "security": 9.0,
                "team": 7.5,
                "tokenomics": 6.5,
                "marketing": 8.0,
                "roadmap": 7.0,
                "clarity": 8.5,
                "partnerships": 7.0
            }},
            "overall_score": 77.5,
            "review": "Strong DeFi protocol with excellent security and experienced team. Highly recommended for serious investors.",
            "keywords": ["excellent", "trusted", "low-risk", "established", "promising"]
        }}

        Respond with ONLY the JSON object, no additional text.
        """

        result = await llm_structured.ainvoke(
            [
                SystemMessage(
                    content="You are an expert DeFi/crypto analyst. Provide accurate, professional assessments in JSON format only."
                ),
                HumanMessage(content=prompt),
            ]
        )

        # Extract the text content from the response
        if hasattr(result, "content"):
            response_text = result.content.strip()
        else:
            response_text = str(result).strip()

        # Clean the response - remove any markdown formatting
        response_text = re.sub(r"^```json\s*", "", response_text)
        response_text = re.sub(r"\s*```$", "", response_text)

        # Parse the JSON response
        assessment_data = json.loads(response_text)

        # Validate and structure the response
        validated_response = {
            "score": float(assessment_data.get("overall_score", 0)),
            # "review": str(assessment_data.get("review", ""))[:140],
            "review": str(assessment_data.get("review", "")),  # Ensure max 140 chars
            "keywords": list(assessment_data.get("keywords", []))[
                :7
            ],  # Ensure max 7 keywords
        }

        return validated_response

    except Exception as e:
        bt.logging.error(f"Error in complete assessment generation: {e}")
        # Return fallback response
        return {
            "score": None,
            "review": None,
            "keywords": [],
        }
    finally:
        time.sleep(0.01)


async def analyze_complete_response(
    prediction: MinerPrediction, actual_score: float
) -> dict:
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
                "quality_keyword_matches": [],
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
           - 40: Within 2% of actual score
           - 30: Within 6% of actual score  
           - 20: Within 8% of actual score
           - 10: Within 10% of actual score
           - 0: More than 10% deviation

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
        result = await result.ainvoke(
            [
                SystemMessage(
                    content="You are an expert at analyzing DeFi/crypto product assessments. Provide comprehensive analysis in JSON format only."
                ),
                HumanMessage(content=prompt),
            ]
        )

        # Extract the text content from the response
        if hasattr(result, "content"):
            response_text = result.content.strip()
        else:
            response_text = str(result).strip()

        # Clean the response - remove any markdown formatting
        response_text = re.sub(r"^```json\s*", "", response_text)
        response_text = re.sub(r"\s*```$", "", response_text)

        # Parse the JSON response
        analysis_data = json.loads(response_text)

        # Validate and structure the response
        validated_response = {
            "sentiment": str(analysis_data.get("sentiment", "unknown")),
            "keyword_verification_score": float(
                analysis_data.get("keyword_verification_score", 0.0)
            ),
            "coherence_score": float(analysis_data.get("coherence_score", 0.0)),
            "score_accuracy": float(analysis_data.get("score_accuracy", 0.0)),
            "total_analysis_score": float(
                analysis_data.get("total_analysis_score", 0.0)
            ),
            "quality_keyword_score": float(
                analysis_data.get("quality_keyword_score", 0.0)
            ),
            "quality_keyword_count": int(analysis_data.get("quality_keyword_count", 0)),
            "quality_keyword_matches": list(
                analysis_data.get("quality_keyword_matches", [])
            ),
        }
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
            "quality_keyword_matches": [],
        }
    finally:
        time.sleep(0.01)


async def analyze_keyword_coherence(
    keywords: List[str], review: str, score: float
) -> float:
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

        result = await llm.ainvoke(
            [
                SystemMessage(
                    content="You are an expert at analyzing quality-descriptive keyword coherence in product assessments."
                ),
                HumanMessage(content=prompt),
            ]
        )

        # Extract the text content from the response
        if hasattr(result, "content"):
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

            numbers = re.findall(r"\d+\.?\d*", response_text)
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

        if score >= 80:
            expected_quality = [
                "excellent",
                "highly-trusted",
                "very-low-risk",
                "leading",
                "established",
            ]
        elif score >= 70:
            expected_quality = ["good", "trusted", "low-risk", "promising", "stable"]
        elif score >= 60:
            expected_quality = [
                "average",
                "moderate",
                "medium-risk",
                "emerging",
                "acceptable",
            ]
        elif score >= 40:
            expected_quality = [
                "poor",
                "untrusted",
                "high-risk",
                "declining",
                "suspicious",
            ]
        else:
            expected_quality = [
                "very-poor",
                "suspicious",
                "very-high-risk",
                "failing",
                "scam",
            ]

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

        result = await llm.ainvoke(
            [
                SystemMessage(
                    content="You are an expert at verifying quality-descriptive keywords in product assessments."
                ),
                HumanMessage(content=prompt),
            ]
        )

        if hasattr(result, "content"):
            response_text = result.content.strip()
        else:
            response_text = str(result).strip()

        try:
            verification_score = float(response_text)
            return max(0, min(5, verification_score))
        except ValueError:
            import re

            numbers = re.findall(r"\d+\.?\d*", response_text)
            if numbers:
                return max(0, min(5, float(numbers[0])))
            return 0.0

    except Exception as e:
        bt.logging.error(f"Quality keyword verification failed: {e}")
        return 0.0


async def analyze_sentiment(review: str) -> str:
    """
    Use OpenAI to analyze the sentiment of a review. Returns 'positive', 'neutral', or 'negative'.
    """
    if not review or len(review.strip()) < 10:
        return "unknown"

    prompt = f"Analyze the sentiment of the following review. Respond with only one word: positive, neutral, or negative.\nReview: {review}"
    try:
        llm = await create_text_llm()
        result = await llm.ainvoke(
            [
                SystemMessage(content="You are a sentiment analysis expert."),
                HumanMessage(content=prompt),
            ]
        )

        if hasattr(result, "content"):
            sentiment = result.content.strip().lower()
        else:
            sentiment = str(result).strip().lower()

        valid_sentiments = ["positive", "neutral", "negative"]
        if sentiment in valid_sentiments:
            return sentiment
        else:
            return "unknown"
    except Exception as e:
        bt.logging.error(f"Sentiment analysis failed: {e}")
        return "unknown"
