#!/usr/bin/env python3
"""
Standalone Validator Test Script
This script tests the validator's reward system without using btcli.
It simulates the validator's reward calculation and shows detailed scoring breakdown.
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any, List
import numpy as np
from unittest.mock import Mock

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from checkerchain.validator.reward import (
    analyze_complete_response,
    reward,
    get_rewards,
    calculate_reward
)
from checkerchain.types.checker_chain import ReviewedProduct
import bittensor as bt

# Import database functions
from checkerchain.database.actions import (
    add_prediction,
    get_predictions_for_product,
    delete_a_product
)
from checkerchain.database.model import MinerPrediction

class MockValidator:
    """Mock validator class for testing"""
    def __init__(self):
        self.metagraph = MockMetagraph()

class MockMetagraph:
    """Mock metagraph for testing"""
    def __init__(self):
        self.S = MockStakes()

class MockStakes:
    """Mock stakes for testing"""
    def __init__(self):
        # Mock stake values
        self.stakes = [1000, 500, 2000, 300, 1500]
    
    def max(self):
        return MockTensor(2000)
    
    def min(self):
        return MockTensor(300)
    
    def __getitem__(self, idx):
        return MockTensor(self.stakes[idx])

class MockTensor:
    """Mock tensor for testing"""
    def __init__(self, value):
        self.value = value
    
    def item(self):
        return self.value

class MockReviewedProduct:
    """Mock reviewed product for testing"""
    def __init__(self, data: Dict[str, Any]):
        self._id = data["_id"]
        self.name = data["name"]
        self.slug = data.get("slug", "test-slug")
        self.trustScore = data["trustScore"]

# Test data
TEST_PRODUCTS = {
    "high_quality": {
        "_id": "high_quality_product",
        "name": "Premium DeFi Protocol",
        "slug": "premium-defi",
        "trustScore": 85.0
    },
    "medium_quality": {
        "_id": "medium_quality_product", 
        "name": "Average DeFi Protocol",
        "slug": "average-defi",
        "trustScore": 65.0
    },
    "low_quality": {
        "_id": "low_quality_product",
        "name": "Suspicious Token",
        "slug": "suspicious-token", 
        "trustScore": 25.0
    }
}

# Test miner responses
TEST_MINER_RESPONSES = {
    "good_miner": {
        "score": 82.0,
        "review": "Excellent DeFi protocol with strong security, experienced team, and proven track record. Highly recommended for serious investors.",
        "keywords": ["excellent", "trusted", "low-risk", "established", "promising"]
    },
    "average_miner": {
        "score": 68.0,
        "review": "Decent DeFi protocol with some good features but room for improvement. Team seems competent but not exceptional.",
        "keywords": ["average", "moderate", "medium-risk", "acceptable", "stable"]
    },
    "poor_miner": {
        "score": 30.0,
        "review": "This looks like a scam. No real team, anonymous developers, and unrealistic promises. Avoid at all costs.",
        "keywords": ["scam", "suspicious", "high-risk", "untrusted", "avoid"]
    },
    "bad_keywords_miner": {
        "score": 75.0,
        "review": "Good DeFi protocol with solid fundamentals and experienced team. Worth considering for investment.",
        "keywords": ["blockchain", "crypto", "defi", "web3", "technology"]  # Technical keywords instead of quality
    },
    "mismatch_miner": {
        "score": 90.0,
        "review": "This is a terrible project with no redeeming qualities. Complete waste of time and money.",
        "keywords": ["excellent", "trusted", "low-risk", "established", "promising"]  # Mismatch between score and review
    }
}

async def test_sentiment_analysis():
    """Test sentiment analysis functionality using single-request approach."""
    print("🧠 Testing Sentiment Analysis")
    print("=" * 50)
    
    test_cases = [
        {
            "prediction": {
                "score": 85.0,
                "review": "Excellent project with great potential!",
                "keywords": ["excellent", "promising", "high-quality"]
            },
            "actual_score": 82.0
        },
        {
            "prediction": {
                "score": 25.0,
                "review": "This is a terrible scam, avoid completely.",
                "keywords": ["scam", "suspicious", "high-risk"]
            },
            "actual_score": 30.0
        },
        {
            "prediction": {
                "score": 65.0,
                "review": "Average project, nothing special.",
                "keywords": ["average", "moderate", "acceptable"]
            },
            "actual_score": 60.0
        }
    ]
    
    for case in test_cases:
        analysis = await analyze_complete_response(case["prediction"], case["actual_score"])
        print(f"Review: {case['prediction']['review']}")
        print(f"Sentiment: {analysis['sentiment']}")
        print(f"Total Analysis Score: {analysis['total_analysis_score']}")
        print()

async def test_keyword_verification():
    """Test keyword verification functionality using single-request approach."""
    print("🔍 Testing Keyword Verification")
    print("=" * 50)
    
    test_cases = [
        {
            "prediction": {
                "score": 85.0,
                "review": "Excellent DeFi protocol with strong security.",
                "keywords": ["excellent", "trusted", "low-risk", "established", "promising"]
            },
            "actual_score": 82.0,
            "description": "Good quality keywords for high score"
        },
        {
            "prediction": {
                "score": 75.0,
                "review": "Good DeFi protocol with solid fundamentals.",
                "keywords": ["blockchain", "crypto", "defi", "web3", "technology"]
            },
            "actual_score": 70.0,
            "description": "Technical keywords instead of quality"
        },
        {
            "prediction": {
                "score": 25.0,
                "review": "This looks like a scam with no real team.",
                "keywords": ["scam", "suspicious", "high-risk", "untrusted", "avoid"]
            },
            "actual_score": 30.0,
            "description": "Good quality keywords for low score"
        }
    ]
    
    for case in test_cases:
        print(f"📋 {case['description']}")
        print(f"   Keywords: {case['prediction']['keywords']}")
        print(f"   Score: {case['prediction']['score']}")
        
        analysis = await analyze_complete_response(case["prediction"], case["actual_score"])
        print(f"   Keyword Verification Score: {analysis['keyword_verification_score']}/5")
        print(f"   Coherence Score: {analysis['coherence_score']}/15")
        print()

async def test_keyword_coherence():
    """Test keyword coherence analysis using single-request approach."""
    print("🔗 Testing Keyword Coherence")
    print("=" * 50)
    
    test_cases = [
        {
            "prediction": {
                "score": 85.0,
                "review": "Excellent DeFi protocol with strong security and experienced team. Highly recommended.",
                "keywords": ["excellent", "trusted", "low-risk", "established", "promising"]
            },
            "actual_score": 82.0,
            "description": "Coherent high-quality assessment"
        },
        {
            "prediction": {
                "score": 85.0,
                "review": "This is a terrible scam with no redeeming qualities.",
                "keywords": ["excellent", "trusted", "low-risk", "established", "promising"]
            },
            "actual_score": 82.0,
            "description": "Incoherent - high score but negative review"
        },
        {
            "prediction": {
                "score": 25.0,
                "review": "This looks like a scam. No real team and unrealistic promises.",
                "keywords": ["scam", "suspicious", "high-risk", "untrusted", "avoid"]
            },
            "actual_score": 30.0,
            "description": "Coherent low-quality assessment"
        }
    ]
    
    for case in test_cases:
        print(f"📋 {case['description']}")
        print(f"   Keywords: {case['prediction']['keywords']}")
        print(f"   Review: {case['prediction']['review']}")
        print(f"   Score: {case['prediction']['score']}")
        
        analysis = await analyze_complete_response(case["prediction"], case["actual_score"])
        print(f"   Coherence Score: {analysis['coherence_score']}/15")
        print(f"   Total Analysis Score: {analysis['total_analysis_score']}")
        print()

async def test_individual_reward():
    """Test individual reward calculation with detailed scoring breakdown."""
    print("💰 Testing Individual Reward Calculation")
    print("=" * 50)
    
    validator = MockValidator()
    
    for product_name, product_data in TEST_PRODUCTS.items():
        print(f"📦 Testing Product: {product_data['name']} (Score: {product_data['trustScore']})")
        print("-" * 40)
        
        product = MockReviewedProduct(product_data)
        
        for miner_name, response in TEST_MINER_RESPONSES.items():
            print(f"   🧑‍💻 Miner: {miner_name}")
            print(f"      Response: Score={response['score']}, Keywords={response['keywords']}")
            
            # Get detailed analysis first
            analysis = await analyze_complete_response(response, product.trustScore)
            
            # Calculate reward
            reward_score = await reward(validator, response, product.trustScore, 0)
            
            print(f"      Reward: {reward_score:.2f}/100")
            
            # Detailed scoring breakdown
            print(f"\n=== Analysis Results ===")
            print(f"Sentiment: {analysis['sentiment']}")
            print(f"Score Accuracy: {analysis['score_accuracy']:.1f}/40")
            print(f"Coherence Score: {analysis['coherence_score']:.1f}/15")
            print(f"Keyword Verification: {analysis['keyword_verification_score']:.1f}/5")
            print(f"Quality Keyword Score: {analysis['quality_keyword_score']:.1f}/5")
            print(f"Quality Keyword Count: {analysis['quality_keyword_count']}")
            print(f"Quality Keyword Matches: {analysis['quality_keyword_matches']}")
            print(f"Total Analysis Score: {analysis['total_analysis_score']:.1f}/100")
            
            # Calculate and display reward
            reward = calculate_reward(analysis)
            print(f"Calculated Reward: {reward:.3f}")
            
            # Display percentages
            print(f"\n=== Score Breakdown ===")
            print(f"Score Accuracy: {analysis['score_accuracy']/40*100:.1f}%")
            print(f"Coherence: {analysis['coherence_score']/15*100:.1f}%")
            print(f"Keyword Verification: {analysis['keyword_verification_score']/5*100:.1f}%")
            print(f"Quality Keywords: {analysis['quality_keyword_score']/5*100:.1f}%")
            print(f"Overall Score: {analysis['total_analysis_score']:.1f}%")
            print()

async def test_batch_rewards():
    """Test batch reward calculation with multiple miners."""
    print("📊 Testing Batch Reward Calculation")
    print("=" * 50)
    
    validator = MockValidator()
    product = MockReviewedProduct(TEST_PRODUCTS["high_quality"])
    
    # Create responses from multiple miners
    responses = list(TEST_MINER_RESPONSES.values())
    miner_uids = list(range(len(responses)))
    
    print(f"📦 Product: {product.name} (Actual Score: {product.trustScore})")
    print(f"👥 Miners: {len(responses)}")
    print()
    
    print("📋 Individual Responses:")
    for i, (miner_name, response) in enumerate(TEST_MINER_RESPONSES.items()):
        print(f"   {i+1}. {miner_name}: Score={response['score']}, Keywords={response['keywords']}")
    print()
    
    # Calculate batch rewards
    rewards = await get_rewards(validator, product, responses, miner_uids)
    
    print("💰 Calculated Rewards:")
    for i, (miner_name, response) in enumerate(TEST_MINER_RESPONSES.items()):
        print(f"   {i+1}. {miner_name}: {rewards[i]:.2f}/100")
    
    print(f"\n📈 Reward Statistics:")
    print(f"   Average Reward: {rewards.mean():.2f}")
    print(f"   Max Reward: {rewards.max():.2f}")
    print(f"   Min Reward: {rewards.min():.2f}")
    print(f"   Miners with >0 reward: {sum(rewards > 0)}/{len(rewards)}")

async def test_database_storage():
    """Test that the database properly stores complete miner responses including reviews and keywords."""
    print("💾 Testing Database Storage")
    print("=" * 50)
    
    # Test product data
    test_product_id = "test_product_123"
    test_miner_id = 42
    
    # Test prediction data
    test_prediction = {
        "score": 85.0,
        "review": "Excellent DeFi protocol with strong security and experienced team. Highly recommended for serious investors.",
        "keywords": ["excellent", "trusted", "low-risk", "established", "promising"]
    }
    
    # Test analysis data
    test_analysis = {
        "sentiment": "positive",
        "keyword_verification_score": 4.5,
        "coherence_score": 12.0,
        "total_reward": 78.5
    }
    
    print(f"📦 Product ID: {test_product_id}")
    print(f"🧑‍💻 Miner ID: {test_miner_id}")
    print(f"📊 Prediction: {test_prediction}")
    print(f"🔍 Analysis: {test_analysis}")
    print()
    
    try:
        # Store the prediction with analysis data
        add_prediction(
            product_id=test_product_id,
            miner_id=test_miner_id,
            prediction_data=test_prediction,
            analysis_data=test_analysis
        )
        print("✅ Successfully stored prediction with analysis data")
        
        # Retrieve the stored prediction
        stored_predictions = get_predictions_for_product(test_product_id)
        
        if stored_predictions:
            stored_prediction = stored_predictions[0]
            print("📥 Retrieved stored prediction:")
            print(f"   Product ID: {stored_prediction.product_id}")
            print(f"   Miner ID: {stored_prediction.miner_id}")
            print(f"   Score: {stored_prediction.prediction}")
            print(f"   Review: {stored_prediction.review}")
            print(f"   Keywords: {stored_prediction.keywords}")
            print(f"   Sentiment: {stored_prediction.sentiment}")
            print(f"   Keyword Verification Score: {stored_prediction.keyword_verification_score}")
            print(f"   Coherence Score: {stored_prediction.coherence_score}")
            print(f"   Total Reward: {stored_prediction.total_reward}")
            print(f"   Created At: {stored_prediction.created_at}")
            print(f"   Updated At: {stored_prediction.updated_at}")
            
            # Verify the data was stored correctly
            assert stored_prediction.product_id == test_product_id
            assert stored_prediction.miner_id == test_miner_id
            assert stored_prediction.prediction == test_prediction["score"]
            assert stored_prediction.review == test_prediction["review"]
            
            # Parse keywords JSON
            stored_keywords = json.loads(stored_prediction.keywords) if stored_prediction.keywords else []
            assert stored_keywords == test_prediction["keywords"]
            
            assert stored_prediction.sentiment == test_analysis["sentiment"]
            assert stored_prediction.keyword_verification_score == test_analysis["keyword_verification_score"]
            assert stored_prediction.coherence_score == test_analysis["coherence_score"]
            assert stored_prediction.total_reward == test_analysis["total_reward"]
            
            print("✅ All data verified correctly!")
            
        else:
            print("❌ No predictions found in database")
            
        # Clean up - delete the test product
        delete_a_product(test_product_id)
        print("🧹 Cleaned up test data")
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_forward_function_storage():
    """Test that the forward function properly stores complete responses."""
    print("🔄 Testing Forward Function Storage")
    print("=" * 50)
    
    # Mock the forward function behavior
    test_queries = ["product_1", "product_2"]
    test_miner_uids = [1, 2]
    test_responses = [
        [
            {
                "score": 85.0,
                "review": "Excellent product with great features.",
                "keywords": ["excellent", "trusted", "low-risk"]
            },
            {
                "score": 65.0,
                "review": "Average product, nothing special.",
                "keywords": ["average", "moderate", "medium-risk"]
            }
        ],
        [
            {
                "score": 90.0,
                "review": "Outstanding quality and performance.",
                "keywords": ["outstanding", "premium", "high-quality"]
            },
            {
                "score": 45.0,
                "review": "Poor product with many issues.",
                "keywords": ["poor", "untrusted", "high-risk"]
            }
        ]
    ]
    
    print(f"📦 Products: {test_queries}")
    print(f"🧑‍💻 Miners: {test_miner_uids}")
    print()
    
    try:
        # Simulate the forward function storage logic
        for miner_uid, miner_predictions in zip(test_miner_uids, test_responses):
            for product_id, prediction in zip(test_queries, miner_predictions):
                print(f"💾 Storing: Miner {miner_uid} -> Product {product_id}")
                print(f"   Score: {prediction['score']}")
                print(f"   Keywords: {prediction['keywords']}")
                
                # Store the prediction
                add_prediction(
                    product_id=product_id,
                    miner_id=miner_uid,
                    prediction_data=prediction
                )
        
        print("✅ Successfully stored all predictions")
        
        # Verify storage for each product
        for product_id in test_queries:
            stored_predictions = get_predictions_for_product(product_id)
            print(f"\n📥 Product {product_id} predictions:")
            for pred in stored_predictions:
                print(f"   Miner {pred.miner_id}: Score={pred.prediction}, Keywords={pred.keywords}")
        
        # Clean up
        for product_id in test_queries:
            delete_a_product(product_id)
        print("🧹 Cleaned up test data")
        
    except Exception as e:
        print(f"❌ Forward function test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to run all tests."""
    print("🚀 Starting Standalone Validator Test")
    print("=" * 60)
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Run all tests
    asyncio.run(test_sentiment_analysis())
    asyncio.run(test_keyword_verification())
    asyncio.run(test_keyword_coherence())
    asyncio.run(test_individual_reward())
    asyncio.run(test_batch_rewards())
    asyncio.run(test_database_storage())
    asyncio.run(test_forward_function_storage())
    
    print("\n✅ Validator testing completed!")

if __name__ == "__main__":
    main() 