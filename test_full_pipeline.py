#!/usr/bin/env python3
"""
Full Pipeline Test Script
This script tests the complete miner-validator pipeline by:
1. Generating miner responses for real products from the API
2. Running validator reward calculations on those responses
3. Showing detailed scoring breakdown
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any, List, Optional

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import miner functions
from checkerchain.miner.llm import (
    generate_complete_assessment,
)
from checkerchain.validator.reward import (
    analyze_complete_response,
    reward,
    get_rewards,
    calculate_reward
)

# Import product fetching
from checkerchain.utils.checker_chain import fetch_product_data

import bittensor as bt

# Actual product IDs from the validator
REAL_PRODUCT_IDS = [
    '686013a8fd5aafa3e424152e',
    '685ffe18fd5aafa3e420f631',
    '685ff863fd5aafa3e420c38f',
    '685fea4efd5aafa3e41f12b9',
    '685fdbc0fd5aafa3e41d1dd0',
    '685fc784fd5aafa3e41a5ecb',
    '685fab9efd5aafa3e41602e0',
    '6860f867fd5aafa3e4471c10'
]

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
    def __init__(self, product_data, actual_score: float):
        self._id = product_data._id
        self.name = product_data.name
        self.slug = getattr(product_data, 'slug', 'test-slug')
        self.trustScore = actual_score

async def generate_miner_response(product_id: str) -> Optional[Dict[str, Any]]:
    """
    Generate a miner response for a real product from the API using single-request approach.
    """
    print(f"🔧 Generating miner response for: {product_id}")
    
    # Fetch real product data
    product = fetch_product_data(product_id)
    if not product:
        print(f"   ❌ Failed to fetch product data for ID: {product_id}")
        return None
    
    print(f"   📋 Product: {product.name}")
    
    try:
        # Convert product to dict format for the single-request function
        product_dict = {
            "name": product.name,
            "description": getattr(product, 'description', ''),
            "website": getattr(product, 'website', ''),
            "category": getattr(product, 'category', 'DeFi')
        }
        
        # Generate complete assessment in single request
        assessment = await generate_complete_assessment(product_dict)
        
        response = {
            "product_id": product_id,
            "score": assessment.get("score"),
            "review": assessment.get("review"),
            "keywords": assessment.get("keywords", [])
        }
        
        print(f"   ✅ Generated response: Score={response['score']}, Keywords={response['keywords']}, Reviews={response['review']}")
        return response
        
    except Exception as e:
        print(f"   ❌ Error generating response: {e}")
        return None

async def analyze_miner_response(response: Dict[str, Any], actual_score: float, product_name: str) -> Optional[Dict[str, Any]]:
    """
    Analyze a miner response using validator's single-request analysis.
    """
    print(f"🔍 Analyzing miner response for: {product_name}")
    
    score = response.get("score")
    review = response.get("review")
    keywords = response.get("keywords", [])
    
    if not score or not review or not keywords:
        print("   ❌ Invalid response - missing required fields")
        return None
    
    try:
        # Use single-request analysis
        analysis_result = await analyze_complete_response(response, actual_score)
        
        print(f"   🧠 Sentiment: {analysis_result['sentiment']}")
        print(f"   🔍 Keyword Verification: {analysis_result['keyword_verification_score']}/5")
        print(f"   🔗 Coherence Score: {analysis_result['coherence_score']}/15")
        print(f"   📊 Score Accuracy: {analysis_result['score_accuracy']}/40")
        print(f"   📈 Total Analysis Score: {analysis_result['total_analysis_score']}")
        
        # Calculate total reward using the new reward function
        validator = MockValidator()
        total_reward = await reward(validator, response, actual_score, 0)
        print(f"   💰 Total Reward: {total_reward:.2f}/100")
        
        return {
            "sentiment": analysis_result['sentiment'],
            "keyword_verification": analysis_result['keyword_verification_score'],
            "coherence_score": analysis_result['coherence_score'],
            "score_accuracy": analysis_result['score_accuracy'],
            "total_analysis_score": analysis_result['total_analysis_score'],
            "quality_keyword_score": analysis_result['quality_keyword_score'],
            "quality_keyword_count": analysis_result['quality_keyword_count'],
            "quality_keyword_matches": analysis_result['quality_keyword_matches'],
            "total_reward": total_reward
        }
        
    except Exception as e:
        print(f"   ❌ Error analyzing response: {e}")
        return None

async def test_single_product_pipeline(product_id: str, actual_score: Optional[float] = None):
    """
    Test the full pipeline for a single real product.
    """
    print(f"\n📦 Testing Product: {product_id}")
    print("=" * 60)
    
    # Fetch product data to get name
    product = fetch_product_data(product_id)
    if not product:
        print(f"❌ Failed to fetch product data for ID: {product_id}")
        return
    
    product_name = product.name
    print(f"Product Name: {product_name}")
    
    # Use actual score if provided, otherwise use a reasonable default
    if actual_score is None:
        actual_score = 65.0  # Default score for testing
    
    # Step 1: Generate miner response
    print("\n🚀 Step 1: Miner Response Generation")
    print("-" * 40)
    miner_response = await generate_miner_response(product_id)
    
    if not miner_response or not miner_response.get("score"):
        print("❌ Failed to generate miner response")
        return
    
    # Step 2: Analyze with validator
    print("\n🔍 Step 2: Validator Analysis")
    print("-" * 40)
    analysis = await analyze_miner_response(
        miner_response, 
        actual_score, 
        product_name
    )
    
    if not analysis:
        print("❌ Failed to analyze response")
        return
    
    # Step 3: Summary
    print("\n📊 Step 3: Summary")
    print("-" * 40)
    print(f"Product: {product_name}")
    print(f"Product ID: {product_id}")
    print(f"Actual Score: {actual_score}/100")
    print(f"Predicted Score: {miner_response['score']}/100")
    print(f"Score Accuracy: {100 - abs(miner_response['score'] - actual_score):.1f}%")
    print(f"Keywords: {miner_response['keywords']}")
    print(f"Review: {miner_response['review'][:100]}...")
    print(f"Sentiment: {analysis['sentiment']}")
    print(f"Keyword Quality: {analysis['keyword_verification']}/5")
    print(f"Coherence: {analysis['coherence_score']}/15")
    print(f"Final Reward: {analysis['total_reward']:.2f}/100")
    
    print(f"\n=== Analysis Results ===")
    print(f"Sentiment: {analysis['sentiment']}")
    print(f"Score Accuracy: {analysis['score_accuracy']:.1f}/40")
    print(f"Coherence Score: {analysis['coherence_score']:.1f}/20")
    print(f"Keyword Verification: {analysis['keyword_verification']:.1f}/5")
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
    print(f"Keyword Verification: {analysis['keyword_verification']/5*100:.1f}%")
    print(f"Quality Keywords: {analysis['quality_keyword_score']/5*100:.1f}%")
    print(f"Overall Score: {analysis['total_analysis_score']:.1f}%")
    
    return {
        "product_id": product_id,
        "product_name": product_name,
        "miner_response": miner_response,
        "analysis": analysis
    }

async def test_multiple_products_pipeline():
    """
    Test the pipeline with multiple real products.
    """
    print("\n🧪 Testing Multiple Real Products Pipeline")
    print("=" * 60)
    
    results = []
    
    for i, product_id in enumerate(REAL_PRODUCT_IDS[:5], 1):  # Test first 5 products
        print(f"\n--- Product {i}: {product_id} ---")
        
        result = await test_single_product_pipeline(product_id)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n📈 Pipeline Test Summary")
    print("=" * 40)
    print(f"Total Products Tested: {len(REAL_PRODUCT_IDS[:5])}")
    print(f"Successful Tests: {len(results)}")
    print(f"Success Rate: {len(results)/len(REAL_PRODUCT_IDS[:5])*100:.1f}%")
    
    if results:
        rewards = [r['analysis']['total_reward'] for r in results]
        scores = [r['miner_response']['score'] for r in results]
        
        print(f"Average Reward: {sum(rewards)/len(rewards):.2f}")
        print(f"Average Predicted Score: {sum(scores)/len(scores):.2f}")
        print(f"Best Reward: {max(rewards):.2f}")
        print(f"Worst Reward: {min(rewards):.2f}")

async def test_batch_rewards_with_real_products():
    """
    Test batch reward calculation with real products.
    """
    print("\n📊 Testing Batch Rewards with Real Products")
    print("=" * 60)
    
    # Generate responses for first 3 products
    all_responses = []
    all_products = []
    
    for i, product_id in enumerate(REAL_PRODUCT_IDS[:3]):
        product = fetch_product_data(product_id)
        if not product:
            continue
            
        # Create mock reviewed product with default score
        mock_product = MockReviewedProduct(product, 65.0)
        all_products.append(mock_product)
        
        # Generate 2 different miner responses for each product
        product_responses = []
        for j in range(2):
            print(f"\n--- Miner {j+1} for {product.name} ---")
            response = await generate_miner_response(product_id)
            if response:
                product_responses.append(response)
        
        all_responses.extend(product_responses)
    
    if not all_responses or not all_products:
        print("❌ No valid responses generated for batch testing")
        return
    
    # Test batch reward calculation
    print(f"\n📊 Batch Reward Calculation")
    print("-" * 40)
    print(f"Total Products: {len(all_products)}")
    print(f"Total Responses: {len(all_responses)}")
    
    validator = MockValidator()
    miner_uids = list(range(len(all_responses)))
    
    # Calculate rewards for each product
    for i, product in enumerate(all_products):
        print(f"\n📦 Product {i+1}: {product.name}")
        print(f"   Actual Score: {product.trustScore}")
        
        # Get responses for this product (2 responses per product)
        start_idx = i * 2
        end_idx = start_idx + 2
        product_responses = all_responses[start_idx:end_idx]
        product_miner_uids = miner_uids[start_idx:end_idx]
        
        if len(product_responses) > 0:
            # Calculate rewards
            rewards = await get_rewards(validator, product, product_responses, product_miner_uids)
            
            print(f"   Rewards: {[f'{r:.2f}' for r in rewards]}")
            print(f"   Average: {rewards.mean():.2f}")
            print(f"   Best: {rewards.max():.2f}")

def main():
    """Main function to run the full pipeline test."""
    print("🚀 Starting Full Pipeline Test with Real Products")
    print("=" * 60)
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    print(f"📋 Available Product IDs: {REAL_PRODUCT_IDS}")
    print()
    
    # Ask user what to test
    print("Choose test mode:")
    print("1. Single product test (quick)")
    print("2. Multiple products test (comprehensive)")
    print("3. Batch rewards test")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        # Test single product
        product_id = REAL_PRODUCT_IDS[0]
        asyncio.run(test_single_product_pipeline(product_id))
    elif choice == "2":
        # Test multiple products
        asyncio.run(test_multiple_products_pipeline())
    elif choice == "3":
        # Test batch rewards
        asyncio.run(test_batch_rewards_with_real_products())
    else:
        print("❌ Invalid choice. Running single product test...")
        product_id = REAL_PRODUCT_IDS[0]
        asyncio.run(test_single_product_pipeline(product_id))
    
    print("\n✅ Full pipeline testing completed!")

if __name__ == "__main__":
    main() 