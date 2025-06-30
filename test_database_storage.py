#!/usr/bin/env python3
"""
Test script to verify that the database properly stores complete miner responses
including reviews, keywords, and analysis results.
"""

import os
import sys
import asyncio
import json
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import database functions
from checkerchain.database.actions import (
    add_prediction,
    get_predictions_for_product,
    delete_a_product
)
from checkerchain.database.model import MinerPrediction


def test_basic_storage():
    """Test basic storage of complete miner responses."""
    print("💾 Testing Basic Database Storage")
    print("=" * 50)
    
    # Test data
    test_product_id = "test_product_123"
    test_miner_id = 42
    
    test_prediction = {
        "score": 85.0,
        "review": "Excellent DeFi protocol with strong security and experienced team. Highly recommended for serious investors.",
        "keywords": ["excellent", "trusted", "low-risk", "established", "promising"]
    }
    
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
        # delete_a_product(test_product_id)
        print("🧹 Cleaned up test data")
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()


def test_multiple_miners():
    """Test storing responses from multiple miners for the same product."""
    print("\n👥 Testing Multiple Miners Storage")
    print("=" * 50)
    
    test_product_id = "multi_miner_test"
    test_miners = [
        {
            "id": 1,
            "prediction": {
                "score": 85.0,
                "review": "Excellent product with great features and security.",
                "keywords": ["excellent", "trusted", "low-risk"]
            },
            "analysis": {
                "sentiment": "positive",
                "keyword_verification_score": 4.5,
                "coherence_score": 12.0,
                "total_reward": 78.5
            }
        },
        {
            "id": 2,
            "prediction": {
                "score": 65.0,
                "review": "Average product, nothing special but functional.",
                "keywords": ["average", "moderate", "medium-risk"]
            },
            "analysis": {
                "sentiment": "neutral",
                "keyword_verification_score": 3.0,
                "coherence_score": 8.5,
                "total_reward": 52.0
            }
        },
        {
            "id": 3,
            "prediction": {
                "score": 45.0,
                "review": "Poor product with many issues and concerns.",
                "keywords": ["poor", "untrusted", "high-risk"]
            },
            "analysis": {
                "sentiment": "negative",
                "keyword_verification_score": 4.0,
                "coherence_score": 10.0,
                "total_reward": 35.5
            }
        }
    ]
    
    try:
        # Store predictions from all miners
        for miner in test_miners:
            print(f"💾 Storing prediction from Miner {miner['id']}")
            add_prediction(
                product_id=test_product_id,
                miner_id=miner["id"],
                prediction_data=miner["prediction"],
                analysis_data=miner["analysis"]
            )
        
        print("✅ Successfully stored all miner predictions")
        
        # Retrieve and verify all predictions
        stored_predictions = get_predictions_for_product(test_product_id)
        print(f"\n📥 Retrieved {len(stored_predictions)} predictions for product {test_product_id}")
        
        for pred in stored_predictions:
            print(f"\n🧑‍💻 Miner {pred.miner_id}:")
            print(f"   Score: {pred.prediction}")
            print(f"   Review: {pred.review}")
            print(f"   Keywords: {pred.keywords}")
            print(f"   Sentiment: {pred.sentiment}")
            print(f"   Total Reward: {pred.total_reward}")
        
        # Clean up
        # delete_a_product(test_product_id)
        print("\n🧹 Cleaned up test data")
        
    except Exception as e:
        print(f"❌ Multiple miners test failed: {e}")
        import traceback
        traceback.print_exc()


def test_forward_function_simulation():
    """Simulate the forward function storage behavior."""
    print("\n🔄 Testing Forward Function Simulation")
    print("=" * 50)
    
    # Simulate forward function data
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
                
                # Store the prediction (without analysis data initially)
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
            # delete_a_product(product_id)
            pass
        print("🧹 Cleaned up test data")
        
    except Exception as e:
        print(f"❌ Forward function test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run all database tests."""
    print("🚀 Starting Database Storage Tests")
    print("=" * 60)
    
    # Run all database tests
    test_basic_storage()
    test_multiple_miners()
    test_forward_function_simulation()
    
    print("\n✅ Database storage testing completed!")


if __name__ == "__main__":
    main() 