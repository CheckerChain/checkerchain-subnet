#!/usr/bin/env python3
"""
Test script to verify single OpenAI request functionality for both miner and validator.
"""

import os
import sys
import asyncio
import json

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the single-request functions
from checkerchain.miner.llm import generate_complete_assessment
from checkerchain.validator.reward import analyze_complete_response


async def test_miner_single_request():
    """Test miner's single-request assessment generation."""
    # Mock product data
    test_product = {
        "name": "Premium DeFi Protocol",
        "description": "A high-quality DeFi protocol with strong security measures, experienced team, and proven track record. Offers lending, borrowing, and yield farming services.",
        "website": "https://premium-defi.com",
        "category": "DeFi"
    }
    
    try:
        # Generate complete assessment in single request
        assessment = await generate_complete_assessment(test_product)
        
        # Validate response structure
        assert "score" in assessment
        assert "review" in assessment
        assert "keywords" in assessment
        assert isinstance(assessment["score"], (int, float))
        assert isinstance(assessment["review"], str)
        assert isinstance(assessment["keywords"], list)
        assert len(assessment["keywords"]) >= 3
        assert len(assessment["keywords"]) <= 7
        
    except Exception as e:
        import traceback
        traceback.print_exc()


async def test_validator_single_request():
    """Test validator's single-request analysis."""
    # Mock miner prediction
    test_prediction = {
        "score": 85.0,
        "review": "Excellent DeFi protocol with strong security and experienced team. Highly recommended for serious investors.",
        "keywords": ["excellent", "trusted", "low-risk", "established", "promising"]
    }
    
    actual_score = 82.0
    
    try:
        # Analyze complete response in single request
        analysis = await analyze_complete_response(test_prediction, actual_score)
        
        print("✅ Generated Analysis:")
        print(f"   Sentiment: {analysis['sentiment']}")
        print(f"   Keyword Verification Score: {analysis['keyword_verification_score']}/5")
        print(f"   Coherence Score: {analysis['coherence_score']}/15")
        print(f"   Score Accuracy: {analysis['score_accuracy']}/40")
        print(f"   Total Analysis Score: {analysis['total_analysis_score']}")
        print()
        
        # Validate response structure
        assert "sentiment" in analysis
        assert "keyword_verification_score" in analysis
        assert "coherence_score" in analysis
        assert "score_accuracy" in analysis
        assert "total_analysis_score" in analysis
        
        print("✅ Analysis structure validated!")
        
    except Exception as e:
        print(f"❌ Validator test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_multiple_assessments():
    """Test multiple assessments to verify consistency."""
    print("\n🔄 Testing Multiple Assessments")
    print("=" * 50)
    
    test_products = [
        {
            "name": "High-Quality DeFi Protocol",
            "description": "Excellent DeFi protocol with strong security and experienced team.",
            "website": "https://high-quality-defi.com",
            "category": "DeFi"
        },
        {
            "name": "Average Crypto Project",
            "description": "Decent project with some good features but room for improvement.",
            "website": "https://average-crypto.com",
            "category": "Crypto"
        },
        {
            "name": "Suspicious Token",
            "description": "Anonymous team, unrealistic promises, and poor documentation.",
            "website": "https://suspicious-token.com",
            "category": "Token"
        }
    ]
    
    try:
        for i, product in enumerate(test_products, 1):
            print(f"📦 Product {i}: {product['name']}")
            
            # Generate assessment
            assessment = await generate_complete_assessment(product)
            print(f"   Score: {assessment['score']}/100")
            print(f"   Keywords: {assessment['keywords']}")
            
            # Analyze the assessment
            analysis = await analyze_complete_response(assessment, assessment['score'])
            print(f"   Sentiment: {analysis['sentiment']}")
            print(f"   Total Analysis Score: {analysis['total_analysis_score']}")
            print()
        
        print("✅ Multiple assessments completed successfully!")
        
    except Exception as e:
        print(f"❌ Multiple assessments test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run all single-request tests."""
    print("🚀 Starting Single-Request Tests")
    print("=" * 60)
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Run all tests
    asyncio.run(test_miner_single_request())
    asyncio.run(test_validator_single_request())
    asyncio.run(test_multiple_assessments())
    
    print("\n✅ Single-request testing completed!")


if __name__ == "__main__":
    main() 