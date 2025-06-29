#!/usr/bin/env python3
"""
Simple test script to verify single OpenAI request functionality.
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
    print("⛏️ Testing Miner Single-Request Assessment")
    print("=" * 50)
    
    # Mock product data
    test_product = {
        "name": "Premium DeFi Protocol",
        "description": "A high-quality DeFi protocol with strong security measures, experienced team, and proven track record. Offers lending, borrowing, and yield farming services.",
        "website": "https://premium-defi.com",
        "category": "DeFi"
    }
    
    print(f"📦 Product: {test_product['name']}")
    print(f"📝 Description: {test_product['description']}")
    print()
    
    try:
        # Generate complete assessment in single request
        assessment = await generate_complete_assessment(test_product)
        
        print("✅ Generated Assessment:")
        print(f"   Score: {assessment['score']}/100")
        print(f"   Review: {assessment['review']}")
        print(f"   Keywords: {assessment['keywords']}")
        print()
        
        # Validate response structure
        assert "score" in assessment
        assert "review" in assessment
        assert "keywords" in assessment
        assert isinstance(assessment["score"], (int, float))
        assert isinstance(assessment["review"], str)
        assert isinstance(assessment["keywords"], list)
        assert len(assessment["keywords"]) >= 3
        assert len(assessment["keywords"]) <= 7
        
        print("✅ Response structure validated!")
        return assessment
        
    except Exception as e:
        print(f"❌ Miner test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_validator_single_request(prediction):
    """Test validator's single-request analysis with detailed scoring breakdown."""
    print("\n🔍 Testing Validator Single-Request Analysis")
    print("=" * 50)
    
    actual_score = 82.0
    
    print(f"📊 Miner Prediction:")
    print(f"   Score: {prediction['score']}/100")
    print(f"   Review: {prediction['review']}")
    print(f"   Keywords: {prediction['keywords']}")
    print(f"   Actual Score: {actual_score}/100")
    print()
    
    try:
        # Analyze complete response in single request
        analysis = await analyze_complete_response(prediction, actual_score)
        
        print("✅ Generated Analysis:")
        print(f"   Sentiment: {analysis['sentiment']}")
        print(f"   Keyword Verification Score: {analysis['keyword_verification_score']}/5")
        print(f"   Coherence Score: {analysis['coherence_score']}/15")
        print(f"   Score Accuracy: {analysis['score_accuracy']}/40")
        print(f"   Total Analysis Score: {analysis['total_analysis_score']}")
        print()
        
        # Detailed scoring breakdown
        print("📈 Detailed Scoring Breakdown:")
        print("   " + "="*40)
        print(f"   Score Accuracy:     {analysis['score_accuracy']:>6.1f}/40.0  ({(analysis['score_accuracy']/40)*100:>5.1f}%)")
        sentiment_score = 20.0 if analysis['sentiment'] != 'unknown' else 5.0
        sentiment_percent = (sentiment_score/20)*100
        print(f"   Sentiment Analysis: {sentiment_score:>6.1f}/20.0  ({sentiment_percent:>5.1f}%)")
        print(f"   Keyword Verification: {analysis['keyword_verification_score']:>6.1f}/5.0   ({(analysis['keyword_verification_score']/5)*100:>5.1f}%)")
        print(f"   Coherence Analysis:  {analysis['coherence_score']:>6.1f}/15.0  ({(analysis['coherence_score']/15)*100:>5.1f}%)")
        print("   " + "-"*40)
        print(f"   Total Performance:   {analysis['total_analysis_score']:>6.1f}/80.0  ({(analysis['total_analysis_score']/80)*100:>5.1f}%)")
        print()
        
        # Validate response structure
        assert "sentiment" in analysis
        assert "keyword_verification_score" in analysis
        assert "coherence_score" in analysis
        assert "score_accuracy" in analysis
        assert "total_analysis_score" in analysis
        
        print("✅ Analysis structure validated!")
        return analysis
        
    except Exception as e:
        print(f"❌ Validator test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_multiple_assessments():
    """Test multiple assessments to verify consistency with detailed scoring."""
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
            print(f"   Score Accuracy: {analysis['score_accuracy']:.1f}/40")
            print(f"   Keyword Verification: {analysis['keyword_verification_score']:.1f}/5")
            print(f"   Coherence Score: {analysis['coherence_score']:.1f}/15")
            print(f"   Total Analysis Score: {analysis['total_analysis_score']:.1f}/80")
            print()
        
        print("✅ Multiple assessments completed successfully!")
        
    except Exception as e:
        print(f"❌ Multiple assessments test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_full_pipeline_simple():
    """Test a simple full pipeline with detailed scoring breakdown."""
    print("\n🚀 Testing Simple Full Pipeline")
    print("=" * 50)
    
    # Test product
    test_product = {
        "name": "Test DeFi Protocol",
        "description": "A test DeFi protocol for pipeline testing.",
        "website": "https://test-defi.com",
        "category": "DeFi"
    }
    
    print(f"📦 Product: {test_product['name']}")
    
    try:
        # Step 1: Generate miner assessment
        print("\n⛏️ Step 1: Miner Assessment")
        assessment = await generate_complete_assessment(test_product)
        print(f"   Score: {assessment['score']}/100")
        print(f"   Review: {assessment['review']}")
        print(f"   Keywords: {assessment['keywords']}")
        
        # Step 2: Analyze with validator
        print("\n🔍 Step 2: Validator Analysis")
        analysis = await analyze_complete_response(assessment, assessment['score'])
        print(f"   Sentiment: {analysis['sentiment']}")
        print(f"   Keyword Verification: {analysis['keyword_verification_score']}/5")
        print(f"   Coherence Score: {analysis['coherence_score']}/15")
        print(f"   Score Accuracy: {analysis['score_accuracy']}/40")
        print(f"   Total Analysis Score: {analysis['total_analysis_score']}/80")
        
        # Detailed scoring breakdown
        print("\n📊 Detailed Scoring Breakdown:")
        print("   " + "="*50)
        print(f"   Score Accuracy:     {analysis['score_accuracy']:>6.1f}/40.0  ({(analysis['score_accuracy']/40)*100:>5.1f}%)")
        sentiment_score = 20.0 if analysis['sentiment'] != 'unknown' else 5.0
        sentiment_percent = (sentiment_score/20)*100
        print(f"   Sentiment Analysis: {sentiment_score:>6.1f}/20.0  ({sentiment_percent:>5.1f}%)")
        print(f"   Keyword Verification: {analysis['keyword_verification_score']:>6.1f}/5.0   ({(analysis['keyword_verification_score']/5)*100:>5.1f}%)")
        print(f"   Coherence Analysis:  {analysis['coherence_score']:>6.1f}/15.0  ({(analysis['coherence_score']/15)*100:>5.1f}%)")
        print("   " + "-"*50)
        print(f"   Total Performance:   {analysis['total_analysis_score']:>6.1f}/80.0  ({(analysis['total_analysis_score']/80)*100:>5.1f}%)")
        
        print("\n✅ Full pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run all single-request tests."""
    print("🚀 Starting Simple Single-Request Tests")
    print("=" * 60)
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Run all tests
    asyncio.run(test_miner_single_request())
    asyncio.run(test_validator_single_request({
        "score": 85.0,
        "review": "Excellent DeFi protocol with strong security and experienced team.",
        "keywords": ["excellent", "trusted", "low-risk", "established", "promising"]
    }))
    asyncio.run(test_multiple_assessments())
    asyncio.run(test_full_pipeline_simple())
    
    print("\n✅ Simple single-request testing completed!")


if __name__ == "__main__":
    main() 