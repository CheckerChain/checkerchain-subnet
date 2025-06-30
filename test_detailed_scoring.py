#!/usr/bin/env python3
"""
Detailed Scoring Test Script
This script shows comprehensive scoring breakdowns for all components.
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


def print_scoring_table(title, data):
    """Print a formatted scoring table."""
    print(f"\n{title}")
    print("=" * 80)
    print(f"{'Component':<25} {'Score':<10} {'Max':<8} {'Percentage':<12} {'Status':<10}")
    print("-" * 80)
    
    for component, score, max_score in data:
        percentage = (score / max_score) * 100
        status = "✅ EXCELLENT" if percentage >= 90 else "🟢 GOOD" if percentage >= 70 else "🟡 AVERAGE" if percentage >= 50 else "🔴 POOR"
        print(f"{component:<25} {score:<10.1f} {max_score:<8.1f} {percentage:<12.1f}% {status:<10}")
    
    print("-" * 80)


async def test_detailed_miner_scoring():
    """Test miner assessment with detailed scoring breakdown."""
    print("⛏️ Testing Miner Assessment with Detailed Scoring")
    print("=" * 60)
    
    # Test product
    test_product = {
        "name": "Premium DeFi Protocol",
        "description": "A high-quality DeFi protocol with strong security measures, experienced team, and proven track record. Offers lending, borrowing, and yield farming services.",
        "website": "https://premium-defi.com",
        "category": "DeFi"
    }
    
    print(f"📦 Product: {test_product['name']}")
    print(f"📝 Description: {test_product['description'][:100]}...")
    print()
    
    try:
        # Generate assessment
        assessment = await generate_complete_assessment(test_product)
        
        # Analyze quality indicators
        quality_indicators = ["excellent", "good", "average", "poor", "scam", "trusted", "untrusted", 
                            "low-risk", "high-risk", "promising", "suspicious", "established", "failing"]
        quality_count = sum(1 for kw in assessment['keywords'] if any(indicator in kw.lower() for indicator in quality_indicators))
        
        # Prepare scoring data
        miner_scores = [
            ("Overall Score", assessment['score'], 100.0),
            ("Review Length", len(assessment['review']), 140.0),
            ("Keywords Count", len(assessment['keywords']), 7.0),
            ("Quality Keywords", quality_count, len(assessment['keywords'])),
        ]
        
        print_scoring_table("📊 Miner Assessment Scoring", miner_scores)
        
        print(f"\n📋 Generated Assessment:")
        print(f"   Score: {assessment['score']}/100")
        print(f"   Review: {assessment['review']}")
        print(f"   Keywords: {assessment['keywords']}")
        
        return assessment
        
    except Exception as e:
        print(f"❌ Miner test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_detailed_validator_scoring(prediction):
    """Test validator analysis with detailed scoring breakdown."""
    print("\n🔍 Testing Validator Analysis with Detailed Scoring")
    print("=" * 60)
    
    actual_score = 82.0
    
    print(f"📊 Miner Prediction:")
    print(f"   Score: {prediction['score']}/100")
    print(f"   Review: {prediction['review']}")
    print(f"   Keywords: {prediction['keywords']}")
    print(f"   Actual Score: {actual_score}/100")
    print()
    
    try:
        # Analyze complete response
        analysis = await analyze_complete_response(prediction, actual_score)
        
        # Calculate sentiment score
        sentiment_score = 20.0 if analysis['sentiment'] != 'unknown' else 5.0
        
        # Prepare scoring data
        validator_scores = [
            ("Score Accuracy", analysis['score_accuracy'], 40.0),
            ("Sentiment Analysis", sentiment_score, 20.0),
            ("Keyword Verification", analysis['keyword_verification_score'], 5.0),
            ("Coherence Analysis", analysis['coherence_score'], 15.0),
            ("Total Analysis", analysis['total_analysis_score'], 80.0),
        ]
        
        print_scoring_table("📊 Validator Analysis Scoring", validator_scores)
        
        print(f"\n📋 Analysis Results:")
        print(f"   Sentiment: {analysis['sentiment']}")
        print(f"   Score Accuracy: {analysis['score_accuracy']:.1f}/40")
        print(f"   Keyword Verification: {analysis['keyword_verification_score']:.1f}/5")
        print(f"   Coherence Score: {analysis['coherence_score']:.1f}/15")
        print(f"   Total Analysis Score: {analysis['total_analysis_score']:.1f}/80")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Validator test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_multiple_products_detailed():
    """Test multiple products with detailed scoring for each."""
    print("\n🔄 Testing Multiple Products with Detailed Scoring")
    print("=" * 60)
    
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
    
    results = []
    
    for i, product in enumerate(test_products, 1):
        print(f"\n📦 Product {i}: {product['name']}")
        print("-" * 50)
        
        # Generate assessment
        assessment = await generate_complete_assessment(product)
        
        # Analyze with validator
        analysis = await analyze_complete_response(assessment, assessment['score'])
        
        # Store results
        results.append({
            "product": product['name'],
            "assessment": assessment,
            "analysis": analysis
        })
        
        # Quick summary
        print(f"   Miner Score: {assessment['score']:.1f}/100")
        print(f"   Validator Score: {analysis['total_analysis_score']:.1f}/80")
        print(f"   Sentiment: {analysis['sentiment']}")
    
    # Summary table
    print(f"\n📊 Summary of All Products")
    print("=" * 80)
    print(f"{'Product':<30} {'Miner Score':<12} {'Validator Score':<15} {'Sentiment':<12} {'Status':<10}")
    print("-" * 80)
    
    for result in results:
        miner_score = result['assessment']['score']
        validator_score = result['analysis']['total_analysis_score']
        sentiment = result['analysis']['sentiment']
        
        # Determine overall status
        avg_score = (miner_score + (validator_score/80)*100) / 2
        status = "✅ EXCELLENT" if avg_score >= 80 else "🟢 GOOD" if avg_score >= 60 else "🟡 AVERAGE" if avg_score >= 40 else "🔴 POOR"
        
        print(f"{result['product']:<30} {miner_score:<12.1f} {validator_score:<15.1f} {sentiment:<12} {status:<10}")
    
    print("-" * 80)
    
    return results


def main():
    """Main function to run all detailed scoring tests."""
    print("🚀 Starting Detailed Scoring Tests")
    print("=" * 60)
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Run all tests
    asyncio.run(test_detailed_miner_scoring())
    asyncio.run(test_detailed_validator_scoring({
        "score": 85.0,
        "review": "Excellent DeFi protocol with strong security and experienced team.",
        "keywords": ["excellent", "trusted", "low-risk", "established", "promising"]
    }))
    asyncio.run(test_multiple_products_detailed())
    
    print("\n✅ Detailed scoring testing completed!")


if __name__ == "__main__":
    main() 