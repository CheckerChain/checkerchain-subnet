#!/usr/bin/env python3
"""
Standalone Miner Test Script
This script tests the miner's response generation without using btcli.
It simulates the miner's forward function and shows the generated responses.
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from checkerchain.miner.llm import generate_complete_assessment
from checkerchain.utils.checker_chain import fetch_product_data
from checkerchain.types.checker_chain import UnreviewedProduct
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

async def test_miner_response_with_real_product(product_id: str):
    """
    Test the miner's response generation for a real product from the API using single-request approach.
    """
    print(f"🔧 Testing Miner Response for Real Product: {product_id}")
    print("=" * 70)
    
    # Fetch real product data
    product = fetch_product_data(product_id)
    if not product:
        print(f"❌ Failed to fetch product data for ID: {product_id}")
        return None
    
    print(f"📋 Product Details:")
    print(f"   Name: {product.name}")
    print(f"   Description: {product.description[:100]}...")
    print(f"   Category: {product.category}")
    print(f"   Network: {product.network}")
    print(f"   Team Size: {len(product.teams)}")
    print(f"   Location: {product.location}")
    print()
    
    try:
        # Convert product to dict format for single-request function
        product_dict = {
            "name": product.name,
            "description": product.description,
            "website": getattr(product, 'website', ''),
            "category": product.category
        }
        
        # Generate complete assessment in single request
        print("🎯 Generating Complete Assessment (Single Request)...")
        assessment = await generate_complete_assessment(product_dict)
        
        print(f"   Overall Score: {assessment['score']}/100")
        print(f"   Review: {assessment['review']}")
        print(f"   Keywords: {assessment['keywords']}")
        print()
        
        # Detailed assessment breakdown
        print("📊 Assessment Details:")
        print("   " + "="*40)
        print(f"   Score:              {assessment['score']:>6.1f}/100.0  ({(assessment['score']/100)*100:>5.1f}%)")
        print(f"   Review Length:      {len(assessment['review']):>6d}/140    ({(len(assessment['review'])/140)*100:>5.1f}%)")
        print(f"   Keywords Count:     {len(assessment['keywords']):>6d}/7      ({(len(assessment['keywords'])/7)*100:>5.1f}%)")
        
        # Quality keyword analysis
        quality_indicators = ["excellent", "good", "average", "poor", "scam", "trusted", "untrusted", 
                            "low-risk", "high-risk", "promising", "suspicious", "established", "failing"]
        quality_count = sum(1 for kw in assessment['keywords'] if any(indicator in kw.lower() for indicator in quality_indicators))
        quality_percent = (quality_count/len(assessment['keywords']))*100 if assessment['keywords'] else 0
        print(f"   Quality Keywords:   {quality_count:>6d}/{len(assessment['keywords'])}     ({quality_percent:>5.1f}%)")
        print()
        
        # Compile final response
        print("📦 Compiling Final Response...")
        final_response = {
            "product_id": product_id,
            "score": assessment['score'],
            "review": assessment['review'],
            "keywords": assessment['keywords']
        }
        
        print("✅ Final Miner Response:")
        print(json.dumps(final_response, indent=2))
        print()
        
        # Validate response format
        print("🔍 Response Validation...")
        score = final_response.get("score")
        review = final_response.get("review")
        keywords = final_response.get("keywords", [])
        
        print(f"   Score Validation: {'✅' if score is not None and 0 <= score <= 100 else '❌'}")
        print(f"   Review Validation: {'✅' if review and len(review) <= 140 else '❌'}")
        print(f"   Keywords Count: {'✅' if 3 <= len(keywords) <= 7 else '❌'} ({len(keywords)} keywords)")
        print(f"   Keywords Quality: {'✅' if all(len(kw.strip()) > 0 for kw in keywords) else '❌'}")
        
        # Check if keywords are quality-descriptive
        quality_indicators = ["excellent", "good", "average", "poor", "scam", "trusted", "untrusted", 
                            "low-risk", "high-risk", "promising", "suspicious", "established", "failing"]
        quality_count = sum(1 for kw in keywords if any(indicator in kw.lower() for indicator in quality_indicators))
        print(f"   Quality Keywords: {'✅' if quality_count >= 3 else '❌'} ({quality_count}/{len(keywords)} are quality indicators)")
        
        return final_response
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_multiple_real_products():
    """
    Test multiple real products from the API.
    """
    print("🧪 Testing Multiple Real Products")
    print("=" * 70)
    
    successful_responses = []
    
    for i, product_id in enumerate(REAL_PRODUCT_IDS, 1):
        print(f"\n🔍 Test {i}: Product ID {product_id}")
        print("-" * 50)
        
        response = await test_miner_response_with_real_product(product_id)
        
        if response:
            successful_responses.append(response)
            print(f"📊 Response Summary for Product {i}:")
            print(f"   Score: {response['score']}/100")
            print(f"   Keywords: {response['keywords']}")
            print(f"   Review: {response['review'][:50]}...")
        else:
            print(f"❌ Failed to generate response for product {i}")
    
    print(f"\n📈 Summary:")
    print(f"   Total Products: {len(REAL_PRODUCT_IDS)}")
    print(f"   Successful Responses: {len(successful_responses)}")
    print(f"   Success Rate: {len(successful_responses)/len(REAL_PRODUCT_IDS)*100:.1f}%")
    
    if successful_responses:
        scores = [r['score'] for r in successful_responses if r['score'] is not None]
        if scores:
            print(f"   Average Score: {sum(scores)/len(scores):.2f}")
            print(f"   Score Range: {min(scores):.2f} - {max(scores):.2f}")

async def test_single_product():
    """
    Test a single product for quick debugging.
    """
    print("🔧 Testing Single Real Product")
    print("=" * 50)
    
    # Test the first product
    product_id = REAL_PRODUCT_IDS[0]
    response = await test_miner_response_with_real_product(product_id)
    
    if response:
        print(f"\n✅ Successfully tested product: {product_id}")
        return response
    else:
        print(f"\n❌ Failed to test product: {product_id}")
        return None

def main():
    """Main function to run the tests."""
    print("🚀 Starting Standalone Miner Test with Real Products")
    print("=" * 70)
    
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
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(test_single_product())
    elif choice == "2":
        asyncio.run(test_multiple_real_products())
    else:
        print("❌ Invalid choice. Running single product test...")
        asyncio.run(test_single_product())
    
    print("\n✅ Miner testing completed!")

if __name__ == "__main__":
    main() 