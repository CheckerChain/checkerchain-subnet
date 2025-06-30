#!/usr/bin/env python3
"""
Test Runner Script
This script allows you to run different tests for the CheckerChain subnet.
"""

import os
import sys
import subprocess

def check_requirements():
    """Check if required environment variables are set."""
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return False
    return True

def run_test(test_name):
    """Run a specific test."""
    if not check_requirements():
        return
    
    test_files = {
        "1": "test_miner_standalone.py",
        "2": "test_validator_standalone.py", 
        "3": "test_full_pipeline.py",
        "4": "test_single_request_simple.py",
        "5": "test_database_storage.py"
    }
    
    if test_name not in test_files:
        print(f"❌ Unknown test: {test_name}")
        return
    
    test_file = test_files[test_name]
    print(f"🚀 Running {test_file}...")
    print("=" * 60)
    
    try:
        result = subprocess.run([sys.executable, test_file], check=True)
        print(f"\n✅ {test_file} completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {test_file} failed with error code: {e.returncode}")
    except FileNotFoundError:
        print(f"\n❌ Test file {test_file} not found!")

def main():
    """Main function."""
    print("🧪 CheckerChain Subnet Test Runner")
    print("=" * 40)
    print()
    print("Available tests:")
    print("1. Miner Standalone Test - Test miner response generation (single-request)")
    print("2. Validator Standalone Test - Test validator reward system (single-request)")
    print("3. Full Pipeline Test - Test complete miner-validator pipeline")
    print("4. Single Request Test - Test single OpenAI request functionality")
    print("5. Database Storage Test - Test database storage of complete responses")
    print()
    
    while True:
        choice = input("Enter test number (1-5) or 'q' to quit: ").strip()
        
        if choice.lower() == 'q':
            print("👋 Goodbye!")
            break
        elif choice in ['1', '2', '3', '4', '5']:
            run_test(choice)
            print("\n" + "=" * 60)
        else:
            print("❌ Invalid choice. Please enter 1-5 or 'q'.")

if __name__ == "__main__":
    main() 