#!/usr/bin/env python3
"""
Test script to check Gemini API connection and basic functionality
"""
import os
from dotenv import load_dotenv
from gemini_api import GeminiClient
from schemas import ToolUseAction

def test_api_connection():
    """Test basic API connection and functionality"""
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment")
        print("Please check your .env file")
        return False
    
    print(f"✅ API key found (length: {len(api_key)})")
    
    try:
        # Create client
        client = GeminiClient(upload_file=False, fps=3.0)
        print("✅ GeminiClient created successfully")
        
        # Test simple text inference
        print("\n--- Testing Simple Text Inference ---")
        simple_prompt = "What is 2+2? Please respond with just the number."
        
        result = client.inference_text(
            prompt=simple_prompt,
            schema=None,
            history=False
        )
        
        if result is not None:
            print(f"✅ Simple text inference successful: {result[:100]}...")
        else:
            print("❌ Simple text inference failed")
            return False
        
        # Test structured response
        print("\n--- Testing Structured Response ---")
        structured_prompt = """
        You are analyzing a tool placement task. Please respond with:
        - toolname: one of "obj1", "obj2", "obj3" 
        - position: a list of two numbers [x, y] where both are between 0 and 600
        
        Choose obj1 at position [300, 300].
        """
        
        structured_result = client.inference_text(
            prompt=structured_prompt,
            schema=ToolUseAction,
            history=False
        )
        
        if structured_result is not None:
            print(f"✅ Structured response successful:")
            print(f"   Toolname: {structured_result.get('toolname', 'N/A')}")
            print(f"   Position: {structured_result.get('position', 'N/A')}")
            
            # Validate structure
            if ('toolname' in structured_result and 
                'position' in structured_result and 
                isinstance(structured_result['position'], list) and 
                len(structured_result['position']) == 2):
                print("✅ Response structure is valid")
            else:
                print("⚠️ Response structure is invalid")
                return False
        else:
            print("❌ Structured response failed")
            return False
        
        print("\n✅ All API tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ API test failed with error: {e}")
        return False

def diagnose_errors():
    """Provide suggestions for common errors"""
    print("\n--- Error Diagnosis ---")
    print("Common issues and solutions:")
    print("1. Rate limits: Wait longer between requests")
    print("2. Quota exceeded: Check your Google Cloud billing")
    print("3. Network issues: Check internet connection")
    print("4. Invalid API key: Verify your GOOGLE_API_KEY")
    print("5. Model access: Ensure you have access to gemini-2.5-flash")

if __name__ == "__main__":
    print("🔧 Testing Gemini API Connection...")
    success = test_api_connection()
    
    if not success:
        diagnose_errors()
    else:
        print("\n🎉 API is working correctly!")
