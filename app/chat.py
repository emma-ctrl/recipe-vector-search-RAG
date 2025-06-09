#!/usr/bin/env python3
"""
Recipe RAG Assistant with Nebius AI Studio
A quick way to chat with your recipe database using Nebius AI
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        "./data/processed/recipe_search_engine.pkl",
        "./data/processed/faiss_index.index"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n🔧 Please run: python scripts/generate_embeddings.py")
        return False
    
    return True

def check_api_key():
    """Check if Nebius API key is set"""
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("NEBIUS_API_KEY"):
        print("❌ NEBIUS_API_KEY not found")
        print("\n🔧 Setup instructions:")
        print("1. Get API key from: https://studio.nebius.ai")
        print("2. Create .env file: cp .env.example .env")
        print("3. Add your API key to .env file")
        return False
    
    return True

def main():
    print("🍳 Recipe RAG Assistant with Nebius AI")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return
    
    if not check_api_key():
        return
    
    try:
        # Import and initialize assistant
        from nebius_rag_assistant import RecipeRAGWithNebius
        
        print("Loading your recipe assistant...")
        assistant = RecipeRAGWithNebius()
        
        print("\n✅ Ready! Ask me about recipes.")
        print("Examples:")
        print("  - 'What's good for a cozy dinner?'")
        print("  - 'I need something quick and healthy'")
        print("  - 'Suggest a chocolate dessert'")
        print("  - 'What can I make in 30 minutes?'")
        print("\nType 'quit' to exit.\n")
        
        # Start chat
        while True:
            try:
                user_input = input("🧑‍🍳 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("👋 Happy cooking!")
                    break
                
                if not user_input:
                    continue
                
                # Get response
                response = assistant.ask(user_input)
                print(f"\n🤖 Assistant: {response}\n")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 Happy cooking!")
                break
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check your .env file has NEBIUS_API_KEY")
        print("2. Ensure you have internet connection")
        print("3. Try running generate_embeddings.py again")
        print("4. Check if your Nebius API key is valid")

if __name__ == "__main__":
    main()