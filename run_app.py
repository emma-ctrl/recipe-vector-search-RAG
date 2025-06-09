#!/usr/bin/env python3
"""
Recipe RAG Assistant Launcher

Simple script to run the Gradio web interface
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if required files exist"""
    required_files = [
        "./data/processed/recipe_search_engine.pkl",
        "./data/processed/faiss_index.index",
        ".env"
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print("❌ Missing required files:")
        for file in missing:
            print(f"   - {file}")
        
        if ".env" in missing:
            print("\n🔧 Setup .env file:")
            print("   1. cp .env.example .env")
            print("   2. Add your NEBIUS_API_KEY to .env")
        
        if any("processed" in f for f in missing):
            print("\n🔧 Generate embeddings:")
            print("   python scripts/generate_embeddings.py")
        
        return False
    
    return True

def main():
    """Launch the Recipe RAG Assistant"""
    print("🍳 Recipe RAG Assistant Launcher")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please fix the issues above before running.")
        return
    
    try:
        print("🚀 Starting Gradio interface...")
        print("   This will open in your browser automatically")
        print("   Press Ctrl+C to stop\n")
        
        # Import and run the Gradio app using proper package import
        from app.gradio_app import create_gradio_interface
        
        demo = create_gradio_interface()
        demo.launch(
            share=True,  # Creates shareable link
            server_port=7860,
            show_error=True
        )
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check your .env file has NEBIUS_API_KEY")
        print("2. Ensure all requirements are installed: pip install -r requirements.txt")
        print("3. Make sure embeddings are generated: python scripts/generate_embeddings.py")

if __name__ == "__main__":
    main()