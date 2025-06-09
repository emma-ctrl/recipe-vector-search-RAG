import os
import pickle
import faiss
import requests
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
from dotenv import load_dotenv

class RecipeRAGWithNebius:
    def __init__(self, search_engine_path: str = "./data/processed/recipe_search_engine.pkl",
                 faiss_index_path: str = "./data/processed/faiss_index.index"):
        """Initialize RAG assistant with Nebius AI Studio"""
        load_dotenv()
        
        # Load search components
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.load_search_engine(search_engine_path, faiss_index_path)
        
        # Nebius AI Studio configuration
        self.nebius_api_key = os.getenv("NEBIUS_API_KEY")
        self.nebius_base_url = "https://api.studio.nebius.ai/v1"
        
        if not self.nebius_api_key:
            raise ValueError("NEBIUS_API_KEY not found in environment variables")
        
    def load_search_engine(self, search_engine_path: str, faiss_index_path: str):
        """Load the pre-built search engine"""
        print("Loading search engine...")
        
        # Load recipe data and embeddings
        with open(search_engine_path, 'rb') as f:
            search_data = pickle.load(f)
        
        self.recipes = search_data['recipes']
        self.embeddings = search_data['embeddings']
        
        # Load FAISS index
        self.index = faiss.read_index(faiss_index_path)
        
        print(f"✅ Search engine loaded with {len(self.recipes)} recipes")
    
    def search_recipes(self, query: str, k: int = 3) -> List[Tuple[Dict, float]]:
        """Search for relevant recipes using vector similarity"""
        # Generate embedding for query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Return results with scores
        results = []
        for idx, score in zip(indices[0], scores[0]):
            recipe = self.recipes[idx]
            results.append((recipe, float(score)))
        
        return results
    
    def format_recipes_for_llm(self, recipes: List[Tuple[Dict, float]]) -> str:
        """Format retrieved recipes for LLM context"""
        context = "Here are relevant recipes from the database:\n\n"
        
        for i, (recipe, score) in enumerate(recipes, 1):
            context += f"Recipe {i}: {recipe['title']}\n"
            
            if recipe.get('cuisines'):
                context += f"Cuisine: {', '.join(recipe['cuisines'])}\n"
            
            if recipe.get('readyInMinutes'):
                context += f"Ready in: {recipe['readyInMinutes']} minutes\n"
            
            if recipe.get('servings'):
                context += f"Serves: {recipe['servings']} people\n"
            
            if recipe.get('ingredients'):
                context += f"Ingredients: {recipe['ingredients'][:200]}...\n"
            
            if recipe.get('instructions'):
                instructions = recipe['instructions'][:300]
                context += f"Instructions: {instructions}...\n"
            
            context += f"Match score: {score:.3f}\n\n"
        
        return context
    
    def generate_response(self, user_query: str, recipe_context: str) -> str:
        """Generate LLM response using Nebius AI Studio"""
        system_prompt = """You are a helpful recipe assistant. Based on the user's question and the provided recipes, give personalized recommendations and cooking advice.

Be conversational, reference specific recipes when relevant, and provide practical cooking tips."""

        user_prompt = f"""User Question: {user_query}

Available Recipes:
{recipe_context}

Please provide a helpful response based on the user's question and the recipes above."""

        headers = {
            "Authorization": f"Bearer {self.nebius_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 400,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                f"{self.nebius_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                response_data = response.json()
                return response_data["choices"][0]["message"]["content"]
            else:
                return f"API Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def ask(self, user_query: str) -> str:
        """Main RAG pipeline: search + generate"""
        print(f"🔍 Searching: '{user_query}'")
        
        # Step 1: Retrieve relevant recipes
        relevant_recipes = self.search_recipes(user_query, k=3)
        
        # Step 2: Format context for LLM
        recipe_context = self.format_recipes_for_llm(relevant_recipes)
        
        # Step 3: Generate response
        print("🤖 Generating response...")
        response = self.generate_response(user_query, recipe_context)
        
        return response

def main():
    """Simple chat interface"""
    print("🍳 Recipe RAG Assistant")
    print("=" * 40)
    
    try:
        assistant = RecipeRAGWithNebius()
        
        print("✅ Ready! Ask about recipes (type 'quit' to exit)\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Happy cooking!")
                break
            
            if not user_input:
                continue
            
            try:
                response = assistant.ask(user_input)
                print(f"\nAssistant: {response}\n")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Check your .env file has NEBIUS_API_KEY")

if __name__ == "__main__":
    main()