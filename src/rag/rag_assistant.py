"""
Recipe RAG Assistant - Core RAG logic separated from UI
"""
import os
import requests
from typing import List, Dict, Tuple
from dotenv import load_dotenv

from .recipe_search import RecipeSearchEngine

class RecipeRAGAssistant:
    """RAG Assistant that combines semantic search with LLM generation"""
    
    def __init__(self, search_engine_path: str = "./data/processed/recipe_search_engine.pkl",
                 faiss_index_path: str = "./data/processed/faiss_index.index",
                 api_provider: str = "nebius"):
        """Initialize RAG assistant"""
        load_dotenv()
        
        # Initialize search engine
        self.search_engine = RecipeSearchEngine()
        self.search_engine.load_search_engine(search_engine_path, faiss_index_path)
        
        # Setup LLM provider
        self.api_provider = api_provider
        self._setup_llm_client()
        
    def _setup_llm_client(self):
        """Setup the LLM client based on provider"""
        if self.api_provider == "nebius":
            self.api_key = os.getenv("NEBIUS_API_KEY")
            self.base_url = "https://api.studio.nebius.ai/v1"
            self.model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            
            if not self.api_key:
                raise ValueError("NEBIUS_API_KEY not found in environment variables")
                
        elif self.api_provider == "openai":
            import openai
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = "gpt-3.5-turbo"
            
        else:
            raise ValueError(f"Unsupported API provider: {self.api_provider}")
    
    def search_recipes(self, query: str, k: int = 3) -> List[Tuple[Dict, float]]:
        """Search for relevant recipes"""
        return self.search_engine.search(query, k)
    
    def format_recipes_for_llm(self, recipes: List[Tuple[Dict, float]]) -> str:
        """Format retrieved recipes for LLM context"""
        if not recipes:
            return "No relevant recipes found in the database."
            
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
            
            context += f"Relevance score: {score:.3f}\n\n"
        
        return context
    
    def generate_response(self, user_query: str, recipe_context: str) -> str:
        """Generate LLM response using the configured provider"""
        
        system_prompt = """You are a helpful recipe assistant. Based on the user's question and the provided recipes, give personalized recommendations and cooking advice.

Guidelines:
- Be conversational and helpful
- Reference specific recipes when relevant
- Consider user preferences, time constraints, and skill level
- Suggest modifications or alternatives when appropriate
- Include practical cooking tips when helpful
- If no good matches, acknowledge this and suggest alternatives"""

        user_prompt = f"""User Question: {user_query}

Available Recipes:
{recipe_context}

Please provide a helpful, personalized response based on the user's question and the recipes above."""

        if self.api_provider == "nebius":
            return self._generate_nebius_response(system_prompt, user_prompt)
        elif self.api_provider == "openai":
            return self._generate_openai_response(system_prompt, user_prompt)
    
    def _generate_nebius_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using Nebius AI"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 400,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
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
            return f"Error generating response: {str(e)}"
    
    def _generate_openai_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=400,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def ask(self, user_query: str, num_recipes: int = 3) -> str:
        """Main RAG pipeline: search + generate"""
        
        # Step 1: Retrieve relevant recipes
        relevant_recipes = self.search_recipes(user_query, k=num_recipes)
        
        # Step 2: Format context for LLM
        recipe_context = self.format_recipes_for_llm(relevant_recipes)
        
        # Step 3: Generate response
        response = self.generate_response(user_query, recipe_context)
        
        return response
    
    def get_recipe_stats(self) -> Dict:
        """Get statistics about the recipe collection"""
        return self.search_engine.get_collection_stats()