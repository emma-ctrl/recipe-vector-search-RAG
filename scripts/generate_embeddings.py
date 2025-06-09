import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from typing import List, Dict, Tuple

class RecipeSearchEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with embedding model and FAISS index"""
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.recipes = []
        self.embeddings = None
        
    def load_recipes(self, json_file: str) -> List[Dict]:
        """Load recipes from JSON file"""
        with open(json_file, 'r', encoding='utf-8') as f:
            recipes = json.load(f)
        print(f"Loaded {len(recipes)} recipes from {json_file}")
        return recipes
    
    def create_embeddings(self, recipes: List[Dict]) -> np.ndarray:
        """Generate embeddings for all recipes"""
        print("Generating embeddings...")
        
        # Extract embedding text from each recipe
        documents = [recipe['embedding_text'] for recipe in recipes]
        
        # Generate embeddings
        embeddings = self.model.encode(
            documents, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def build_index(self, embeddings: np.ndarray):
        """Build FAISS index for fast similarity search"""
        print("Building FAISS index...")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index.add(embeddings)
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def setup_search_engine(self, recipes: List[Dict]):
        """Complete setup: embeddings + index + data storage"""
        self.recipes = recipes
        self.embeddings = self.create_embeddings(recipes)
        self.build_index(self.embeddings)
        
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for similar recipes"""
        if self.index is None:
            raise ValueError("Search engine not initialized. Run setup_search_engine first.")
        
        # Generate embedding for query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Return results with scores
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            recipe = self.recipes[idx]
            results.append((recipe, float(score)))
        
        return results
    
    def print_search_results(self, query: str, results: List[Tuple[Dict, float]]):
        """Pretty print search results"""
        print(f"\n🔍 Search: '{query}'")
        print("=" * 60)
        
        for i, (recipe, score) in enumerate(results, 1):
            print(f"\n{i}. {recipe['title']}")
            print(f"   Similarity: {score:.3f}")
            
            if recipe.get('cuisines'):
                print(f"   Cuisines: {', '.join(recipe['cuisines'])}")
            
            if recipe.get('readyInMinutes'):
                print(f"   Ready in: {recipe['readyInMinutes']} min")
            
            if recipe.get('ingredients'):
                ingredients_preview = recipe['ingredients'][:100]
                print(f"   Ingredients: {ingredients_preview}...")
    
    def save_search_engine(self, filename: str = "recipe_search_engine.pkl"):
        """Save the complete search engine"""
        os.makedirs('./data/processed', exist_ok=True)
        filepath = f'./data/processed/{filename}'
        
        search_data = {
            'recipes': self.recipes,
            'embeddings': self.embeddings,
            'model_name': self.model._modules['0'].auto_model.name_or_path if hasattr(self.model, '_modules') else 'all-MiniLM-L6-v2'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(search_data, f)
        
        # Save FAISS index separately
        index_path = f'./data/processed/faiss_index.index'
        faiss.write_index(self.index, index_path)
        
        print(f"💾 Search engine saved to {filepath}")
        print(f"💾 FAISS index saved to {index_path}")
    
    def load_search_engine(self, filename: str = "recipe_search_engine.pkl"):
        """Load a saved search engine"""
        filepath = f'./data/processed/{filename}'
        index_path = f'./data/processed/faiss_index.index'
        
        with open(filepath, 'rb') as f:
            search_data = pickle.load(f)
        
        self.recipes = search_data['recipes']
        self.embeddings = search_data['embeddings']
        self.index = faiss.read_index(index_path)
        
        print(f"✅ Search engine loaded from {filepath}")

def main():
    # Initialize search engine
    search_engine = RecipeSearchEngine()
    
    # Load recipe data (update path to your actual file)
    recipe_file = "./data/raw/recipes_20250609_161929.json"

    recipes = search_engine.load_recipes(recipe_file)
    
    # Setup complete search engine
    search_engine.setup_search_engine(recipes)
    
    # Save for later use
    search_engine.save_search_engine()
    
    # Test with various queries
    test_queries = [
        "creamy comfort food for dinner",
        "quick vegetarian pasta", 
        "healthy breakfast with oats",
        "chocolate dessert",
        "easy weeknight meal"
    ]
    
    print("\n🧪 Testing semantic search:")
    for query in test_queries:
        results = search_engine.search(query, k=3)
        search_engine.print_search_results(query, results)
        print("\n" + "="*60)

if __name__ == "__main__":
    main()