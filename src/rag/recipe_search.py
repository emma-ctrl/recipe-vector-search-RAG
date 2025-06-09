"""
Recipe Search Engine - Handles vector search and embeddings
"""
import pickle
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Dict, Tuple

class RecipeSearchEngine:
    """Handles vector search and recipe retrieval"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.recipes = []
        self.embeddings = None
        self.recipe_df = None
        
    def load_search_engine(self, search_engine_path: str, faiss_index_path: str):
        """Load pre-built search engine and prepare for visualization"""
        print("Loading search engine...")
        
        # Load recipe data and embeddings
        with open(search_engine_path, 'rb') as f:
            search_data = pickle.load(f)
        
        self.recipes = search_data['recipes']
        self.embeddings = search_data['embeddings']
        
        # Load FAISS index
        self.index = faiss.read_index(faiss_index_path)
        
        # Prepare DataFrame for visualization and stats
        self._prepare_recipe_dataframe()
        
        print(f"✅ Search engine loaded with {len(self.recipes)} recipes")
        
    def _prepare_recipe_dataframe(self):
        """Create pandas DataFrame for easier analysis and visualization"""
        self.recipe_df = pd.DataFrame([
            {
                'id': recipe.get('id'),
                'title': recipe['title'],
                'cuisine': ', '.join(recipe.get('cuisines', [])) if recipe.get('cuisines') else 'Mixed/Other',
                'ready_time': recipe.get('readyInMinutes', 0),
                'servings': recipe.get('servings', 0),
                'dish_type': ', '.join(recipe.get('dishTypes', [])) if recipe.get('dishTypes') else 'General',
                'diets': ', '.join(recipe.get('diets', [])) if recipe.get('diets') else 'No specific diet',
                'health_score': recipe.get('healthScore', 0),
                'ingredients_preview': recipe.get('ingredients', '')[:100] + '...' if recipe.get('ingredients') else '',
                'source_url': recipe.get('sourceUrl', '')
            }
            for recipe in self.recipes
        ])
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for similar recipes"""
        if self.index is None:
            raise ValueError("Search engine not loaded. Call load_search_engine first.")
        
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
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the recipe collection"""
        if self.recipe_df is None:
            return {}
            
        stats = {
            'total_recipes': len(self.recipes),
            'unique_cuisines': len(self.recipe_df['cuisine'].unique()),
            'avg_ready_time': self.recipe_df['ready_time'].mean(),
            'time_range': (self.recipe_df['ready_time'].min(), self.recipe_df['ready_time'].max()),
            'top_cuisines': self.recipe_df['cuisine'].value_counts().head(5).to_dict(),
            'unique_dish_types': len(self.recipe_df['dish_type'].unique())
        }
        return stats
    
    def get_embedding_coordinates(self, method: str = "PCA") -> np.ndarray:
        """Get 2D coordinates for visualization"""
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded")
            
        if method == "PCA":
            reducer = PCA(n_components=2, random_state=42)
        elif method == "TSNE":
            perplexity = min(30, len(self.recipes) - 1)
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        else:
            raise ValueError(f"Unsupported method: {method}")
            
        return reducer.fit_transform(self.embeddings)
    
    def get_recipe_dataframe(self) -> pd.DataFrame:
        """Get the recipe DataFrame for analysis"""
        return self.recipe_df.copy() if self.recipe_df is not None else pd.DataFrame()