import requests
import json
import time
from typing import List, Dict, Any
import os
from datetime import datetime

class RecipeCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.spoonacular.com/recipes"
        self.collected_recipes = []
        
    def fetch_recipes(self, query: str = "", num_recipes: int = 100, sort_by: str = "popularity") -> List[Dict]:
        """Fetch recipes from Spoonacular API with full details"""
        
        params = {
            'apiKey': self.api_key,
            'addRecipeInformation': 'true',
            'fillIngredients': 'true',
            'number': num_recipes,
            'sort': sort_by
        }
        
        if query:
            params['query'] = query
            
        url = f"{self.base_url}/complexSearch"
        
        try:
            print(f"Fetching {num_recipes} recipes for query: '{query}'...")
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            recipes = data.get('results', [])
            
            print(f"Successfully fetched {len(recipes)} recipes")
            return recipes
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching recipes: {e}")
            return []
    
    def process_recipe(self, recipe: Dict[Any, Any]) -> Dict[str, Any]:
        """Extract and clean key information from a recipe"""
        
        # Extract ingredients as readable text
        ingredients_text = ""
        if recipe.get('extendedIngredients'):
            ingredients = [ing.get('original', '') for ing in recipe['extendedIngredients']]
            ingredients_text = "; ".join(ingredients)
        
        # Extract instructions as readable text
        instructions_text = ""
        if recipe.get('analyzedInstructions'):
            for instruction_set in recipe['analyzedInstructions']:
                steps = instruction_set.get('steps', [])
                step_texts = [step.get('step', '') for step in steps]
                instructions_text = " ".join(step_texts)
        elif recipe.get('instructions'):
            instructions_text = recipe['instructions']
        
        # Combine categorical information
        categories = []
        for cat_type in ['cuisines', 'dishTypes', 'diets', 'occasions']:
            if recipe.get(cat_type):
                categories.extend(recipe[cat_type])
        
        # Create rich text for embedding
        embedding_text = f"""
        Title: {recipe.get('title', '')}
        Summary: {recipe.get('summary', '')}
        Cuisine: {', '.join(recipe.get('cuisines', []))}
        Dish Type: {', '.join(recipe.get('dishTypes', []))}
        Diet: {', '.join(recipe.get('diets', []))}
        Ingredients: {ingredients_text}
        Instructions: {instructions_text}
        Ready in: {recipe.get('readyInMinutes', 'N/A')} minutes
        Serves: {recipe.get('servings', 'N/A')} people
        """.strip()
        
        return {
            'id': recipe.get('id'),
            'title': recipe.get('title', ''),
            'summary': recipe.get('summary', ''),
            'image': recipe.get('image', ''),
            'readyInMinutes': recipe.get('readyInMinutes'),
            'servings': recipe.get('servings'),
            'cuisines': recipe.get('cuisines', []),
            'dishTypes': recipe.get('dishTypes', []),
            'diets': recipe.get('diets', []),
            'ingredients': ingredients_text,
            'instructions': instructions_text,
            'spoonacularScore': recipe.get('spoonacularScore'),
            'healthScore': recipe.get('healthScore'),
            'pricePerServing': recipe.get('pricePerServing'),
            'sourceUrl': recipe.get('sourceUrl'),
            'embedding_text': embedding_text,
            'categories': categories
        }
    
    def collect_diverse_dataset(self) -> List[Dict[str, Any]]:
        """Collect a diverse set of recipes using multiple queries"""
        
        # Define diverse queries to get variety
        queries = [
            ("", 60),  # Popular recipes
            ("vegetarian", 30),  # Vegetarian focus
            ("quick easy", 30),  # Quick meals
            ("comfort food", 30),  # Comfort food
        ]
        
        all_recipes = []
        recipe_ids = set()  # Track IDs to avoid duplicates
        
        for query, count in queries:
            recipes = self.fetch_recipes(query=query, num_recipes=count)
            
            for recipe in recipes:
                recipe_id = recipe.get('id')
                if recipe_id and recipe_id not in recipe_ids:
                    processed_recipe = self.process_recipe(recipe)
                    all_recipes.append(processed_recipe)
                    recipe_ids.add(recipe_id)
                    
            # Small delay between API calls
            time.sleep(1)
        
        print(f"\nCollected {len(all_recipes)} unique recipes")
        return all_recipes
    
    def save_recipes(self, recipes: List[Dict], filename: str = None):
        """Save recipes to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recipes_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(recipes, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(recipes)} recipes to {filename}")
        return filename
    
    def print_sample_recipe(self, recipes: List[Dict]):
        """Print a sample recipe to verify data quality"""
        if recipes:
            sample = recipes[0]
            print("\n=== SAMPLE RECIPE ===")
            print(f"Title: {sample['title']}")
            print(f"Cuisines: {sample['cuisines']}")
            print(f"Ready in: {sample['readyInMinutes']} minutes")
            print(f"Ingredients preview: {sample['ingredients'][:100]}...")
            print(f"Embedding text length: {len(sample['embedding_text'])} characters")

def main():
    # Replace with your actual API key
    API_KEY = "YOUR_SPOONACULAR_API_KEY"
    
    if API_KEY == "YOUR_SPOONACULAR_API_KEY":
        print("Please replace 'YOUR_SPOONACULAR_API_KEY' with your actual API key")
        return
    
    collector = RecipeCollector(API_KEY)
    
    # Collect diverse recipe dataset
    recipes = collector.collect_diverse_dataset()
    
    if recipes:
        # Save to file
        filename = collector.save_recipes(recipes)
        
        # Show sample
        collector.print_sample_recipe(recipes)
        
        print(f"\n✅ Data collection complete!")
        print(f"📁 Recipes saved to: {filename}")
        print(f"📊 Total recipes: {len(recipes)}")
        print(f"🔍 Ready for embedding and vector search!")
    else:
        print("❌ No recipes collected. Please check your API key and connection.")

if __name__ == "__main__":
    main()