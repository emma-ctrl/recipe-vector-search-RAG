"""
Gradio Web Interface for Recipe RAG Assistant
"""
import sys
from pathlib import Path

# Add project root to Python path for proper package imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.rag import RecipeRAGAssistant, RecipeSearchEngine

class RecipeGradioApp:
    """Gradio web interface for the Recipe RAG Assistant"""
    
    def __init__(self):
        """Initialize the Gradio app"""
        print("Initializing Recipe RAG Assistant...")
        
        # Initialize components
        self.assistant = RecipeRAGAssistant()
        self.search_engine = self.assistant.search_engine
        
        # Session storage for recipe memory
        self.session_recipes = {}
        
        # Pre-compute embeddings for visualization
        self._prepare_visualizations()
        
        print("✅ App ready!")
    
    def _prepare_visualizations(self):
        """Pre-compute visualization coordinates"""
        print("Preparing visualizations...")
        
        # Get embedding coordinates
        self.pca_coords = self.search_engine.get_embedding_coordinates("PCA")
        self.tsne_coords = self.search_engine.get_embedding_coordinates("TSNE")
        
        # Get recipe dataframe
        self.recipe_df = self.search_engine.get_recipe_dataframe()
        
    def _get_session_id(self, history):
        """Generate a simple session ID"""
        return "main_session"
    
    def _store_recipes_in_session(self, session_id: str, recipes: list):
        """Store the current search results in session memory"""
        self.session_recipes[session_id] = {
            i + 1: recipe for i, (recipe, score) in enumerate(recipes)
        }
    
    def _get_recipe_from_session(self, session_id: str, recipe_number: int):
        """Get a specific recipe from session memory"""
        session_data = self.session_recipes.get(session_id, {})
        return session_data.get(recipe_number)
    
    def _is_recipe_reference(self, message: str):
        """Check if message is asking about a specific recipe number"""
        import re
        
        patterns = [
            r'recipe\s+(\d+)',
            r'recipe\s+number\s+(\d+)', 
            r'try\s+recipe\s+(\d+)',
            r'^(\d+)$',
            r'option\s+(\d+)',
            r'choice\s+(\d+)'
        ]
        
        message_lower = message.lower().strip()
        
        for pattern in patterns:
            match = re.search(pattern, message_lower)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def chat_with_recipes(self, message: str, history: list) -> tuple:
        """Handle chat interface with recipe memory"""
        if not message.strip():
            return history, ""
        
        try:
            session_id = self._get_session_id(history)
            
            # Check if user is asking about a specific recipe
            recipe_number = self._is_recipe_reference(message)
            
            if recipe_number:
                # User is asking about a specific recipe
                stored_recipe = self._get_recipe_from_session(session_id, recipe_number)
                
                if stored_recipe:
                    response = self._generate_recipe_details_with_buttons(stored_recipe, message)
                else:
                    response = f"I don't have a Recipe {recipe_number} from our recent search. Try asking a new question!"
            
            else:
                # Regular search query
                relevant_recipes = self.assistant.search_recipes(message, k=3)
                self._store_recipes_in_session(session_id, relevant_recipes)
                
                # Generate response with recipe buttons
                base_response = self.assistant.ask(message)
                response = self._add_recipe_buttons_to_response(base_response, relevant_recipes)
            
            # Update chat history
            history = history or []
            history.append([message, response])
            
            return history, ""
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            history = history or []
            history.append([message, error_msg])
            return history, ""
    
    def _add_recipe_buttons_to_response(self, response: str, recipes: list) -> str:
        """Add interactive buttons to chat response for each recipe"""
        if not recipes:
            return response
        
        # Add recipe action buttons
        buttons_html = "\n\n---\n\n"
        
        for i, (recipe, score) in enumerate(recipes, 1):
            source_url = recipe.get('sourceUrl', '')
            recipe_title = recipe.get('title', f'Recipe {i}')
            
            if source_url:
                # Create clickable buttons for each recipe
                buttons_html += f"""**Recipe {i}: {recipe_title}**
🔗 [View Full Recipe]({source_url}) | ⏱️ {recipe.get('readyInMinutes', '?')} min | 🍽️ Serves {recipe.get('servings', '?')}

"""
            else:
                buttons_html += f"""**Recipe {i}: {recipe_title}**
⏱️ {recipe.get('readyInMinutes', '?')} min | 🍽️ Serves {recipe.get('servings', '?')} | ℹ️ *No source link available*

"""
        
        return response + buttons_html
    
    def _generate_recipe_details_with_buttons(self, recipe: dict, user_message: str) -> str:
        """Generate detailed recipe info with action buttons"""
        
        base_response = self._generate_recipe_details(recipe, user_message)
        
        # Add action buttons for this specific recipe
        source_url = recipe.get('sourceUrl', '')
        
        buttons_html = "\n\n---\n\n"
        if source_url:
            buttons_html += f"""**🔗 [Get Complete Recipe & Instructions]({source_url})**

**Quick Info:**
⏱️ Ready in: {recipe.get('readyInMinutes', 'Unknown')} minutes
🍽️ Serves: {recipe.get('servings', 'Unknown')} people
🌟 Health Score: {recipe.get('healthScore', 'N/A')}/100
"""
        else:
            buttons_html += "\n*Full recipe link not available for this recipe.*"
        
        return base_response + buttons_html
    
    def _generate_recipe_details(self, recipe: dict, user_message: str) -> str:
        """Generate detailed information about a specific recipe"""
        
        context = f"""
        Recipe Details: {recipe['title']}
        
        Cuisine: {', '.join(recipe.get('cuisines', ['General']))}
        Ready in: {recipe.get('readyInMinutes', 'Unknown')} minutes
        Serves: {recipe.get('servings', 'Unknown')} people
        
        Ingredients: {recipe.get('ingredients', 'Not available')[:500]}...
        
        Instructions: {recipe.get('instructions', 'Not available')[:500]}...
        
        Source: {recipe.get('sourceUrl', 'Not available')}
        """
        
        return self.assistant.generate_response(
            f"The user is asking about this specific recipe: {user_message}",
            context
        )
    
    def get_enhanced_recipe_dataframe(self):
        """Get recipe dataframe with clickable links for the explorer"""
        df = self.recipe_df.copy()
        
        # Add source URLs from original recipe data
        source_urls = []
        for recipe in self.search_engine.recipes:
            url = recipe.get('sourceUrl', '')
            if url and url.strip():
                source_urls.append(f'<a href="{url}" target="_blank">🔗 View Recipe</a>')
            else:
                source_urls.append("No link available")
        
        df['recipe_link'] = source_urls
        
        # Select and reorder columns for display
        display_df = df[['title', 'cuisine', 'ready_time', 'dish_type', 'diets', 'recipe_link']].head(15)
        display_df.columns = ['Recipe Name', 'Cuisine', 'Ready Time (min)', 'Dish Type', 'Diets', 'Full Recipe']
        
        return display_df
    
    def visualize_embeddings(self, method: str, color_by: str, search_query: str = ""):
        """Create embedding visualization"""
        
        # Choose coordinates
        coords = self.pca_coords if method == "PCA" else self.tsne_coords
        
        # Create visualization dataframe
        viz_df = self.recipe_df.copy()
        viz_df['x'] = coords[:, 0]
        viz_df['y'] = coords[:, 1]
        
        # Handle search highlighting
        if search_query.strip():
            try:
                similar_recipes = self.search_engine.search(search_query, k=10)
                recipe_ids = {recipe['id']: i for i, recipe in enumerate(self.search_engine.recipes)}
                highlighted_indices = [
                    recipe_ids[recipe[0]['id']] 
                    for recipe, score in similar_recipes 
                    if recipe[0]['id'] in recipe_ids
                ]
                viz_df['highlighted'] = viz_df.index.isin(highlighted_indices)
            except:
                viz_df['highlighted'] = False
        else:
            viz_df['highlighted'] = False
        
        # Create hover text
        viz_df['hover_text'] = (
            "<b>" + viz_df['title'] + "</b><br>" +
            "Cuisine: " + viz_df['cuisine'] + "<br>" +
            "Ready in: " + viz_df['ready_time'].astype(str) + " min<br>" +
            "Serves: " + viz_df['servings'].astype(str) + "<br>" +
            "Type: " + viz_df['dish_type'] + "<br>" +
            "Ingredients: " + viz_df['ingredients_preview']
        )
        
        # Create the plot
        if search_query.strip() and any(viz_df['highlighted']):
            # Highlight search results
            fig = go.Figure()
            
            # Regular points
            regular_df = viz_df[~viz_df['highlighted']]
            fig.add_trace(go.Scatter(
                x=regular_df['x'], y=regular_df['y'],
                mode='markers',
                marker=dict(size=8, color='lightgray', opacity=0.6),
                text=regular_df['hover_text'],
                hovertemplate='%{text}<extra></extra>',
                name='Other recipes'
            ))
            
            # Highlighted points
            highlight_df = viz_df[viz_df['highlighted']]
            fig.add_trace(go.Scatter(
                x=highlight_df['x'], y=highlight_df['y'],
                mode='markers',
                marker=dict(size=12, color='red', opacity=0.9),
                text=highlight_df['hover_text'],
                hovertemplate='%{text}<extra></extra>',
                name=f'Similar to "{search_query}"'
            ))
            
            title = f"{method} Visualization - Search: '{search_query}'"
            
        else:
            # Regular colored plot
            color_map = {
                "Cuisine": 'cuisine',
                "Ready Time": 'ready_time', 
                "Dish Type": 'dish_type'
            }
            color_col = color_map.get(color_by, 'cuisine')
            
            if color_by == "Ready Time":
                fig = px.scatter(
                    viz_df, x='x', y='y', color=color_col,
                    color_continuous_scale='viridis',
                    hover_data={'hover_text': True, 'x': False, 'y': False},
                    title=f"{method} Visualization - Colored by {color_by}"
                )
            else:
                fig = px.scatter(
                    viz_df, x='x', y='y', color=color_col,
                    hover_data={'hover_text': True, 'x': False, 'y': False},
                    title=f"{method} Visualization - Colored by {color_by}"
                )
            
            fig.update_traces(hovertemplate='%{customdata[0]}<extra></extra>')
        
        # Update layout
        fig.update_layout(
            width=800, height=600,
            xaxis_title=f"{method} Component 1",
            yaxis_title=f"{method} Component 2"
        )
        
        return fig
    
    def get_stats_markdown(self) -> str:
        """Generate helpful suggestions for users"""
        return """
        ## 💡 Try Asking Me:
        
        **🍽️ Meal Ideas:**
        - "What's good for dinner tonight?"
        - "I need something quick and healthy"
        - "Suggest a comfort food recipe"
        
        **🎯 Specific Requests:**
        - "Show me chocolate desserts"
        - "What can I make in 30 minutes?"
        - "I want something vegetarian"
        
        **🔍 Ingredient-Based:**
        - "Recipes with chicken and pasta"
        - "Something with oats for breakfast"
        - "Easy weeknight meals"
        
        **Just start typing your question below! 👇**
        """

def create_gradio_interface():
    """Create and configure the Gradio interface"""
    
    app = RecipeGradioApp()
    
    with gr.Blocks(title="Recipe RAG Assistant", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("# 🍳 Recipe RAG Assistant")
        gr.Markdown("Chat with your recipe database and explore how recipes are organized in embedding space!")
        
        with gr.Tab("💬 Chat with Recipes"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        height=500,
                        placeholder="Ask me about recipes! Try: 'What's good for dinner?' or 'I need something quick'"
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Type your recipe question here...",
                            container=False, scale=4
                        )
                        submit_btn = gr.Button("Send", scale=1, variant="primary")
                
                with gr.Column(scale=1):
                    stats_display = gr.Markdown(app.get_stats_markdown())
        
        with gr.Tab("🗺️ Recipe Embedding Visualization"):
            gr.Markdown("### Explore Recipe Relationships")
            gr.Markdown("Each point represents a recipe. Similar recipes cluster together!")
            
            with gr.Row():
                with gr.Column():
                    method = gr.Radio(
                        choices=["PCA", "TSNE"],
                        value="PCA",
                        label="Visualization Method"
                    )
                    
                    color_by = gr.Radio(
                        choices=["Cuisine", "Ready Time", "Dish Type"],
                        value="Cuisine",
                        label="Color By"
                    )
                    
                    search_query = gr.Textbox(
                        placeholder="Search to highlight similar recipes",
                        label="Search & Highlight"
                    )
                    
                    viz_btn = gr.Button("Update Visualization", variant="primary")
            
            plot = gr.Plot(label="Recipe Embedding Space")
        
        with gr.Tab("📊 Recipe Explorer"):
            gr.Markdown("### Browse Your Recipe Collection")
            gr.Markdown("*Click the recipe links to view complete recipes with full instructions!*")
            
            recipe_table = gr.Dataframe(
                value=app.get_enhanced_recipe_dataframe(),
                interactive=False,
                wrap=True
            )
        
        # Event handlers
        submit_btn.click(
            app.chat_with_recipes,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        msg.submit(
            app.chat_with_recipes,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        viz_btn.click(
            app.visualize_embeddings,
            inputs=[method, color_by, search_query],
            outputs=[plot]
        )
        
        # Load initial visualization
        demo.load(
            lambda: app.visualize_embeddings("PCA", "Cuisine", ""),
            outputs=[plot]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True, server_port=7860)