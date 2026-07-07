"""
CineMatch AI - Gradio Interactive Demo

Provides interactive interface for exploring ML model outputs and testing recommendations.
"""

import logging
from typing import List, Tuple
import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DEMO FUNCTIONS
# ============================================================================

def get_user_recommendations(
    user_id: int,
    k: int,
    strategy: str,
) -> pd.DataFrame:
    """
    Get recommendations for a user.
    
    Args:
        user_id: User ID
        k: Number of recommendations
        strategy: Recommendation strategy
        
    Returns:
        DataFrame with recommendations
    """
    logger.info(f"Getting {k} recommendations for user {user_id} using {strategy}")
    
    # TODO: Call actual API endpoint
    # For demo, return sample data
    
    data = {
        'Rank': list(range(1, k + 1)),
        'Movie ID': [318 + i for i in range(k)],
        'Title': [
            f'Movie {i+1}' for i in range(k)
        ],
        'Genres': ['Drama, Crime'] * k,
        'Score': [4.9 - (i * 0.1) for i in range(k)],
        'Match %': [95 - (i * 3) for i in range(k)],
        'Explanation': [
            f'Based on your ratings' for i in range(k)
        ]
    }
    
    return pd.DataFrame(data)


def get_similar_movies(
    movie_title: str,
    k: int,
) -> pd.DataFrame:
    """
    Get similar movies.
    
    Args:
        movie_title: Movie title
        k: Number of similar movies
        
    Returns:
        DataFrame with similar movies
    """
    logger.info(f"Finding {k} movies similar to '{movie_title}'")
    
    # TODO: Call actual API endpoint
    
    data = {
        'Rank': list(range(1, k + 1)),
        'Title': [f'Similar Movie {i+1}' for i in range(k)],
        'Genres': ['Drama, Crime'] * k,
        'Similarity Score': [0.92 - (i * 0.05) for i in range(k)],
    }
    
    return pd.DataFrame(data)


def get_metrics_history(
    metric: str,
    days: int,
) -> Tuple[go.Figure, dict]:
    """
    Get metrics history.
    
    Args:
        metric: Metric to display (Hit@10, NDCG@10, etc.)
        days: Number of days to show
        
    Returns:
        Plotly figure and summary stats
    """
    logger.info(f"Getting {metric} history for last {days} days")
    
    # TODO: Query metrics from MLflow or PostgreSQL
    
    # Sample data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
    values = [0.60 + (i * 0.001) for i in range(days)]
    
    df = pd.DataFrame({
        'Date': dates,
        'Value': values
    })
    
    fig = px.line(
        df,
        x='Date',
        y='Value',
        title=f'{metric} Trend',
        labels={'Value': metric},
        template='plotly_dark',
    )
    
    fig.update_layout(
        hovermode='x unified',
        plot_bgcolor='#0A0E1A',
        paper_bgcolor='#0A0E1A',
        font=dict(color='#F0F4FF'),
    )
    
    stats = {
        'Current': f"{values[-1]:.4f}",
        'Average': f"{pd.Series(values).mean():.4f}",
        'Min': f"{pd.Series(values).min():.4f}",
        'Max': f"{pd.Series(values).max():.4f}",
    }
    
    return fig, stats


def test_cold_start(
    num_ratings: int,
    movie_ids: str,
) -> pd.DataFrame:
    """
    Test cold-start handling with minimal user history.
    
    Args:
        num_ratings: Number of ratings provided
        movie_ids: Comma-separated movie IDs
        
    Returns:
        DataFrame with recommendations
    """
    logger.info(f"Testing cold-start with {num_ratings} ratings")
    
    # TODO: Call API endpoint with test user
    
    return pd.DataFrame({
        'Rank': [1, 2, 3, 4, 5],
        'Title': ['Popular Movie 1', 'Popular Movie 2', 'Popular Movie 3', 
                  'Popular Movie 4', 'Popular Movie 5'],
        'Strategy': ['Popularity (Cold-Start)'] * 5,
    })


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface() -> gr.Blocks:
    """Create Gradio interface."""
    
    with gr.Blocks(
        title="CineMatch AI - ML Demo",
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css="""
            body { background-color: #0A0E1A; color: #F0F4FF; }
            .container { background-color: #111827; }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🎬 CineMatch AI - ML Exploration Demo
        
        Interactive interface for exploring the recommendation system's capabilities.
        """)
        
        # ====================================================================
        # TAB 1: USER RECOMMENDATIONS
        # ====================================================================
        
        with gr.Tab("User Recommendations"):
            gr.Markdown("## Get Personalized Recommendations")
            
            with gr.Row():
                with gr.Column():
                    user_id = gr.Number(
                        label="User ID",
                        value=1,
                        precision=0,
                    )
                    k = gr.Slider(
                        label="Number of Recommendations",
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                    )
                    strategy = gr.Radio(
                        choices=["svd", "knn", "ensemble", "popularity"],
                        value="ensemble",
                        label="Strategy",
                    )
                
                with gr.Column():
                    get_recs_btn = gr.Button("Get Recommendations", scale=2)
            
            recs_output = gr.Dataframe(
                label="Recommendations",
                interactive=False,
            )
            
            latency = gr.Textbox(label="Latency (ms)", interactive=False)
            
            get_recs_btn.click(
                fn=get_user_recommendations,
                inputs=[user_id, k, strategy],
                outputs=[recs_output],
            )
        
        # ====================================================================
        # TAB 2: SIMILAR MOVIES
        # ====================================================================
        
        with gr.Tab("Item Similarity"):
            gr.Markdown("## Find Similar Movies")
            
            with gr.Row():
                with gr.Column():
                    movie_title = gr.Textbox(
                        label="Movie Title",
                        value="The Shawshank Redemption",
                    )
                    k_similar = gr.Slider(
                        label="Number of Similar Movies",
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                    )
                
                with gr.Column():
                    similar_btn = gr.Button("Find Similar", scale=2)
            
            similar_output = gr.Dataframe(
                label="Similar Movies",
                interactive=False,
            )
            
            similar_btn.click(
                fn=get_similar_movies,
                inputs=[movie_title, k_similar],
                outputs=[similar_output],
            )
        
        # ====================================================================
        # TAB 3: METRICS & DIAGNOSTICS
        # ====================================================================
        
        with gr.Tab("Model Metrics"):
            gr.Markdown("## Model Performance Metrics")
            
            with gr.Row():
                with gr.Column():
                    metric = gr.Dropdown(
                        choices=["Hit@10", "NDCG@10", "Coverage", "RMSE"],
                        value="Hit@10",
                        label="Metric",
                    )
                    days = gr.Slider(
                        label="Days to Show",
                        minimum=7,
                        maximum=90,
                        value=30,
                        step=1,
                    )
                
                with gr.Column():
                    metrics_btn = gr.Button("Load Metrics", scale=2)
            
            metrics_chart = gr.Plot(label="Trend")
            stats_output = gr.JSON(label="Summary Statistics")
            
            metrics_btn.click(
                fn=get_metrics_history,
                inputs=[metric, days],
                outputs=[metrics_chart, stats_output],
            )
        
        # ====================================================================
        # TAB 4: COLD-START TESTING
        # ====================================================================
        
        with gr.Tab("Cold-Start Testing"):
            gr.Markdown("## Test Recommendations for New Users")
            
            with gr.Row():
                with gr.Column():
                    num_ratings = gr.Number(
                        label="Number of Initial Ratings",
                        value=0,
                        precision=0,
                    )
                    movie_ids_input = gr.Textbox(
                        label="Movie IDs (comma-separated)",
                        placeholder="318,858,50",
                    )
                
                with gr.Column():
                    test_btn = gr.Button("Test Cold-Start", scale=2)
            
            coldstart_output = gr.Dataframe(
                label="Fallback Recommendations",
                interactive=False,
            )
            
            test_btn.click(
                fn=test_cold_start,
                inputs=[num_ratings, movie_ids_input],
                outputs=[coldstart_output],
            )
        
        # ====================================================================
        # TAB 5: SYSTEM STATUS
        # ====================================================================
        
        with gr.Tab("System Status"):
            gr.Markdown("## System Health & Status")
            
            status_info = gr.JSON(
                value={
                    "api_status": "healthy",
                    "database": "connected",
                    "redis_cache": "connected",
                    "model_loaded": True,
                    "uptime_hours": 48,
                    "cache_hit_rate": "82.3%",
                    "avg_latency_ms": 145,
                },
                label="System Status",
            )
            
            gr.Markdown("""
            ### Recent Activity
            - Models trained: 5
            - Total recommendations served: 1,234,567
            - Cache hits: 1,012,893 (82.3%)
            - Errors: 123 (0.01%)
            """)
    
    return demo


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting CineMatch AI Gradio Demo...")
    
    demo = create_interface()
    
    # Launch interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        show_api=False,
    )
