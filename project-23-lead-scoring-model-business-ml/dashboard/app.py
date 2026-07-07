"""
LeadForge AI - Plotly Dash Analytics Dashboard
Real-time model performance and lead scoring analytics
"""

import os
import logging
import pandas as pd
from datetime import datetime
import pickle

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px

import asyncpg
import asyncio

logger = logging.getLogger(__name__)

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://leadforge:password@localhost:5432/leadforge_db")
MODEL_PATH = "models/xgboost_model.pkl"


class DashboardData:
    """Data provider for dashboard"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load trained model"""
        try:
            if os.path.exists(MODEL_PATH):
                with open(MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
    
    async def connect(self):
        """Create database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=5
            )
            logger.info("Connected to database")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
    
    async def get_statistics(self) -> dict:
        """Get overall statistics"""
        try:
            if not self.pool:
                return {}
            
            async with self.pool.acquire() as conn:
                total = await conn.fetchval("SELECT COUNT(*) FROM lead_scores")
                avg_score = await conn.fetchval("SELECT AVG(score) FROM lead_scores")
                hot = await conn.fetchval("SELECT COUNT(*) FROM lead_scores WHERE tier = 'hot'")
                warm = await conn.fetchval("SELECT COUNT(*) FROM lead_scores WHERE tier = 'warm'")
                cold = await conn.fetchval("SELECT COUNT(*) FROM lead_scores WHERE tier = 'cold'")
            
            return {
                'total_leads': total or 0,
                'avg_score': round(avg_score or 0, 1),
                'hot_leads': hot or 0,
                'warm_leads': warm or 0,
                'cold_leads': cold or 0
            }
        except Exception as e:
            logger.error(f"Error fetching statistics: {e}")
            return {}
    
    async def get_recent_scores(self, limit: int = 50) -> pd.DataFrame:
        """Get recent lead scores"""
        try:
            if not self.pool:
                return pd.DataFrame()
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT lead_id, score, tier, created_at FROM lead_scores ORDER BY created_at DESC LIMIT $1",
                    limit
                )
            
            data = [dict(row) for row in rows]
            return pd.DataFrame(data) if data else pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching recent scores: {e}")
            return pd.DataFrame()


# Initialize dashboard data
dashboard_data = DashboardData(DATABASE_URL)


# Create Dash app
app = dash.Dash(
    __name__,
    title="LeadForge AI - Analytics Dashboard",
    suppress_callback_exceptions=True
)

# Define layout
app.layout = html.Div([
    html.Div([
        html.H1("🎯 LeadForge AI - Analytics Dashboard", className="header-title"),
        html.P("Real-time Lead Scoring Analytics", className="header-subtitle")
    ], className="header"),
    
    html.Div([
        html.Div([
            html.H3("Total Leads", className="stat-label"),
            html.H2(id="stat-total-leads", children="—", className="stat-value")
        ], className="stat-card"),
        
        html.Div([
            html.H3("Average Score", className="stat-label"),
            html.H2(id="stat-avg-score", children="—", className="stat-value")
        ], className="stat-card"),
        
        html.Div([
            html.H3("🔥 Hot", className="stat-label"),
            html.H2(id="stat-hot", children="—", className="stat-value stat-hot")
        ], className="stat-card"),
    ], className="stats-container"),
    
    html.Div([
        html.H2("Recent Scores", className="section-title"),
        html.Div(id="recent-scores-table", className="table-container")
    ], className="section"),
    
    dcc.Interval(id="interval", interval=30*1000, n_intervals=0),
    
    html.Style("""
        body { font-family: Arial, sans-serif; background: #f5f5f5; }
        .header { background: white; padding: 30px; text-align: center; }
        .header-title { color: #667eea; margin: 0; }
        .stats-container { display: flex; gap: 20px; padding: 20px; justify-content: center; }
        .stat-card { background: white; padding: 20px; border-radius: 8px; text-align: center; min-width: 150px; }
        .stat-label { color: #999; margin: 0; font-size: 0.9em; }
        .stat-value { color: #667eea; margin: 10px 0 0 0; font-size: 2em; }
        .stat-hot { color: #ff6b6b; }
        .section { background: white; padding: 20px; margin: 20px; border-radius: 8px; }
        .section-title { margin-top: 0; }
        table { width: 100%; border-collapse: collapse; }
        th { background: #f5f5f5; padding: 10px; text-align: left; }
        td { padding: 10px; border-bottom: 1px solid #eee; }
    """)
], style={"padding": "0", "margin": "0"})


@callback(
    Output("stat-total-leads", "children"),
    Output("stat-avg-score", "children"),
    Output("stat-hot", "children"),
    Input("interval", "n_intervals")
)
def update_stats(n):
    """Update statistics"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        if not dashboard_data.pool:
            loop.run_until_complete(dashboard_data.connect())
        
        stats = loop.run_until_complete(dashboard_data.get_statistics())
        
        return (
            f"{stats.get('total_leads', 0):,}",
            f"{stats.get('avg_score', 0):.0f}",
            f"{stats.get('hot_leads', 0):,}"
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        return "—", "—", "—"


@callback(
    Output("recent-scores-table", "children"),
    Input("interval", "n_intervals")
)
def update_table(n):
    """Update scores table"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        if not dashboard_data.pool:
            loop.run_until_complete(dashboard_data.connect())
        
        df = loop.run_until_complete(dashboard_data.get_recent_scores(limit=20))
        
        if df.empty:
            return html.P("No data yet")
        
        rows = []
        for _, row in df.iterrows():
            rows.append(html.Tr([
                html.Td(row['lead_id']),
                html.Td(f"{row['score']}"),
                html.Td(row['tier']),
                html.Td(str(row['created_at']) if pd.notna(row['created_at']) else "—")
            ]))
        
        return html.Table([
            html.Thead(html.Tr([
                html.Th("Lead ID"),
                html.Th("Score"),
                html.Th("Tier"),
                html.Th("Created")
            ])),
            html.Tbody(rows)
        ])
    except Exception as e:
        logger.error(f"Error: {e}")
        return html.P(f"Error: {str(e)}")


def startup():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(dashboard_data.connect())


if __name__ == '__main__':
    startup()
    app.run_server(debug=False, host='0.0.0.0', port=8050)
