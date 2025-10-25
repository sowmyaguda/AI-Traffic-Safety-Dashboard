"""
Interactive CV Analytics Dashboard
Fully reactive chat-driven interface with RAPIDS cuDF
Every command instantly updates map and plots
Smart analysis with GPU-accelerated performance
"""
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium import plugins
import cudf
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from groq import Groq
from openai import OpenAI
import google.generativeai as genai
from datetime import datetime
import pandas as pd
import numpy as np
from functools import lru_cache
import re

# ====== PAGE CONFIG ======
st.set_page_config(
    page_title="CV Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clear cache to ensure fresh data processing (especially for deceleration calculations)
if 'cache_cleared' not in st.session_state:
    st.cache_data.clear()
    st.session_state.cache_cleared = True

# ====== CUSTOM CSS (OPTIMIZED FOR SPEED) ======
st.markdown("""
<style>
/* Fast-loading font (display=swap prevents blocking) */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

/* Apply Inter font */
html, body, .main, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
.stMarkdown, .stText, .stHtml, .element-container {
  font-family: 'Inter', -apple-system, sans-serif !important;
}


/* ---------- Keep Streamlit chrome visible ---------- */
#MainMenu { visibility: visible !important; }    /* 3-dot menu */
footer    { visibility: hidden  !important; }
header[data-testid="stHeader"]{
  visibility: visible !important;
  position: sticky; top: 0; z-index: 9999;
  background: #ffffff !important;
  box-shadow: 0 1px 0 rgba(0,0,0,.06);
}

/* ---------- Page container ---------- */
.main .block-container{
  max-width: 100% !important;
  padding: 0rem 1.25rem 1.25rem !important;  /* no top padding */
  padding-top: 0 !important;
  margin-top: 0 !important;
  background: #F5F7FA !important;
}

/* Remove all top spacing from first element */
.main .block-container > div:first-child{
  padding-top: 0 !important;
  margin-top: 0 !important;
}

/* Remove spacing from element container */
.element-container:first-child{
  margin-top: 0 !important;
  padding-top: 0 !important;
}

/* ---------- PROFESSIONAL SIDEBAR ---------- */
[data-testid="stSidebar"]{
  background: linear-gradient(to bottom, #ffffff 0%, #f8f9ff 100%) !important;
  border-right: none !important;
  box-shadow: 2px 0 12px rgba(102,126,234,0.08) !important;
}

/* Sidebar gradient header accent */
[data-testid="stSidebar"]::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
  z-index: 1;
}

[data-testid="stSidebar"] *{ color:#1f2937 !important; }

/* Sidebar headers with gradient accent */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3{
  color:#111827 !important;
  font-weight:700 !important;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 0.75rem !important;
}

/* Sidebar section labels */
[data-testid="stSidebar"] label p{
  color:#4b5563 !important;
  font-weight:600 !important;
  font-size: 0.875rem !important;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 0.5rem !important;
}

/* Modern inputs with gradient border on hover */
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextArea textarea,
[data-testid="stSidebar"] .stFileUploader,
[data-testid="stSidebar"] .stFileUploader div[role="button"],
[data-testid="stSidebar"] div[data-baseweb="select"]>div{
  background:#ffffff !important;
  color:#111827 !important;
  border:2px solid #e5e7eb !important;
  border-radius:10px !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}

/* Input hover effect */
[data-testid="stSidebar"] .stTextInput input:hover,
[data-testid="stSidebar"] .stTextArea textarea:hover,
[data-testid="stSidebar"] div[data-baseweb="select"]>div:hover{
  border-color: #667eea !important;
  box-shadow: 0 2px 8px rgba(102,126,234,0.15) !important;
}

/* Force select internals to stay white + readable */
[data-testid="stSidebar"] div[data-baseweb="select"] [class*="ValueContainer"],
[data-testid="stSidebar"] div[data-baseweb="select"] [class*="SingleValue"],
[data-testid="stSidebar"] div[data-baseweb="select"] [class*="Input"],
[data-testid="stSidebar"] div[data-baseweb="select"] [class*="SelectArrow"],
[data-testid="stSidebar"] div[data-baseweb="select"] *{
  background: transparent !important;
  color:#111827 !important;
}

/* Focus ring with gradient glow */
.stTextInput>div>div>input:focus,
.stTextArea textarea:focus,
div[data-baseweb="select"]>div:focus{
  border-color:#667eea !important;
  box-shadow:0 0 0 3px rgba(102,126,234,.20), 0 4px 12px rgba(102,126,234,0.15) !important;
  outline: none !important;
}

/* Buttons in sidebar with gradient */
[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] .stButton>button{
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
  color: white !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 0.6rem 1.2rem !important;
  font-weight: 600 !important;
  box-shadow: 0 4px 12px rgba(102,126,234,0.3) !important;
  transition: all 0.2s ease !important;
}

[data-testid="stSidebar"] button[kind="primary"]:hover,
[data-testid="stSidebar"] .stButton>button:hover{
  transform: translateY(-1px);
  box-shadow: 0 6px 16px rgba(102,126,234,0.4) !important;
}

/* ---------- Full-Width Header ---------- */
.main-header{
  background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
  padding: 2rem 0 !important;
  border-radius: 0;
  text-align: center;
  color:#F9FAFB;
  margin: -0.5rem -1.25rem 1.5rem -1.25rem !important;
  box-shadow: 0 2px 10px rgba(102,126,234,.18);
  width: calc(100% + 2.5rem);
}
.main-header h1{ font-size: 2.5rem !important; margin:0; font-weight:700; letter-spacing:-.5px; }

/* ---------- Modern Chat Interface ---------- */
.chat-outer-wrapper{
  background: linear-gradient(135deg, rgba(102,126,234,0.08) 0%, rgba(118,75,162,0.08) 100%);
  border-radius:16px; padding:1.5rem; box-shadow:0 4px 12px rgba(0,0,0,0.08);
}

.chat-history-box{
  background:#ffffff; border-radius:12px; padding:1rem; margin-bottom:1rem;
  max-height:500px; overflow-y:auto; overflow-x:hidden;
  box-shadow: inset 0 1px 3px rgba(0,0,0,0.06);
  border:1px solid #E5E7EB;
}

/* Scrollbar styling */
.chat-history-box::-webkit-scrollbar{ width:6px; }
.chat-history-box::-webkit-scrollbar-track{ background:#F1F3F5; border-radius:3px; }
.chat-history-box::-webkit-scrollbar-thumb{ background:#ADB5BD; border-radius:3px; }
.chat-history-box::-webkit-scrollbar-thumb:hover{ background:#868E96; }

/* Message cards with animations */
.msg-card{
  margin-bottom:12px; padding:12px 16px; border-radius:12px;
  animation: slideIn 0.3s ease-out; word-wrap:break-word;
  overflow-wrap:break-word; max-width:100%; box-sizing:border-box;
}
@keyframes slideIn{
  from{ opacity:0; transform:translateY(-8px); }
  to{ opacity:1; transform:translateY(0); }
}

.msg-card.user{
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color:white; margin-left:50px; box-shadow:0 2px 8px rgba(102,126,234,0.25);
}
.msg-card.assistant{
  background:#F8F9FA; color:#212529; margin-right:50px;
  border-left:3px solid #667eea; box-shadow:0 1px 3px rgba(0,0,0,0.08);
}

.msg-header{
  font-size:0.7rem; font-weight:700; margin-bottom:6px;
  text-transform:uppercase; letter-spacing:0.8px; opacity:0.9;
}
.msg-text{
  font-size:0.95rem; line-height:1.6; white-space:pre-wrap;
  word-break:break-word;
}

/* Empty state */
.chat-empty{
  text-align:center; padding:80px 20px; color:#868E96;
}
.chat-empty-icon{ font-size:3.5rem; margin-bottom:16px; opacity:0.6; }

/* ---------- Buttons ---------- */
.stButton>button{
  background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
  color:#fff; border:none; border-radius:12px; padding:.75rem 1.5rem; font-weight:600;
  box-shadow:0 2px 8px rgba(102,126,234,.2); transition:transform .2s, box-shadow .2s;
}
.stButton>button:hover{
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102,126,234,.30);
}
.stButton>button:active{ transform: translateY(0); }

/* ---------- Inputs in MAIN area (match sidebar look) ---------- */
.stTextInput input,
.stTextArea textarea,
div[data-baseweb="select"]>div{
  background:#ffffff !important;
  color:#111827 !important;
  border:1.5px solid #E5E7EB !important;
  border-radius:12px !important;
}
div[data-baseweb="select"] *{ color:#111827 !important; }

/* ---------- Map & cards ---------- */
.map-container, .chart-card{
  background:#fff; border-radius:16px; padding:1.25rem;
  border:1px solid #E5E7EB; box-shadow:0 2px 4px rgba(0,0,0,.04);
}
.map-header{ display:flex; align-items:center; justify-content:space-between;
  margin-bottom:.75rem; padding-bottom:.75rem; border-bottom:2px solid #F0F0F0; }
.map-title{ font-size:1.25rem; font-weight:700; color:#1A1A1A; }

/* ---------- Tighter layout on wide screens ---------- */
@media (min-width: 1200px){
  .main .block-container{ padding: .75rem 1.25rem !important; }
  /* Give more room to the map column when using st.columns([2,1]) */
  .stHorizontalBlock{ gap: 1.25rem !important; }
}

/* ---------- Alerts & metrics ---------- */
.stAlert{ border-radius:12px; border-left:4px solid #667eea; }
[data-testid="stMetricValue"]{ font-size:2rem; font-weight:700; color:#1A1A1A; }
[data-testid="stMetricLabel"]{ font-size:.875rem; color:#6B7280; font-weight:500; letter-spacing:.5px; }

/* ---------- Small utilities ---------- */
.hover-lift{ transition: transform .3s ease, box-shadow .3s ease; }
.hover-lift:hover{ transform: translateY(-4px); box-shadow:0 8px 16px rgba(0,0,0,.10); }

/* Make Folium map edges crisp on light background */
.leaflet-container{ border-radius:12px; overflow:hidden; }

/* Make map larger and use full width */
iframe[title="folium"], .folium-map, .streamlit-folium {
  width: 100% !important;
  min-height: 700px !important;
}

/* Remove all top spacing from vertical blocks */
[data-testid="stVerticalBlock"]{
  padding-top: 0 !important;
  margin-top: 0 !important;
}
[data-testid="stVerticalBlock"] > div:first-child{
  padding-top: 0 !important;
  margin-top: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ====== INITIALIZE SESSION STATE ======
if 'vehicles' not in st.session_state:
    st.session_state.vehicles = None
if 'crashes' not in st.session_state:
    st.session_state.crashes = None
if 'filtered_vehicles' not in st.session_state:
    st.session_state.filtered_vehicles = None
if 'filtered_crashes' not in st.session_state:
    st.session_state.filtered_crashes = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_view' not in st.session_state:
    st.session_state.current_view = "all"
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
if 'api_provider' not in st.session_state:
    st.session_state.api_provider = "Gemini"  # Default to Gemini (best free tier)
if 'groq_model' not in st.session_state:
    st.session_state.groq_model = "llama-3.3-70b-versatile"  # Default Groq model
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'show_plots' not in st.session_state:
    st.session_state.show_plots = set()  # Set of plot types to show
if 'color_mode' not in st.session_state:
    st.session_state.color_mode = "neutral"  # "neutral" or "speed"
if 'color_scheme' not in st.session_state:
    st.session_state.color_scheme = {
        'high': '#FF4444',    # Red for >70 mph
        'medium': '#FFA500',  # Orange for 50-70 mph
        'low': '#4CAF50',     # Green for <50 mph
        'neutral': '#888888'  # Gray for neutral
    }
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}  # Cache for fast analytics
if 'braking_mode' not in st.session_state:
    st.session_state.braking_mode = "off"  # "off", "moderate" (-0.3g), "hard" (-0.5g), "emergency" (-0.7g)
if 'braking_threshold' not in st.session_state:
    st.session_state.braking_threshold = -0.3  # Default threshold

# Advanced filter session state
if 'active_filters' not in st.session_state:
    st.session_state.active_filters = {
        'speed_range': None,  # (min, max) tuple
        'date_range': None,   # (start_date, end_date) tuple
        'roads': [],          # list of road names
        'event_types': [],    # list of event types
        'days_of_week': [],   # list of day numbers (0=Monday)
        'filter_mode': 'AND' # 'AND' or 'OR' for combining filters
    }

# Map version to prevent unnecessary flickering
if 'map_version' not in st.session_state:
    st.session_state.map_version = 0

# ====== HELPER FUNCTIONS ======
def parse_color_request(text):
    """Parse custom color scheme from user message"""
    colors = {}
    text_lower = text.lower()

    # Color mapping
    color_map = {
        'red': '#FF0000', 'white': '#FFFFFF', 'pink': '#FFC0CB',
        'green': '#00FF00', 'blue': '#0000FF', 'yellow': '#FFFF00',
        'orange': '#FFA500', 'purple': '#800080', 'black': '#000000',
        'gray': '#808080', 'grey': '#808080'
    }

    # Extract colors for different speed ranges
    patterns = {
        'high': r'(?:high|fast|>70|above 70|over 70).*?(?:color|shows?|be)\s+(\w+)',
        'medium': r'(?:medium|moderate|50-70|50 to 70).*?(?:color|shows?|be)\s+(\w+)',
        'low': r'(?:low|slow|<50|below 50|under 50).*?(?:color|shows?|be)\s+(\w+)'
    }

    for speed_range, pattern in patterns.items():
        match = re.search(pattern, text_lower)
        if match:
            color_name = match.group(1)
            if color_name in color_map:
                colors[speed_range] = color_map[color_name]

    return colors if colors else None

@st.cache_data(ttl=600)
def get_top_crash_roads(_crashes_df, top_n=5):
    """Fast cached analysis of top crash locations (GPU-accelerated)"""
    if _crashes_df is None or len(_crashes_df) == 0:
        return None

    df = _crashes_df.to_pandas() if hasattr(_crashes_df, 'to_pandas') else _crashes_df

    if 'RoadName' not in df.columns:
        return None

    df_clean = df.dropna(subset=['RoadName'])
    if len(df_clean) == 0:
        return None

    road_counts = df_clean['RoadName'].value_counts().head(top_n)
    return road_counts

@st.cache_data(ttl=600)
def get_speed_stats(_vehicles_df):
    """Fast cached speed statistics (GPU-accelerated)"""
    if _vehicles_df is None or len(_vehicles_df) == 0:
        return None

    df = _vehicles_df.to_pandas() if hasattr(_vehicles_df, 'to_pandas') else _vehicles_df

    if 'speed' not in df.columns:
        return None

    return {
        'mean': df['speed'].mean(),
        'median': df['speed'].median(),
        'max': df['speed'].max(),
        'min': df['speed'].min(),
        'speeding_count': len(df[df['speed'] > 70]),
        'total_count': len(df)
    }

# ====== AUTO-LOAD CV DATA ======
if not st.session_state.data_loaded:
    cv_data_path = "/home/ubuntu/cvdata/cv_processed.csv"
    if os.path.exists(cv_data_path):
        try:
            # Load sample of vehicle data (100K rows for performance)
            df = cudf.read_csv(cv_data_path, nrows=100000)

            # Rename columns to match app expectations
            column_mapping = {
                'SnappedLatitude': 'latitude',
                'SnappedLongitude': 'longitude',
                'SpeedMPH': 'speed',
                'RoadName': 'road',
                'TimestampUTC': 'timestamp',
                'SpeedLimitMPH': 'speed_limit'
            }
            df = df.rename(columns=column_mapping)

            # Keep necessary columns (including optional ones for enhanced popups)
            required_cols = ['latitude', 'longitude', 'speed', 'road', 'timestamp']
            optional_cols = ['VehicleID', 'speed_limit', 'VehicleType', 'Bearing', 'AccMagnitude', 'AccX', 'AccY']
            all_cols = required_cols + [col for col in optional_cols if col in df.columns]
            existing_cols = [col for col in all_cols if col in df.columns]
            df = df[existing_cols]

            # Remove rows with missing coordinates
            df = df.dropna(subset=['latitude', 'longitude'])

            # Add deceleration_g from existing AccMagnitude if available
            # Negative AccMagnitude = braking/deceleration
            if 'AccMagnitude' in df.columns:
                # Convert to deceleration_g (negative values = braking)
                # AccMagnitude is already in g-force, just negate for braking convention
                df['deceleration_g'] = -df['AccMagnitude']

            st.session_state.vehicles = df
            st.session_state.filtered_vehicles = df
            st.session_state.data_loaded = True
        except Exception as e:
            # Silent fail - user can upload manually
            pass

# ====== GROQ CLIENT ======
def get_llm_client():
    """Get LLM client based on selected provider"""
    if st.session_state.api_provider == "Groq":
        if st.session_state.groq_api_key:
            return Groq(api_key=st.session_state.groq_api_key), "groq"
    elif st.session_state.api_provider == "OpenAI":
        if st.session_state.openai_api_key:
            return OpenAI(api_key=st.session_state.openai_api_key), "openai"
    elif st.session_state.api_provider == "Gemini":
        if st.session_state.gemini_api_key:
            genai.configure(api_key=st.session_state.gemini_api_key)
            return genai.GenerativeModel('gemini-1.5-pro'), "gemini"
    return None, None

# ====== LLM INTENT PARSER ======
def parse_intent(user_query):
    """
    Use LLM (Groq or OpenAI) to parse user intent and extract parameters
    Returns: (action, parameters dict)
    """
    client, provider = get_llm_client()
    if not client:
        return "error", {"message": "No API key"}

    system_prompt = """You are a command parser for a vehicle analytics dashboard with crash prediction capabilities.
Parse the user's query and return ONLY a JSON object with:
- action: one of ["show_all", "filter_road", "show_speeding", "show_crashes", "filter_speed", "filter_time", "analyze", "plot_speed", "plot_crashes", "analyze_crashes", "color_by_speed", "timespace", "show_hard_braking", "plot_hard_braking", "analyze_crash_delays", "analyze_proximity", "road_safety_score", "temporal_analysis", "crash_hotspots", "braking_intensity", "ai_insights", "help"]
- params: dictionary of parameters

IMPORTANT: Only include "show_crashes": true if the user EXPLICITLY mentions crashes in their query.

Time filtering uses 24-hour format. Extract start_hour and end_hour from queries about time ranges.

Examples:
"Show speeding on I-70" -> {"action": "filter_road", "params": {"road": "I-70", "show_speeding": true}}
"Filter by MO 141 North" -> {"action": "filter_road", "params": {"road": "MO 141 North"}}
"Filter by MO 141 North and speed less than 50 mph" -> {"action": "filter_road", "params": {"road": "MO 141 North", "max_speed": 50}}
"Filter by I-70 and speed over 60" -> {"action": "filter_road", "params": {"road": "I-70", "min_speed": 60}}
"Filter by I-70 and show crashes" -> {"action": "filter_road", "params": {"road": "I-70", "show_crashes": true}}
"Show MO 141 with crashes" -> {"action": "filter_road", "params": {"road": "MO 141", "show_crashes": true}}
"Filter by speed over 80" -> {"action": "filter_speed", "params": {"min_speed": 80}}
"Filter by crashes on MO 141" -> {"action": "filter_road", "params": {"road": "MO 141", "crashes_only": true}}
"Show crash hotspots" -> {"action": "show_crashes", "params": {}}
"Show crashes" -> {"action": "show_crashes", "params": {}}
"Show only crashes" -> {"action": "show_crashes", "params": {}}
"Show all data" -> {"action": "show_all", "params": {}}
"Show all data and crashes" -> {"action": "show_all", "params": {"show_crashes": true}}
"Show crashes and all data" -> {"action": "show_all", "params": {"show_crashes": true}}
"Integrate crashes with all data" -> {"action": "show_all", "params": {"show_crashes": true}}
"Show both vehicles and crashes" -> {"action": "show_all", "params": {"show_crashes": true}}
"Filter by time 8am to 10am" -> {"action": "filter_time", "params": {"start_hour": 8, "end_hour": 10}}
"Show data from 2pm to 5pm" -> {"action": "filter_time", "params": {"start_hour": 14, "end_hour": 17}}
"Filter by morning rush hour" -> {"action": "filter_time", "params": {"start_hour": 7, "end_hour": 9}}
"Show vehicles between 14:00 and 16:00" -> {"action": "filter_time", "params": {"start_hour": 14, "end_hour": 16}}
"Filter I-70 from 8am to 5pm" -> {"action": "filter_road", "params": {"road": "I-70", "start_hour": 8, "end_hour": 17}}
"Plot speed distribution" -> {"action": "plot_speed", "params": {}}
"Where are the highest crashes?" -> {"action": "analyze_crashes", "params": {}}
"Color by speed" -> {"action": "color_by_speed", "params": {}}
"Corridor plot" -> {"action": "timespace", "params": {}}
"Timespace diagram for I-70" -> {"action": "timespace", "params": {"road": "I-70"}}
"Show hard braking events" -> {"action": "show_hard_braking", "params": {}}
"Where are vehicles braking hard?" -> {"action": "show_hard_braking", "params": {}}
"Show hard braking on I-70" -> {"action": "show_hard_braking", "params": {"road": "I-70"}}
"Show hard braking from 8am to 9am" -> {"action": "show_hard_braking", "params": {"start_hour": 8, "end_hour": 9}}
"Show hard braking on I-70 from 7am to 9am" -> {"action": "show_hard_braking", "params": {"road": "I-70", "start_hour": 7, "end_hour": 9}}
"Show emergency braking" -> {"action": "show_hard_braking", "params": {"threshold": -0.7}}
"Show moderate braking" -> {"action": "show_hard_braking", "params": {"threshold": -0.3}}
"Plot hard braking" -> {"action": "plot_hard_braking", "params": {}}
"Plot braking distribution" -> {"action": "plot_hard_braking", "params": {}}
"Analyze crash delays" -> {"action": "analyze_crash_delays", "params": {}}
"Show crash impact on traffic" -> {"action": "analyze_crash_delays", "params": {}}
"Calculate delay from crashes" -> {"action": "analyze_crash_delays", "params": {}}
"Which crashes caused most delay?" -> {"action": "analyze_crash_delays", "params": {}}
"Analyze crash-braking correlation" -> {"action": "analyze_proximity", "params": {}}
"Find near-miss events" -> {"action": "analyze_proximity", "params": {}}
"Predict future crash locations" -> {"action": "analyze_proximity", "params": {}}
"Show risk zones" -> {"action": "analyze_proximity", "params": {}}
"Where will crashes happen next?" -> {"action": "analyze_proximity", "params": {}}
"Calculate road safety scores" -> {"action": "road_safety_score", "params": {}}
"Which roads are most dangerous?" -> {"action": "road_safety_score", "params": {}}
"Show safety analysis" -> {"action": "road_safety_score", "params": {}}
"Rank roads by safety" -> {"action": "road_safety_score", "params": {}}
"When are crashes most likely?" -> {"action": "temporal_analysis", "params": {}}
"What time of day is most dangerous?" -> {"action": "temporal_analysis", "params": {}}
"Show me crash patterns by hour" -> {"action": "temporal_analysis", "params": {}}
"Analyze crash timing" -> {"action": "temporal_analysis", "params": {}}
"When do hard braking events happen?" -> {"action": "temporal_analysis", "params": {}}
"Show rush hour risk" -> {"action": "temporal_analysis", "params": {}}
"Temporal risk analysis" -> {"action": "temporal_analysis", "params": {}}
"How far in advance can we predict crashes?" -> {"action": "temporal_analysis", "params": {}}
"Show leading indicator analysis" -> {"action": "temporal_analysis", "params": {}}
"Find crash hotspots" -> {"action": "crash_hotspots", "params": {}}
"Where are crashes clustered?" -> {"action": "crash_hotspots", "params": {}}
"Show me the most dangerous spots" -> {"action": "crash_hotspots", "params": {}}
"Identify crash clusters" -> {"action": "crash_hotspots", "params": {}}
"Show spatial crash patterns" -> {"action": "crash_hotspots", "params": {}}
"Analyze braking intensity" -> {"action": "braking_intensity", "params": {}}
"Which roads have the most extreme braking?" -> {"action": "braking_intensity", "params": {}}
"Show braking severity analysis" -> {"action": "braking_intensity", "params": {}}
"Correlate braking with crashes" -> {"action": "braking_intensity", "params": {}}
"Show emergency braking patterns" -> {"action": "braking_intensity", "params": {}}
"AI insights" -> {"action": "ai_insights", "params": {}}
"Run AI analysis" -> {"action": "ai_insights", "params": {}}
"Show me comprehensive insights" -> {"action": "ai_insights", "params": {}}

Return ONLY valid JSON, no markdown."""

    try:
        # Handle different API structures based on provider
        if provider == "gemini":
            # Gemini uses a different API structure
            prompt = f"{system_prompt}\n\nUser query: {user_query}"
            response = client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=200
                )
            )
            text = response.text.strip()
        else:
            # OpenAI and Groq use the same interface
            if provider == "groq":
                model = st.session_state.groq_model
            else:  # openai
                model = "gpt-4o-mini"

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.2,
                max_tokens=200
            )
            text = response.choices[0].message.content.strip()
        # Clean markdown if present
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()

        parsed = json.loads(text)
        return parsed.get("action", "unknown"), parsed.get("params", {})

    except Exception as e:
        st.error(f"Error parsing intent: {str(e)}")
        return "error", {"message": str(e)}

# ====== DATA OPERATIONS ======
def enrich_crash_data_with_roads(crashes, vehicles):
    """Enrich crash data with road names from vehicle data using spatial lookup"""
    try:
        # Convert to pandas for easier manipulation
        crashes_pd = crashes.to_pandas() if hasattr(crashes, 'to_pandas') else crashes
        vehicles_pd = vehicles.to_pandas() if hasattr(vehicles, 'to_pandas') else vehicles

        # Determine crash coordinate column names
        crash_lat_col = None
        crash_lon_col = None

        # Check for standard lat/lon columns
        if 'Latitude' in crashes_pd.columns:
            crash_lat_col = 'Latitude'
        elif 'latitude' in crashes_pd.columns:
            crash_lat_col = 'latitude'
        elif 'lat' in crashes_pd.columns:
            crash_lat_col = 'lat'

        if 'Longitude' in crashes_pd.columns:
            crash_lon_col = 'Longitude'
        elif 'longitude' in crashes_pd.columns:
            crash_lon_col = 'longitude'
        elif 'lon' in crashes_pd.columns:
            crash_lon_col = 'lon'

        # If no lat/lon columns found, check for geometry column
        if (crash_lat_col is None or crash_lon_col is None) and 'geometry' in crashes_pd.columns:
            st.info("Extracting coordinates from geometry column...")
            try:
                # Parse geometry column (supports WKT POINT format and other formats)
                def extract_coords(geom):
                    if pd.isna(geom):
                        return None, None
                    geom_str = str(geom)

                    # Handle WKT POINT format: "POINT (lon lat)"
                    if 'POINT' in geom_str.upper():
                        coords = geom_str.split('(')[1].split(')')[0].strip()
                        lon, lat = coords.split()
                        return float(lat), float(lon)

                    # Handle other formats (you can extend this)
                    return None, None

                coords = crashes_pd['geometry'].apply(extract_coords)
                crashes_pd['latitude'] = coords.apply(lambda x: x[0] if x else None)
                crashes_pd['longitude'] = coords.apply(lambda x: x[1] if x else None)

                crash_lat_col = 'latitude'
                crash_lon_col = 'longitude'
                st.success("Extracted coordinates from geometry!")
            except Exception as e:
                st.error(f"Error parsing geometry: {str(e)}")
                return crashes

        if crash_lat_col is None or crash_lon_col is None:
            st.error(f"No usable coordinate data found. Available columns: {crashes_pd.columns.tolist()}")
            return crashes

        # Check if vehicle data has required columns
        if 'road' not in vehicles_pd.columns:
            st.warning("Vehicle data doesn't have 'road' column. Cannot enrich crash data.")
            return crashes

        # Remove NaN values
        vehicles_clean = vehicles_pd.dropna(subset=['latitude', 'longitude', 'road'])
        crashes_clean = crashes_pd.dropna(subset=[crash_lat_col, crash_lon_col])

        # For each crash, find the nearest vehicle and get its road name
        road_names = []
        for _, crash in crashes_clean.iterrows():
            # Calculate distances to all vehicles (simple euclidean for speed)
            distances = ((vehicles_clean['latitude'] - crash[crash_lat_col])**2 +
                        (vehicles_clean['longitude'] - crash[crash_lon_col])**2)**0.5

            # Find nearest vehicle
            nearest_idx = distances.idxmin()
            road_name = vehicles_clean.loc[nearest_idx, 'road']
            road_names.append(road_name)

        # Add RoadName column to crash data
        crashes_clean['RoadName'] = road_names

        # Convert back to cudf
        return cudf.from_pandas(crashes_clean)
    except Exception as e:
        st.error(f"Error enriching crash data: {str(e)}")
        return crashes

def filter_by_road(road_name):
    """Filter vehicles and crashes by road name"""
    if st.session_state.vehicles is None:
        return None, None

    # GPU-accelerated filtering (cudf requires regex=False when using case=False)
    v_mask = st.session_state.vehicles['road'].str.contains(road_name, case=False, regex=False)
    # Fill NaN mask values with False
    v_mask = v_mask.fillna(False)
    filtered_v = st.session_state.vehicles[v_mask]

    filtered_c = None
    if st.session_state.crashes is not None and 'RoadName' in st.session_state.crashes.columns:
        c_mask = st.session_state.crashes['RoadName'].str.contains(road_name, case=False, regex=False)
        c_mask = c_mask.fillna(False)
        filtered_c = st.session_state.crashes[c_mask]

    return filtered_v, filtered_c

def filter_by_speed(min_speed=None, max_speed=None):
    """Filter vehicles by speed range"""
    if st.session_state.vehicles is None:
        return None

    filtered = st.session_state.vehicles

    if min_speed is not None:
        filtered = filtered[filtered['speed'] >= min_speed]
    if max_speed is not None:
        filtered = filtered[filtered['speed'] <= max_speed]

    return filtered

def filter_by_time(vehicles_df, crashes_df, start_hour, end_hour):
    """Filter vehicles and crashes by time range (hour of day)

    Args:
        vehicles_df: Vehicle DataFrame with timestamp column
        crashes_df: Crash DataFrame with timestamp column
        start_hour: Start hour (0-23)
        end_hour: End hour (0-23)

    Returns:
        Tuple of (filtered_vehicles, filtered_crashes)
    """
    filtered_v = None
    filtered_c = None

    if vehicles_df is not None and len(vehicles_df) > 0:
        # Convert to pandas
        v_df = vehicles_df.to_pandas() if hasattr(vehicles_df, 'to_pandas') else vehicles_df

        # Check if timestamp column exists
        if 'timestamp' not in v_df.columns:
            st.error("Timestamp column not found in vehicle data. Cannot filter by time.")
            return None, None

        # Convert timestamp to datetime and extract hour
        v_df['timestamp'] = pd.to_datetime(v_df['timestamp'])
        v_df['hour'] = v_df['timestamp'].dt.hour

        # Filter by time range
        if start_hour <= end_hour:
            # Normal range (e.g., 8am to 5pm)
            time_mask = (v_df['hour'] >= start_hour) & (v_df['hour'] < end_hour)
        else:
            # Overnight range (e.g., 10pm to 2am)
            time_mask = (v_df['hour'] >= start_hour) | (v_df['hour'] < end_hour)

        filtered_v = v_df[time_mask].drop(columns=['hour'])

        # Convert back to cudf
        filtered_v = cudf.from_pandas(filtered_v)

    if crashes_df is not None and len(crashes_df) > 0:
        # Convert to pandas
        c_df = crashes_df.to_pandas() if hasattr(crashes_df, 'to_pandas') else crashes_df

        if 'timestamp' in c_df.columns:
            # Convert timestamp to datetime and extract hour
            c_df['timestamp'] = pd.to_datetime(c_df['timestamp'])
            c_df['hour'] = c_df['timestamp'].dt.hour

            if start_hour <= end_hour:
                time_mask = (c_df['hour'] >= start_hour) & (c_df['hour'] < end_hour)
            else:
                time_mask = (c_df['hour'] >= start_hour) | (c_df['hour'] < end_hour)

            filtered_c = c_df[time_mask].drop(columns=['hour'])
            filtered_c = cudf.from_pandas(filtered_c)

    return filtered_v, filtered_c

def ensure_deceleration(vehicles_df):
    """
    Ensure deceleration_g column exists - use AccMagnitude if available, otherwise calculate

    Args:
        vehicles_df: Vehicle DataFrame

    Returns:
        DataFrame with deceleration_g column
    """
    if vehicles_df is None:
        return None

    df = vehicles_df.to_pandas() if hasattr(vehicles_df, 'to_pandas') else vehicles_df

    # If deceleration_g already exists, return as-is
    if 'deceleration_g' in df.columns:
        return vehicles_df

    # If AccMagnitude exists, use it (already in g-force, negate for braking convention)
    if 'AccMagnitude' in df.columns:
        df['deceleration_g'] = -df['AccMagnitude']
        return cudf.from_pandas(df) if hasattr(vehicles_df, 'to_pandas') else df

    # Otherwise calculate from speed changes
    return calculate_deceleration(vehicles_df)

def calculate_deceleration(vehicles_df):
    """Calculate deceleration/acceleration from consecutive speed measurements

    Args:
        vehicles_df: Vehicle DataFrame with speed and timestamp columns

    Returns:
        DataFrame with added 'deceleration_g' column (in g-force units)
        Negative values = deceleration/braking
        Positive values = acceleration
    """
    if vehicles_df is None or len(vehicles_df) == 0:
        return None

    # Convert to pandas for easier manipulation
    df = vehicles_df.to_pandas() if hasattr(vehicles_df, 'to_pandas') else vehicles_df.copy()

    # Check required columns
    if 'speed' not in df.columns or 'timestamp' not in df.columns:
        st.error("Speed or timestamp column not found. Cannot calculate deceleration.")
        return None

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by timestamp to ensure proper ordering
    df = df.sort_values('timestamp')

    # Calculate speed change (mph)
    df['speed_change'] = df['speed'].diff()

    # Calculate time difference (seconds)
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()

    # Calculate acceleration (mph/s)
    # Handle division by zero
    df['acceleration_mph_s'] = df['speed_change'] / df['time_diff'].replace(0, float('nan'))

    # Convert to ft/s² (1 mph/s = 1.467 ft/s²)
    df['acceleration_ft_s2'] = df['acceleration_mph_s'] * 1.467

    # Convert to g-force (1g = 32.174 ft/s²)
    df['deceleration_g'] = df['acceleration_ft_s2'] / 32.174

    # Remove first row (NaN from diff) and rows with invalid time differences
    df = df.dropna(subset=['deceleration_g'])
    df = df[df['time_diff'] > 0]  # Remove invalid time differences
    df = df[df['time_diff'] < 60]  # Remove gaps larger than 60 seconds

    # Filter out physically impossible deceleration values (data errors)
    # Maximum realistic braking: -1.5g (most vehicles can't exceed -1.2g)
    # Minimum realistic acceleration: +1.5g (sports cars can do 0-60 in ~3s = ~0.9g)
    count_before = len(df)
    df = df[(df['deceleration_g'] >= -1.5) & (df['deceleration_g'] <= 1.5)]
    count_after = len(df)

    # Notify if significant data was filtered
    filtered_count = count_before - count_after
    if filtered_count > 0:
        st.info(f"Filtered out {filtered_count:,} impossible deceleration values (|g| > 1.5) caused by GPS/data errors")

    # Store time_diff for popup display before dropping
    df['time_gap'] = df['time_diff']

    # Drop intermediate calculation columns (keep time_gap)
    df = df.drop(columns=['speed_change', 'time_diff', 'acceleration_mph_s', 'acceleration_ft_s2'])

    # Convert back to cudf
    return cudf.from_pandas(df)

def filter_hard_braking(vehicles_df, threshold_g=-0.3):
    """Filter vehicle data to show only hard braking events

    Args:
        vehicles_df: Vehicle DataFrame
        threshold_g: Deceleration threshold in g-force (default: -0.3g)
                    More negative = harder braking (e.g., -0.5g, -0.7g)

    Returns:
        Filtered DataFrame with only hard braking events
    """
    if vehicles_df is None or len(vehicles_df) == 0:
        return None

    # Always recalculate deceleration to ensure filters are applied
    # (Don't reuse existing deceleration_g column as it may contain unfiltered data)
    df_with_decel = calculate_deceleration(vehicles_df)
    if df_with_decel is None:
        return None

    # Filter for hard braking events (deceleration < threshold)
    # More negative = harder braking
    hard_braking = df_with_decel[df_with_decel['deceleration_g'] <= threshold_g]

    return hard_braking

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two lat/lon points"""
    from math import radians, sin, cos, sqrt, atan2

    R = 6371000  # Earth radius in meters

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c

def calculate_baseline_speeds(vehicles_df):
    """Calculate baseline (normal) speeds per road from vehicle data

    Returns dict mapping road names to median speed in mph
    """
    if vehicles_df is None or len(vehicles_df) == 0:
        return {}

    df = vehicles_df.to_pandas() if hasattr(vehicles_df, 'to_pandas') else vehicles_df

    # Check required columns
    if 'road_name' not in df.columns or 'speed' not in df.columns:
        return {}

    # Group by road and calculate median speed (robust to outliers)
    baseline_speeds = df.groupby('road_name')['speed'].median().to_dict()

    return baseline_speeds

def analyze_crash_delays(vehicles_df, crashes_df, radius_meters=500, time_window_minutes=60):
    """Analyze traffic delays caused by crashes (OPTIMIZED)

    Args:
        vehicles_df: Vehicle data
        crashes_df: Crash data
        radius_meters: Consider vehicles within this radius of crash (default 500m)
        time_window_minutes: Time window after crash to analyze (default 60 min)

    Returns:
        Dictionary with crash delay analysis results
    """
    import numpy as np

    if vehicles_df is None or len(vehicles_df) == 0:
        return None
    if crashes_df is None or len(crashes_df) == 0:
        return None

    # Convert to pandas for easier processing
    v_df = vehicles_df.to_pandas() if hasattr(vehicles_df, 'to_pandas') else vehicles_df
    c_df = crashes_df.to_pandas() if hasattr(crashes_df, 'to_pandas') else crashes_df

    # Sample crashes if too many (analyze top 100 by default for speed)
    if len(c_df) > 100:
        c_df = c_df.sample(100, random_state=42)

    # Get crash location columns
    lat_col = 'Latitude' if 'Latitude' in c_df.columns else 'latitude'
    lon_col = 'Longitude' if 'Longitude' in c_df.columns else 'longitude'
    road_col = 'RoadName' if 'RoadName' in c_df.columns else 'road_name'

    # Check required vehicle columns
    if 'latitude' not in v_df.columns or 'speed' not in v_df.columns:
        return None

    # Calculate baseline speeds once
    baseline_speeds = calculate_baseline_speeds(v_df)

    crash_results = []

    # Convert vehicle coordinates to numpy arrays for faster computation
    v_lats = v_df['latitude'].values
    v_lons = v_df['longitude'].values
    v_speeds = v_df['speed'].values

    # Analyze each crash
    for idx, crash in c_df.iterrows():
        crash_lat = crash[lat_col]
        crash_lon = crash[lon_col]
        crash_road = crash.get(road_col, 'Unknown')

        if pd.isna(crash_lat) or pd.isna(crash_lon):
            continue

        # Vectorized distance calculation (much faster!)
        # Use simple bounding box first to reduce calculations
        lat_diff = np.abs(v_lats - crash_lat)
        lon_diff = np.abs(v_lons - crash_lon)

        # Rough filter: ~0.01 degrees ≈ 1km
        radius_deg = (radius_meters / 1000) * 0.01
        rough_filter = (lat_diff < radius_deg) & (lon_diff < radius_deg)

        if not rough_filter.any():
            continue

        # Only calculate precise distance for nearby vehicles
        nearby_indices = np.where(rough_filter)[0]

        distances = np.array([
            haversine_distance(v_lats[i], v_lons[i], crash_lat, crash_lon)
            for i in nearby_indices
        ])

        # Find vehicles within radius
        within_radius = distances <= radius_meters
        affected_indices = nearby_indices[within_radius]

        if len(affected_indices) == 0:
            continue

        # Get affected vehicle speeds
        affected_speeds = v_speeds[affected_indices]

        # Get baseline speed for this road
        baseline_speed = baseline_speeds.get(crash_road, np.median(affected_speeds))

        if baseline_speed == 0 or pd.isna(baseline_speed):
            baseline_speed = 55  # Default assumption

        # Calculate actual average speed near crash
        actual_speed = np.mean(affected_speeds)

        # Calculate speed reduction
        speed_reduction = baseline_speed - actual_speed
        speed_reduction_pct = (speed_reduction / baseline_speed) * 100 if baseline_speed > 0 else 0

        # Estimate delay per vehicle (assumes 1km affected segment)
        segment_length_km = (radius_meters / 1000)  # Convert to km

        if actual_speed > 0:
            actual_time = (segment_length_km / actual_speed) * 60  # minutes
        else:
            actual_time = 999  # Very high delay

        baseline_time = (segment_length_km / baseline_speed) * 60  # minutes
        delay_per_vehicle = actual_time - baseline_time  # minutes

        # Total delay across all affected vehicles
        total_delay_minutes = delay_per_vehicle * len(affected_indices)
        total_delay_hours = total_delay_minutes / 60

        crash_results.append({
            'crash_id': idx,
            'road': crash_road,
            'latitude': crash_lat,
            'longitude': crash_lon,
            'affected_vehicles': len(affected_indices),
            'baseline_speed_mph': round(baseline_speed, 1),
            'actual_speed_mph': round(actual_speed, 1),
            'speed_reduction_mph': round(speed_reduction, 1),
            'speed_reduction_pct': round(speed_reduction_pct, 1),
            'delay_per_vehicle_min': round(delay_per_vehicle, 2),
            'total_delay_hours': round(total_delay_hours, 2)
        })

    return crash_results

# ====== VISUALIZATION FUNCTIONS ======
@st.cache_data(ttl=300, max_entries=10)
def prepare_map_data(_vehicles_df, _crashes_df, is_filtered, color_mode, v_count, c_count):
    """Fast data preparation for map rendering (GPU-accelerated, cached)

    v_count and c_count are added to cache key to ensure different datasets don't share cache
    """
    # This function is cached to avoid reprocessing data unnecessarily
    v_pd = None
    c_pd = None
    all_coords = []

    if _vehicles_df is not None and len(_vehicles_df) > 0:
        v_pd = _vehicles_df.to_pandas() if hasattr(_vehicles_df, 'to_pandas') else _vehicles_df
        v_pd = v_pd.dropna(subset=['latitude', 'longitude'])

        if len(v_pd) > 0:
            # Smart sampling based on view
            if is_filtered:
                if len(v_pd) > 5000:
                    v_pd = v_pd.sample(5000)
            else:
                if len(v_pd) > 1000:
                    v_pd = v_pd.sample(1000)

            all_coords.extend(v_pd[['latitude', 'longitude']].values.tolist())

    if _crashes_df is not None and len(_crashes_df) > 0:
        c_pd = _crashes_df.to_pandas() if hasattr(_crashes_df, 'to_pandas') else _crashes_df
        lat_col = 'Latitude' if 'Latitude' in c_pd.columns else ('latitude' if 'latitude' in c_pd.columns else None)
        lon_col = 'Longitude' if 'Longitude' in c_pd.columns else ('longitude' if 'longitude' in c_pd.columns else None)

        if lat_col and lon_col:
            c_pd = c_pd.dropna(subset=[lat_col, lon_col])

            # Sample crashes for better performance and visibility when showing all data
            if not is_filtered and len(c_pd) > 200:
                c_pd = c_pd.sample(200, random_state=42)

            all_coords.extend(c_pd[[lat_col, lon_col]].values.tolist())
        else:
            c_pd = None

    return v_pd, c_pd, all_coords

def create_map(vehicles_df=None, crashes_df=None, is_filtered=False, hotspot_results=None, intensity_results=None):
    """Create interactive Folium map with auto-zoom for filtered data (FAST)"""

    # Calculate counts for cache key
    v_count = len(vehicles_df) if vehicles_df is not None else 0
    c_count = len(crashes_df) if crashes_df is not None else 0

    # Use cached data preparation
    v_pd, c_pd, all_coords = prepare_map_data(
        vehicles_df, crashes_df, is_filtered, st.session_state.color_mode, v_count, c_count
    )

    # Get crash column names if we have crash data
    lat_col = None
    lon_col = None
    if c_pd is not None:
        lat_col = 'Latitude' if 'Latitude' in c_pd.columns else ('latitude' if 'latitude' in c_pd.columns else None)
        lon_col = 'Longitude' if 'Longitude' in c_pd.columns else ('longitude' if 'longitude' in c_pd.columns else None)

    # Calculate center and bounds
    if len(all_coords) > 0:
        lats = [coord[0] for coord in all_coords]
        lons = [coord[1] for coord in all_coords]
        center = [sum(lats) / len(lats), sum(lons) / len(lons)]

        # Calculate bounds for auto-zoom
        if is_filtered and len(all_coords) > 1:
            bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
        else:
            bounds = None
    else:
        center = [38.55, -90.27]
        bounds = None

    # Create map with dark theme and labels
    m = folium.Map(
        location=center,
        zoom_start=10,
        tiles='https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
    )

    # Fit bounds if we have filtered data
    if bounds is not None:
        m.fit_bounds(bounds, padding=[30, 30])

    # Add vehicle markers
    if v_pd is not None and len(v_pd) > 0:
        for _, row in v_pd.iterrows():
            speed = row.get('speed', 0)

            # Color based on mode
            if st.session_state.braking_mode == "on" and 'deceleration_g' in row:
                # Color by braking severity
                decel = row['deceleration_g']
                if decel <= -0.7:
                    color = '#8B0000'  # Dark red - Emergency braking
                    severity = "Emergency"
                elif decel <= -0.5:
                    color = '#FF0000'  # Red - Hard braking
                    severity = "Hard"
                else:
                    color = '#FF8C00'  # Dark orange - Moderate braking
                    severity = "Moderate"

                # Build styled popup for hard braking
                severity_colors = {
                    'Emergency': '#8B0000',
                    'Hard': '#FF0000',
                    'Moderate': '#FF8C00'
                }
                header_color = severity_colors.get(severity, '#FF8C00')

                popup_html = f"""
                <div style="font-family: Arial, sans-serif; min-width: 200px;">
                    <div style="background: {header_color}; color: white; padding: 8px; margin: -10px -10px 8px -10px; border-radius: 3px 3px 0 0;">
                        <strong style="font-size: 14px;">{severity} Braking</strong>
                    </div>
                    <div style="padding: 4px 0;">
                        <div style="margin: 4px 0;"><strong>Force:</strong> {decel:.3f}g</div>
                        <div style="margin: 4px 0;"><strong>Speed:</strong> {speed:.1f} mph</div>
                        <div style="margin: 4px 0;"><strong>Road:</strong> {row.get('road', 'N/A')}</div>
                """

                if 'timestamp' in row and pd.notna(row['timestamp']):
                    popup_html += f"""<div style="margin: 4px 0;"><strong>Time:</strong> {str(row['timestamp'])[11:19]}</div>"""

                if 'speed_limit' in row and pd.notna(row['speed_limit']):
                    popup_html += f"""<div style="margin: 4px 0;"><strong>Limit:</strong> {row['speed_limit']:.0f} mph</div>"""

                if 'time_gap' in row and pd.notna(row['time_gap']):
                    popup_html += f"""<div style="margin: 4px 0; font-size: 11px; color: #666;"><strong>Gap:</strong> {row['time_gap']:.1f}s</div>"""

                popup_html += """</div></div>"""
                popup_text = popup_html
            elif st.session_state.color_mode == "speed":
                # Color by speed using custom scheme
                scheme = st.session_state.color_scheme
                if speed > 70:
                    color = scheme['high']
                    speed_category = "High Speed"
                elif speed > 50:
                    color = scheme['medium']
                    speed_category = "Medium Speed"
                else:
                    color = scheme['low']
                    speed_category = "Low Speed"

                # Build styled popup for speed mode
                popup_html = f"""
                <div style="font-family: Arial, sans-serif; min-width: 200px;">
                    <div style="background: {color}; color: white; padding: 8px; margin: -10px -10px 8px -10px; border-radius: 3px 3px 0 0;">
                        <strong style="font-size: 14px;">{speed_category}</strong>
                    </div>
                    <div style="padding: 4px 0;">
                        <div style="margin: 4px 0;"><strong>Speed:</strong> {speed:.1f} mph</div>
                        <div style="margin: 4px 0;"><strong>Road:</strong> {row.get('road', 'N/A')}</div>
                """

                if 'timestamp' in row and pd.notna(row['timestamp']):
                    popup_html += f"""<div style="margin: 4px 0;"><strong>Time:</strong> {str(row['timestamp'])[11:19]}</div>"""

                if 'speed_limit' in row and pd.notna(row['speed_limit']):
                    popup_html += f"""<div style="margin: 4px 0;"><strong>Limit:</strong> {row['speed_limit']:.0f} mph</div>"""
                    if speed > row['speed_limit']:
                        over = speed - row['speed_limit']
                        popup_html += f"""<div style="margin: 4px 0; padding: 4px; background: #fff3cd; border-left: 3px solid #ff9800; color: #856404;"><strong>Speeding:</strong> {over:.1f} mph over</div>"""

                popup_html += """</div></div>"""
                popup_text = popup_html
            else:
                # Neutral gray color - default view
                color = scheme.get('neutral', '#888888') if st.session_state.color_mode == "speed" else '#888888'

                # Build styled popup for normal view
                popup_html = f"""
                <div style="font-family: Arial, sans-serif; min-width: 200px;">
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 8px; margin: -10px -10px 8px -10px; border-radius: 3px 3px 0 0;">
                        <strong style="font-size: 14px;">Vehicle Data</strong>
                    </div>
                    <div style="padding: 4px 0;">
                        <div style="margin: 4px 0;"><strong>Speed:</strong> {speed:.1f} mph</div>
                        <div style="margin: 4px 0;"><strong>Road:</strong> {row.get('road', 'N/A')}</div>
                """

                if 'timestamp' in row and pd.notna(row['timestamp']):
                    popup_html += f"""<div style="margin: 4px 0;"><strong>Time:</strong> {str(row['timestamp'])[11:19]}</div>"""

                if 'speed_limit' in row and pd.notna(row['speed_limit']):
                    popup_html += f"""<div style="margin: 4px 0;"><strong>Limit:</strong> {row['speed_limit']:.0f} mph</div>"""

                if 'VehicleType' in row and pd.notna(row['VehicleType']):
                    popup_html += f"""<div style="margin: 4px 0;"><strong>Type:</strong> {row['VehicleType']}</div>"""

                popup_html += """</div></div>"""
                popup_text = popup_html

            # Uniform sizing for all circles
            radius = 5  # Same size for all
            opacity = 0.7  # Consistent opacity

            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius,
                color=color,  # Border matches fill color
                weight=1,  # Thin subtle border
                fill=True,
                fillColor=color,
                fillOpacity=opacity,
                popup=popup_text
            ).add_to(m)

    # Add crash markers
    if c_pd is not None and len(c_pd) > 0:
        for _, row in c_pd.iterrows():
            # Build styled crash popup
            road_name = row.get('RoadName', 'Unknown')
            severity = row.get('severity', row.get('Severity', 'Not Specified'))

            crash_popup_html = f"""
            <div style="font-family: Arial, sans-serif; min-width: 200px;">
                <div style="background: #dc3545; color: white; padding: 8px; margin: -10px -10px 8px -10px; border-radius: 3px 3px 0 0;">
                    <strong style="font-size: 14px;">Traffic Crash</strong>
                </div>
                <div style="padding: 4px 0;">
                    <div style="margin: 4px 0;"><strong>Road:</strong> {road_name}</div>
            """

            # Add severity if available and not N/A
            if severity and str(severity).upper() not in ['N/A', 'NONE', 'UNKNOWN', 'NOT SPECIFIED']:
                crash_popup_html += f"""<div style="margin: 4px 0;"><strong>Severity:</strong> {severity}</div>"""

            # Add timestamp if available
            if 'timestamp' in row and pd.notna(row['timestamp']):
                crash_popup_html += f"""<div style="margin: 4px 0;"><strong>Time:</strong> {str(row['timestamp'])[11:19]}</div>"""
            elif 'Timestamp' in row and pd.notna(row['Timestamp']):
                crash_popup_html += f"""<div style="margin: 4px 0;"><strong>Time:</strong> {str(row['Timestamp'])[11:19]}</div>"""
            elif 'CrashDate' in row and pd.notna(row['CrashDate']):
                crash_popup_html += f"""<div style="margin: 4px 0;"><strong>Date:</strong> {str(row['CrashDate'])[:10]}</div>"""

            # Add crash type if available
            if 'CrashType' in row and pd.notna(row['CrashType']):
                crash_popup_html += f"""<div style="margin: 4px 0;"><strong>Type:</strong> {row['CrashType']}</div>"""

            # Add number of vehicles if available
            if 'NumVehicles' in row and pd.notna(row['NumVehicles']):
                crash_popup_html += f"""<div style="margin: 4px 0;"><strong>Vehicles:</strong> {int(row['NumVehicles'])}</div>"""

            # Add injuries if available
            if 'Injuries' in row and pd.notna(row['Injuries']) and row['Injuries'] > 0:
                crash_popup_html += f"""<div style="margin: 4px 0; color: #d9534f;"><strong>Injuries:</strong> {int(row['Injuries'])}</div>"""

            crash_popup_html += """</div></div>"""

            folium.Marker(
                location=[row[lat_col], row[lon_col]],
                popup=crash_popup_html,
                icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa')
            ).add_to(m)

    # Add crash hotspot cluster circles
    if hotspot_results is not None and hotspot_results.get('total_clusters', 0) > 0:
        clusters = hotspot_results['clusters']
        for cluster in clusters:
            # Determine color based on severity (crash count)
            crash_count = cluster['crash_count']
            if crash_count >= 5:
                color = '#8B0000'  # Dark red for high concentration
                opacity = 0.6
            elif crash_count >= 3:
                color = '#FF0000'  # Red for moderate concentration
                opacity = 0.5
            else:
                color = '#FF4444'  # Light red for low concentration
                opacity = 0.4

            # Create cluster popup
            cluster_popup_html = f"""
            <div style="font-family: Arial, sans-serif; min-width: 200px;">
                <div style="background: {color}; color: white; padding: 8px; margin: -10px -10px 8px -10px; border-radius: 3px 3px 0 0;">
                    <strong style="font-size: 14px;">Crash Hotspot #{cluster['cluster_id']}</strong>
                </div>
                <div style="padding: 4px 0;">
                    <div style="margin: 4px 0;"><strong>Crashes:</strong> {crash_count}</div>
                    <div style="margin: 4px 0;"><strong>Radius:</strong> {cluster['radius_meters']:.0f}m</div>
                    <div style="margin: 4px 0;"><strong>Road:</strong> {cluster['primary_road']}</div>
                    <div style="margin: 4px 0;"><strong>Center:</strong> ({cluster['center_lat']:.6f}, {cluster['center_lon']:.6f})</div>
                    <div style="margin: 8px 0; padding: 6px; background: #fff3cd; border-left: 3px solid #ff9800; color: #856404;">
                        <strong>High-risk zone!</strong> Multiple crashes within small area.
                    </div>
                </div>
            </div>
            """

            # Add circle for cluster
            folium.Circle(
                location=[cluster['center_lat'], cluster['center_lon']],
                radius=cluster['radius_meters'],
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=opacity,
                opacity=0.8,
                weight=2,
                popup=cluster_popup_html,
                tooltip=f"Hotspot: {crash_count} crashes"
            ).add_to(m)

            # Add center marker
            folium.CircleMarker(
                location=[cluster['center_lat'], cluster['center_lon']],
                radius=6,
                color='white',
                fillColor=color,
                fillOpacity=1.0,
                weight=2,
                popup=cluster_popup_html
            ).add_to(m)

    return m

def plot_speed_distribution(data, title="Speed Distribution"):
    """Create speed distribution plot"""
    if data is None or len(data) == 0:
        st.warning("No data to plot")
        return

    df = data.to_pandas() if hasattr(data, 'to_pandas') else data

    fig = px.histogram(
        df,
        x='speed',
        nbins=40,
        title=title,
        labels={'speed': 'Speed (mph)', 'count': 'Count'},
        color_discrete_sequence=['#667eea']
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def load_timespace_data(road=None, max_records=50000):
    """Load vehicle data with timestamps for timespace diagram"""
    try:
        cv_data_path = "/home/ubuntu/cvdata/cv_processed.csv"
        if not os.path.exists(cv_data_path):
            st.warning("CV data file not found")
            return None

        # Load with timestamp column
        cols = ['VehicleID', 'TimestampUTC', 'SnappedLatitude', 'SnappedLongitude', 'SpeedMPH', 'RoadName']
        df = cudf.read_csv(cv_data_path, usecols=cols, nrows=max_records)

        # Filter by road if specified (using contains for partial matching)
        if road:
            mask = df['RoadName'].str.contains(road, case=False, regex=False)
            mask = mask.fillna(False)
            df = df[mask]

        if len(df) == 0:
            return None

        return df.to_pandas()

    except Exception as e:
        st.error(f"Error loading timespace data: {str(e)}")
        return None

def plot_timespace_diagram(road=None):
    """Create timespace diagram showing vehicle trajectories along a corridor"""

    # Load data with timestamps
    with st.spinner("Loading vehicle trajectory data..."):
        df = load_timespace_data(road=road)

    if df is None or len(df) == 0:
        st.warning(f"No trajectory data found{f' for {road}' if road else ''}")
        return

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['TimestampUTC'])

    # Calculate cumulative distance along corridor (simplified using lat/lon)
    # Group by vehicle and sort by timestamp
    df = df.sort_values(['VehicleID', 'timestamp'])

    # Calculate distance from starting point of each vehicle
    def calculate_distance(group):
        # Use haversine formula or simple euclidean distance
        if len(group) < 2:
            group['distance'] = 0
            return group

        lat1, lon1 = group['SnappedLatitude'].iloc[0], group['SnappedLongitude'].iloc[0]

        # Simple euclidean distance (scaled)
        group['distance'] = ((group['SnappedLatitude'] - lat1)**2 +
                           (group['SnappedLongitude'] - lon1)**2)**0.5 * 69  # Approx miles

        return group

    df = df.groupby('VehicleID').apply(calculate_distance).reset_index(drop=True)

    # Limit to top vehicles by data point count for visualization clarity
    top_vehicles = df['VehicleID'].value_counts().head(20).index
    df_plot = df[df['VehicleID'].isin(top_vehicles)]

    # Create timespace diagram
    fig = px.line(
        df_plot,
        x='timestamp',
        y='distance',
        color='VehicleID',
        title=f"Timespace Diagram{f' - {road}' if road else ' - All Roads'}",
        labels={'timestamp': 'Time', 'distance': 'Distance Along Corridor (miles)'},
        hover_data=['SpeedMPH']
    )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Distance Along Corridor (miles)",
        showlegend=False,
        hovermode='closest'
    )

    st.plotly_chart(fig, use_container_width=True)
    st.info(f"Showing trajectories for top 20 vehicles by data points{f' on {road}' if road else ''}")

def plot_crashes_by_road(data, top_n=10):
    """Create crashes by road plot"""
    if data is None or len(data) == 0:
        st.warning("No crash data to plot")
        return

    df = data.to_pandas() if hasattr(data, 'to_pandas') else data

    if 'RoadName' not in df.columns:
        st.warning("No RoadName column in crash data. Upload crash data again - the system will automatically look up road names from vehicle data!")
        return

    # Remove NaN values
    df_clean = df.dropna(subset=['RoadName'])

    if len(df_clean) == 0:
        st.warning("No valid road names found in crash data")
        return

    road_counts = df_clean['RoadName'].value_counts().head(top_n)

    fig = px.bar(
        x=road_counts.index,
        y=road_counts.values,
        title=f"Top {top_n} Roads by Crash Count",
        labels={'x': 'Road', 'y': 'Crash Count'},
        color_discrete_sequence=['#764ba2']
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_hard_braking_distribution(data, title="Hard Braking Distribution"):
    """Create hard braking severity distribution plot"""
    if data is None or len(data) == 0:
        st.warning("No hard braking data to plot")
        return

    df = data.to_pandas() if hasattr(data, 'to_pandas') else data

    if 'deceleration_g' not in df.columns:
        st.warning("No deceleration data available. Run 'Show hard braking events' first.")
        return

    # Create histogram with severity color coding
    fig = go.Figure()

    # Define severity bins
    emergency = df[df['deceleration_g'] <= -0.7]
    hard = df[(df['deceleration_g'] > -0.7) & (df['deceleration_g'] <= -0.5)]
    moderate = df[(df['deceleration_g'] > -0.5) & (df['deceleration_g'] <= -0.3)]

    # Add histogram traces for each severity level
    fig.add_trace(go.Histogram(
        x=emergency['deceleration_g'],
        name='Emergency (≤-0.7g)',
        marker_color='#8B0000',
        opacity=0.7,
        xbins=dict(size=0.05)
    ))

    fig.add_trace(go.Histogram(
        x=hard['deceleration_g'],
        name='Hard (≤-0.5g)',
        marker_color='#FF0000',
        opacity=0.7,
        xbins=dict(size=0.05)
    ))

    fig.add_trace(go.Histogram(
        x=moderate['deceleration_g'],
        name='Moderate (≤-0.3g)',
        marker_color='#FF8C00',
        opacity=0.7,
        xbins=dict(size=0.05)
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Deceleration (g-force)',
        yaxis_title='Count',
        barmode='stack',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Emergency Braking", f"{len(emergency):,}", f"{(len(emergency)/len(df)*100):.1f}%")
    with col2:
        st.metric("Hard Braking", f"{len(hard):,}", f"{(len(hard)/len(df)*100):.1f}%")
    with col3:
        st.metric("Moderate Braking", f"{len(moderate):,}", f"{(len(moderate)/len(df)*100):.1f}%")

# ====== AI-POWERED CRASH-BRAKING ANALYSIS ======
def analyze_proximity_risk(vehicles_df, crashes_df, distance_meters=200):
    """Analyze hard braking events near crash locations to identify risk zones

    Args:
        vehicles_df: DataFrame with vehicle data including deceleration
        crashes_df: DataFrame with crash locations
        distance_meters: Proximity threshold in meters (default 200m)

    Returns:
        dict with analysis results
    """
    if vehicles_df is None or crashes_df is None:
        return None

    # Ensure deceleration_g exists
    vehicles_df = ensure_deceleration(vehicles_df)
    if vehicles_df is None:
        return None

    # Convert to pandas for easier manipulation
    v_pd = vehicles_df.to_pandas() if hasattr(vehicles_df, 'to_pandas') else vehicles_df
    c_pd = crashes_df.to_pandas() if hasattr(crashes_df, 'to_pandas') else crashes_df

    # Calculate hard braking first if not already done
    if 'deceleration_g' not in v_pd.columns:
        v_pd = calculate_deceleration(vehicles_df)
        if v_pd is None:
            return None
        v_pd = v_pd.to_pandas() if hasattr(v_pd, 'to_pandas') else v_pd

    # Filter for hard braking events (≤-0.3g)
    hard_braking = v_pd[v_pd['deceleration_g'] <= -0.3].copy()

    if len(hard_braking) == 0 or len(c_pd) == 0:
        return {
            'total_hard_braking': 0,
            'total_crashes': 0,
            'near_misses': [],
            'risk_zones': []
        }

    # Simple distance calculation (approximate)
    # 1 degree lat/lon ≈ 111km, distance_meters/111000 degrees
    threshold_deg = distance_meters / 111000

    near_misses = []

    # Get crash location columns (check various naming conventions)
    if 'SnappedLatitude' in c_pd.columns:
        lat_col = 'SnappedLatitude'
    elif 'Latitude' in c_pd.columns:
        lat_col = 'Latitude'
    elif 'latitude' in c_pd.columns:
        lat_col = 'latitude'
    else:
        lat_col = 'lat'  # fallback

    if 'SnappedLongitude' in c_pd.columns:
        lon_col = 'SnappedLongitude'
    elif 'Longitude' in c_pd.columns:
        lon_col = 'Longitude'
    elif 'longitude' in c_pd.columns:
        lon_col = 'longitude'
    else:
        lon_col = 'lon'  # fallback

    # For each crash, find nearby hard braking events
    for _, crash in c_pd.iterrows():
        crash_lat = crash[lat_col]
        crash_lon = crash[lon_col]

        # Find hard braking within distance
        nearby = hard_braking[
            (abs(hard_braking['latitude'] - crash_lat) < threshold_deg) &
            (abs(hard_braking['longitude'] - crash_lon) < threshold_deg)
        ]

        if len(nearby) > 0:
            near_misses.append({
                'crash_location': (crash_lat, crash_lon),
                'crash_road': crash.get('RoadName', 'Unknown'),
                'nearby_hard_braking': len(nearby),
                'avg_deceleration': nearby['deceleration_g'].mean(),
                'max_deceleration': nearby['deceleration_g'].min()  # Most negative = strongest braking
            })

    # Identify risk zones (areas with high hard braking but no crashes yet)
    # Group hard braking by road
    if 'road' in hard_braking.columns:
        road_braking = hard_braking.groupby('road').agg({
            'deceleration_g': ['count', 'mean', 'min']
        }).reset_index()
        road_braking.columns = ['road', 'hard_braking_count', 'avg_deceleration', 'max_deceleration']

        # Count crashes per road
        if 'RoadName' in c_pd.columns:
            road_crashes = c_pd.groupby('RoadName').size().reset_index(name='crash_count')

            # Merge and find high braking, low crash roads
            risk_analysis = road_braking.merge(
                road_crashes,
                left_on='road',
                right_on='RoadName',
                how='left'
            )
            risk_analysis['crash_count'] = risk_analysis['crash_count'].fillna(0)

            # Calculate risk score: high braking count + severe braking - crashes
            # (Negative deceleration, so more negative = higher risk)
            risk_analysis['risk_score'] = (
                risk_analysis['hard_braking_count'] *
                abs(risk_analysis['avg_deceleration']) *
                (1 / (risk_analysis['crash_count'] + 1))  # Inverse of crashes
            )

            # Sort by risk score
            risk_zones = risk_analysis.sort_values('risk_score', ascending=False).head(10)
            risk_zones_list = risk_zones.to_dict('records')
        else:
            risk_zones_list = []
    else:
        risk_zones_list = []

    return {
        'total_hard_braking': len(hard_braking),
        'total_crashes': len(c_pd),
        'near_misses': near_misses,
        'near_miss_count': len(near_misses),
        'risk_zones': risk_zones_list,
        'correlation_rate': len(near_misses) / len(c_pd) * 100 if len(c_pd) > 0 else 0
    }

def calculate_road_safety_scores(vehicles_df, crashes_df):
    """Calculate comprehensive safety scores for each road

    Returns:
        DataFrame with road safety metrics
    """
    if vehicles_df is None:
        return None

    # Ensure deceleration_g exists
    vehicles_df = ensure_deceleration(vehicles_df)
    if vehicles_df is None:
        return None

    v_pd = vehicles_df.to_pandas() if hasattr(vehicles_df, 'to_pandas') else vehicles_df

    # Calculate metrics per road
    if 'road' not in v_pd.columns:
        return None

    road_stats = v_pd.groupby('road').agg({
        'speed': ['mean', 'std', 'count'],
        'deceleration_g': lambda x: (x <= -0.3).sum() if 'deceleration_g' in v_pd.columns else 0
    }).reset_index()

    road_stats.columns = ['road', 'avg_speed', 'speed_variance', 'total_records', 'hard_braking_count']

    # Add crash counts if available
    if crashes_df is not None:
        c_pd = crashes_df.to_pandas() if hasattr(crashes_df, 'to_pandas') else crashes_df
        if 'RoadName' in c_pd.columns:
            crash_counts = c_pd.groupby('RoadName').size().reset_index(name='crash_count')
            road_stats = road_stats.merge(crash_counts, left_on='road', right_on='RoadName', how='left')
            road_stats['crash_count'] = road_stats['crash_count'].fillna(0)
        else:
            road_stats['crash_count'] = 0
    else:
        road_stats['crash_count'] = 0

    # Calculate safety score (lower = more dangerous)
    # Factor in: crashes, hard braking frequency, speed variance
    road_stats['hard_braking_rate'] = road_stats['hard_braking_count'] / road_stats['total_records']
    road_stats['crash_rate'] = road_stats['crash_count'] / road_stats['total_records'] * 1000  # Per 1000 records

    # Composite safety score (0-100, higher = safer)
    max_crash_rate = road_stats['crash_rate'].max() if road_stats['crash_rate'].max() > 0 else 1
    max_braking_rate = road_stats['hard_braking_rate'].max() if road_stats['hard_braking_rate'].max() > 0 else 1

    road_stats['safety_score'] = 100 - (
        (road_stats['crash_rate'] / max_crash_rate * 40) +
        (road_stats['hard_braking_rate'] / max_braking_rate * 40) +
        (road_stats['speed_variance'] / road_stats['speed_variance'].max() * 20)
    )

    # Classify risk level
    road_stats['risk_level'] = pd.cut(
        road_stats['safety_score'],
        bins=[0, 30, 60, 80, 100],
        labels=['Critical', 'High', 'Moderate', 'Low']
    )

    return road_stats.sort_values('safety_score')

def analyze_temporal_risk(vehicles_df, crashes_df, distance_meters=200):
    """Analyze when hard braking and crashes occur - temporal risk patterns

    Args:
        vehicles_df: DataFrame with vehicle data including timestamp and deceleration
        crashes_df: DataFrame with crash locations and timestamps
        distance_meters: Proximity threshold for matching braking to crashes

    Returns:
        dict with temporal analysis results including:
        - hourly_risk: Risk score by hour of day
        - daily_risk: Risk score by day of week
        - time_to_crash: Distribution of time gaps between braking and crashes
        - high_risk_windows: Time windows with elevated crash risk
        - road_temporal_patterns: Per-road temporal risk patterns
    """
    if vehicles_df is None or crashes_df is None:
        return None

    # Ensure deceleration_g exists
    vehicles_df = ensure_deceleration(vehicles_df)
    if vehicles_df is None:
        return None

    # Convert to pandas
    v_pd = vehicles_df.to_pandas() if hasattr(vehicles_df, 'to_pandas') else vehicles_df
    c_pd = crashes_df.to_pandas() if hasattr(crashes_df, 'to_pandas') else crashes_df

    # Ensure we have timestamps - check various naming conventions
    vehicle_time_col = None
    crash_time_col = None

    # Check vehicle timestamp
    if 'timestamp' in v_pd.columns:
        vehicle_time_col = 'timestamp'
    elif 'Timestamp' in v_pd.columns:
        vehicle_time_col = 'Timestamp'
    elif 'time' in v_pd.columns:
        vehicle_time_col = 'time'
    else:
        return None

    # Check crash timestamp
    if 'timestamp' in c_pd.columns:
        crash_time_col = 'timestamp'
    elif 'Timestamp' in c_pd.columns:
        crash_time_col = 'Timestamp'
    elif 'CrashDate' in c_pd.columns:
        crash_time_col = 'CrashDate'
    elif 'crash_time' in c_pd.columns:
        crash_time_col = 'crash_time'
    elif 'pub_millis' in c_pd.columns:
        crash_time_col = 'pub_millis'
    else:
        return None

    # Standardize to 'timestamp' column name
    if vehicle_time_col != 'timestamp':
        v_pd['timestamp'] = v_pd[vehicle_time_col]
    if crash_time_col != 'timestamp':
        # pub_millis is actually a datetime string, not milliseconds
        c_pd['timestamp'] = c_pd[crash_time_col]

    # Calculate hard braking if not done
    if 'deceleration_g' not in v_pd.columns:
        v_pd = calculate_deceleration(vehicles_df)
        if v_pd is None:
            return None
        v_pd = v_pd.to_pandas() if hasattr(v_pd, 'to_pandas') else v_pd

    # Convert timestamps to datetime and remove timezone info to avoid comparison issues
    v_pd['timestamp'] = pd.to_datetime(v_pd['timestamp'])
    c_pd['timestamp'] = pd.to_datetime(c_pd['timestamp'])

    # Remove timezone information if present (make timezone-naive)
    if hasattr(v_pd['timestamp'].dt, 'tz') and v_pd['timestamp'].dt.tz is not None:
        v_pd['timestamp'] = v_pd['timestamp'].dt.tz_localize(None)
    if hasattr(c_pd['timestamp'].dt, 'tz') and c_pd['timestamp'].dt.tz is not None:
        c_pd['timestamp'] = c_pd['timestamp'].dt.tz_localize(None)

    # Extract time features
    v_pd['hour'] = v_pd['timestamp'].dt.hour
    v_pd['day_of_week'] = v_pd['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
    v_pd['day_name'] = v_pd['timestamp'].dt.day_name()

    c_pd['hour'] = c_pd['timestamp'].dt.hour
    c_pd['day_of_week'] = c_pd['timestamp'].dt.dayofweek
    c_pd['day_name'] = c_pd['timestamp'].dt.day_name()

    # Filter hard braking events
    hard_braking = v_pd[v_pd['deceleration_g'] <= -0.3].copy()

    if len(hard_braking) == 0 or len(c_pd) == 0:
        return None

    # 1. HOURLY RISK ANALYSIS
    crashes_by_hour = c_pd.groupby('hour').size()
    braking_by_hour = hard_braking.groupby('hour').size()
    total_vehicles_by_hour = v_pd.groupby('hour').size()

    hourly_risk = pd.DataFrame({
        'hour': range(24),
        'crashes': crashes_by_hour.reindex(range(24), fill_value=0),
        'hard_braking': braking_by_hour.reindex(range(24), fill_value=0),
        'total_vehicles': total_vehicles_by_hour.reindex(range(24), fill_value=1)
    })

    # Calculate risk score per hour (normalized)
    hourly_risk['crash_rate'] = hourly_risk['crashes'] / hourly_risk['total_vehicles'] * 1000
    hourly_risk['braking_rate'] = hourly_risk['hard_braking'] / hourly_risk['total_vehicles'] * 100

    # Composite hourly risk score (0-100, higher = more dangerous)
    max_crash_rate = hourly_risk['crash_rate'].max() if hourly_risk['crash_rate'].max() > 0 else 1
    max_braking_rate = hourly_risk['braking_rate'].max() if hourly_risk['braking_rate'].max() > 0 else 1

    hourly_risk['risk_score'] = (
        (hourly_risk['crash_rate'] / max_crash_rate * 50) +
        (hourly_risk['braking_rate'] / max_braking_rate * 50)
    )

    # 2. DAY OF WEEK ANALYSIS
    crashes_by_day = c_pd.groupby('day_name').size()
    braking_by_day = hard_braking.groupby('day_name').size()

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_risk = pd.DataFrame({
        'day': day_order,
        'crashes': [crashes_by_day.get(d, 0) for d in day_order],
        'hard_braking': [braking_by_day.get(d, 0) for d in day_order]
    })

    # 3. TIME-TO-CRASH ANALYSIS (leading indicator)
    # For each crash, find hard braking events nearby in time AND space
    threshold_deg = distance_meters / 111000
    time_gaps = []

    # Get crash location columns
    if 'SnappedLatitude' in c_pd.columns:
        lat_col, lon_col = 'SnappedLatitude', 'SnappedLongitude'
    elif 'Latitude' in c_pd.columns:
        lat_col, lon_col = 'Latitude', 'Longitude'
    elif 'latitude' in c_pd.columns:
        lat_col, lon_col = 'latitude', 'longitude'
    else:
        lat_col, lon_col = 'lat', 'lon'

    for _, crash in c_pd.iterrows():
        crash_time = crash['timestamp']
        crash_lat = crash[lat_col]
        crash_lon = crash[lon_col]

        # Find hard braking within distance that happened BEFORE crash
        nearby_braking = hard_braking[
            (abs(hard_braking['latitude'] - crash_lat) < threshold_deg) &
            (abs(hard_braking['longitude'] - crash_lon) < threshold_deg) &
            (hard_braking['timestamp'] < crash_time)
        ]

        if len(nearby_braking) > 0:
            # Calculate time gaps in minutes
            for _, brake in nearby_braking.iterrows():
                time_gap_minutes = (crash_time - brake['timestamp']).total_seconds() / 60
                if time_gap_minutes <= 1440:  # Within 24 hours
                    time_gaps.append(time_gap_minutes)

    # 4. HIGH-RISK TIME WINDOWS
    high_risk_hours = hourly_risk[hourly_risk['risk_score'] >= hourly_risk['risk_score'].quantile(0.75)]['hour'].tolist()

    # 5. PER-ROAD TEMPORAL PATTERNS
    road_temporal = []
    if 'road' in hard_braking.columns and 'RoadName' in c_pd.columns:
        for road in hard_braking['road'].unique():
            road_braking = hard_braking[hard_braking['road'] == road]
            road_crashes = c_pd[c_pd['RoadName'] == road]

            if len(road_crashes) > 0:
                # Peak crash hour for this road
                crash_hour_counts = road_crashes.groupby('hour').size()
                peak_hour = crash_hour_counts.idxmax() if len(crash_hour_counts) > 0 else None

                # Peak braking hour
                braking_hour_counts = road_braking.groupby('hour').size()
                peak_braking_hour = braking_hour_counts.idxmax() if len(braking_hour_counts) > 0 else None

                road_temporal.append({
                    'road': road,
                    'total_crashes': len(road_crashes),
                    'total_hard_braking': len(road_braking),
                    'peak_crash_hour': peak_hour,
                    'peak_braking_hour': peak_braking_hour,
                    'crashes_during_peak': crash_hour_counts.max() if len(crash_hour_counts) > 0 else 0
                })

    return {
        'hourly_risk': hourly_risk,
        'daily_risk': daily_risk,
        'time_gaps': time_gaps,
        'high_risk_hours': high_risk_hours,
        'road_temporal_patterns': road_temporal,
        'total_time_gaps_found': len(time_gaps),
        'median_time_to_crash': np.median(time_gaps) if len(time_gaps) > 0 else None,
        'crashes_with_warning': len([t for t in time_gaps if t <= 60])  # Within 1 hour
    }

def analyze_crash_hotspots(crashes_df, cluster_radius_meters=50, min_crashes=2):
    """Identify crash hotspot clusters using spatial clustering

    Args:
        crashes_df: DataFrame with crash locations
        cluster_radius_meters: Maximum distance between crashes in same cluster (default 50m)
        min_crashes: Minimum crashes to form a cluster (default 2)

    Returns:
        dict with hotspot analysis results
    """
    if crashes_df is None or len(crashes_df) == 0:
        return None

    # Convert to pandas
    c_pd = crashes_df.to_pandas() if hasattr(crashes_df, 'to_pandas') else crashes_df

    # Get crash location columns
    if 'SnappedLatitude' in c_pd.columns:
        lat_col, lon_col = 'SnappedLatitude', 'SnappedLongitude'
    elif 'Latitude' in c_pd.columns:
        lat_col, lon_col = 'Latitude', 'Longitude'
    elif 'latitude' in c_pd.columns:
        lat_col, lon_col = 'latitude', 'longitude'
    else:
        lat_col, lon_col = 'lat', 'lon'

    # Remove NaN coordinates
    c_pd = c_pd.dropna(subset=[lat_col, lon_col])

    if len(c_pd) == 0:
        return None

    # Prepare coordinates for clustering
    coords = c_pd[[lat_col, lon_col]].values

    # Convert radius from meters to degrees (approximate)
    # 1 degree ≈ 111km, so divide meters by 111000
    eps_degrees = cluster_radius_meters / 111000

    # Use DBSCAN clustering
    from sklearn.cluster import DBSCAN

    clustering = DBSCAN(eps=eps_degrees, min_samples=min_crashes, metric='euclidean')
    c_pd['cluster'] = clustering.fit_predict(coords)

    # Extract clusters (excluding noise points labeled as -1)
    clusters = []
    for cluster_id in c_pd['cluster'].unique():
        if cluster_id == -1:  # Skip noise points
            continue

        cluster_crashes = c_pd[c_pd['cluster'] == cluster_id]

        # Calculate cluster center (centroid)
        center_lat = cluster_crashes[lat_col].mean()
        center_lon = cluster_crashes[lon_col].mean()

        # Calculate cluster radius (max distance from center)
        def calc_distance(row):
            # Haversine distance approximation
            dlat = row[lat_col] - center_lat
            dlon = row[lon_col] - center_lon
            return np.sqrt(dlat**2 + dlon**2) * 111000  # Convert to meters

        cluster_crashes['dist_from_center'] = cluster_crashes.apply(calc_distance, axis=1)
        radius_meters = cluster_crashes['dist_from_center'].max()

        # Get road name if available
        road_names = cluster_crashes['RoadName'].value_counts() if 'RoadName' in cluster_crashes.columns else pd.Series()
        primary_road = road_names.index[0] if len(road_names) > 0 else "Unknown"

        clusters.append({
            'cluster_id': int(cluster_id),
            'center_lat': center_lat,
            'center_lon': center_lon,
            'crash_count': len(cluster_crashes),
            'radius_meters': radius_meters,
            'primary_road': primary_road,
            'crashes': cluster_crashes[[lat_col, lon_col]].values.tolist()
        })

    # Sort by crash count (most dangerous first)
    clusters = sorted(clusters, key=lambda x: x['crash_count'], reverse=True)

    return {
        'total_crashes': len(c_pd),
        'total_clusters': len(clusters),
        'clustered_crashes': sum(c['crash_count'] for c in clusters),
        'isolated_crashes': len(c_pd[c_pd['cluster'] == -1]),
        'clusters': clusters,
        'cluster_radius': cluster_radius_meters
    }

def analyze_braking_intensity(vehicles_df, crashes_df):
    """Analyze braking intensity patterns and correlation with crashes

    Args:
        vehicles_df: DataFrame with vehicle data including deceleration
        crashes_df: DataFrame with crash locations

    Returns:
        dict with intensity analysis results
    """
    if vehicles_df is None:
        return None

    # Ensure deceleration_g exists
    vehicles_df = ensure_deceleration(vehicles_df)
    if vehicles_df is None:
        return None

    # Convert to pandas
    v_pd = vehicles_df.to_pandas() if hasattr(vehicles_df, 'to_pandas') else vehicles_df

    # Calculate deceleration if not done
    if 'deceleration_g' not in v_pd.columns:
        v_pd = calculate_deceleration(vehicles_df)
        if v_pd is None:
            return None
        v_pd = v_pd.to_pandas() if hasattr(v_pd, 'to_pandas') else v_pd

    # Filter for hard braking events
    hard_braking = v_pd[v_pd['deceleration_g'] <= -0.3].copy()

    if len(hard_braking) == 0:
        return None

    # Classify braking severity
    def classify_severity(g):
        if g <= -0.7:
            return 'Emergency'
        elif g <= -0.5:
            return 'Hard'
        else:
            return 'Moderate'

    hard_braking['severity'] = hard_braking['deceleration_g'].apply(classify_severity)

    # Count by severity
    severity_counts = hard_braking['severity'].value_counts()

    # Per-road intensity analysis
    road_intensity = []
    if 'road' in hard_braking.columns:
        for road in hard_braking['road'].unique():
            road_braking = hard_braking[hard_braking['road'] == road]

            emergency = len(road_braking[road_braking['severity'] == 'Emergency'])
            hard = len(road_braking[road_braking['severity'] == 'Hard'])
            moderate = len(road_braking[road_braking['severity'] == 'Moderate'])

            avg_decel = road_braking['deceleration_g'].mean()
            max_decel = road_braking['deceleration_g'].min()  # Most negative

            # Calculate intensity score (weighted by severity)
            intensity_score = (emergency * 3) + (hard * 2) + (moderate * 1)

            road_intensity.append({
                'road': road,
                'total_braking': len(road_braking),
                'emergency': emergency,
                'hard': hard,
                'moderate': moderate,
                'avg_deceleration': avg_decel,
                'max_deceleration': max_decel,
                'intensity_score': intensity_score
            })

    # Sort by intensity score
    road_intensity = sorted(road_intensity, key=lambda x: x['intensity_score'], reverse=True)

    # If crash data available, correlate intensity with crashes
    crash_correlation = None
    if crashes_df is not None and 'road' in hard_braking.columns:
        c_pd = crashes_df.to_pandas() if hasattr(crashes_df, 'to_pandas') else crashes_df
        if 'RoadName' in c_pd.columns:
            crash_counts = c_pd.groupby('RoadName').size()

            # For top intensity roads, get crash counts
            for road_data in road_intensity[:10]:
                road_crashes = crash_counts.get(road_data['road'], 0)
                road_data['crashes'] = road_crashes

    return {
        'total_hard_braking': len(hard_braking),
        'emergency_count': severity_counts.get('Emergency', 0),
        'hard_count': severity_counts.get('Hard', 0),
        'moderate_count': severity_counts.get('Moderate', 0),
        'avg_deceleration': hard_braking['deceleration_g'].mean(),
        'max_deceleration': hard_braking['deceleration_g'].min(),
        'road_intensity': road_intensity,
        'severity_distribution': severity_counts.to_dict()
    }

def plot_crash_hotspots(hotspot_results):
    """Visualize crash hotspot clustering analysis"""
    if hotspot_results is None or hotspot_results.get('total_clusters', 0) == 0:
        st.warning("No crash hotspots found. Try adjusting the cluster radius or minimum crash count.")
        return

    st.subheader("Crash Hotspot Clustering Analysis")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Hotspots Identified", hotspot_results['total_clusters'])
    with col2:
        st.metric("Clustered Crashes", hotspot_results['clustered_crashes'])
    with col3:
        total_crashes = hotspot_results.get('total_crashes', hotspot_results['clustered_crashes'])
        cluster_rate = (hotspot_results['clustered_crashes'] / total_crashes * 100) if total_crashes > 0 else 0
        st.metric("Clustering Rate", f"{cluster_rate:.1f}%",
                 help="% of crashes that occur in hotspots")
    with col4:
        avg_crashes = hotspot_results['clustered_crashes'] / hotspot_results['total_clusters'] if hotspot_results['total_clusters'] > 0 else 0
        st.metric("Avg Crashes/Hotspot", f"{avg_crashes:.1f}")

    # Top dangerous spots table
    st.markdown("### Top 10 Most Dangerous Crash Hotspots")
    st.info("These are spatial clusters where multiple crashes occurred within a small radius. Urgent intervention needed!")

    clusters = hotspot_results['clusters'][:10]
    cluster_df = pd.DataFrame(clusters)

    # Format for display
    display_df = cluster_df[['cluster_id', 'crash_count', 'center_lat', 'center_lon', 'radius_meters', 'primary_road']].copy()
    display_df.columns = ['Cluster ID', 'Crashes', 'Latitude', 'Longitude', 'Radius (m)', 'Primary Road']
    display_df['Latitude'] = display_df['Latitude'].round(6)
    display_df['Longitude'] = display_df['Longitude'].round(6)
    display_df['Radius (m)'] = display_df['Radius (m)'].round(1)

    st.dataframe(display_df, use_container_width=True)

    # Bar chart of hotspot crash counts
    fig_crashes = px.bar(
        cluster_df.head(10),
        x='cluster_id',
        y='crash_count',
        title="Crash Count by Hotspot Cluster",
        color='crash_count',
        color_continuous_scale='Reds',
        labels={'cluster_id': 'Cluster ID', 'crash_count': 'Number of Crashes'}
    )
    fig_crashes.update_layout(showlegend=False)
    st.plotly_chart(fig_crashes, use_container_width=True)

    # Cluster size distribution
    cluster_sizes = [c['crash_count'] for c in hotspot_results['clusters']]
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=cluster_sizes,
        nbinsx=10,
        marker_color='#FF4444',
        name='Hotspots'
    ))
    fig_dist.update_layout(
        title="Hotspot Cluster Size Distribution",
        xaxis_title="Crashes per Hotspot",
        yaxis_title="Number of Hotspots",
        showlegend=False
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # Spatial concentration insight
    if hotspot_results['total_clusters'] > 0:
        st.markdown("### Spatial Concentration Insights")
        top_3_crashes = sum(c['crash_count'] for c in hotspot_results['clusters'][:3])
        top_3_pct = (top_3_crashes / hotspot_results['clustered_crashes'] * 100) if hotspot_results['clustered_crashes'] > 0 else 0

        st.info(f"""
        **Key Finding:** The top 3 hotspots account for **{top_3_crashes} crashes ({top_3_pct:.1f}%)** of all clustered crashes.

        This extreme concentration suggests targeted interventions at these locations could have significant impact.
        """)

def plot_braking_intensity(intensity_results):
    """Visualize braking intensity analysis"""
    if intensity_results is None or intensity_results.get('total_hard_braking', 0) == 0:
        st.warning("No braking intensity data available")
        return

    st.subheader("Braking Intensity & Crash Severity Analysis")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Hard Braking", f"{intensity_results['total_hard_braking']:,}")
    with col2:
        emergency_pct = (intensity_results['emergency_count'] / intensity_results['total_hard_braking'] * 100) if intensity_results['total_hard_braking'] > 0 else 0
        st.metric("Emergency Braking", f"{intensity_results['emergency_count']:,}",
                 delta=f"{emergency_pct:.1f}%", delta_color="inverse")
    with col3:
        st.metric("Avg Deceleration", f"{intensity_results['avg_deceleration']:.2f}g")
    with col4:
        st.metric("Max Deceleration", f"{intensity_results['max_deceleration']:.2f}g",
                 help="Most extreme braking event recorded")

    # Severity distribution pie chart
    st.markdown("### Braking Severity Distribution")
    severity_dist = intensity_results['severity_distribution']

    fig_pie = go.Figure(data=[go.Pie(
        labels=list(severity_dist.keys()),
        values=list(severity_dist.values()),
        marker=dict(colors=['#8B0000', '#FF4444', '#FFA500']),
        hole=0.4
    )])
    fig_pie.update_layout(
        title="Braking Events by Severity Level",
        annotations=[dict(text='Severity', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Top intensity roads table
    if len(intensity_results['road_intensity']) > 0:
        st.markdown("### Top 10 Roads by Braking Intensity")
        st.info("Intensity score = (Emergency × 3) + (Hard × 2) + (Moderate × 1)")

        road_df = pd.DataFrame(intensity_results['road_intensity'][:10])
        display_df = road_df[['road', 'emergency', 'hard', 'moderate', 'intensity_score', 'crashes']].copy()
        display_df.columns = ['Road', 'Emergency', 'Hard', 'Moderate', 'Intensity Score', 'Crashes']

        st.dataframe(display_df, use_container_width=True)

        # Stacked bar chart of severity by road
        fig_stacked = go.Figure()
        fig_stacked.add_trace(go.Bar(
            name='Emergency (≤-0.7g)',
            x=road_df['road'].head(10),
            y=road_df['emergency'].head(10),
            marker_color='#8B0000'
        ))
        fig_stacked.add_trace(go.Bar(
            name='Hard (≤-0.5g)',
            x=road_df['road'].head(10),
            y=road_df['hard'].head(10),
            marker_color='#FF4444'
        ))
        fig_stacked.add_trace(go.Bar(
            name='Moderate (≤-0.3g)',
            x=road_df['road'].head(10),
            y=road_df['moderate'].head(10),
            marker_color='#FFA500'
        ))

        fig_stacked.update_layout(
            title='Braking Intensity Breakdown by Road',
            xaxis_title='Road',
            yaxis_title='Number of Events',
            barmode='stack',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_stacked, use_container_width=True)

        # Intensity vs Crashes correlation
        if 'crashes' in road_df.columns:
            st.markdown("### Intensity-to-Crash Correlation")

            fig_corr = px.scatter(
                road_df.head(20),
                x='intensity_score',
                y='crashes',
                size='emergency',
                color='crashes',
                hover_data=['road'],
                title='Braking Intensity vs Crash Occurrence',
                labels={'intensity_score': 'Intensity Score', 'crashes': 'Number of Crashes'},
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            # Calculate correlation coefficient
            if len(road_df) > 1:
                corr = road_df['intensity_score'].corr(road_df['crashes'])
                st.info(f"""
                **Correlation Coefficient:** {corr:.3f}

                {'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.4 else 'Weak'} correlation between braking intensity and crash occurrence.
                Roads with higher emergency braking frequency {'are more likely' if corr > 0.4 else 'may be associated'} to have crashes.
                """)

def plot_proximity_analysis(analysis_results):
    """Visualize proximity analysis results"""
    if analysis_results is None or len(analysis_results.get('near_misses', [])) == 0:
        st.warning("No proximity analysis results to display")
        return

    st.subheader("AI-Powered Crash-Braking Proximity Analysis")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Hard Braking", f"{analysis_results['total_hard_braking']:,}")
    with col2:
        st.metric("Total Crashes", f"{analysis_results['total_crashes']:,}")
    with col3:
        st.metric("Near-Misses Found", f"{analysis_results['near_miss_count']:,}",
                 help="Hard braking events within 200m of crashes")
    with col4:
        st.metric("Correlation Rate", f"{analysis_results['correlation_rate']:.1f}%",
                 help="% of crashes with nearby hard braking")

    # Near-miss details table
    if len(analysis_results['near_misses']) > 0:
        st.markdown("### Near-Miss Events (Predicted Crashes)")
        near_miss_df = pd.DataFrame(analysis_results['near_misses'])
        near_miss_df = near_miss_df.sort_values('nearby_hard_braking', ascending=False).head(15)

        # Format for display
        display_df = near_miss_df[['crash_road', 'nearby_hard_braking', 'avg_deceleration', 'max_deceleration']].copy()
        display_df.columns = ['Road', 'Hard Braking Events', 'Avg Deceleration (g)', 'Max Deceleration (g)']
        display_df['Avg Deceleration (g)'] = display_df['Avg Deceleration (g)'].round(3)
        display_df['Max Deceleration (g)'] = display_df['Max Deceleration (g)'].round(3)

        st.dataframe(display_df, use_container_width=True)

        # Bar chart of top roads
        fig = px.bar(
            display_df.head(10),
            x='Road',
            y='Hard Braking Events',
            title="Top 10 Roads with Hard Braking Near Crash Sites",
            color='Max Deceleration (g)',
            color_continuous_scale='Reds_r',
            labels={'Hard Braking Events': 'Near-Miss Count'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Risk zones (high braking, no crashes yet)
    if len(analysis_results.get('risk_zones', [])) > 0:
        st.markdown("### Predictive Risk Zones (Future Crash Hotspots)")
        st.info("These roads show high hard braking but haven't had crashes yet - prime candidates for intervention!")

        risk_df = pd.DataFrame(analysis_results['risk_zones'])
        display_risk = risk_df[['road', 'hard_braking_count', 'avg_deceleration', 'crash_count', 'risk_score']].head(10)
        display_risk.columns = ['Road', 'Hard Braking Count', 'Avg Deceleration (g)', 'Current Crashes', 'Risk Score']
        display_risk['Avg Deceleration (g)'] = display_risk['Avg Deceleration (g)'].round(3)
        display_risk['Risk Score'] = display_risk['Risk Score'].round(2)

        st.dataframe(display_risk, use_container_width=True)

        # Risk score chart
        fig = px.bar(
            display_risk,
            x='Road',
            y='Risk Score',
            title="Top 10 Future Crash Risk Zones by AI Score",
            color='Hard Braking Count',
            color_continuous_scale='YlOrRd'
        )
        st.plotly_chart(fig, use_container_width=True)

def plot_road_safety_scores(safety_scores):
    """Visualize road safety score analysis"""
    if safety_scores is None or len(safety_scores) == 0:
        st.warning("No safety score data available")
        return

    st.subheader("Road Safety Score Analysis")

    # Get top dangerous and top safe roads
    most_dangerous = safety_scores.head(10)
    safest = safety_scores.tail(10)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Most Dangerous Roads")
        fig = px.bar(
            most_dangerous,
            y='road',
            x='safety_score',
            orientation='h',
            title="Lowest Safety Scores",
            color='risk_level',
            color_discrete_map={
                'Critical': '#8B0000',
                'High': '#FF0000',
                'Moderate': '#FF8C00',
                'Low': '#90EE90'
            }
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Safest Roads")
        fig = px.bar(
            safest,
            y='road',
            x='safety_score',
            orientation='h',
            title="Highest Safety Scores",
            color='risk_level',
            color_discrete_map={
                'Critical': '#8B0000',
                'High': '#FF0000',
                'Moderate': '#FF8C00',
                'Low': '#90EE90'
            }
        )
        fig.update_layout(yaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig, use_container_width=True)

    # Detailed metrics table
    st.markdown("### Detailed Safety Metrics")
    display_cols = ['road', 'safety_score', 'risk_level', 'crash_count', 'hard_braking_count',
                   'hard_braking_rate', 'avg_speed', 'speed_variance']

    available_cols = [col for col in display_cols if col in safety_scores.columns]
    detail_df = safety_scores[available_cols].head(20)

    # Format numeric columns
    if 'safety_score' in detail_df.columns:
        detail_df['safety_score'] = detail_df['safety_score'].round(1)
    if 'hard_braking_rate' in detail_df.columns:
        detail_df['hard_braking_rate'] = (detail_df['hard_braking_rate'] * 100).round(2)
    if 'avg_speed' in detail_df.columns:
        detail_df['avg_speed'] = detail_df['avg_speed'].round(1)
    if 'speed_variance' in detail_df.columns:
        detail_df['speed_variance'] = detail_df['speed_variance'].round(1)

    st.dataframe(detail_df, use_container_width=True)

def plot_temporal_analysis(temporal_results):
    """Visualize temporal risk analysis results"""
    if temporal_results is None:
        st.warning("No temporal analysis results to display")
        return

    st.subheader("Temporal Risk Pattern Analysis")
    st.info("Discover WHEN crashes and hard braking occur - identify high-risk time windows")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Leading Indicators Found", f"{temporal_results['total_time_gaps_found']:,}",
                 help="Hard braking events before crashes at same location")
    with col2:
        median_time = temporal_results['median_time_to_crash']
        if median_time is not None:
            st.metric("Median Warning Time", f"{median_time:.0f} min",
                     help="Median time between hard braking and crash")
        else:
            st.metric("Median Warning Time", "N/A")
    with col3:
        crashes_1hr = temporal_results['crashes_with_warning']
        total_gaps = temporal_results['total_time_gaps_found']
        if total_gaps > 0:
            pct = crashes_1hr / total_gaps * 100
            st.metric("Crashes Within 1 Hour", f"{pct:.1f}%",
                     help="% of crashes with hard braking warning within 1 hour")
        else:
            st.metric("Crashes Within 1 Hour", "N/A")
    with col4:
        high_risk_count = len(temporal_results['high_risk_hours'])
        st.metric("High-Risk Hours", f"{high_risk_count}",
                 help="Hours with elevated crash risk")

    # 1. HOURLY RISK HEATMAP
    st.markdown("### Hourly Risk Pattern (24-Hour)")
    hourly_risk = temporal_results['hourly_risk']

    # Create dual-axis chart
    fig = go.Figure()

    # Add crashes as bars
    fig.add_trace(go.Bar(
        x=hourly_risk['hour'],
        y=hourly_risk['crashes'],
        name='Crashes',
        marker_color='#FF4444',
        yaxis='y'
    ))

    # Add hard braking as line
    fig.add_trace(go.Scatter(
        x=hourly_risk['hour'],
        y=hourly_risk['hard_braking'],
        name='Hard Braking',
        line=dict(color='#FFA500', width=3),
        yaxis='y2'
    ))

    # Add risk score as area
    fig.add_trace(go.Scatter(
        x=hourly_risk['hour'],
        y=hourly_risk['risk_score'],
        name='Risk Score',
        fill='tozeroy',
        fillcolor='rgba(255,100,100,0.2)',
        line=dict(color='#8B0000', width=2, dash='dot'),
        yaxis='y3'
    ))

    fig.update_layout(
        title="Crash Risk by Hour of Day",
        xaxis=dict(title="Hour of Day", tickmode='linear', tick0=0, dtick=2),
        yaxis=dict(title=dict(text="Crashes", font=dict(color='#FF4444'))),
        yaxis2=dict(title=dict(text="Hard Braking Events", font=dict(color='#FFA500')), overlaying='y', side='right'),
        yaxis3=dict(title=dict(text="Risk Score", font=dict(color='#8B0000')), overlaying='y', side='right', position=0.95),
        hovermode='x unified',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Highlight high-risk hours
    if len(temporal_results['high_risk_hours']) > 0:
        # Convert to 12-hour AM/PM format
        def hour_to_12hr(h):
            if h == 0:
                return "12:00 AM"
            elif h < 12:
                return f"{h}:00 AM"
            elif h == 12:
                return "12:00 PM"
            else:
                return f"{h-12}:00 PM"

        high_risk_str = ", ".join([hour_to_12hr(h) for h in temporal_results['high_risk_hours']])
        st.warning(f"**High-Risk Hours Identified:** {high_risk_str}")

    # 2. DAY OF WEEK ANALYSIS
    st.markdown("### Day of Week Risk Pattern")
    daily_risk = temporal_results['daily_risk']

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            daily_risk,
            x='day',
            y='crashes',
            title="Crashes by Day of Week",
            color='crashes',
            color_continuous_scale='Reds',
            labels={'crashes': 'Number of Crashes', 'day': 'Day'}
        )
        fig.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': daily_risk['day'].tolist()})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            daily_risk,
            x='day',
            y='hard_braking',
            title="Hard Braking by Day of Week",
            color='hard_braking',
            color_continuous_scale='Oranges',
            labels={'hard_braking': 'Hard Braking Events', 'day': 'Day'}
        )
        fig.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': daily_risk['day'].tolist()})
        st.plotly_chart(fig, use_container_width=True)

    # 3. TIME-TO-CRASH DISTRIBUTION (Leading Indicator)
    if len(temporal_results['time_gaps']) > 0:
        st.markdown("### Leading Indicator Analysis: Time Between Hard Braking → Crash")
        st.info("This shows the 'warning window' - how much advance notice hard braking gives before a crash")

        time_gaps = temporal_results['time_gaps']

        # Create histogram with bins
        bins = [0, 15, 30, 60, 120, 360, 720, 1440]  # minutes
        labels = ['<15 min', '15-30 min', '30-60 min', '1-2 hrs', '2-6 hrs', '6-12 hrs', '12-24 hrs']

        gap_df = pd.DataFrame({'time_gap': time_gaps})
        gap_df['time_category'] = pd.cut(gap_df['time_gap'], bins=bins, labels=labels, include_lowest=True)
        gap_counts = gap_df['time_category'].value_counts().reindex(labels, fill_value=0)

        fig = px.bar(
            x=labels,
            y=gap_counts.values,
            title="Distribution of Warning Times (Hard Braking → Crash)",
            labels={'x': 'Time Before Crash', 'y': 'Number of Events'},
            color=gap_counts.values,
            color_continuous_scale='YlOrRd'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Calculate statistics
        within_1hr = len([t for t in time_gaps if t <= 60])
        within_30min = len([t for t in time_gaps if t <= 30])
        total = len(time_gaps)

        st.markdown(f"""
        **Key Findings:**
        - **{within_1hr} of {total} crashes ({within_1hr/total*100:.1f}%)** had hard braking within 1 hour
        - **{within_30min} of {total} crashes ({within_30min/total*100:.1f}%)** had hard braking within 30 minutes
        - **Median warning time:** {temporal_results['median_time_to_crash']:.1f} minutes

        **Conclusion:** Hard braking is a strong **leading indicator** that can predict crashes in advance!
        """)

    # 4. PER-ROAD TEMPORAL PATTERNS
    if len(temporal_results['road_temporal_patterns']) > 0:
        st.markdown("### Peak Risk Hours by Road")
        road_temporal = pd.DataFrame(temporal_results['road_temporal_patterns'])
        road_temporal = road_temporal.sort_values('total_crashes', ascending=False).head(10)

        display_road = road_temporal[['road', 'total_crashes', 'peak_crash_hour', 'crashes_during_peak']].copy()
        display_road.columns = ['Road', 'Total Crashes', 'Peak Crash Hour', 'Crashes During Peak']

        # Format peak hour to 12-hour AM/PM format
        def format_hour_ampm(x):
            if pd.isna(x):
                return "N/A"
            h = int(x)
            if h == 0:
                return "12:00 AM"
            elif h < 12:
                return f"{h}:00 AM"
            elif h == 12:
                return "12:00 PM"
            else:
                return f"{h-12}:00 PM"

        display_road['Peak Crash Hour'] = display_road['Peak Crash Hour'].apply(format_hour_ampm)

        st.dataframe(display_road, use_container_width=True)

        # Peak hour visualization
        fig = px.scatter(
            road_temporal,
            x='peak_crash_hour',
            y='road',
            size='crashes_during_peak',
            color='total_crashes',
            title="Most Dangerous Roads and Their Peak Crash Hours",
            labels={'peak_crash_hour': 'Hour of Day', 'road': 'Road', 'crashes_during_peak': 'Crashes in Peak Hour'},
            color_continuous_scale='Reds'
        )
        fig.update_xaxes(tickmode='linear', tick0=0, dtick=2)
        st.plotly_chart(fig, use_container_width=True)

def plot_crash_delay_impact(crash_results, top_n=15):
    """Create crash delay impact visualization showing crashes ranked by total delay

    Args:
        crash_results: List of crash delay analysis dictionaries
        top_n: Number of top crashes to display (default 15)
    """
    if crash_results is None or len(crash_results) == 0:
        st.warning("No crash delay analysis results to plot")
        return

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(crash_results)

    # Sort by total delay and get top N
    df_sorted = df.sort_values('total_delay_hours', ascending=True).tail(top_n)

    # Create labels combining road name and crash ID
    df_sorted['label'] = df_sorted.apply(
        lambda row: f"{row['road'][:20]}... (ID: {row['crash_id']})" if len(row['road']) > 20
                    else f"{row['road']} (ID: {row['crash_id']})",
        axis=1
    )

    # Color code by severity (based on total delay hours)
    colors = []
    for delay in df_sorted['total_delay_hours']:
        if delay >= 10:
            colors.append('#8B0000')  # Dark red - severe
        elif delay >= 5:
            colors.append('#FF0000')  # Red - high
        elif delay >= 2:
            colors.append('#FF8C00')  # Dark orange - moderate
        else:
            colors.append('#FFA500')  # Orange - low

    # Create horizontal bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df_sorted['label'],
        x=df_sorted['total_delay_hours'],
        orientation='h',
        marker=dict(color=colors),
        text=df_sorted['total_delay_hours'].apply(lambda x: f"{x:.1f}h"),
        textposition='outside',
        hovertemplate=(
            '<b>%{y}</b><br>' +
            'Total Delay: %{x:.1f} hours<br>' +
            'Affected Vehicles: %{customdata[0]}<br>' +
            'Speed Reduction: %{customdata[1]:.1f} mph (%{customdata[2]:.1f}%)<br>' +
            'Delay per Vehicle: %{customdata[3]:.1f} min<br>' +
            '<extra></extra>'
        ),
        customdata=df_sorted[['affected_vehicles', 'speed_reduction_mph',
                               'speed_reduction_pct', 'delay_per_vehicle_min']].values
    ))

    fig.update_layout(
        title=f"Top {min(top_n, len(df))} Crashes by Traffic Delay Impact",
        xaxis_title="Total Delay (Vehicle-Hours)",
        yaxis_title="",
        showlegend=False,
        height=max(400, top_n * 35),  # Dynamic height based on number of crashes
        margin=dict(l=20, r=100, t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show summary statistics
    total_crashes_analyzed = len(df)
    total_delay_all = df['total_delay_hours'].sum()
    total_vehicles_affected = df['affected_vehicles'].sum()
    avg_delay_per_crash = df['total_delay_hours'].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Crashes Analyzed", f"{total_crashes_analyzed:,}")
    with col2:
        st.metric("Total Delay", f"{total_delay_all:.1f} hours")
    with col3:
        st.metric("Vehicles Affected", f"{total_vehicles_affected:,}")
    with col4:
        st.metric("Avg Delay/Crash", f"{avg_delay_per_crash:.1f} hours")

# ====== COMMAND EXECUTOR ======
def execute_command(action, params):
    """Execute parsed command and update UI"""

    if action == "show_all":
        show_crashes = params.get("show_crashes", False)

        st.session_state.filtered_vehicles = st.session_state.vehicles
        st.session_state.current_view = "all"

        v_count = len(st.session_state.vehicles) if st.session_state.vehicles is not None else 0

        # Only show crashes if explicitly requested
        if show_crashes:
            st.session_state.filtered_crashes = st.session_state.crashes
            c_count = len(st.session_state.crashes) if st.session_state.crashes is not None else 0
            crash_display = min(c_count, 200)

            display_msg = f"Showing **{v_count:,} vehicles** and **{c_count} crashes** on map"
            st.success(display_msg)
            return f"Displaying BOTH vehicles and crashes on the map!\n- Vehicles: {v_count:,} (colored circles)\n- Crashes: {c_count} (red pin markers)\n\nAll data is now visible!"
        else:
            st.session_state.filtered_crashes = None

            display_msg = f"Showing all **{v_count:,} vehicles** (crashes hidden)"
            st.success(display_msg)
            return f"Displaying {v_count:,} vehicles on the map. Crashes are hidden. Say 'show all data and crashes' to see both!"

    elif action == "filter_road":
        road = params.get("road", "")
        crashes_only = params.get("crashes_only", False)
        show_crashes = params.get("show_crashes", False)
        min_speed = params.get("min_speed")
        max_speed = params.get("max_speed")

        # Filter by road
        v, c = filter_by_road(road)

        # Handle crashes only mode
        if crashes_only:
            st.session_state.filtered_vehicles = None
            st.session_state.filtered_crashes = c
            st.session_state.current_view = f"crashes: {road}"

            c_count = len(c) if c is not None else 0
            st.success(f"Showing {c_count} crashes on {road}")
            return f"Displaying **{c_count} crash locations** on {road}. Look for red pin markers on the map!"

        # Apply time filtering if requested
        start_hour = params.get("start_hour")
        end_hour = params.get("end_hour")
        if start_hour is not None and end_hour is not None:
            v, c = filter_by_time(v, c, start_hour, end_hour)

        # Apply speed filtering if requested
        if min_speed is not None or max_speed is not None:
            if v is not None:
                if min_speed is not None:
                    v = v[v['speed'] >= min_speed]
                if max_speed is not None:
                    v = v[v['speed'] <= max_speed]

        # Apply speeding filter if requested
        if params.get("show_speeding"):
            if v is not None:
                v = v[v['speed'] >= 65]

        st.session_state.filtered_vehicles = v

        # CRITICAL: Only show crashes if explicitly requested
        if show_crashes:
            st.session_state.filtered_crashes = c
        else:
            st.session_state.filtered_crashes = None

        st.session_state.current_view = f"road: {road}"

        v_count = len(v) if v is not None else 0
        c_count = len(c) if c is not None and show_crashes else 0

        # Smart analysis with speed insights
        response = f"**{road} Analysis:**\n\n"
        response += f"Found **{v_count:,} vehicles** "

        # Add speed filter description
        if min_speed or max_speed:
            if min_speed and max_speed:
                response += f"(speed {min_speed}-{max_speed} mph) "
            elif min_speed:
                response += f"(speed ≥{min_speed} mph) "
            elif max_speed:
                response += f"(speed ≤{max_speed} mph) "

        response += f"on this road.\n\n"

        # Only mention crashes if explicitly shown
        if show_crashes:
            if c_count > 0:
                response += f"**Crashes:** {c_count} crash locations shown as red pin markers on the map.\n\n"
            else:
                response += f"**No crashes recorded** on {road}.\n\n"

        if v is not None and len(v) > 0:
            # Get speed stats
            speed_stats = get_speed_stats(v)
            if speed_stats:
                response += f"**Speed Insights:**\n"
                response += f"- Average speed: **{speed_stats['mean']:.1f} mph**\n"
                response += f"- Max speed: **{speed_stats['max']:.1f} mph**\n"
                response += f"- Speeding vehicles (>70 mph): **{speed_stats['speeding_count']:,}** ({(speed_stats['speeding_count']/speed_stats['total_count']*100):.1f}%)\n\n"

        response += f"Map zoomed to show **{road}**. Use 'Reset View' to see full data."

        st.success(f"Analyzed {road}")
        return response

    elif action == "show_speeding":
        min_speed = params.get("min_speed", 65)
        v = filter_by_speed(min_speed=min_speed)
        st.session_state.filtered_vehicles = v
        st.session_state.filtered_crashes = None
        st.session_state.current_view = f"speeding > {min_speed} mph"

        v_count = len(v) if v is not None else 0
        st.success(f"Found **{v_count:,} speeding vehicles** (>{min_speed} mph)")
        return f"I found {v_count:,} vehicles going over {min_speed} mph. These are shown on the map in red/orange colors!"

    elif action == "filter_speed":
        min_speed = params.get("min_speed")
        max_speed = params.get("max_speed")
        v = filter_by_speed(min_speed, max_speed)
        st.session_state.filtered_vehicles = v
        st.session_state.filtered_crashes = None  # Clear crashes when filtering by speed
        st.session_state.current_view = f"speed filter"

        v_count = len(v) if v is not None else 0

        # Build response based on parameters
        if min_speed and max_speed:
            filter_desc = f"between {min_speed} and {max_speed} mph"
        elif min_speed:
            filter_desc = f"above {min_speed} mph"
        elif max_speed:
            filter_desc = f"below {max_speed} mph"
        else:
            filter_desc = "with all speeds"

        st.success(f"Found **{v_count:,} vehicles** {filter_desc}")
        return f"Showing {v_count:,} vehicles {filter_desc} on the map. Crashes hidden to focus on vehicle data."

    elif action == "filter_time":
        start_hour = params.get("start_hour", 0)
        end_hour = params.get("end_hour", 24)
        show_crashes = params.get("show_crashes", False)

        # Apply time filtering
        v, c = filter_by_time(st.session_state.vehicles, st.session_state.crashes, start_hour, end_hour)

        st.session_state.filtered_vehicles = v
        st.session_state.current_view = f"time: {start_hour}:00-{end_hour}:00"

        # Only show crashes if explicitly requested
        if show_crashes:
            st.session_state.filtered_crashes = c
        else:
            st.session_state.filtered_crashes = None

        v_count = len(v) if v is not None else 0
        c_count = len(c) if c is not None and show_crashes else 0

        # Convert to 12-hour format for display
        def format_hour(h):
            if h == 0:
                return "12am"
            elif h < 12:
                return f"{h}am"
            elif h == 12:
                return "12pm"
            else:
                return f"{h-12}pm"

        time_desc = f"from {format_hour(start_hour)} to {format_hour(end_hour)}"

        if show_crashes:
            st.success(f"Showing **{v_count:,} vehicles** and **{c_count} crashes** {time_desc}")
            return f"Filtered to time range {time_desc}:\n- Vehicles: {v_count:,}\n- Crashes: {c_count}\n\nAll data now shows only this time period!"
        else:
            st.success(f"Showing **{v_count:,} vehicles** {time_desc}")
            return f"Filtered to {v_count:,} vehicles {time_desc}. Try combining with road filters for more specific analysis!"

    elif action == "show_crashes":
        st.session_state.filtered_vehicles = None
        st.session_state.filtered_crashes = st.session_state.crashes
        st.session_state.current_view = "crashes only"

        c_count = len(st.session_state.crashes) if st.session_state.crashes is not None else 0
        st.success(f"Showing **{c_count} crash locations** on the map")
        return f"Displaying **{c_count} crash locations** on the map as red pin markers. Click any marker to see crash details!"

    elif action == "plot_speed":
        st.session_state.show_plots.add('speed')
        st.success("Speed distribution plot shown below")
        return "I've created a speed distribution histogram below the map. It shows how many vehicles are traveling at different speeds!"

    elif action == "plot_crashes":
        st.session_state.show_plots.add('crashes')
        st.success("Crash analysis plot shown below")
        return "I've created a bar chart below showing which roads have the most crashes. This helps identify hotspots!"

    elif action == "analyze_crashes":
        # Analyze crash data and provide specific insights with road names
        if st.session_state.crashes is None or len(st.session_state.crashes) == 0:
            st.warning("No crash data available to analyze")
            return "I don't have any crash data to analyze. Please upload crash data first!"

        # Use cached analysis for speed
        top_roads = get_top_crash_roads(st.session_state.crashes, top_n=10)

        if top_roads is None:
            st.warning("No RoadName column in crash data")
            return "The crash data doesn't have road name information. Upload it again and I'll look it up automatically!"

        total_crashes = st.session_state.crashes.to_pandas()['RoadName'].notna().sum()

        # Create comprehensive analysis
        response = f"## Crash Hotspot Analysis\n\n"
        response += f"Analyzed **{total_crashes} crashes** across the dataset. Critical findings:\n\n"

        response += f"### Top 5 Most Dangerous Roads:\n\n"
        for i, (road, count) in enumerate(top_roads.head(5).items(), 1):
            percentage = (count / total_crashes) * 100
            severity = "CRITICAL" if percentage > 10 else "HIGH" if percentage > 5 else "MODERATE"
            response += f"{i}. **{road}**: {count} crashes ({percentage:.1f}%) [{severity}]\n"

        response += f"\n### Key Insights:\n"
        response += f"- **Highest crash location:** {top_roads.index[0]} with **{top_roads.iloc[0]} crashes**\n"
        response += f"- Top 5 roads account for **{sum(top_roads.head(5).values)}** crashes ({(sum(top_roads.head(5).values)/total_crashes*100):.1f}% of total)\n"

        if len(top_roads) > 5:
            response += f"- **{len(top_roads)}** different roads have recorded crashes\n"

        response += f"\n**Recommendation:** Focus safety improvements on {top_roads.index[0]}, {top_roads.index[1]}, and {top_roads.index[2]}.\n"

        # Show the plot
        st.session_state.show_plots.add('crashes')
        st.success(f"Crash analysis complete - {top_roads.index[0]} is the most dangerous road")

        return response

    elif action == "color_by_speed":
        # Parse custom colors from user message if provided
        if params.get('user_query'):
            custom_colors = parse_color_request(params['user_query'])
            if custom_colors:
                # Update color scheme with custom colors
                for speed_range, color_hex in custom_colors.items():
                    st.session_state.color_scheme[speed_range] = color_hex

        # Toggle color mode
        if st.session_state.color_mode == "neutral":
            st.session_state.color_mode = "speed"
            st.success("Color mode activated")

            # Build response with current color scheme
            scheme = st.session_state.color_scheme
            color_names = {
                '#FF0000': 'Red', '#FFFFFF': 'White', '#FFC0CB': 'Pink',
                '#FF4444': 'Red', '#FFA500': 'Orange', '#4CAF50': 'Green',
                '#00FF00': 'Green', '#0000FF': 'Blue', '#FFFF00': 'Yellow'
            }

            high_name = color_names.get(scheme['high'], scheme['high'])
            med_name = color_names.get(scheme['medium'], scheme['medium'])
            low_name = color_names.get(scheme['low'], scheme['low'])

            return f"Map colored by speed: {high_name} for high speeds (>70 mph), {med_name} for medium speeds (50-70 mph), {low_name} for low speeds (<50 mph). Check the map to see patterns."
        else:
            st.session_state.color_mode = "neutral"
            st.success("Switched to neutral colors")
            return "Switched back to neutral gray colors. Ask me to color by speed again when needed."

    elif action == "timespace":
        road = params.get("road")

        # Add timespace to show_plots
        st.session_state.show_plots.add('timespace')

        # Store the road parameter in session state for rendering
        if road:
            st.session_state.timespace_road = road
            st.success(f"Generating timespace diagram for {road}...")
            return f"Loading vehicle trajectory data for **{road}**. The timespace diagram will show how vehicles move along this corridor over time."
        else:
            # If we have filtered vehicles by road, use that road
            if st.session_state.current_view.startswith("road: "):
                road = st.session_state.current_view.split("road: ")[1]
                st.session_state.timespace_road = road
            else:
                st.session_state.timespace_road = None

            st.success("Generating timespace diagram for current view...")
            if road:
                return f"Creating timespace diagram showing vehicle trajectories over time. This visualizes traffic flow patterns."
            else:
                return "Creating timespace diagram for all roads. This may take a moment. Tip: Filter by a specific road first for clearer trajectories!"

    elif action == "show_hard_braking":
        # Get parameters
        threshold = params.get("threshold", -0.3)  # Default -0.3g (moderate to hard braking)
        road = params.get("road")
        start_hour = params.get("start_hour")
        end_hour = params.get("end_hour")

        # Start with all vehicles or filtered by road
        if road:
            v, _ = filter_by_road(road)
            if v is None or len(v) == 0:
                st.warning(f"No vehicle data found for {road}")
                return f"No vehicle data found for **{road}**. Try a different road name."
        else:
            v = st.session_state.vehicles

        if v is None or len(v) == 0:
            st.warning("No vehicle data available")
            return "No vehicle data loaded. Please upload or wait for data to load."

        # Show data count before filtering
        initial_count = len(v)

        # Filter by time if requested
        if start_hour is not None and end_hour is not None:
            st.info(f"Filtering {initial_count:,} vehicle records for time range {start_hour}:00-{end_hour}:00...")
            v, _ = filter_by_time(v, None, start_hour, end_hour)
            if v is None or len(v) == 0:
                st.warning(f"No vehicle data found between {start_hour}:00 and {end_hour}:00")
                # Show available time range
                if initial_count > 0:
                    sample_v = st.session_state.vehicles.to_pandas() if hasattr(st.session_state.vehicles, 'to_pandas') else st.session_state.vehicles
                    if 'timestamp' in sample_v.columns:
                        sample_v['timestamp'] = pd.to_datetime(sample_v['timestamp'])
                        min_time = sample_v['timestamp'].min()
                        max_time = sample_v['timestamp'].max()
                        st.info(f"Available data spans: {min_time} to {max_time}")
                return f"No vehicle data found between **{start_hour}:00 and {end_hour}:00**. Check the data time range above."
            else:
                st.success(f"Found {len(v):,} vehicle records in time range")

        # Filter for hard braking events
        with st.spinner("Calculating deceleration from speed changes..."):
            records_before_calc = len(v)
            hard_braking_events = filter_hard_braking(v, threshold_g=threshold)

        if hard_braking_events is None or len(hard_braking_events) == 0:
            st.warning(f"No hard braking events found (threshold: {threshold}g)")
            st.info(f"Started with {records_before_calc:,} vehicle records, but found no hard braking events. This could mean:\n- No consecutive speed measurements (needed to calculate deceleration)\n- All deceleration values were below threshold\n- Data quality issues filtered out all events")
            return f"No hard braking events detected{f' on {road}' if road else ''}{f' between {start_hour}:00-{end_hour}:00' if start_hour else ''} with threshold {threshold}g. Try -0.2g for more events."

        # Update session state
        st.session_state.filtered_vehicles = hard_braking_events
        st.session_state.filtered_crashes = None
        st.session_state.braking_mode = "on"
        st.session_state.braking_threshold = threshold

        # Determine severity level
        if threshold <= -0.7:
            severity = "emergency"
            severity_text = "Emergency"
        elif threshold <= -0.5:
            severity = "hard"
            severity_text = "Hard"
        else:
            severity = "moderate"
            severity_text = "Moderate to Hard"

        # Build descriptive view label
        view_parts = [f"hard braking ({threshold}g)"]
        if road:
            view_parts.append(f"on {road}")
        if start_hour is not None and end_hour is not None:
            view_parts.append(f"{start_hour}:00-{end_hour}:00")
        st.session_state.current_view = " ".join(view_parts)

        event_count = len(hard_braking_events)

        # Calculate statistics
        decel_stats = hard_braking_events.to_pandas()['deceleration_g'] if hasattr(hard_braking_events, 'to_pandas') else hard_braking_events['deceleration_g']
        avg_decel = decel_stats.mean()
        max_decel = decel_stats.min()  # Min because more negative = harder braking

        # Build response message
        response = f"## {severity_text} Braking Analysis\n\n"
        location_filter = f" on {road}" if road else ""
        time_filter = f" between **{start_hour}:00 and {end_hour}:00**" if (start_hour is not None and end_hour is not None) else ""
        response += f"Found **{event_count:,} hard braking events**{location_filter}{time_filter} (deceleration ≤ {threshold}g)\n\n"

        response += f"**Braking Statistics:**\n"
        response += f"- Average deceleration: **{avg_decel:.3f}g**\n"
        response += f"- Maximum braking force: **{max_decel:.3f}g** (highest deceleration recorded)\n"
        response += f"- Severity threshold: {severity_text} braking (≤ {threshold}g)\n\n"

        response += f"**What this means:**\n"
        if threshold <= -0.7:
            response += f"These are **emergency braking events** - very hard stops that could indicate dangerous situations, obstacles, or near-misses.\n"
        elif threshold <= -0.5:
            response += f"These are **hard braking events** - significant decelerations that may indicate traffic congestion, sharp curves, or sudden stops.\n"
        else:
            response += f"These are **moderate to hard braking events** - noticeable decelerations that could indicate areas requiring attention.\n"

        # Build visualization message
        viz_parts = ["Hard braking locations are now shown on the map"]
        if road:
            viz_parts.append(f"for {road}")
        if start_hour is not None and end_hour is not None:
            viz_parts.append(f"from {start_hour}:00 to {end_hour}:00")
        response += f"\n**Visualization:** {' '.join(viz_parts)}. Each point represents a hard braking event."

        st.success(f"Found {event_count:,} hard braking events{location_filter}{time_filter}!")
        return response

    elif action == "plot_hard_braking":
        # Check if we have hard braking data with deceleration calculated
        if st.session_state.filtered_vehicles is None or len(st.session_state.filtered_vehicles) == 0:
            st.warning("No hard braking data available. Run 'Show hard braking events' first!")
            return "I need hard braking data to plot. Try asking me to 'Show hard braking events' first, then ask for the plot again."

        # Check if deceleration column exists
        df = st.session_state.filtered_vehicles.to_pandas() if hasattr(st.session_state.filtered_vehicles, 'to_pandas') else st.session_state.filtered_vehicles
        if 'deceleration_g' not in df.columns:
            st.warning("No deceleration data available. Run 'Show hard braking events' first!")
            return "I need to calculate deceleration data first. Ask me to 'Show hard braking events' and then I can plot the distribution."

        st.session_state.show_plots.add('hard_braking')
        st.success("Hard braking distribution plot shown below")
        return "I've created a stacked histogram showing the distribution of hard braking events by severity (Emergency, Hard, and Moderate). Check below the map to see which severity levels are most common!"

    elif action == "analyze_crash_delays":
        # Check if we have both vehicle and crash data
        if st.session_state.vehicles is None or len(st.session_state.vehicles) == 0:
            st.warning("No vehicle data available for analysis")
            return "I need vehicle data to analyze traffic delays. Please upload vehicle data first."

        if st.session_state.crashes is None or len(st.session_state.crashes) == 0:
            st.warning("No crash data available for analysis")
            return "I need crash data to analyze delays. Please upload crash data first."

        # Perform crash delay analysis
        with st.spinner("Analyzing traffic delays caused by crashes... This may take a moment."):
            crash_results = analyze_crash_delays(
                st.session_state.vehicles,
                st.session_state.crashes,
                radius_meters=500,
                time_window_minutes=60
            )

        if crash_results is None or len(crash_results) == 0:
            st.warning("No crashes with measurable delay impact found")
            return "I couldn't find any crashes with measurable traffic delay. The crashes may be too far from vehicle data points, or there may not be enough vehicle data near crash locations."

        # Store results in session state for plotting
        st.session_state.crash_delay_results = crash_results

        # Show the plot
        st.session_state.show_plots.add('crash_delays')

        # Generate response with key findings
        total_crashes = len(crash_results)
        total_delay_hours = sum([r['total_delay_hours'] for r in crash_results])
        total_vehicles = sum([r['affected_vehicles'] for r in crash_results])

        # Find top 3 crashes by delay
        top_crashes = sorted(crash_results, key=lambda x: x['total_delay_hours'], reverse=True)[:3]

        response = f"## Crash Delay Impact Analysis\n\n"
        response += f"Analyzed **{total_crashes} crashes** with measurable traffic impact:\n\n"
        response += f"**Overall Impact:**\n"
        response += f"- Total traffic delay: **{total_delay_hours:.1f} vehicle-hours**\n"
        response += f"- Vehicles affected: **{total_vehicles:,}**\n"
        response += f"- Average delay per crash: **{total_delay_hours/total_crashes:.1f} hours**\n\n"

        response += f"**Top 3 Crashes by Delay Impact:**\n"
        for i, crash in enumerate(top_crashes, 1):
            response += f"{i}. **{crash['road']}** (Crash ID: {crash['crash_id']})\n"
            response += f"   - Total delay: {crash['total_delay_hours']:.1f} hours\n"
            response += f"   - Affected vehicles: {crash['affected_vehicles']}\n"
            response += f"   - Speed reduction: {crash['speed_reduction_mph']:.1f} mph ({crash['speed_reduction_pct']:.1f}%)\n\n"

        response += f"Check the chart below for a complete ranking of all crashes by delay impact!"

        st.success(f"Analyzed {total_crashes} crashes with traffic delay impact")
        return response

    elif action == "analyze_proximity":
        # AI-powered proximity analysis: Find hard braking near crashes
        if st.session_state.vehicles is None or len(st.session_state.vehicles) == 0:
            st.warning("No vehicle data available for proximity analysis")
            return "I need vehicle data to analyze crash-braking correlations. Please ensure data is loaded."

        if st.session_state.crashes is None or len(st.session_state.crashes) == 0:
            st.warning("No crash data available for proximity analysis")
            return "I need crash data to find near-miss events. Please upload crash data."

        # Perform AI proximity analysis
        with st.spinner("AI is analyzing crash-braking patterns... This AI analysis identifies near-miss events and predicts future crash locations!"):
            analysis_results = analyze_proximity_risk(
                st.session_state.vehicles,
                st.session_state.crashes,
                distance_meters=200
            )

        if analysis_results is None:
            st.error("Could not perform proximity analysis")
            return "Analysis failed. Please check your data."

        # Store results and show visualization
        st.session_state.proximity_results = analysis_results
        st.session_state.show_plots.add('proximity_analysis')

        # Generate AI insights response
        near_miss_count = analysis_results['near_miss_count']
        total_crashes = analysis_results['total_crashes']
        correlation_rate = analysis_results['correlation_rate']
        risk_zone_count = len(analysis_results.get('risk_zones', []))

        response = f"## AI-Powered Crash Prediction Analysis\n\n"
        response += f"**AI has discovered critical safety insights:**\n\n"

        response += f"**Near-Miss Detection:**\n"
        response += f"- Found **{near_miss_count} near-miss events** (hard braking within 200m of crashes)\n"
        response += f"- **{correlation_rate:.1f}% of crashes** had warning signs (hard braking nearby)\n"
        response += f"- These near-misses could have become crashes!\n\n"

        if risk_zone_count > 0:
            response += f"**Predictive Risk Zones:**\n"
            response += f"- AI identified **{risk_zone_count} high-risk roads** that haven't had crashes YET\n"
            response += f"- These roads show dangerous braking patterns similar to crash locations\n"
            response += f"- **Intervention recommended** to prevent future crashes!\n\n"

        response += f"**Key Finding:** Hard braking events are strong predictors of crash locations. "
        response += f"The AI can now forecast where crashes are likely to occur next based on braking patterns!\n\n"
        response += f"See detailed analysis, risk scores, and AI predictions below the map."

        st.success(f"AI Analysis Complete: Found {near_miss_count} near-misses and {risk_zone_count} risk zones")
        return response

    elif action == "road_safety_score":
        # Calculate comprehensive road safety scores
        if st.session_state.vehicles is None or len(st.session_state.vehicles) == 0:
            st.warning("No vehicle data available for safety scoring")
            return "I need vehicle data to calculate safety scores. Please ensure data is loaded."

        # Perform safety score calculation
        with st.spinner("Calculating comprehensive road safety scores..."):
            safety_scores = calculate_road_safety_scores(
                st.session_state.vehicles,
                st.session_state.crashes
            )

        if safety_scores is None or len(safety_scores) == 0:
            st.warning("Could not calculate safety scores")
            return "Unable to calculate road safety scores. Data may be missing required columns."

        # Store results and show visualization
        st.session_state.safety_scores = safety_scores
        st.session_state.show_plots.add('safety_scores')

        # Get most dangerous roads
        most_dangerous = safety_scores.head(5)
        safest = safety_scores.tail(5)

        response = f"## Road Safety Score Analysis\n\n"
        response += f"Analyzed **{len(safety_scores)} roads** with comprehensive safety metrics.\n\n"

        response += f"**Most Dangerous Roads:**\n"
        for i, row in enumerate(most_dangerous.itertuples(), 1):
            score = row.safety_score if hasattr(row, 'safety_score') else 0
            risk = row.risk_level if hasattr(row, 'risk_level') else 'Unknown'
            hard_braking = row.hard_braking_count if hasattr(row, 'hard_braking_count') else 0
            crashes = row.crash_count if hasattr(row, 'crash_count') else 0
            response += f"{i}. **{row.road}** (Score: {score:.1f}/100, Risk: {risk})\n"
            response += f"   - Hard braking: {hard_braking}, Crashes: {crashes}\n"

        response += f"\n**Safest Roads:**\n"
        for i, row in enumerate(safest.itertuples(), 1):
            score = row.safety_score if hasattr(row, 'safety_score') else 0
            risk = row.risk_level if hasattr(row, 'risk_level') else 'Unknown'
            response += f"{i}. **{row.road}** (Score: {score:.1f}/100, Risk: {risk})\n"

        response += f"\nSee complete rankings and detailed metrics below the map."

        st.success(f"Analyzed {len(safety_scores)} roads - safety scores calculated")
        return response

    elif action == "temporal_analysis":
        # Analyze temporal risk patterns - when crashes and hard braking occur
        if st.session_state.vehicles is None or st.session_state.crashes is None:
            st.warning("Need both vehicle and crash data for temporal analysis")
            return "I need both vehicle and crash data to analyze temporal patterns. Please upload both datasets."

        with st.spinner("Analyzing temporal risk patterns... Discovering when crashes are most likely to occur!"):
            temporal_results = analyze_temporal_risk(
                st.session_state.vehicles,
                st.session_state.crashes,
                distance_meters=200
            )

        if temporal_results is None:
            st.warning("Could not perform temporal analysis - missing timestamp data")
            return "Unable to analyze temporal patterns. Data may be missing timestamp information."

        # Store results and show visualization
        st.session_state.temporal_results = temporal_results
        st.session_state.show_plots.add('temporal_analysis')

        # Filter map to show only high-risk hours
        high_risk_hours = temporal_results['high_risk_hours']
        if len(high_risk_hours) > 0 and st.session_state.vehicles is not None:
            # Filter vehicles to high-risk hours
            v_df = st.session_state.vehicles.to_pandas() if hasattr(st.session_state.vehicles, 'to_pandas') else st.session_state.vehicles

            # Ensure timestamp column exists
            if 'timestamp' in v_df.columns:
                v_df['timestamp'] = pd.to_datetime(v_df['timestamp'])
                # Remove timezone if present
                if hasattr(v_df['timestamp'].dt, 'tz') and v_df['timestamp'].dt.tz is not None:
                    v_df['timestamp'] = v_df['timestamp'].dt.tz_localize(None)
                v_df['hour'] = v_df['timestamp'].dt.hour

                # Filter to high-risk hours
                filtered_v = v_df[v_df['hour'].isin(high_risk_hours)]

                # Convert back to cudf if needed
                if hasattr(st.session_state.vehicles, 'to_pandas'):
                    st.session_state.filtered_vehicles = cudf.from_pandas(filtered_v)
                else:
                    st.session_state.filtered_vehicles = filtered_v

            # Filter crashes to high-risk hours if available
            if st.session_state.crashes is not None:
                c_df = st.session_state.crashes.to_pandas() if hasattr(st.session_state.crashes, 'to_pandas') else st.session_state.crashes

                # Check for timestamp column (could be pub_millis or other)
                time_col = None
                if 'timestamp' in c_df.columns:
                    time_col = 'timestamp'
                elif 'pub_millis' in c_df.columns:
                    c_df['timestamp'] = c_df['pub_millis']
                    time_col = 'timestamp'

                if time_col:
                    c_df['timestamp'] = pd.to_datetime(c_df['timestamp'])
                    # Remove timezone if present
                    if hasattr(c_df['timestamp'].dt, 'tz') and c_df['timestamp'].dt.tz is not None:
                        c_df['timestamp'] = c_df['timestamp'].dt.tz_localize(None)
                    c_df['hour'] = c_df['timestamp'].dt.hour

                    # Filter to high-risk hours
                    filtered_c = c_df[c_df['hour'].isin(high_risk_hours)]

                    # Convert back to cudf if needed
                    if hasattr(st.session_state.crashes, 'to_pandas'):
                        st.session_state.filtered_crashes = cudf.from_pandas(filtered_c)
                    else:
                        st.session_state.filtered_crashes = filtered_c

            st.session_state.current_view = f"high_risk_hours_{len(high_risk_hours)}"

        # Generate response
        median_warning = temporal_results['median_time_to_crash']
        crashes_1hr = temporal_results['crashes_with_warning']
        total_gaps = temporal_results['total_time_gaps_found']

        response = f"## Temporal Risk Pattern Analysis\n\n"
        response += f"**When are crashes most likely to occur?**\n\n"

        # High-risk hours (convert to 12-hour AM/PM format)
        if len(high_risk_hours) > 0:
            def hour_to_ampm(h):
                if h == 0:
                    return "12:00 AM"
                elif h < 12:
                    return f"{h}:00 AM"
                elif h == 12:
                    return "12:00 PM"
                else:
                    return f"{h-12}:00 PM"

            high_risk_str = ", ".join([hour_to_ampm(h) for h in high_risk_hours])
            response += f"**High-Risk Time Windows:** {high_risk_str}\n\n"

        # Leading indicator findings
        if total_gaps > 0:
            pct_1hr = crashes_1hr / total_gaps * 100
            response += f"**Leading Indicator Discovery:**\n"
            response += f"- Found **{total_gaps} instances** where hard braking preceded crashes at same location\n"
            response += f"- **{pct_1hr:.1f}% of crashes** had hard braking warning within 1 hour\n"
            if median_warning is not None:
                response += f"- **Median warning time:** {median_warning:.1f} minutes before crash\n\n"

            response += f"**Key Finding:** Hard braking is a **predictive leading indicator** that can forecast crashes in advance!\n\n"

        # Road-specific patterns
        if len(temporal_results['road_temporal_patterns']) > 0:
            response += f"**Per-Road Timing Patterns:**\n"
            top_roads = sorted(temporal_results['road_temporal_patterns'],
                             key=lambda x: x['total_crashes'], reverse=True)[:3]
            for i, road_data in enumerate(top_roads, 1):
                peak_hour = road_data['peak_crash_hour']
                if peak_hour is not None:
                    # Convert to 12-hour AM/PM format
                    h = int(peak_hour)
                    if h == 0:
                        time_str = "12:00 AM"
                    elif h < 12:
                        time_str = f"{h}:00 AM"
                    elif h == 12:
                        time_str = "12:00 PM"
                    else:
                        time_str = f"{h-12}:00 PM"
                    response += f"{i}. **{road_data['road']}** - Most dangerous at {time_str} ({road_data['crashes_during_peak']} crashes)\n"

        response += f"\n**Map Updated:** The map now shows only data from the {len(high_risk_hours)} high-risk time windows, so you can see WHERE dangerous WHEN happens!\n\n"
        response += f"See detailed hourly patterns, day-of-week analysis, and leading indicator distribution below!"

        # Get filtered counts
        v_count = len(st.session_state.filtered_vehicles) if st.session_state.filtered_vehicles is not None else 0
        c_count = len(st.session_state.filtered_crashes) if st.session_state.filtered_crashes is not None else 0

        st.success(f"Temporal Analysis Complete: Map filtered to {len(high_risk_hours)} high-risk hours ({v_count:,} vehicles, {c_count} crashes)")
        return response

    elif action == "crash_hotspots":
        # Identify crash hotspot clusters using spatial analysis
        if st.session_state.crashes is None:
            st.warning("Need crash data for hotspot clustering")
            return "I need crash data to identify hotspots. Please upload crash data first."

        # Get optional parameters
        cluster_radius = params.get('radius_meters', 50)
        min_crashes = params.get('min_crashes', 2)

        with st.spinner("Analyzing crash hotspot clusters... Identifying the most dangerous spots!"):
            hotspot_results = analyze_crash_hotspots(
                st.session_state.crashes,
                cluster_radius_meters=cluster_radius,
                min_crashes=min_crashes
            )

        if hotspot_results is None or hotspot_results['total_clusters'] == 0:
            st.warning("No crash hotspots found. Try adjusting parameters.")
            return "No crash hotspots were detected in the data. This could mean crashes are well-distributed spatially, or you may need to adjust the clustering parameters."

        # Store results and show visualization
        st.session_state.hotspot_results = hotspot_results
        st.session_state.show_plots.add('crash_hotspots')

        # Trigger map refresh to show cluster circles
        st.session_state.map_version += 1

        # Generate response
        clusters = hotspot_results['clusters']
        top_cluster = clusters[0] if len(clusters) > 0 else None

        response = f"## Crash Hotspot Clustering Analysis\n\n"
        response += f"**Spatial Pattern Discovery:**\n"
        response += f"- Identified **{hotspot_results['total_clusters']} distinct crash hotspots**\n"
        response += f"- **{hotspot_results['clustered_crashes']} crashes** occur in these concentrated areas\n"

        if hotspot_results.get('total_crashes', 0) > 0:
            cluster_rate = (hotspot_results['clustered_crashes'] / hotspot_results['total_crashes'] * 100)
            response += f"- **{cluster_rate:.1f}%** of all crashes happen in hotspots (spatial concentration!)\n\n"

        # Top dangerous spots
        if top_cluster:
            response += f"**Most Dangerous Hotspot:**\n"
            response += f"- Location: {top_cluster['primary_road']}\n"
            response += f"- GPS: ({top_cluster['center_lat']:.6f}, {top_cluster['center_lon']:.6f})\n"
            response += f"- **{top_cluster['crash_count']} crashes** within {top_cluster['radius_meters']:.0f}m radius\n\n"

        # Top 3 concentration
        if len(clusters) >= 3:
            top_3_crashes = sum(c['crash_count'] for c in clusters[:3])
            top_3_pct = (top_3_crashes / hotspot_results['clustered_crashes'] * 100) if hotspot_results['clustered_crashes'] > 0 else 0
            response += f"**Extreme Concentration Alert:**\n"
            response += f"The top 3 hotspots alone account for **{top_3_crashes} crashes ({top_3_pct:.1f}%)**!\n\n"
            response += f"**Recommendation:** Targeted interventions at these {hotspot_results['total_clusters']} locations could significantly reduce crash rates.\n\n"

        response += f"**Map Updated:** The map now shows red circles marking each crash hotspot cluster. Circle size represents the cluster radius.\n\n"
        response += f"See complete hotspot details, cluster visualization, and spatial patterns below the map!"

        st.success(f"Found {hotspot_results['total_clusters']} crash hotspots covering {hotspot_results['clustered_crashes']} crashes")
        return response

    elif action == "braking_intensity":
        # Analyze braking intensity patterns and correlation with crashes
        if st.session_state.vehicles is None:
            st.warning("Need vehicle data for braking intensity analysis")
            return "I need vehicle data to analyze braking intensity. Please upload vehicle data first."

        with st.spinner("Analyzing braking intensity patterns... Classifying emergency vs hard vs moderate braking!"):
            intensity_results = analyze_braking_intensity(
                st.session_state.vehicles,
                st.session_state.crashes if st.session_state.crashes is not None else None
            )

        if intensity_results is None or intensity_results.get('total_hard_braking', 0) == 0:
            st.warning("No hard braking events found")
            return "No hard braking events were detected in the vehicle data."

        # Store results and show visualization
        st.session_state.braking_intensity_results = intensity_results
        st.session_state.show_plots.add('braking_intensity')

        # Enable braking mode on map to visualize intensity
        st.session_state.braking_mode = "on"
        st.session_state.map_version += 1

        # Generate response
        response = f"## Braking Intensity & Crash Severity Analysis\n\n"
        response += f"**Braking Severity Classification:**\n"
        response += f"- **Total hard braking events:** {intensity_results['total_hard_braking']:,}\n"
        response += f"- **Emergency braking** (≤-0.7g): {intensity_results['emergency_count']:,} events\n"
        response += f"- **Hard braking** (≤-0.5g): {intensity_results['hard_count']:,} events\n"
        response += f"- **Moderate braking** (≤-0.3g): {intensity_results['moderate_count']:,} events\n\n"

        emergency_pct = (intensity_results['emergency_count'] / intensity_results['total_hard_braking'] * 100) if intensity_results['total_hard_braking'] > 0 else 0
        response += f"**{emergency_pct:.1f}%** of all hard braking is classified as EMERGENCY (extremely dangerous!)\n\n"

        # Most extreme deceleration
        response += f"**Most extreme braking:** {intensity_results['max_deceleration']:.2f}g\n"
        response += f"**Average deceleration:** {intensity_results['avg_deceleration']:.2f}g\n\n"

        # Top intensity roads
        if len(intensity_results['road_intensity']) > 0:
            top_road = intensity_results['road_intensity'][0]
            response += f"**Highest Intensity Road:** {top_road['road']}\n"
            response += f"- Emergency: {top_road['emergency']}, Hard: {top_road['hard']}, Moderate: {top_road['moderate']}\n"
            response += f"- **Intensity Score:** {top_road['intensity_score']}\n"

            if 'crashes' in top_road:
                response += f"- **Crashes:** {top_road['crashes']}\n"
                if top_road['crashes'] > 0:
                    response += f"\n**Warning:** This road shows both extreme braking AND crashes - urgent intervention needed!\n\n"
                else:
                    response += f"\n**Alert:** High braking intensity but no crashes yet - potential future crash zone!\n\n"

        # Correlation insight (if crash data available)
        if len(intensity_results['road_intensity']) > 1 and 'crashes' in intensity_results['road_intensity'][0]:
            road_df = pd.DataFrame(intensity_results['road_intensity'][:20])
            if 'intensity_score' in road_df.columns and 'crashes' in road_df.columns:
                corr = road_df['intensity_score'].corr(road_df['crashes'])
                response += f"**Intensity-Crash Correlation:** {corr:.3f}\n"
                if abs(corr) > 0.7:
                    response += f"{'Strong' if corr > 0 else 'Strong negative'} correlation found! Braking intensity {'predicts' if corr > 0 else 'inversely relates to'} crash occurrence.\n\n"
                elif abs(corr) > 0.4:
                    response += f"Moderate correlation - roads with extreme braking patterns tend to have {'more' if corr > 0 else 'fewer'} crashes.\n\n"

        response += f"\n**Map Updated:** The map now shows hard braking events colored by severity (Emergency=Dark Red, Hard=Red, Moderate=Orange).\n\n"
        response += f"See detailed severity distribution, per-road intensity rankings, and correlation charts below!"

        st.success(f"Analyzed {intensity_results['total_hard_braking']:,} braking events - {intensity_results['emergency_count']:,} emergency-level")
        return response

    elif action == "ai_insights":
        # Run comprehensive AI analysis combining all features
        if st.session_state.vehicles is None or st.session_state.crashes is None:
            st.warning("Need both vehicle and crash data for AI insights")
            return "I need both vehicle and crash data to generate comprehensive AI insights. Please upload both datasets."

        with st.spinner("Running comprehensive AI analysis... This will generate multiple insights!"):
            # Run both analyses
            proximity_results = analyze_proximity_risk(
                st.session_state.vehicles,
                st.session_state.crashes,
                distance_meters=200
            )

            safety_scores = calculate_road_safety_scores(
                st.session_state.vehicles,
                st.session_state.crashes
            )

        # Store results
        st.session_state.proximity_results = proximity_results
        st.session_state.safety_scores = safety_scores
        st.session_state.show_plots.add('proximity_analysis')
        st.session_state.show_plots.add('safety_scores')

        # Generate comprehensive insights
        near_miss_count = proximity_results['near_miss_count'] if proximity_results else 0
        risk_zone_count = len(proximity_results.get('risk_zones', [])) if proximity_results else 0
        road_count = len(safety_scores) if safety_scores is not None else 0

        response = f"## Comprehensive AI Insights Report\n\n"
        response += f"**Analysis Summary:**\n"
        response += f"- Analyzed **{road_count} roads** across the network\n"
        response += f"- Discovered **{near_miss_count} near-miss events** (crash predictors)\n"
        response += f"- Identified **{risk_zone_count} high-risk zones** for intervention\n\n"

        response += f"**Key AI Discoveries:**\n"
        response += f"1. Hard braking patterns successfully predict {proximity_results['correlation_rate']:.1f}% of crash locations\n"
        response += f"2. AI identified future crash hotspots that haven't had crashes yet\n"
        response += f"3. Safety scores reveal which roads need immediate attention\n"
        response += f"4. Pattern recognition shows crash-braking correlation is statistically significant\n\n"

        response += f"**Actionable Recommendations:**\n"
        if risk_zone_count > 0:
            response += f"- Install warning signs on {risk_zone_count} high-risk roads\n"
            response += f"- Implement speed reduction measures on most dangerous segments\n"
            response += f"- Monitor near-miss locations for safety interventions\n\n"

        response += f"See detailed visualizations, rankings, and predictions below!"

        st.success(f"AI Analysis Complete: Generated comprehensive safety insights")
        return response

    elif action == "help":
        st.info("""
**Available Commands:**
- "Show all data"
- "Filter by I-70" or "Show data on Highway 40"
- "Show speeding vehicles"
- "Filter by speed over 80"
- "Show crash hotspots"
- "Plot speed distribution"
- "Plot crashes by road"
- "Where are the highest crashes happening?" - Analyzes crash data
- "Color by speed" - Toggle speed-based coloring
- "Corridor plot" or "Timespace diagram" - Show vehicle trajectories over time
- "Show hard braking events" - Find areas with hard braking (moderate: -0.3g, hard: -0.5g, emergency: -0.7g)
- "Plot hard braking" - Show distribution chart of braking severity levels
- "Analyze crash delays" - Calculate traffic delays caused by crashes
        """)

    else:
        st.warning("❓ Command not recognized. Try 'help' for available commands.")

# ====== MAIN APP ======
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Interactive CV Analytics Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar - Configuration & Upload
    with st.sidebar:
        st.header("Configuration")

        # API Provider Selection
        providers = ["Gemini", "Groq", "OpenAI"]
        try:
            default_index = providers.index(st.session_state.api_provider)
        except ValueError:
            default_index = 0

        provider = st.selectbox(
            "LLM Provider",
            providers,
            index=default_index,
            help="Gemini recommended - best free tier"
        )
        st.session_state.api_provider = provider

        # API Key Input (conditional based on provider)
        if provider == "Gemini":
            api_key = st.text_input(
                "Gemini API Key",
                value=st.session_state.gemini_api_key,
                type="password",
                help="Get your FREE key from https://aistudio.google.com/app/apikey"
            )
            if api_key:
                st.session_state.gemini_api_key = api_key
        elif provider == "Groq":
            api_key = st.text_input(
                "Groq API Key",
                value=st.session_state.groq_api_key,
                type="password",
                help="Get your free key from https://console.groq.com/"
            )
            if api_key:
                st.session_state.groq_api_key = api_key

            # Model selector for Groq
            groq_models = [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "llama3-70b-8192",
                "gemma-7b-it"
            ]
            try:
                model_index = groq_models.index(st.session_state.groq_model)
            except ValueError:
                model_index = 0

            selected_model = st.selectbox(
                "Groq Model",
                groq_models,
                index=model_index,
                help="llama-3.3-70b-versatile recommended for accuracy"
            )
            st.session_state.groq_model = selected_model
        else:  # OpenAI
            api_key = st.text_input(
                "OpenAI API Key",
                value=st.session_state.openai_api_key,
                type="password",
                help="Get your API key from https://platform.openai.com/api-keys"
            )
            if api_key:
                st.session_state.openai_api_key = api_key

        st.markdown("---")
        st.header("Upload Data")

        # File upload
        vehicle_file = st.file_uploader("Vehicle Data (CSV)", type=['csv'])
        crash_file = st.file_uploader("Crash Data (CSV)", type=['csv'])

        if vehicle_file:
            try:
                st.session_state.vehicles = cudf.read_csv(vehicle_file)
                st.session_state.filtered_vehicles = st.session_state.vehicles
                st.success(f"Loaded {len(st.session_state.vehicles):,} vehicle records")
            except Exception as e:
                st.error(f"Error loading vehicle data: {e}")

        if crash_file:
            try:
                crashes = cudf.read_csv(crash_file)

                # Enrich crash data if RoadName is missing
                if 'RoadName' not in crashes.columns and st.session_state.vehicles is not None:
                    st.info("RoadName not found in crash data. Looking up roads from vehicle data...")
                    crashes = enrich_crash_data_with_roads(crashes, st.session_state.vehicles)

                st.session_state.crashes = crashes
                st.session_state.filtered_crashes = crashes
                st.success(f"Loaded {len(crashes):,} crash records")
            except Exception as e:
                st.error(f"Error loading crash data: {e}")
        else:
            # If no crash file is uploaded, clear crash data
            if st.session_state.crashes is not None:
                st.session_state.crashes = None
                st.session_state.filtered_crashes = None
                st.info("Crash data cleared")

        st.markdown("---")
        st.header("Data Status")

        # AUTO-FIX: Add deceleration_g if missing but AccMagnitude exists
        if st.session_state.vehicles is not None:
            df = st.session_state.vehicles.to_pandas() if hasattr(st.session_state.vehicles, 'to_pandas') else st.session_state.vehicles
            if 'deceleration_g' not in df.columns and 'AccMagnitude' in df.columns:
                df['deceleration_g'] = -df['AccMagnitude']
                st.session_state.vehicles = cudf.from_pandas(df) if hasattr(st.session_state.vehicles, 'to_pandas') else df
                st.session_state.filtered_vehicles = st.session_state.vehicles

        if st.session_state.vehicles is not None:
            st.write(f"Vehicles: {len(st.session_state.vehicles):,}")
        else:
            st.write("No vehicle data")

        if st.session_state.crashes is not None:
            st.write(f"Crashes: {len(st.session_state.crashes):,}")
        else:
            st.write("No crash data")

        st.markdown("---")
        st.markdown(f"**Current View:** {st.session_state.current_view}")

        # Color scheme legend
        if st.session_state.color_mode == "speed":
            st.markdown("---")
            st.markdown("**Color Scheme:**")
            st.markdown("• Red: >70 mph (High speed)")
            st.markdown("• Orange: 50-70 mph (Medium)")
            st.markdown("• Green: <50 mph (Low speed)")

        # Braking severity legend
        if st.session_state.braking_mode == "on":
            st.markdown("---")
            st.markdown("**Braking Severity:**")
            st.markdown("• Dark Red: Emergency (≤-0.7g)")
            st.markdown("• Red: Hard Braking (≤-0.5g)")
            st.markdown("• Orange: Moderate (≤-0.3g)")

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        # Map header with reset button
        map_col1, map_col2 = st.columns([3, 1])
        with map_col1:
            st.subheader("Interactive Map")
        with map_col2:
            # Reset button beside the map
            if st.session_state.current_view != "all":
                if st.button("Reset View", type="primary", use_container_width=True):
                    st.session_state.filtered_vehicles = st.session_state.vehicles
                    st.session_state.filtered_crashes = st.session_state.crashes
                    st.session_state.current_view = "all"
                    st.session_state.show_plots = set()
                    st.session_state.color_mode = "neutral"
                    st.session_state.map_version += 1  # Force map update
                    st.rerun()

        # Display map (stable container to prevent flickering!)
        map_container = st.empty()

        with map_container.container():
            try:
                # Check if we're showing filtered data
                is_filtered = st.session_state.current_view != "all"

                # Create map with optional hotspot and intensity overlays
                hotspot_results = getattr(st.session_state, 'hotspot_results', None)
                intensity_results = getattr(st.session_state, 'braking_intensity_results', None)

                map_obj = create_map(
                    st.session_state.filtered_vehicles,
                    st.session_state.filtered_crashes,
                    is_filtered=is_filtered,
                    hotspot_results=hotspot_results,
                    intensity_results=intensity_results
                )

                # Use version-based key - only changes when data actually changes!
                map_key = f"map_v{st.session_state.map_version}"
                st_folium(map_obj, width=1200, height=700, key=map_key, returned_objects=[])
            except Exception as e:
                st.error(f"Error creating map: {str(e)}")

    with col2:
        st.subheader("AI Chat")

        # Chat input
        user_query = st.text_input(
            "Ask me anything:",
            placeholder="e.g., Show speeding on I-70",
            key="chat_input"
        )

        if st.button("Send", type="primary") and user_query:
            # Check if API key is set based on provider
            api_key_missing = (
                (st.session_state.api_provider == "Groq" and not st.session_state.groq_api_key) or
                (st.session_state.api_provider == "OpenAI" and not st.session_state.openai_api_key) or
                (st.session_state.api_provider == "Gemini" and not st.session_state.gemini_api_key)
            )
            if api_key_missing:
                st.error(f"Please enter your {st.session_state.api_provider} API key in the sidebar first!")
            elif st.session_state.vehicles is None:
                st.error("No vehicle data loaded. Please upload data or wait for auto-load.")
            else:
                # Add to history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_query,
                    "time": datetime.now()
                })

                # Parse and execute
                try:
                    with st.spinner("Processing..."):
                        action, params = parse_intent(user_query)

                        if action != "error":
                            # Add user query to params for custom color parsing
                            params['user_query'] = user_query
                            response = execute_command(action, params)
                            # Increment map version to force update
                            st.session_state.map_version += 1
                            # Add response to history
                            if response:
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": response,
                                    "time": datetime.now()
                                })
                            else:
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": f"Done! Updated the map.",
                                    "time": datetime.now()
                                })
                            # Force rerun to update map immediately
                            st.rerun()
                        else:
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"I couldn't understand that. {params.get('message', 'Try asking something like: show all data, filter by I-70, or show speeding vehicles')}",
                                "time": datetime.now()
                            })
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Sorry, I encountered an error: {str(e)}",
                        "time": datetime.now()
                    })

                # Force rerun after command execution to update map

        # Show chat history with modern design
        st.markdown("### Chat History")

        # Build chat HTML
        chat_html = '<div class="chat-outer-wrapper"><div class="chat-history-box">'

        if st.session_state.chat_history:
            # Show last 20 messages, newest first
            for msg in reversed(st.session_state.chat_history[-20:]):
                # Escape HTML content to prevent issues
                content = str(msg['content']).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')

                if msg["role"] == "user":
                    chat_html += f'''<div class="msg-card user">
<div class="msg-header">You</div>
<div class="msg-text">{content}</div>
</div>'''
                else:
                    chat_html += f'''<div class="msg-card assistant">
<div class="msg-header">Assistant</div>
<div class="msg-text">{content}</div>
</div>'''
        else:
            chat_html += '''<div class="chat-empty">
<div style="font-weight:600; font-size:1.1rem; margin-bottom:8px;">No messages yet</div>
<div style="font-size:0.9rem;">Ask a question above to get started!</div>
</div>'''

        chat_html += '</div></div>'
        st.markdown(chat_html, unsafe_allow_html=True)

    # Plots section - Only show if user requested
    if len(st.session_state.show_plots) > 0:
        st.markdown("---")
        st.subheader("Analytics")

        plot_col1, plot_col2 = st.columns(2)

        with plot_col1:
            if 'speed' in st.session_state.show_plots and st.session_state.filtered_vehicles is not None:
                # Add close button (closes instantly without full page reload!)
                if st.button("✕", key="close_speed", help="Close Speed Plot"):
                    st.session_state.show_plots.discard('speed')

                plot_speed_distribution(
                    st.session_state.filtered_vehicles,
                    f"Speed Distribution - {st.session_state.current_view}"
                )

        with plot_col2:
            if 'crashes' in st.session_state.show_plots and st.session_state.filtered_crashes is not None:
                # Add close button (closes instantly without full page reload!)
                if st.button("✕", key="close_crashes", help="Close Crash Plot"):
                    st.session_state.show_plots.discard('crashes')

                plot_crashes_by_road(st.session_state.filtered_crashes)

        # Hard braking distribution (full width)
        if 'hard_braking' in st.session_state.show_plots:
            st.markdown("---")

            # Add close button (closes instantly without full page reload!)
            if st.button("✕", key="close_hard_braking", help="Close Hard Braking Plot"):
                st.session_state.show_plots.discard('hard_braking')

            # Use filtered vehicles as the data source
            if st.session_state.filtered_vehicles is not None:
                plot_hard_braking_distribution(st.session_state.filtered_vehicles)

        # Crash delay analysis (full width)
        if 'crash_delays' in st.session_state.show_plots:
            st.markdown("---")

            # Add close button (closes instantly without full page reload!)
            if st.button("✕", key="close_crash_delays", help="Close Crash Delay Analysis"):
                st.session_state.show_plots.discard('crash_delays')

            # Use crash_delay_results from session state
            if hasattr(st.session_state, 'crash_delay_results') and st.session_state.crash_delay_results:
                plot_crash_delay_impact(st.session_state.crash_delay_results)

        # AI Proximity Analysis (full width)
        if 'proximity_analysis' in st.session_state.show_plots:
            st.markdown("---")

            # Add close button
            if st.button("✕", key="close_proximity", help="Close Proximity Analysis"):
                st.session_state.show_plots.discard('proximity_analysis')

            # Display proximity analysis results
            if hasattr(st.session_state, 'proximity_results') and st.session_state.proximity_results:
                plot_proximity_analysis(st.session_state.proximity_results)

        # Road Safety Scores (full width)
        if 'safety_scores' in st.session_state.show_plots:
            st.markdown("---")

            # Add close button
            if st.button("✕", key="close_safety", help="Close Safety Analysis"):
                st.session_state.show_plots.discard('safety_scores')

            # Display safety score results
            if hasattr(st.session_state, 'safety_scores') and st.session_state.safety_scores is not None:
                plot_road_safety_scores(st.session_state.safety_scores)

        # Temporal Risk Analysis (full width)
        if 'temporal_analysis' in st.session_state.show_plots:
            st.markdown("---")

            # Add close button
            if st.button("✕", key="close_temporal", help="Close Temporal Analysis"):
                st.session_state.show_plots.discard('temporal_analysis')

            # Display temporal analysis results
            if hasattr(st.session_state, 'temporal_results') and st.session_state.temporal_results:
                plot_temporal_analysis(st.session_state.temporal_results)

        # Timespace diagram (full width)
        if 'timespace' in st.session_state.show_plots:
            st.markdown("---")

            # Add close button (closes instantly without full page reload!)
            if st.button("✕", key="close_timespace", help="Close Timespace Diagram"):
                st.session_state.show_plots.discard('timespace')

            # Use the timespace_road from session state
            road = getattr(st.session_state, 'timespace_road', None)

            plot_timespace_diagram(road=road)

        # Crash Hotspot Clustering (full width)
        if 'crash_hotspots' in st.session_state.show_plots:
            st.markdown("---")

            # Add close button
            if st.button("✕", key="close_hotspots", help="Close Crash Hotspot Analysis"):
                st.session_state.show_plots.discard('crash_hotspots')

            # Display hotspot clustering results
            if hasattr(st.session_state, 'hotspot_results') and st.session_state.hotspot_results:
                plot_crash_hotspots(st.session_state.hotspot_results)

        # Braking Intensity Analysis (full width)
        if 'braking_intensity' in st.session_state.show_plots:
            st.markdown("---")

            # Add close button
            if st.button("✕", key="close_intensity", help="Close Braking Intensity Analysis"):
                st.session_state.show_plots.discard('braking_intensity')

            # Display braking intensity results
            if hasattr(st.session_state, 'braking_intensity_results') and st.session_state.braking_intensity_results:
                plot_braking_intensity(st.session_state.braking_intensity_results)

if __name__ == "__main__":
    main()
