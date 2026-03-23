import streamlit as st
import joblib
import re
import nltk
import plotly.graph_objects as go
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# 1. Page Config (Must be the first Streamlit command)
st.set_page_config(
    page_title="Sentiment Pro",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Modern Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global styling */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #f8fafc; /* Soft modern slate background */
    }

    /* Headers */
    .main-title {
        text-align: center;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4f46e5, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        padding-top: 1rem;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 3rem;
    }

    /* Styling Streamlit Native Elements */
    div[data-testid="stTextArea"] textarea {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        background-color: #ffffff;
        padding: 1rem;
        font-size: 1.05rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #4f46e5;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.2);
    }

    /* Beautiful Result Cards */
    .result-card {
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.5s ease-in;
    }
    .result-positive {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    .result-negative {
        background: linear-gradient(135deg, #f43f5e 0%, #e11d48 100%);
    }
    .result-emoji {
        font-size: 4.5rem;
        margin-bottom: 0.5rem;
        line-height: 1;
    }
    .result-text {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 1px;
    }

    /* Empty State */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        min-height: 400px;
        color: #94a3b8;
        background: white;
        border-radius: 20px;
        border: 2px dashed #cbd5e1;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# 3. Cache Model & NLTK
@st.cache_resource(show_spinner="Loading AI Model...")
def load_model():
    try:
        return joblib.load('models/sentiment_pipeline.joblib')
    except FileNotFoundError:
        st.error("Model file not found! Please ensure 'models/sentiment_pipeline.joblib' exists.")
        return None

model = load_model()

@st.cache_data
def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

download_nltk_data()

# 4. Text Cleaning Function
def clean_text(text, remove_stopwords=True, use_stemming=True):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
    if use_stemming:
        stemmer = PorterStemmer()
        words = [stemmer.stem(w) for w in words]
    return ' '.join(words)

# 5. Callbacks for smooth state management
if "user_text" not in st.session_state:
    st.session_state.user_text = ""
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "proba" not in st.session_state:
    st.session_state.proba = None

def set_example(text):
    st.session_state.user_text = text
    st.session_state.prediction = None # Clear previous results

def clear_input():
    st.session_state.user_text = ""
    st.session_state.prediction = None

# 6. UI Layout
st.markdown('<div class="main-title">Sentiment Analysis Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Uncover the emotion behind any text with AI-powered precision ✨</div>', unsafe_allow_html=True)

col_left, col_space, col_right = st.columns([1, 0.1, 1])

# --- LEFT COLUMN: INPUT ---
with col_left:
    st.markdown("### 📝 Enter your text")
    
    # Text area bound directly to session state
    st.text_area(
        label="Input Text",
        height=180,
        placeholder="e.g., I absolutely loved this movie! The cinematography was breathtaking.",
        key="user_text", # Binds to st.session_state.user_text automatically
        label_visibility="collapsed"
    )

    st.markdown("#### ✨ Try an example")
    ex_col1, ex_col2 = st.columns(2)
    
    examples = [
        ("👍 Masterpiece", "An absolute masterpiece, I'll watch it again! The acting was superb."),
        ("👎 Waste of time", "Terrible film. The plot was boring and a complete waste of time."),
        ("😐 Just okay", "It was okay, nothing special but not the worst I've seen."),
        ("🥰 Amazing product", "I am so happy with this purchase. Highly recommended!")
    ]

    # Render example buttons
    for i, (label, text) in enumerate(examples):
        target_col = ex_col1 if i % 2 == 0 else ex_col2
        target_col.button(
            label, 
            use_container_width=True, 
            on_click=set_example, 
            args=(text,)
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Action Buttons
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        predict_btn = st.button("🔮 Predict Sentiment", type="primary", use_container_width=True)
    with btn_col2:
        st.button("🗑️ Clear", on_click=clear_input, use_container_width=True)

    # Process Prediction
    if predict_btn:
        if not st.session_state.user_text.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        elif model is None:
            st.error("⚠️ Model is not loaded. Cannot predict.")
        else:
            with st.spinner("🧠 Analyzing emotion..."):
                cleaned = clean_text(st.session_state.user_text)
                # Note: Adjust logic here depending on how your specific joblib model outputs
                st.session_state.prediction = model.predict([cleaned])[0]
                st.session_state.proba = model.predict_proba([cleaned])[0]


# --- RIGHT COLUMN: RESULTS ---
with col_right:
    if st.session_state.prediction is not None:
        st.markdown("### 📊 Analysis Results")
        
        # Determine styling based on result (Assuming 1=Positive, 0=Negative)
        is_positive = (st.session_state.prediction == 1)
        sentiment_text = "Positive" if is_positive else "Negative"
        pos_prob = st.session_state.proba[1]
        neg_prob = st.session_state.proba[0]
        
        css_class = "result-positive" if is_positive else "result-negative"
        emoji = "✨ Awesome!" if is_positive else "😞 Oh no..."

        # 1. Result Card
        st.markdown(f"""
        <div class="result-card {css_class}">
            <div class="result-emoji">{emoji}</div>
            <div class="result-text">{sentiment_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        # 2. Native Streamlit Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Positive", f"{pos_prob:.1%}")
        m2.metric("Negative", f"{neg_prob:.1%}")
        m3.metric("Confidence", f"{max(pos_prob, neg_prob):.1%}")

        # 3. Modernized Plotly Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pos_prob * 100,
            number={'suffix': "%", 'font': {'size': 40, 'color': '#1e293b'}},
            title={'text': "Positivity Score", 'font': {'size': 16, 'color': '#64748b'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
                'bar': {'color': "#4f46e5", 'thickness': 0.25}, # Indigo bar
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#e2e8f0",
                'steps': [
                    {'range': [0, 45], 'color': '#ffe4e6'},  # Soft rose
                    {'range': [45, 55], 'color': '#f1f5f9'},  # Slate neutral
                    {'range': [55, 100], 'color': '#d1fae5'} # Soft emerald
                ],
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=250,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # 4. Cleaned text expander
        with st.expander("🔍 View Preprocessed Text"):
            st.caption("This is how the AI model sees your text after stripping punctuation, stop words, and applying stemming:")
            st.code(clean_text(st.session_state.user_text), language="text")

    else:
        # Beautiful Empty State
        st.markdown("""
        <div class="empty-state">
            <h1 style="font-size: 4rem; margin-bottom: 0;">🔮</h1>
            <h3>Awaiting Input</h3>
            <p>Enter some text and hit predict to see the magic happen.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>"
    "Built with ❤️ using Streamlit & scikit-learn | <a href='#' style='color: #4f46e5; text-decoration: none;'>View Source</a>"
    "</p>", 
    unsafe_allow_html=True
)