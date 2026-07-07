FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    streamlit==1.28.1 \
    requests==2.31.0 \
    pandas==2.1.0 \
    plotly==5.17.0

# Copy dashboard
COPY streamlit_dashboard.py .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Streamlit config
RUN mkdir -p ~/.streamlit && echo "\
[server]\n\
port = 8501\n\
headless = true\n\
" > ~/.streamlit/config.toml

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_dashboard.py"]
