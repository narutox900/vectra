# 🔍 Vectra: Query Fan-Out Simulator

Streamlit app that simulates Google's AI Mode query fan-out process using Gemini or OpenAI.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run vectra.py
```

## Usage

1. Enter API key (Gemini or OpenAI)
2. Enter query(s) - single or bulk
3. Select mode: AI Overview (simple) or AI Mode (complex)
4. Click "Run Fan-Out 🚀"
5. Download results as CSV

## Optional: SerpAPI

Enable SerpAPI integration to fetch Google search results and AI Overview data for each generated query.

## Files

- `vectra.py`: Full-featured version (bulk processing, SerpAPI, analysis)
- `vectra-single.py`: Simplified single-query version
