# Travel Companion — Streamlit App

A minimal, deployable flight-search demo with:
- Natural-language parsing (regex + dateparser)
- Optional Amadeus flight offers
- Offline synthetic fallback
- Folium route map

## Quick Start (local)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py