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

# ✈️ Travel Companion – Intelligent Flight Assistant

## 🔍 Purpose
The Travel Companion project is a multimodal AI-powered assistant that helps users search for flights using natural-language queries — via text **or** voice input. It integrates multiple NLP engines, real-time APIs, and map-based visualizations to deliver an intuitive and intelligent travel planning experience.

---

## ⚙️ Core Features

| Component | Description |
|---|---|
| **Multimodal Input** | Supports typed and voice-based queries via OpenAI Whisper API |
| **NLP Engine Selection** | Switch between spaCy + NLTK, OpenAI GPT-4o, or DeepSeek |
| **Flight Data Source** | Connects to Amadeus API for real-time availability; uses synthetic fallback |
| **City Gazetteer** | Uses `rapidfuzz` for fuzzy city matching and alias handling |
| **Date Parsing** | Translates phrases like “next week” using `dateparser` |
| **Dynamic Visualization** | Flight routes displayed on interactive maps via Folium |
| **Voice Interaction** | Records audio via browser (ipywebrtc), transcribes using Whisper |
| **Interactive UI** | Built with ipywidgets for inputs, buttons, tables, and recording |

---

## 🧠 Technology Stack

| Layer | Tools & Libraries |
|---|---|
| **NLP** | spaCy, NLTK, OpenAI GPT-4o, DeepSeek |
| **Voice Input** | OpenAI Whisper, ipywebrtc, SciPy |
| **Flight API** | Amadeus (OAuth-secured) |
| **Map Visualization** | Folium, Leaflet.js (Python wrapper) |
| **Fuzzy Matching** | rapidfuzz |
| **UI Layer** | ipywidgets, IPython display |
| **Audio Preprocessing** | FFmpeg (via subprocess), `scipy.io.wavfile` |
| **Date Parsing** | dateparser |

---

## 📦 Modular Design
- **Query** — Data model for user requests  
- **Flight** — Represents individual flight offers  
- **NLPService** — Extracts structured info from queries via chosen NLP engine  
- **FlightService** — Handles synthetic and real-time flight search  
- **CityService** — Manages city normalization and metadata  
- **MapService** — Generates route maps with curved paths  
- **WhisperService** — Converts audio to text using Whisper

---

## ✅ Strengths
- **Flexible NLP pipeline**: easily extendable with other LLMs or NLP tools  
- **Fail-safety**: synthetic fallback when APIs fail  
- **User experience**: friendly output handling, even with partial input  
- **Educational value**: demonstrates end-to-end AI integration (NLP + API + UI + voice + maps)

---

## 🚀 Potential Extensions
- Integrate hotel/car rental APIs (e.g., Booking.com, Expedia)  
- Add user authentication and itinerary saving  
- Deploy as a web app via Streamlit or Flask  
- Add translation for multilingual queries

---

## 🔗 References
- OpenAI Whisper  
- Amadeus Travel APIs  
- spaCy  
- NLTK  
- Folium  
- rapidfuzz  
