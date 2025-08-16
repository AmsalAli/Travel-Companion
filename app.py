# file: app.py
"""Travel Companion — Streamlit Web App

Features
- Natural-language parsing for flight search (no LLM required)
- Optional Amadeus API integration for real flight offers
- Offline synthetic flight generator as a fallback
- Folium map to visualize routes

Environment Variables (optional)
- AMADEUS_API_KEY
- AMADEUS_API_SECRET

Notes
- NLP keeps dependencies light (regex + dateparser).
- City/IATA/coords coverage includes common global cities; extend CITY_DB as needed.
"""
from __future__ import annotations

import os
import math
import json
import random
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import dateparser
import requests
import streamlit as st
from streamlit_folium import st_folium
import folium

# --------- Models ---------
@dataclass
class Query:
    origin: Optional[str] = None            # IATA or city name
    destination: Optional[str] = None       # IATA or city name
    departure_date: Optional[date] = None
    return_date: Optional[date] = None
    adults: int = 1
    cabin_class: str = "ECONOMY"            # ECONOMY | PREMIUM_ECONOMY | BUSINESS | FIRST
    max_price: Optional[float] = None
    preferred_airline: Optional[str] = None
    nonstop_only: bool = False

    def is_roundtrip(self) -> bool:
        return self.return_date is not None

    def to_dict(self) -> Dict:
        d = asdict(self)
        # convert dates to ISO
        if self.departure_date:
            d["departure_date"] = self.departure_date.isoformat()
        if self.return_date:
            d["return_date"] = self.return_date.isoformat()
        return d

# --------- City / Airport DB ---------
# Minimal curated city -> (IATA, lat, lon). Extendable.
CITY_DB: Dict[str, Tuple[str, float, float]] = {
    # Europe
    "berlin": ("BER", 52.5200, 13.4050),
    "munich": ("MUC", 48.1351, 11.5820),
    "frankfurt": ("FRA", 50.1109, 8.6821),
    "hamburg": ("HAM", 53.5511, 9.9937),
    "cologne": ("CGN", 50.9375, 6.9603),
    "dusseldorf": ("DUS", 51.2277, 6.7735),
    "stuttgart": ("STR", 48.7758, 9.1829),
    "paris": ("CDG", 48.8566, 2.3522),
    "london": ("LHR", 51.5074, -0.1278),
    "amsterdam": ("AMS", 52.3676, 4.9041),
    "brussels": ("BRU", 50.8503, 4.3517),
    "madrid": ("MAD", 40.4168, -3.7038),
    "barcelona": ("BCN", 41.3874, 2.1686),
    "lisbon": ("LIS", 38.7223, -9.1393),
    "rome": ("FCO", 41.9028, 12.4964),
    "milan": ("MXP", 45.4642, 9.1900),
    "vienna": ("VIE", 48.2082, 16.3738),
    "zurich": ("ZRH", 47.3769, 8.5417),
    "geneva": ("GVA", 46.2044, 6.1432),
    "copenhagen": ("CPH", 55.6761, 12.5683),
    "stockholm": ("ARN", 59.3293, 18.0686),
    "oslo": ("OSL", 59.9139, 10.7522),
    "helsinki": ("HEL", 60.1699, 24.9384),
    "dublin": ("DUB", 53.3498, -6.2603),
    "prague": ("PRG", 50.0755, 14.4378),
    "budapest": ("BUD", 47.4979, 19.0402),
    "warsaw": ("WAW", 52.2297, 21.0122),
    "athens": ("ATH", 37.9838, 23.7275),
    "istanbul": ("IST", 41.0082, 28.9784),
    "reyjkavik": ("KEF", 64.1466, -21.9426),

    # North America
    "new york": ("JFK", 40.7128, -74.0060),
    "nyc": ("JFK", 40.7128, -74.0060),
    "boston": ("BOS", 42.3601, -71.0589),
    "washington": ("IAD", 38.9072, -77.0369),
    "chicago": ("ORD", 41.8781, -87.6298),
    "miami": ("MIA", 25.7617, -80.1918),
    "atlanta": ("ATL", 33.7490, -84.3880),
    "dallas": ("DFW", 32.7767, -96.7970),
    "houston": ("IAH", 29.7604, -95.3698),
    "denver": ("DEN", 39.7392, -104.9903),
    "seattle": ("SEA", 47.6062, -122.3321),
    "san francisco": ("SFO", 37.7749, -122.4194),
    "los angeles": ("LAX", 34.0522, -118.2437),
    "las vegas": ("LAS", 36.1699, -115.1398),
    "san diego": ("SAN", 32.7157, -117.1611),
    "toronto": ("YYZ", 43.6532, -79.3832),
    "vancouver": ("YVR", 49.2827, -123.1207),
    "montreal": ("YUL", 45.5019, -73.5674),

    # LATAM
    "mexico city": ("MEX", 19.4326, -99.1332),
    "bogota": ("BOG", 4.7110, -74.0721),
    "lima": ("LIM", -12.0464, -77.0428),
    "sao paulo": ("GRU", -23.5505, -46.6333),
    "rio de janeiro": ("GIG", -22.9068, -43.1729),
    "buenos aires": ("EZE", -34.6037, -58.3816),

    # Middle East / Africa
    "dubai": ("DXB", 25.2048, 55.2708),
    "doha": ("DOH", 25.2854, 51.5310),
    "tel aviv": ("TLV", 32.0853, 34.7818),
    "johannesburg": ("JNB", -26.2041, 28.0473),
    "nairobi": ("NBO", -1.2921, 36.8219),
    "cairo": ("CAI", 30.0444, 31.2357),

    # Asia-Pacific
    "tokyo": ("HND", 35.6762, 139.6503),
    "osaka": ("KIX", 34.6937, 135.5023),
    "seoul": ("ICN", 37.5665, 126.9780),
    "beijing": ("PEK", 39.9042, 116.4074),
    "shanghai": ("PVG", 31.2304, 121.4737),
    "hong kong": ("HKG", 22.3193, 114.1694),
    "singapore": ("SIN", 1.3521, 103.8198),
    "bangkok": ("BKK", 13.7563, 100.5018),
    "kuala lumpur": ("KUL", 3.1390, 101.6869),
    "taipei": ("TPE", 25.0330, 121.5654),
    "sydney": ("SYD", -33.8688, 151.2093),
    "melbourne": ("MEL", -37.8136, 144.9631),
    "auckland": ("AKL", -36.8509, 174.7645),
}

IATA_TO_COORDS = {v[0]: (v[1], v[2]) for v in CITY_DB.values()}

# --------- Helpers ---------
CABINS = {"economy": "ECONOMY", "premium economy": "PREMIUM_ECONOMY", "business": "BUSINESS", "first": "FIRST"}

AIRLINES = {
    "lufthansa": "LH", "british airways": "BA", "air france": "AF", "klm": "KL", "emirates": "EK",
    "qatar": "QR", "turkish": "TK", "american": "AA", "delta": "DL", "united": "UA",
}


def parse_date(text: str) -> Optional[date]:
    if not text:
        return None
    dt = dateparser.parse(text, settings={"PREFER_DATES_FROM": "future"})
    return dt.date() if dt else None


def norm_city(text: str) -> str:
    return text.strip().lower()


def resolve_city_or_iata(s: str) -> Optional[str]:
    s = s.strip()
    if len(s) == 3 and s.isalpha():
        return s.upper()
    key = norm_city(s)
    if key in CITY_DB:
        return CITY_DB[key][0]
    # last resort: take 3 letters
    return s[:3].upper() if s else None


# Simple haversine
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# --------- NLP Parser ---------
import re

ROUTE_PATTERNS = [
    re.compile(r"from\s+(?P<orig>[a-zA-Z\s]+)\s+(to|->)\s+(?P<dest>[a-zA-Z\s]+)", re.I),
    re.compile(r"(?P<orig>[a-zA-Z\s]+)\s+to\s+(?P<dest>[a-zA-Z\s]+)", re.I),
]

ADULTS_PATTERN = re.compile(r"(for|with)\s+(?P<n>\d+)\s+(adult|adults|people|passengers?)", re.I)
CABIN_PATTERN = re.compile(r"(in|class)\s+(economy|premium economy|business|first)", re.I)
MAX_PRICE_PATTERN = re.compile(r"under\s*\$?(?P<price>\d{2,5})", re.I)
AIRLINE_PATTERN = re.compile(r"on\s+([A-Za-z\s]+)", re.I)
NONSTOP_PATTERN = re.compile(r"(non[-\s]?stop|direct)\s+only", re.I)
DATE_RANGE_PATTERN = re.compile(r"between\s+(?P<d1>[A-Za-z0-9,\s/-]+)\s+and\s+(?P<d2>[A-Za-z0-9,\s/-]+)", re.I)
DEPARTURE_ON_PATTERN = re.compile(r"on\s+(?P<d>[A-Za-z0-9,\s/-]+)", re.I)
RETURN_ON_PATTERN = re.compile(r"return(ing)?\s+(on\s+)?(?P<d>[A-Za-z0-9,\s/-]+)", re.I)


def parse_nl_query(text: str) -> Query:
    q = Query()
    t = text.strip()

    # route
    for pat in ROUTE_PATTERNS:
        m = pat.search(t)
        if m:
            q.origin = resolve_city_or_iata(m.group("orig"))
            q.destination = resolve_city_or_iata(m.group("dest"))
            break

    # dates
    m = DATE_RANGE_PATTERN.search(t)
    if m:
        q.departure_date = parse_date(m.group("d1"))
        q.return_date = parse_date(m.group("d2"))
    else:
        m = DEPARTURE_ON_PATTERN.search(t)
        if m:
            q.departure_date = parse_date(m.group("d"))
        m = RETURN_ON_PATTERN.search(t)
        if m:
            q.return_date = parse_date(m.group("d"))

    # people
    m = ADULTS_PATTERN.search(t)
    if m:
        try:
            q.adults = max(1, int(m.group("n")))
        except Exception:
            pass

    # cabin
    m = CABIN_PATTERN.search(t)
    if m:
        q.cabin_class = CABINS.get(m.group(2).lower(), "ECONOMY")

    # price
    m = MAX_PRICE_PATTERN.search(t)
    if m:
        q.max_price = float(m.group("price"))

    # airline
    m = AIRLINE_PATTERN.search(t)
    if m:
        name = m.group(1).strip().lower()
        if name in AIRLINES:
            q.preferred_airline = AIRLINES[name]

    # direct only
    if NONSTOP_PATTERN.search(t):
        q.nonstop_only = True

    return q


# --------- Amadeus Client (optional) ---------
class AmadeusClient:
    base = "https://test.api.amadeus.com"

    def __init__(self, key: str, secret: str):
        self.key = key
        self.secret = secret
        self._token: Optional[str] = None
        self._exp: datetime = datetime.min

    def _get_token(self) -> Optional[str]:
        if self._token and datetime.utcnow() < self._exp:
            return self._token
        try:
            resp = requests.post(
                f"{self.base}/v1/security/oauth2/token",
                data={"grant_type": "client_credentials", "client_id": self.key, "client_secret": self.secret},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            self._token = data.get("access_token")
            expires_in = int(data.get("expires_in", 0))
            self._exp = datetime.utcnow() + timedelta(seconds=expires_in - 60)
            return self._token
        except Exception:
            return None

    def search_offers(self, q: Query) -> Optional[List[Dict]]:
        token = self._get_token()
        if not token:
            return None
        try:
            params = {
                "originLocationCode": q.origin,
                "destinationLocationCode": q.destination,
                "departureDate": q.departure_date.isoformat() if q.departure_date else (date.today() + timedelta(days=7)).isoformat(),
                "adults": str(q.adults),
                "currencyCode": "EUR",
                "nonStop": str(q.nonstop_only).lower(),
                "max": "20",
            }
            if q.return_date:
                params["returnDate"] = q.return_date.isoformat()
            if q.cabin_class:
                params["travelClass"] = q.cabin_class

            headers = {"Authorization": f"Bearer {token}"}
            r = requests.get(f"{self.base}/v2/shopping/flight-offers", params=params, headers=headers, timeout=20)
            r.raise_for_status()
            data = r.json()
            # Normalize into our schema list
            offers = []
            for item in data.get("data", []):
                price = float(item.get("price", {}).get("grandTotal", 0.0))
                itineraries = item.get("itineraries", [])
                duration = item.get("itineraries", [{}])[0].get("duration", "") if itineraries else ""
                carrier = item.get("validatingAirlineCodes", [""])[0]
                offers.append({
                    "airline": carrier,
                    "price": price,
                    "currency": data.get("dictionaries", {}).get("currencyCodes", {}).get("EUR", "EUR"),
                    "duration": duration,
                    "stops": 0,  # not parsed here
                    "departure": params["departureDate"],
                    "return": params.get("returnDate"),
                })
            return offers
        except Exception:
            return None


# --------- Synthetic Flight Generator ---------
class SyntheticFlights:
    @staticmethod
    def generate(q: Query, n: int = 10) -> List[Dict]:
        # Coordinates for distance-based pricing
        lat1, lon1 = IATA_TO_COORDS.get(q.origin, (0.0, 0.0))
        lat2, lon2 = IATA_TO_COORDS.get(q.destination, (0.0, 0.0))
        distance_km = haversine_km(lat1, lon1, lat2, lon2) if lat1 or lat2 else 1500

        base = max(50, distance_km * 0.09)  # € per km heuristic
        cabin_mult = {"ECONOMY": 1.0, "PREMIUM_ECONOMY": 1.4, "BUSINESS": 2.1, "FIRST": 3.2}[q.cabin_class]
        nonstop_mult = 1.15 if q.nonstop_only else 1.0

        dep = q.departure_date or date.today() + timedelta(days=random.randint(3, 45))
        ret = q.return_date if q.is_roundtrip() else None

        flights = []
        for _ in range(n):
            stops = 0 if q.nonstop_only else random.choice([0, 0, 1])
            dur_hours = max(1, int(distance_km / 750) + stops * random.choice([1, 2]))
            dep_time = datetime.combine(dep, datetime.min.time()) + timedelta(hours=random.randint(5, 22))
            if ret:
                ret_time = datetime.combine(ret, datetime.min.time()) + timedelta(hours=random.randint(5, 22))
            else:
                ret_time = None

            airline = q.preferred_airline or random.choice(["LH", "BA", "AF", "KL", "EK", "QR", "AA", "DL", "UA", "TK"])  # noqa: E501
            price = round(base * cabin_mult * nonstop_mult * random.uniform(0.8, 1.3), 2)
            if q.max_price and price > q.max_price:
                continue

            flights.append({
                "airline": airline,
                "price": price,
                "currency": "EUR",
                "duration": f"{dur_hours}h",
                "stops": stops,
                "departure": dep_time.isoformat(),
                "return": ret_time.isoformat() if ret_time else None,
            })
        return sorted(flights, key=lambda x: x["price"])[:n]


# --------- Folium Map ---------
class MapView:
    @staticmethod
    def render(origin_iata: str, dest_iata: str) -> folium.Map:
        (lat1, lon1) = IATA_TO_COORDS.get(origin_iata, (0.0, 0.0))
        (lat2, lon2) = IATA_TO_COORDS.get(dest_iata, (0.0, 0.0))
        center = [(lat1 + lat2) / 2, (lon1 + lon2) / 2]
        m = folium.Map(location=center, zoom_start=3)
        folium.Marker([lat1, lon1], tooltip=origin_iata).add_to(m)
        folium.Marker([lat2, lon2], tooltip=dest_iata).add_to(m)
        folium.PolyLine([[lat1, lon1], [lat2, lon2]], weight=3).add_to(m)
        return m


# --------- UI ---------
st.set_page_config(page_title="Travel Companion", page_icon="✈️", layout="centered")

st.title("✈️ Travel Companion")
st.caption("Search flights from natural language or structured inputs. Amadeus integration optional.")

with st.expander("Enter a natural language request", expanded=True):
    nl = st.text_input(
        "e.g. 'Berlin to New York on Oct 20 returning Oct 27 for 2 adults in business under 1200'",
        value="",
    )
    if st.button("Parse Query"):
        q = parse_nl_query(nl)
        st.session_state["q_parsed"] = q.to_dict()

# Keep a Query object in session
if "q_parsed" not in st.session_state:
    st.session_state["q_parsed"] = Query().to_dict()
q_state = st.session_state["q_parsed"]

st.subheader("Search Parameters")
col1, col2 = st.columns(2)
with col1:
    origin = st.text_input("Origin (city or IATA)", value=q_state.get("origin") or "Berlin")
    dep = st.date_input("Departure", value=(parse_date(q_state.get("departure_date")) or (date.today() + timedelta(days=14))))
    cabin = st.selectbox("Cabin", options=list(CABINS.values()), index=list(CABINS.values()).index(q_state.get("cabin_class", "ECONOMY")))
with col2:
    dest = st.text_input("Destination (city or IATA)", value=q_state.get("destination") or "New York")
    rtrip = st.checkbox("Round trip", value=bool(q_state.get("return_date")))
    ret = st.date_input("Return", value=(parse_date(q_state.get("return_date")) or (date.today() + timedelta(days=21))), disabled=not rtrip)

adults = st.number_input("Adults", min_value=1, max_value=9, value=int(q_state.get("adults", 1)), step=1)
nonstop = st.checkbox("Nonstop only", value=bool(q_state.get("nonstop_only", False)))
max_price = st.number_input("Max Price (EUR)", min_value=0, value=int(q_state.get("max_price") or 0), step=50)
preferred_airline = st.selectbox("Preferred airline (optional)", options=["None"] + sorted(AIRLINES.keys()))

# Build Query
query = Query(
    origin=resolve_city_or_iata(origin),
    destination=resolve_city_or_iata(dest),
    departure_date=dep,
    return_date=ret if rtrip else None,
    adults=adults,
    cabin_class=cabin,
    max_price=float(max_price) if max_price else None,
    preferred_airline=AIRLINES.get(preferred_airline) if preferred_airline != "None" else None,
    nonstop_only=nonstop,
)

# Search
if st.button("Search Flights", type="primary"):
    st.session_state["last_query"] = query.to_dict()

if "last_query" in st.session_state:
    q = Query(**{**st.session_state["last_query"],
                 "departure_date": parse_date(st.session_state["last_query"].get("departure_date")) or dep,
                 "return_date": parse_date(st.session_state["last_query"].get("return_date")) or (ret if rtrip else None)})

    st.write("### Results")

    # Try Amadeus
    offers: List[Dict]
    api_key, api_secret = os.getenv("AMADEUS_API_KEY"), os.getenv("AMADEUS_API_SECRET")
    offers = []
    used_amadeus = False
    if api_key and api_secret:
        am = AmadeusClient(api_key, api_secret)
        data = am.search_offers(q)
        if data:
            offers = data
            used_amadeus = True

    # Fallback synthetic
    if not offers:
        offers = SyntheticFlights.generate(q)

    # Map
    if q.origin and q.destination:
        try:
            st.write("#### Route Map")
            m = MapView.render(q.origin, q.destination)
            st_folium(m, width=700, height=350)
        except Exception:
            st.info("Map not available for these codes.")

    # Table
    if offers:
        st.write("#### Flight Options")
        st.dataframe(
            [{
                "Airline": f["airline"],
                "Price": f"{f['price']} {f.get('currency', 'EUR')}",
                "Duration": f.get("duration", "–"),
                "Stops": f.get("stops", 0),
                "Departure": f.get("departure", ""),
                "Return": f.get("return", ""),
            } for f in offers],
            use_container_width=True,
        )
        st.success("Results shown from {}".format("Amadeus API" if used_amadeus else "synthetic generator"))
    else:
        st.warning("No offers found. Try relaxing filters (price, nonstop, cabin).")

# Footer
st.markdown("---")
st.caption("Tip: you can deploy this app in minutes on Streamlit Community Cloud.")