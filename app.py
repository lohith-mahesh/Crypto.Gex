import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import aiohttp
import asyncio
import numpy as np
from datetime import datetime, timedelta
import webbrowser
import logging
import json
import contextlib
import os
import math

# Good ol' logging setup. Kept it at INFO so we don't spam the console unless things actually break.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CRYPTO-GEX")

# Deribit's public API. No auth needed for these endpoints, which is nice.
BASE_URL = "https://www.deribit.com/api/v2/public/"

# Mapping for Deribit's string date format (e.g., '28MAR26')
MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}

class Greeks:
    """
    Standard Black-Scholes-Merton implementation. 
    Assuming 0 dividend yield here because it's crypto.
    """
    @staticmethod
    def nd1(d):
        return 0.5 * (1.0 + math.erf(d / math.sqrt(2.0)))

    @staticmethod
    def npd1(d):
        return math.exp(-0.5 * d**2) / math.sqrt(2 * math.pi)

    @classmethod
    def calculate(cls, S, K, T, sigma, r, type):
        # Prevent division by zero if the option expires literally right now or has flat IV
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0, 0.0, 0.0
        
        try:
            # Core BS math
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            if type == 'C': # Calls
                delta = cls.nd1(d1)
                gamma = cls.npd1(d1) / (S * sigma * math.sqrt(T))
                theta = -(S * cls.npd1(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * cls.nd1(d2)
                vega = S * cls.npd1(d1) * math.sqrt(T)
            else: # Puts
                delta = cls.nd1(d1) - 1
                gamma = cls.npd1(d1) / (S * sigma * math.sqrt(T))
                theta = -(S * cls.npd1(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * cls.nd1(-d2)
                vega = S * cls.npd1(d1) * math.sqrt(T)
            
            return delta, gamma, theta, vega
        except:
            # Catch-all for extreme deep ITM/OTM edge cases that cause math overflows.
            # Better to return zeros than crash the whole data ingestion loop.
            return 0.0, 0.0, 0.0, 0.0

class Cache:
    """
    Dirty little TTL cache so we don't hammer Deribit's API and get IP banned.
    Defaults to 5 seconds which is plenty fast for macro options data.
    """
    def __init__(self):
        self.store = {}

    def get(self, key):
        entry = self.store.get(key)
        if entry and datetime.now() < entry['expiry']:
            return entry['data']
        return None

    def set(self, key, data, ttl=5):
        self.store[key] = {
            'data': data,
            'expiry': datetime.now() + timedelta(seconds=ttl)
        }

class Analytics:
    @staticmethod
    def max_pain(chain):
        # Calculate where options buyers bleed the most at expiry. Market makers love this metric.
        if not chain: return 0
        strikes = sorted({o['k'] for o in chain})
        min_pain = float('inf')
        pain_price = 0
        
        for price in strikes:
            loss = 0
            for o in chain:
                if o['ty'] == 'C' and price > o['k']: 
                    loss += (price - o['k']) * o['oi']
                elif o['ty'] == 'P' and price < o['k']: 
                    loss += (o['k'] - price) * o['oi']
            
            if loss < min_pain:
                min_pain = loss
                pain_price = price
        return pain_price

    @staticmethod
    def pcr(chain):
        # Basic sentiment check. High PCR = bearish/hedging heavily.
        c = sum(o['oi'] for o in chain if o['ty'] == 'C')
        p = sum(o['oi'] for o in chain if o['ty'] == 'P')
        return round(p/c, 2) if c > 0 else 0

    @staticmethod
    def weighted_iv(chain):
        # Raw average IV is useless because illiquid strikes skew it. Weighting by OI fixes this.
        total_oi = sum(o['oi'] for o in chain)
        if not total_oi: return 0
        return sum(o['iv'] * o['oi'] for o in chain) / total_oi

    @staticmethod
    def vwap(chain):
        total_vol = sum(o['vol'] for o in chain)
        if not total_vol: return 0
        return sum(o['k'] * o['vol'] for o in chain) / total_vol

    @staticmethod
    def skew_25d(chain, spot):
        # Calling this "25 delta" is a bit of a lie. It's actually just a +/- 10% spot 
        # approximation to save compute. Doing actual 25d root finding is too slow for a websocket feed.
        calls = [o for o in chain if o['ty'] == 'C' and abs(o['k'] - spot*1.1) < spot*0.05]
        puts = [o for o in chain if o['ty'] == 'P' and abs(o['k'] - spot*0.9) < spot*0.05]
        if not calls or not puts: return 0
        c_iv = sum(o['iv'] for o in calls) / len(calls)
        p_iv = sum(o['iv'] for o in puts) / len(puts)
        return round(p_iv - c_iv, 4)

    @staticmethod
    def term_structure(chain):
        # Groups IV by expiry to see if we're in contango or backwardation.
        expiries = sorted({o['exp'] for o in chain})
        structure = []
        for e in expiries:
            subset = [o for o in chain if o['exp'] == e]
            structure.append({"exp": e, "iv": Analytics.weighted_iv(subset)})
        return structure

class MarketData:
    def __init__(self):
        self.session = None
        self.cache = Cache()

    async def start(self):
        # Reusing the aiohttp session is critical for performance. Don't tear it down per request.
        if not self.session:
            self.session = aiohttp.ClientSession(headers={"User-Agent": "CryptoGEX/2.0"})

    async def stop(self):
        if self.session: 
            await self.session.close()

    async def fetch(self, url):
        cached = self.cache.get(url)
        if cached: return cached
        try:
            # 5s timeout. If Deribit is taking longer than that, their matching engine is probably struggling anyway.
            async with self.session.get(url, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.cache.set(url, data)
                    return data
                return None
        except Exception as e:
            logger.error(f"Network error: {e}")
            return None

    def _parse_expiry(self, date_str):
        # Manually ripping apart Deribit's date strings. 
        # Note: Added +2000 to the year. Will break in the year 2100. Not my problem.
        try:
            day = int(date_str[:2])
            mon = date_str[2:5].upper()
            year = int(date_str[5:]) + 2000
            if mon in MONTH_MAP: return datetime(year, MONTH_MAP[mon], day)
        except: return None
        return None

    async def snapshot(self, ticker):
        spot = 0
        # Try finding the spot price. USDC first, fallback to USD if it fails.
        for idx in [f"{ticker.lower()}_usdc", f"{ticker.lower()}_usd"]:
            res = await self.fetch(f"{BASE_URL}get_index_price?index_name={idx}")
            if res and 'result' in res:
                spot = res['result']['index_price']
                break
        
        if not spot: return {"error": "Spot price not found. Is the ticker correct?"}

        # Fetch both coin-margined and USDC-margined options to get the full picture.
        urls = [
            f"{BASE_URL}get_book_summary_by_currency?currency={ticker}&kind=option",
            f"{BASE_URL}get_book_summary_by_currency?currency=USDC&kind=option"
        ]
        results = await asyncio.gather(*[self.fetch(u) for u in urls])
        
        raw = []
        for r in results:
            if r and 'result' in r: raw.extend(r['result'])

        chain = []
        seen = set() # To deduplicate overlapping USDC/Coin margined instruments
        now = datetime.now()
        prefix = ticker.upper()

        for d in raw:
            name = d.get('instrument_name')
            if not name or name in seen: continue
            
            # Ensure we're only looking at the requested coin.
            if not (name.startswith(f"{prefix}-") or name.startswith(f"{prefix}_")): continue
            seen.add(name)
            
            try:
                # Instrument names look like BTC-29MAR24-65000-C
                parts = name.split('-')
                if len(parts) < 4: continue
                
                exp = self._parse_expiry(parts[1])
                # Ignore expired contracts
                if not exp or exp <= now: continue
                k = float(parts[2])
                
                # Filter out extreme tail strikes (e.g., $400k BTC calls) to reduce noise.
                if not (spot * 0.2 < k < spot * 2.5): continue

                iv = d.get('mark_iv', 0) / 100
                # Time to expiry in years. 31536000 = seconds in a 365-day year.
                t = (exp - now).total_seconds() / 31536000
                oi = d.get('open_interest', 0)
                ty = parts[3]

                # Risk free rate hardcoded to 5% (0.05). Could be pulled dynamically but this is fine for now.
                delta, gamma, theta, vega = Greeks.calculate(spot, k, t, iv, 0.05, ty)

                chain.append({
                    "k": k, "t": t, "iv": iv, "oi": oi,
                    "vol": d.get('volume', 0), "ty": ty, "exp": parts[1],
                    "delta": delta, "gamma": gamma, "theta": theta, "vega": vega
                })
            except: 
                # Skip unparseable rows silently
                continue

        if not chain: return {"error": "No valid options found in that range."}
        
        # Sort by strike price so the frontend doesn't have to deal with out-of-order data
        chain.sort(key=lambda x: x['k'])

        return {
            "spot": spot,
            "vwap": Analytics.vwap(chain),
            "max_pain": Analytics.max_pain(chain),
            "pcr": Analytics.pcr(chain),
            "avg_iv": Analytics.weighted_iv(chain),
            "skew": Analytics.skew_25d(chain, spot),
            "structure": Analytics.term_structure(chain),
            "chain": chain,
            "ts": datetime.now().strftime("%H:%M:%S")
        }

engine = MarketData()

# Lifespan context handles startup and shutdown gracefully. 
# Replaces the deprecated @app.on_event("startup")
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    await engine.start()
    # Auto-open the browser because I'm lazy and hate copying localhost URLs
    webbrowser.open("http://127.0.0.1:8000")
    yield
    await engine.stop()

app = FastAPI(title="Crypto.GEX Terminal", lifespan=lifespan)

@app.websocket("/ws")
async def ws_handler(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_json()
            if msg.get('action') == 'sub':
                ticker = msg.get('ticker', '').upper()
                res = await engine.snapshot(ticker)
                
                # Send errors back cleanly so the UI can show a toast notification or something
                if "error" in res:
                    await websocket.send_json({"type": "error", "msg": res['error']})
                else:
                    await websocket.send_json({"type": "data", "payload": res})
    except WebSocketDisconnect:
        # Client dropped. Totally normal, don't spam the logs.
        pass
    except Exception as e:
        logger.error(f"WS Error: {e}")
    finally:
        try: 
            await websocket.close()
        except: 
            pass # Socket already dead, just ignore.

@app.get("/", response_class=HTMLResponse)
async def home():
    # Serve the frontend statically.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "index.html")
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: index.html not found. Did you forget to create the frontend?"

@app.get("/api/v1/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/v1/snapshot/{ticker}")
async def get_ticker_snapshot(ticker: str):
    # Standard REST fallback in case websockets aren't desired.
    data = await engine.snapshot(ticker.upper())
    return data

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
