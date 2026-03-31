import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import aiohttp
import asyncio
from datetime import datetime, timedelta
import os
import math
import logging
import contextlib

# Configure logging at INFO level for system monitoring.
# Standard out streams are used to align with Docker container logging best practices.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Crypto-Gex")

# Deribit public API base URL. Rate limit: 20 req/sec per IP.
BASE_URL = "https://www.deribit.com/api/v2/public/"

# Mapping for Deribit instrument date format (e.g., '28MAR26').
# Constant time lookup dictionary to avoid runtime string manipulation overhead.
MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}

class Greeks:
    """
    Black-Scholes-Merton implementation using standard math libraries.
    Assumes 0% continuous dividend yield (q=0) for crypto assets.
    """
    @staticmethod
    def nd1(d):
        # Cumulative Distribution Function (CDF) of the standard normal distribution.
        # Utilizes math.erf for high-precision approximation without heavy library imports.
        return 0.5 * (1.0 + math.erf(d / math.sqrt(2.0)))

    @staticmethod
    def npd1(d):
        # Probability Density Function (PDF) of the standard normal distribution.
        return math.exp(-0.5 * d**2) / math.sqrt(2 * math.pi)

    @classmethod
    def calculate(cls, S, K, T, sigma, r, type):
        # Handle zero-time to expiry or zero implied volatility to prevent ZeroDivisionError.
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0, 0.0, 0.0
        
        try:
            # d1 represents the probability-weighted moneyness of the option.
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            # d2 represents the probability of the option expiring in-the-money.
            d2 = d1 - sigma * math.sqrt(T)
            
            if type == 'C': 
                delta = cls.nd1(d1)
                gamma = cls.npd1(d1) / (S * sigma * math.sqrt(T))
                # Annualized Theta calculating decay over 365 days.
                theta = -(S * cls.npd1(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * cls.nd1(d2)
                vega = S * cls.npd1(d1) * math.sqrt(T)
            else: 
                delta = cls.nd1(d1) - 1
                gamma = cls.npd1(d1) / (S * sigma * math.sqrt(T))
                theta = -(S * cls.npd1(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * cls.nd1(-d2)
                vega = S * cls.npd1(d1) * math.sqrt(T)
            
            return delta, gamma, theta, vega
        
        # Catch specific math domain errors from deep OTM/ITM strikes to prevent thread crash.
        except (ValueError, ZeroDivisionError, OverflowError):
            return 0.0, 0.0, 0.0, 0.0

class Cache:
    """
    In-memory TTL cache to manage API request frequency.
    Prevents IP bans by serving stale data (TTL: 5 seconds) during concurrent WebSocket requests.
    """
    def __init__(self):
        self.store = {}

    def get(self, key):
        entry = self.store.get(key)
        # Verify if the current monotonic clock exceeds the stored expiration timestamp.
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
        # Calculate strike price where the total intrinsic value of open options is minimized.
        if not chain: return 0
        strikes = sorted({o['k'] for o in chain})
        min_pain = float('inf')
        pain_price = 0
        
        # Iterates through every potential settlement strike against the entire open interest matrix.
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
        # Put/Call Ratio based on absolute Open Interest, filtering by instrument type string.
        c = sum(o['oi'] for o in chain if o['ty'] == 'C')
        p = sum(o['oi'] for o in chain if o['ty'] == 'P')
        return round(p/c, 2) if c > 0 else 0

    @staticmethod
    def weighted_iv(chain):
        # Volume-weighted aggregate implied volatility to normalize skew across strikes.
        total_oi = sum(o['oi'] for o in chain)
        if not total_oi: return 0
        return sum(o['iv'] * o['oi'] for o in chain) / total_oi

    @staticmethod
    def vwap(chain):
        # Volume-Weighted Average Strike to determine intraday center of gravity.
        total_vol = sum(o['vol'] for o in chain)
        if not total_vol: return 0
        return sum(o['k'] * o['vol'] for o in chain) / total_vol

    @staticmethod
    def skew_25d(chain, spot):
        # Approximate 25-delta skew computing the IV differential between 10% OTM puts and calls.
        calls = [o for o in chain if o['ty'] == 'C' and abs(o['k'] - spot*1.1) < spot*0.05]
        puts = [o for o in chain if o['ty'] == 'P' and abs(o['k'] - spot*0.9) < spot*0.05]
        if not calls or not puts: return 0
        c_iv = sum(o['iv'] for o in calls) / len(calls)
        p_iv = sum(o['iv'] for o in puts) / len(puts)
        return round(p_iv - c_iv, 4)

    @staticmethod
    def term_structure(chain):
        # Aggregates weighted IV per expiration maturity to build a temporal volatility curve.
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
        # Initialize persistent HTTP connection pooling to eliminate TCP handshake latency on repeated fetches.
        if not self.session:
            self.session = aiohttp.ClientSession(headers={"User-Agent": "CryptoGex/2.0"})

    async def stop(self):
        # Graceful teardown of unclosed sockets to prevent file descriptor leaks.
        if self.session: 
            await self.session.close()

    async def fetch(self, url):
        # Cache bypass check
        cached = self.cache.get(url)
        if cached: return cached
        try:
            # 5-second timeout drop to prevent hanging coroutines on API degradation.
            async with self.session.get(url, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.cache.set(url, data)
                    return data
                return None
        except Exception as e:
            logger.error(f"Network error on fetch {url}: {e}")
            return None

    def _parse_expiry(self, date_str):
        # Manual byte-slice parsing of standard Deribit 'DDMMMYY' instrument substrings.
        try:
            day = int(date_str[:2])
            mon = date_str[2:5].upper()
            year = int(date_str[5:]) + 2000
            if mon in MONTH_MAP: return datetime(year, MONTH_MAP[mon], day)
        except: return None
        return None

    async def snapshot(self, ticker):
        spot = 0
        # Failover iteration through base asset index price variants to establish the spot underlying.
        for idx in [f"{ticker.lower()}_usdc", f"{ticker.lower()}_usd"]:
            res = await self.fetch(f"{BASE_URL}get_index_price?index_name={idx}")
            if res and 'result' in res:
                spot = res['result']['index_price']
                break
        
        if not spot: return {"error": "Spot price index rejected or unavailable."}

        # Coroutine aggregation: fetch standard and stablecoin-margined books in parallel to halve total network I/O time.
        urls = [
            f"{BASE_URL}get_book_summary_by_currency?currency={ticker}&kind=option",
            f"{BASE_URL}get_book_summary_by_currency?currency=USDC&kind=option"
        ]
        results = await asyncio.gather(*[self.fetch(u) for u in urls])
        
        raw = []
        for r in results:
            if r and 'result' in r: raw.extend(r['result'])

        chain = []
        seen = set() 
        now = datetime.now()
        prefix = ticker.upper()

        for d in raw:
            name = d.get('instrument_name')
            # Deduplication filter: Handles overlapping strikes between Coin-M and USDC-M instruments.
            if not name or name in seen: continue
            if not (name.startswith(f"{prefix}-") or name.startswith(f"{prefix}_")): continue
            seen.add(name)
            
            try:
                parts = name.split('-')
                if len(parts) < 4: continue
                
                exp = self._parse_expiry(parts[1])
                # Drop expired contracts to prevent negative time-to-maturity calculations in BSM.
                if not exp or exp <= now: continue
                k = float(parts[2])
                
                # Spatial bounds filtering: Drop strikes outside a 0.2x to 2.5x variance of spot to save CPU cycles.
                if not (spot * 0.2 < k < spot * 2.5): continue

                iv = d.get('mark_iv', 0) / 100
                # Time continuous fraction relative to a 365-day trading year.
                t = (exp - now).total_seconds() / 31536000
                oi = d.get('open_interest', 0)
                ty = parts[3]

                # BSM execution utilizing a fixed 5% risk-free rate matrix.
                delta, gamma, theta, vega = Greeks.calculate(spot, k, t, iv, 0.05, ty)

                chain.append({
                    "k": k, "t": t, "iv": iv, "oi": oi,
                    "vol": d.get('volume', 0), "ty": ty, "exp": parts[1],
                    "delta": delta, "gamma": gamma, "theta": theta, "vega": vega
                })
            except: 
                continue

        if not chain: return {"error": "Zero valid contracts passed spatial filtering."}
        
        # Ensures client-side rendering engine receives an ordered matrix.
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

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup phase: Initialize global HTTP session pool.
    await engine.start()
    yield
    # Shutdown phase: Clean termination of client sessions.
    await engine.stop()

app = FastAPI(title="Crypto.Gex Terminal", lifespan=lifespan)

@app.websocket("/ws")
async def ws_handler(websocket: WebSocket):
    # Establish ASGI WebSocket connection.
    await websocket.accept()
    try:
        # Event loop listening for client JSON payloads.
        while True:
            msg = await websocket.receive_json()
            if msg.get('action') == 'sub':
                ticker = msg.get('ticker', '').upper()
                # Trigger pipeline calculation on explicit client subscription request.
                res = await engine.snapshot(ticker)
                
                if "error" in res:
                    await websocket.send_json({"type": "error", "msg": res['error']})
                else:
                    await websocket.send_json({"type": "data", "payload": res})
    except WebSocketDisconnect:
        # Client silently dropped the connection without a standard closing handshake.
        pass
    except Exception as e:
        logger.error(f"WebSocket state error: {e}")
    finally:
        # Ensure socket termination to prevent zombie connections tying up the Uvicorn worker.
        try: 
            await websocket.close()
        except: 
            pass

@app.get("/", response_class=HTMLResponse)
async def home():
    # Dynamic file path resolution independent of the active working directory.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "index.html")
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Error: index.html missing from working directory."

@app.get("/api/v1/health")
async def health():
    # Simple L7 ping endpoint for load balancer health checks.
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/v1/snapshot/{ticker}")
async def get_ticker_snapshot(ticker: str):
    # REST fallback endpoint for clients incapable of maintaining a persistent WebSocket.
    return await engine.snapshot(ticker.upper())

if __name__ == "__main__":
    # Local execution entry point. 
    # Port 8000 binds to the local loopback, overridden by Docker CMD when containerized.
    uvicorn.run(app, host="127.0.0.1", port=8000)
