import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ==========================================
# KONFIGURATION
# ==========================================
MODEL_NAME = "gemini-1.5-flash"

class FilteredHeadlines(BaseModel):
    relevant_headlines: list[str] = Field(description="Liste von harten, echten Schlagzeilen.")

def get_robust_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "TrueFin Terminal"})
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session

# ==========================================
# 1. MARKTDATEN (Finnhub API - Gefixt!)
# ==========================================
def get_ticker_from_name(query: str, api_key: str) -> str:
    if not api_key: return query.strip().upper()
    session = get_robust_session()
    try:
        res = session.get(f"https://finnhub.io/api/v1/search?q={query}&token={api_key}", timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data.get("result") and len(data["result"]) > 0:
                # FIX: Wir suchen explizit nach dem US-Ticker (ohne Punkt im Namen), 
                # da Finnhub im Free-Tier nur US-Historienkurse erlaubt!
                for item in data["result"]:
                    if "." not in item["symbol"] and item["type"] == "Common Stock":
                        return item["symbol"]
                # Fallback: Nimm das erste
                return data["result"][0]["symbol"]
    except Exception as e:
        print(f"Fehler bei der Ticker-Suche: {e}")
    return query.strip().upper()

def load_stock_data(ticker: str, api_key: str) -> tuple[pd.DataFrame, list[dict], dict]:
    if not api_key: return pd.DataFrame(), [], {}
    
    session = get_robust_session()
    base_url = "https://finnhub.io/api/v1"
    
    now = int(time.time())
    three_months_ago = now - (90 * 24 * 60 * 60)
    today_str = datetime.now().strftime('%Y-%m-%d')
    week_ago_str = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

    # 1. Historische Kurse (Candles)
    df = pd.DataFrame()
    try:
        res = session.get(f"{base_url}/stock/candle?symbol={ticker}&resolution=D&from={three_months_ago}&to={now}&token={api_key}")
        if res.status_code == 200:
            data = res.json()
            if data.get("s") == "ok":
                df = pd.DataFrame({"Close": data["c"], "Open": data["o"]}, index=pd.to_datetime(data["t"], unit='s'))
            else:
                print(f"Finnhub Candle Status für {ticker}: {data.get('s')} (Oft ein Zeichen für non-US Ticker im Free-Tier)")
        else:
            print(f"Finnhub Candle Fehler Code: {res.status_code}")
    except Exception as e: 
        print(f"Exception bei Candles: {e}")

    # 2. Metadaten (Profil & Metriken)
    info = {}
    try:
        prof = session.get(f"{base_url}/stock/profile2?symbol={ticker}&token={api_key}").json()
        mets = session.get(f"{base_url}/stock/metric?symbol={ticker}&metric=all&token={api_key}").json()

        info["shortName"] = prof.get("name", ticker)
        info["country"] = prof.get("country", "")
        if prof.get("marketCapitalization"):
            info["marketCap"] = prof.get("marketCapitalization") * 1000000 

        if mets.get("metric"):
            info["trailingPE"] = mets["metric"].get("peExclExtraTTM")
            dy = mets["metric"].get("dividendYieldIndicatedAnnual")
            if dy is not None: info["dividendYield"] = dy / 100.0
    except Exception as e: 
        print(f"Exception bei Profil/Metriken: {e}")

    # 3. News
    news_mapped = []
    try:
        news_res = session.get(f"{base_url}/company-news?symbol={ticker}&from={week_ago_str}&to={today_str}&token={api_key}").json()
        if isinstance(news_res, list): # Sicherstellen, dass es eine Liste ist und kein Error-Dict
            for n in news_res[:15]: 
                news_mapped.append({"title": n.get("headline", ""), "publisher": n.get("source", "")})
    except Exception as e: 
        print(f"Exception bei News: {e}")

    return df, news_mapped, info

def calculate_metrics(hist: pd.DataFrame, info: dict) -> dict:
    if hist.empty: return {}
    close_price = hist["Close"].iloc[-1]
    reference = hist["Close"].iloc[-2] if len(hist) >= 2 else hist["Open"].iloc[-1]
    change_pct = ((close_price - reference) / reference) * 100 if reference else 0.0

    if len(hist) >= 20:
        sma20 = hist["Close"].rolling(window=20).mean().iloc[-1]
        if close_price > sma20 * 1.02: trend = "🟢 Bullisch"
        elif close_price < sma20 * 0.98: trend = "🔴 Bärisch"
        else: trend = "⚪ Neutral"
        
        daily_returns = hist["Close"].pct_change().dropna()
        vol = daily_returns.tail(20).std() * np.sqrt(252) * 100
        vol_str = f"{vol:.2f}%"
    else:
        trend = "N/A"
        vol_str = "N/A"

    mc = info.get("marketCap")
    if mc is None or pd.isna(mc): mc_str = "N/A"
    elif mc >= 1e12: mc_str = f"${mc/1e12:.2f} Bio."
    elif mc >= 1e9: mc_str = f"${mc/1e9:.2f} Mrd."
    else: mc_str = f"${mc/1e6:.2f} Mio."

    div = info.get("dividendYield")
    div_str = f"{div*100:.2f}%" if div and not pd.isna(div) else "N/A"

    pe = info.get("trailingPE")
    pe_str = f"{pe:.2f}" if pe and not pd.isna(pe) else "N/A"

    return {
        "close": close_price,
        "change_pct": change_pct,
        "market_cap": mc_str,
        "pe_ratio": pe_str,
        "dividend_yield": div_str,
        "trend_signal": trend,
        "volatility": vol_str
    }

# ==========================================
# 2. KI ANALYSE
# ==========================================
def filter_news_with_ai(client: genai.Client, ticker: str, user_input: str, raw_news: list[dict]) -> list[str]:
    if not raw_news: return []
    BLOCKED_PUBLISHERS = ["motley fool", "zacks", "seeking alpha", "investorplace", "tipranks", "thestreet"]
    pre_filtered = []
    for n in raw_news:
        pub = n.get("publisher", "Unbekannt").lower()
        title = n.get("title", "Kein Titel")
        if any(b in pub for b in BLOCKED_PUBLISHERS): continue
        pre_filtered.append(f"[{pub.title()}] {title}")
        if len(pre_filtered) >= 15: break

    if not pre_filtered: return []

    prompt = f"""Hier sind aktuelle Schlagzeilen für '{ticker}':\n{chr(10).join(f"- {t}" for t in pre_filtered)}
    REGELN:
    1. Behalte AUSSCHLIESSLICH harte Fakten (Quartalszahlen, M&A, Management-Wechsel, Klagen).
    2. Lösche Clickbait, Ratgeber oder "Aktie X vs Y".
    3. Gib [] zurück, falls nichts Relevantes dabei ist."""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME, contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=FilteredHeadlines, temperature=0.0)
        )
        return response.parsed.relevant_headlines[:5]
    except: return []

def generate_analysis(client: genai.Client, ticker: str, metrics: dict, news_list: list[str], macro: dict, euro_macro: dict, sec: list, finnhub: dict) -> str:
    direction = "gestiegen" if metrics.get("change_pct", 0) >= 0 else "gefallen"
    news_text = "\n".join([f"- {t}" for t in news_list]) if news_list else "KEINE RELEVANTEN NEWS."
    
    prompt = f"""Du bist ein institutioneller Analyst. Die Aktie {ticker} ist heute um {abs(metrics.get('change_pct', 0)):.2f}% {direction} (Kurs: ${metrics.get('close', 0):.2f}).
    Harte Fakten:\n{news_text}\n
    Erkläre kurz und professionell, warum sich die Aktie heute so bewegt.
    Struktur (Markdown H3):
    ### 🗞️ Was bewegt den Kurs?
    ### 📊 Fundamentale Einordnung
    ### 🤖 Sektor & Makro-Kontext"""
    
    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt, config=types.GenerateContentConfig(temperature=0.2))
        return response.text
    except Exception as e: return f"⚠️ Analyse-Fehler: {e}"

# ==========================================
# 3. EXTERNE APIs
# ==========================================
def get_macro_data(api_key: str) -> dict:
    if not api_key: return {}
    session = get_robust_session()
    data = {}
    endpoints = {"US Leitzins": "FEDFUNDS", "US Inflation": "CPIAUCSL"}
    for name, series_id in endpoints.items():
        try:
            res = session.get(f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&sort_order=desc&limit=1", timeout=3)
            if res.status_code == 200:
                data[name] = {"value": round(float(res.json()['observations'][0]['value']), 2)}
        except: pass
    return data

def get_euro_macro_data() -> dict:
    session = get_robust_session()
    data = {}
    try:
        res = session.get("https://api.db.nomics.world/v22/series/ECB/FM/M.U2.EUR.4F.KR.MRR_RT.LEV", timeout=3)
        if res.status_code == 200:
            val = res.json()['series']['docs'][0]['value'][-1]
            if val is not None: data["EZB Leitzins"] = {"value": round(val, 2)}
    except: pass
    return data

def get_sec_filings(ticker: str, email: str) -> list[dict]:
    if not email: return []
    session = get_robust_session()
    session.headers.update({"User-Agent": f"TrueFin App {email}"})
    try:
        cik_res = session.get("https://www.sec.gov/files/company_tickers.json", timeout=3).json()
        cik = next((str(entry['cik_str']).zfill(10) for entry in cik_res.values() if entry['ticker'].upper() == ticker.upper()), None)
        if not cik: return []
        
        recent = session.get(f"https://data.sec.gov/submissions/CIK{cik}.json", timeout=3).json()['filings']['recent']
        results = []
        for i in range(len(recent['form'])):
            if recent['form'][i] in ["10-K", "10-Q", "8-K"]:
                acc_num = recent['accessionNumber'][i].replace("-", "")
                results.append({
                    "form": recent['form'][i],
                    "filingDate": recent['filingDate'][i],
                    "reportDate": recent['reportDate'][i] or "N/A",
                    "description": recent['primaryDocDescription'][i],
                    "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_num}/{recent['primaryDocument'][i]}"
                })
            if len(results) >= 3: break
        return results
    except: return []

def get_finnhub_data(ticker: str, api_key: str) -> dict:
    if not api_key: return {}
    session = get_robust_session()
    data = {}
    try:
        rec_res = session.get(f"https://finnhub.io/api/v1/stock/recommendation?symbol={ticker}&token={api_key}", timeout=3)
        if rec_res.status_code == 200 and len(rec_res.json()) > 0: data["recommendations"] = rec_res.json()[0]
            
        ins_res = session.get(f"https://finnhub.io/api/v1/stock/insider-sentiment?symbol={ticker}&from=2024-01-01&to=2024-12-31&token={api_key}", timeout=3)
        if ins_res.status_code == 200 and ins_res.json().get('data'):
            avg_mspr = round(sum(d['mspr'] for d in ins_res.json()['data']) / len(ins_res.json()['data']), 2)
            data["insider"] = {"mspr": avg_mspr}
    except: pass
    return data