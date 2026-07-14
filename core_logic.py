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

# Das offizielle DBnomics Paket importieren!
from dbnomics import fetch_series

# ==========================================
# KONFIGURATION
# ==========================================
MODEL_NAME = "gemini-3.1-flash-lite"

class FilteredHeadlines(BaseModel):
    relevant_headlines: list[str] = Field(description="Liste von harten, echten Schlagzeilen.")

def get_robust_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session

# ==========================================
# 1. MARKTDATEN (Finnhub)
# ==========================================
def get_ticker_from_name(query: str, api_key: str) -> str:
    if not api_key: return query.strip().upper()
    session = get_robust_session()
    try:
        res = session.get(f"https://finnhub.io/api/v1/search?q={query}&token={api_key.strip()}", timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data.get("result") and len(data["result"]) > 0:
                for item in data["result"]:
                    if "." not in item["symbol"] and item["type"] == "Common Stock":
                        return item["symbol"]
                return data["result"][0]["symbol"]
    except: pass
    return query.strip().upper()

def load_stock_data(ticker: str, api_key: str) -> tuple[pd.DataFrame, list[dict], dict]:
    df = pd.DataFrame()
    info = {}
    news_mapped = []
    if not api_key: return df, news_mapped, info
    
    session = get_robust_session()
    token = api_key.strip()
    
    # 1. Unternehmensprofil & Metriken
    try:
        prof = session.get(f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token={token}", timeout=5).json()
        mets = session.get(f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={token}", timeout=5).json()
        
        info["shortName"] = prof.get("name", ticker)
        # FIX für den SEC Tab: Wenn country leer ist, standardmäßig US setzen
        info["country"] = prof.get("country") or "US" 
        if prof.get("marketCapitalization"):
            info["marketCap"] = prof.get("marketCapitalization") * 1000000 
            
        if mets.get("metric"):
            info["trailingPE"] = mets["metric"].get("peExclExtraTTM")
            dy = mets["metric"].get("dividendYieldIndicatedAnnual")
            if dy: info["dividendYield"] = dy / 100.0
    except: pass

    # 2. Kurse (Historie für Trendlinien)
    now = int(time.time())
    three_months_ago = now - (90 * 24 * 60 * 60)
    try:
        res = session.get(f"https://finnhub.io/api/v1/stock/candle?symbol={ticker}&resolution=D&from={three_months_ago}&to={now}&token={token}", timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data.get("s") == "ok":
                df = pd.DataFrame({"Close": data["c"], "Open": data["o"]}, index=pd.to_datetime(data["t"], unit='s'))
    except: pass

    # FIX FÜR EUROPÄISCHE AKTIEN: Fallback für Live-Kurs, falls Finnhub historische Daten für EU blockiert
    if df.empty:
        try:
            quote = session.get(f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={token}", timeout=5).json()
            if quote and quote.get("c", 0) > 0:
                df = pd.DataFrame({
                    "Close": [quote.get("pc", quote["c"]), quote["c"]],
                    "Open": [quote.get("o", quote["c"]), quote.get("o", quote["c"])]
                }, index=[pd.Timestamp.now() - pd.Timedelta(days=1), pd.Timestamp.now()])
        except: pass

    # 3. News
    try:
        today_str = datetime.now().strftime('%Y-%m-%d')
        week_ago_str = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        news_res = session.get(f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={week_ago_str}&to={today_str}&token={token}", timeout=5).json()
        if isinstance(news_res, list):
            for n in news_res[:15]: 
                news_mapped.append({"title": n.get("headline", ""), "publisher": n.get("source", "")})
    except: pass
    
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
        "close": close_price, "change_pct": change_pct, "market_cap": mc_str,
        "pe_ratio": pe_str, "dividend_yield": div_str,
        "trend_signal": trend, "volatility": vol_str
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
    endpoints = {
        "US Leitzins": {"id": "FEDFUNDS", "units": "lin"},
        "US Inflation": {"id": "CPIAUCSL", "units": "pc1"}
    }
    for name, cfg in endpoints.items():
        try:
            res = session.get(f"https://api.stlouisfed.org/fred/series/observations?series_id={cfg['id']}&units={cfg['units']}&api_key={api_key}&file_type=json&sort_order=desc&limit=1", timeout=5)
            if res.status_code == 200:
                val = float(res.json()['observations'][0]['value'])
                data[name] = {"value": round(val, 2)}
        except: pass
    return data

def get_euro_macro_data() -> dict:
    data = {}
    try:
        # DBnomics API Aufruf mit offiziellem Paket (exakt aus deinem Backup)
        df = fetch_series("ECB/FM/M.U2.EUR.4F.KR.MRR_RT.LEV")
        if df is not None and not df.empty and 'value' in df.columns:
            # Sichert ab, dass das NA/NaN ignoriert wird
            valid_vals = df['value'].dropna()
            if not valid_vals.empty:
                last_val = valid_vals.iloc[-1]
                data["EZB Leitzins"] = {"value": round(float(last_val), 2)}
    except Exception as e: 
        print(f"DBnomics Fehler: {e}")
    return data

def get_sec_filings(ticker: str, email: str) -> list[dict]:
    if not email: return []
    session = get_robust_session()
    
    clean_email = email.strip()
    session.headers.update({
        "User-Agent": f"TrueFin_Terminal ({clean_email})",
        "Accept-Encoding": "gzip, deflate"
    })
    
    try:
        cik_res = session.get("https://www.sec.gov/files/company_tickers.json", timeout=10)
        if cik_res.status_code != 200: return []
        
        cik_data = cik_res.json()
        cik = next((str(entry['cik_str']).zfill(10) for entry in cik_data.values() if entry['ticker'].upper() == ticker.upper()), None)
        if not cik: return []
        
        recent = session.get(f"https://data.sec.gov/submissions/CIK{cik}.json", timeout=10).json()['filings']['recent']
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
    except Exception as e: 
        print(f"SEC Fehler: {e}")
        return []

def get_finnhub_data(ticker: str, api_key: str) -> dict:
    if not api_key: return {}
    session = get_robust_session()
    data = {}
    try:
        rec_res = session.get(f"https://finnhub.io/api/v1/stock/recommendation?symbol={ticker}&token={api_key.strip()}", timeout=5)
        if rec_res.status_code == 200 and len(rec_res.json()) > 0: data["recommendations"] = rec_res.json()[0]
            
        ins_res = session.get(f"https://finnhub.io/api/v1/stock/insider-sentiment?symbol={ticker}&from=2024-01-01&to=2024-12-31&token={api_key.strip()}", timeout=5)
        if ins_res.status_code == 200 and ins_res.json().get('data'):
            avg_mspr = round(sum(d['mspr'] for d in ins_res.json()['data']) / len(ins_res.json()['data']), 2)
            data["insider"] = {"mspr": avg_mspr}
    except: pass
    return data