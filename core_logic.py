import pandas as pd
import numpy as np
import requests
import yfinance as yf
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ==========================================
# KONFIGURATION
# ==========================================
MODEL_NAME = "gemini-1.5-flash" # Robustes, schnelles Modell für Textaufgaben

class FilteredHeadlines(BaseModel):
    relevant_headlines: list[str] = Field(
        description="Liste von harten, echten Schlagzeilen. Keine Meinung, kein Clickbait."
    )

# ==========================================
# HELPER: ROBUSTE SESSION (YFINANCE FIX)
# ==========================================
def get_robust_session() -> requests.Session:
    """
    Täuscht einen echten Chrome-Browser vor und nutzt automatische Wiederholungsversuche.
    Das verhindert den 'YFRateLimitError' auf Streamlit Cloud!
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })
    
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# ==========================================
# 1. MARKTDATEN (Yahoo Finance)
# ==========================================
def get_ticker_from_name(query: str) -> str:
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=1&newsCount=0"
    session = get_robust_session()
    try:
        response = session.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get('quotes'):
            return data['quotes'][0]['symbol']
    except requests.RequestException:
        pass 
    return query.strip().upper()

def load_stock_data(ticker: str) -> tuple[pd.DataFrame, list[dict], dict]:
    """Lädt die Aktienkurse mit der geschützten Session."""
    session = get_robust_session()
    stock = yf.Ticker(ticker, session=session)
    
    # 3 Monate für die Volatilitäts- und SMA-Berechnung
    hist = stock.history(period="3mo")
    return hist, stock.news, stock.info

def calculate_metrics(hist: pd.DataFrame, info: dict) -> dict:
    """Berechnet Finanzkennzahlen, Trend und Volatilität."""
    if hist.empty: 
        return {}
        
    close_price = hist["Close"].iloc[-1]
    reference = hist["Close"].iloc[-2] if len(hist) >= 2 else hist["Open"].iloc[-1]
    change_pct = ((close_price - reference) / reference) * 100 if reference else 0.0

    # Trend (SMA 20)
    if len(hist) >= 20:
        sma20 = hist["Close"].rolling(window=20).mean().iloc[-1]
        if close_price > sma20 * 1.02: trend = "🟢 Bullisch"
        elif close_price < sma20 * 0.98: trend = "🔴 Bärisch"
        else: trend = "⚪ Neutral"
    else:
        trend = "N/A"

    # Volatilität (Standardabweichung der letzten 20 Tage annualisiert)
    if len(hist) >= 20:
        daily_returns = hist["Close"].pct_change().dropna()
        vol = daily_returns.tail(20).std() * np.sqrt(252) * 100
        vol_str = f"{vol:.2f}%"
    else:
        vol_str = "N/A"

    # Formatierungen
    mc = info.get("marketCap")
    if mc is None: mc_str = "N/A"
    elif mc >= 1e12: mc_str = f"${mc/1e12:.2f} Bio."
    elif mc >= 1e9: mc_str = f"${mc/1e9:.2f} Mrd."
    else: mc_str = f"${mc/1e6:.2f} Mio."

    div = info.get("dividendYield") or info.get("trailingAnnualDividendYield")
    div_str = f"{div*100:.2f}%" if div else "N/A"

    pe = info.get("trailingPE")
    pe_str = f"{pe:.2f}" if pe else "N/A"

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
# 2. KI ANALYSE (Gemini)
# ==========================================
def filter_news_with_ai(client: genai.Client, ticker: str, user_input: str, raw_news: list[dict]) -> list[str]:
    BLOCKED_PUBLISHERS = ["motley fool", "zacks", "seeking alpha", "investorplace", "tipranks", "thestreet"]
    
    pre_filtered = []
    for n in raw_news:
        pub = n.get("publisher", n.get("provider", "Unbekannt")).lower()
        title = n.get("content", {}).get("title", n.get("title", "Kein Titel"))
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
            model=MODEL_NAME, 
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=FilteredHeadlines, temperature=0.0)
        )
        return response.parsed.relevant_headlines[:5]
    except: 
        return []

def generate_analysis(client: genai.Client, ticker: str, metrics: dict, news_list: list[str], macro: dict, euro_macro: dict, sec: list, finnhub: dict) -> str:
    direction = "gestiegen" if metrics["change_pct"] >= 0 else "gefallen"
    news_text = "\n".join([f"- {t}" for t in news_list]) if news_list else "KEINE RELEVANTEN NEWS."
    
    prompt = f"""Du bist ein institutioneller Analyst. Die Aktie {ticker} ist heute um {abs(metrics['change_pct']):.2f}% {direction} (Kurs: ${metrics['close']:.2f}).
    
    Harte Fakten:\n{news_text}
    
    Erkläre kurz und professionell, warum sich die Aktie heute so bewegt. Nutze gegebenenfalls makroökonomische Faktoren, falls keine spezifischen News vorliegen.
    
    Struktur (Markdown H3):
    ### 🗞️ Was bewegt den Kurs?
    ### 📊 Fundamentale Einordnung
    ### 🤖 Sektor & Makro-Kontext
    """
    
    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt, config=types.GenerateContentConfig(temperature=0.2))
        return response.text
    except Exception as e: 
        return f"⚠️ Analyse-Fehler: {e}"

# ==========================================
# 3. EXTERNE APIs (Robust & Fail-Safe)
# ==========================================
def get_macro_data(api_key: str) -> dict:
    """Holt US-Makrodaten von FRED."""
    if not api_key: return {}
    session = get_robust_session()
    data = {}
    endpoints = {
        "US Leitzins": "FEDFUNDS",
        "US Inflation": "CPIAUCSL"
    }
    for name, series_id in endpoints.items():
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&sort_order=desc&limit=1"
        try:
            res = session.get(url, timeout=3)
            if res.status_code == 200:
                val = float(res.json()['observations'][0]['value'])
                data[name] = {"value": round(val, 2)}
        except: pass
    return data

def get_euro_macro_data() -> dict:
    """Holt EU-Makrodaten von DBnomics (Kein API Key nötig)."""
    session = get_robust_session()
    data = {}
    try:
        # EZB Leitzins (Main refinancing operations)
        url = "https://api.db.nomics.world/v22/series/ECB/FM/M.U2.EUR.4F.KR.MRR_RT.LEV"
        res = session.get(url, timeout=3)
        if res.status_code == 200:
            val = res.json()['series']['docs'][0]['value'][-1]
            if val is not None: data["EZB Leitzins"] = {"value": round(val, 2)}
    except: pass
    return data

def get_sec_filings(ticker: str, email: str) -> list[dict]:
    """Holt die neuesten 10-K / 10-Q Berichte von der SEC."""
    if not email: return []
    session = get_robust_session()
    session.headers.update({"User-Agent": f"TrueFin App {email}"})
    
    try:
        # 1. CIK Nummer herausfinden
        cik_res = session.get("https://www.sec.gov/files/company_tickers.json", timeout=3)
        cik_res.raise_for_status()
        cik_dict = cik_res.json()
        
        cik = None
        for entry in cik_dict.values():
            if entry['ticker'].upper() == ticker.upper():
                cik = str(entry['cik_str']).zfill(10)
                break
                
        if not cik: return []
        
        # 2. Filings abrufen
        filings_res = session.get(f"https://data.sec.gov/submissions/CIK{cik}.json", timeout=3)
        filings_res.raise_for_status()
        recent = filings_res.json()['filings']['recent']
        
        results = []
        for i in range(len(recent['form'])):
            if recent['form'][i] in ["10-K", "10-Q", "8-K"]:
                acc_num = recent['accessionNumber'][i].replace("-", "")
                results.append({
                    "form": recent['form'][i],
                    "filingDate": recent['filingDate'][i],
                    "reportDate": recent['reportDate'][i] if recent['reportDate'][i] else "N/A",
                    "description": recent['primaryDocDescription'][i],
                    "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_num}/{recent['primaryDocument'][i]}"
                })
            if len(results) >= 3: break # Nur die 3 neuesten
        return results
    except:
        return []

def get_finnhub_data(ticker: str, api_key: str) -> dict:
    """Holt Analysten-Konsens und Insider-Sentiment von Finnhub."""
    if not api_key: return {}
    session = get_robust_session()
    data = {}
    
    try:
        # Analysten
        rec_url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={ticker}&token={api_key}"
        rec_res = session.get(rec_url, timeout=3)
        if rec_res.status_code == 200 and len(rec_res.json()) > 0:
            data["recommendations"] = rec_res.json()[0]
            
        # Insider
        ins_url = f"https://finnhub.io/api/v1/stock/insider-sentiment?symbol={ticker}&from=2024-01-01&to=2024-12-31&token={api_key}"
        ins_res = session.get(ins_url, timeout=3)
        if ins_res.status_code == 200 and ins_res.json().get('data'):
            # Durchschnitt der MSPR (Management Sentiment)
            mspr_list = [d['mspr'] for d in ins_res.json()['data']]
            avg_mspr = round(sum(mspr_list) / len(mspr_list), 2)
            data["insider"] = {"mspr": avg_mspr}
    except: pass
    
    return data