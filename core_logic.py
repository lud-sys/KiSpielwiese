import datetime
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dbnomics import fetch_series

# --- KONFIGURATION ---
MODEL_NAME = "gemini-3.1-flash-lite"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

# --- KI-SCHEMA ---
class HeadlineWithSentiment(BaseModel):
    headline: str = Field(description="Die bereinigte, relevante Schlagzeile.")
    sentiment: str = Field(description="Bewertung der Auswirkung: 'BULLISH', 'BEARISH' oder 'NEUTRAL'")

class FilteredHeadlines(BaseModel):
    relevant_headlines: list[HeadlineWithSentiment] = Field(description="Liste der relevanten Schlagzeilen inklusive Sentiment.")

# --- UTILS ---
def get_requests_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def format_large_number(num: float | int | None) -> str:
    if num is None or pd.isna(num): return "N/A"
    if num >= 1_000_000_000_000: return f"{num / 1_000_000_000_000:.2f} Bio. $"
    if num >= 1_000_000_000: return f"{num / 1_000_000_000:.2f} Mrd. $"
    if num >= 1_000_000: return f"{num / 1_000_000:.2f} Mio. $"
    return f"{num:,.2f} $"

# --- DATENBESCHAFFUNG ---
def get_ticker_from_name(query: str) -> str:
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=1&newsCount=0"
    session = get_requests_session()
    try:
        response = session.get(url, headers={'User-Agent': USER_AGENT}, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get('quotes'):
            return data['quotes'][0]['symbol']
    except requests.RequestException:
        pass
    return query.strip().upper()

def load_stock_data(ticker: str) -> tuple[pd.DataFrame, list[dict], dict]:
    stock = yf.Ticker(ticker)
    hist = stock.history(period="3mo")
    return hist, stock.news, stock.info

def get_macro_data(api_key: str) -> dict:
    if not api_key: return {}
    session = get_requests_session()
    indicators = {
        "US-Leitzins (FED)": "FEDFUNDS",
        "10J US-Staatsanleihen": "DGS10",
        "US-Arbeitslosenquote": "UNRATE"
    }
    macro_context = {}
    for name, series_id in indicators.items():
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&sort_order=desc&limit=1"
        try:
            resp = session.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if "observations" in data and len(data["observations"]) > 0:
                    obs = data["observations"][0]
                    macro_context[name] = {"date": obs["date"], "value": obs["value"]}
        except Exception as e:
            print(f"Fehler FRED: {e}")
    return macro_context

def get_euro_macro_data() -> dict:
    euro_context = {}
    indicators = {
        "EZB Leitzins": "ECB/FM/D.U2.EUR.4F.KR.MRR_RT.LEV",
        "EU Arbeitslosenquote": "Eurostat/une_rt_m/M.SA.TOTAL.PC_ACT.EA20",
        "EU Inflation (HICP)": "Eurostat/prc_hicp_manr/M.RCH_A.CP00.EA20"
    }
    try:
        for name, series_id in indicators.items():
            df = fetch_series(series_id)
            if df is not None and not df.empty:
                df_valid = df.dropna(subset=['value'])
                if not df_valid.empty:
                    latest = df_valid.iloc[-1]
                    date_str = latest['period'].strftime('%Y-%m') if hasattr(latest['period'], 'strftime') else str(latest['period'])
                    euro_context[name] = {"date": date_str, "value": round(latest['value'], 2)}
    except Exception as e:
        print(f"Fehler DBnomics: {e}")
    return euro_context

def get_sec_filings(ticker: str, email: str) -> list[dict]:
    if not email: return []
    clean_ticker = ticker.split(".")[0].upper()
    session = get_requests_session()
    headers = {"User-Agent": f"FinAITerminal/1.0 ({email})"}
    try:
        ticker_map_url = "https://www.sec.gov/files/company_tickers.json"
        resp = session.get(ticker_map_url, headers=headers, timeout=5)
        if resp.status_code != 200: return []
        
        ticker_data = resp.json()
        cik = None
        for item in ticker_data.values():
            if item["ticker"] == clean_ticker:
                cik = str(item["cik_str"]).zfill(10)
                break
                
        if not cik: return []
        
        sec_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        sec_resp = session.get(sec_url, headers=headers, timeout=5)
        if sec_resp.status_code != 200: return []
        
        filings_data = sec_resp.json().get("filings", {}).get("recent", {})
        if not filings_data: return []
        
        extracted_filings = []
        WICHTIGE_FORMS = ["10-K", "10-Q", "8-K"]
        for i in range(len(filings_data.get("form", []))):
            form_type = filings_data["form"][i]
            
            # URL Konstruktion
            # CIK ohne führende Nullen für den URL-Pfad
            url_cik = str(int(cik)) 
            # Accession Number ohne Bindestriche
            acc_no = filings_data["accessionNumber"][i]
            acc_no_clean = acc_no.replace("-", "")
            prim_doc = filings_data["primaryDocument"][i]
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{url_cik}/{acc_no_clean}/{prim_doc}"

            if form_type in WICHTIGE_FORMS or len(extracted_filings) < 2:
                extracted_filings.append({
                    "form": form_type,
                    "filingDate": filings_data["filingDate"][i],
                    "reportDate": filings_data["reportDate"][i],
                    "description": filings_data["primaryDocDescription"][i] or filings_data["primaryDocument"][i],
                    "url": doc_url # HIER IST DAS NEUE FELD
                })
            if len(extracted_filings) >= 5: break
        return extracted_filings
    except Exception:
        return []

def get_finnhub_data(ticker: str, api_key: str) -> dict:
    if not api_key: return {}
    clean_ticker = ticker.split(".")[0].upper()
    session = get_requests_session()
    results = {"recommendations": None, "insider": None}
    
    try:
        rec_url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={clean_ticker}&token={api_key}"
        rec_resp = session.get(rec_url, timeout=5)
        if rec_resp.status_code == 200:
            rec_data = rec_resp.json()
            if rec_data and len(rec_data) > 0:
                results["recommendations"] = rec_data[0] 
    except Exception:
        pass
        
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    start_date = (datetime.date.today() - datetime.timedelta(days=180)).strftime('%Y-%m-%d')
    try:
        ins_url = f"https://finnhub.io/api/v1/stock/insider-sentiment?symbol={clean_ticker}&from={start_date}&to={end_date}&token={api_key}"
        ins_resp = session.get(ins_url, timeout=5)
        if ins_resp.status_code == 200:
            ins_data = ins_resp.json()
            if ins_data.get("data") and len(ins_data["data"]) > 0:
                df = pd.DataFrame(ins_data["data"])
                results["insider"] = {
                    "mspr": round(df["mspr"].mean(), 2), 
                    "change": df["change"].sum()
                }
    except Exception:
        pass
        
    return results

def calculate_metrics(hist: pd.DataFrame, info: dict) -> dict:
    if hist.empty: return {}
    close_price = hist["Close"].iloc[-1]
    reference = hist["Close"].iloc[-2] if len(hist) >= 2 else hist["Open"].iloc[-1]
    ref_label = "Vortag" if len(hist) >= 2 else "Handelsbeginn"
    change_pct = ((close_price - reference) / reference) * 100 if reference else 0.0
    
    hist['SMA20'] = hist['Close'].rolling(window=20).mean()
    sma20_latest = hist['SMA20'].iloc[-1] if len(hist) >= 20 else None
    trend_signal = "N/A"
    if sma20_latest:
        trend_signal = "BULLISH" if close_price > sma20_latest else "BEARISH"
        
    log_returns = np.log(hist['Close'] / hist['Close'].shift(1))
    volatility = log_returns.tail(20).std() * (252 ** 0.5) * 100
    vol_label = "Hoch" if volatility > 30 else ("Moderat" if volatility > 15 else "Niedrig")

    div_rate = info.get("dividendRate") or info.get("trailingAnnualDividendRate")
    dividend_yield = f"{(div_rate / close_price) * 100:.2f}%" if div_rate and close_price > 0 else "N/A"
    
    return {
        "close": close_price, 
        "change_pct": change_pct, 
        "ref_label": ref_label, 
        "market_cap": format_large_number(info.get("marketCap")), 
        "pe_ratio": round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else "N/A", 
        "dividend_yield": dividend_yield,
        "trend_signal": trend_signal,
        "volatility": f"{volatility:.2f}% ({vol_label})" if not pd.isna(volatility) else "N/A"
    }

# --- KI LOGIK ---
def filter_news_with_ai(client: genai.Client, ticker: str, user_input: str, raw_news: list[dict]) -> list[str]:
    BLOCKED_PUBLISHERS = ["motley fool", "zacks", "seeking alpha", "investorplace", "tipranks", "thestreet", "kiplinger", "barron's"]
    CLICKBAIT_KEYWORDS = ["vs", "better buy", "stock to buy", "should you", "buy right now", "dividend stock", "is it too late", "buy or sell"]
    
    pre_filtered_news = []
    for n in raw_news:
        pub = n.get("publisher", n.get("provider", "Unbekannt")).lower()
        title = n.get("content", {}).get("title", n.get("title", "Kein Titel"))
        if any(b in pub for b in BLOCKED_PUBLISHERS) or any(k in title.lower() for k in CLICKBAIT_KEYWORDS): continue
        pre_filtered_news.append(f"[{pub.title()}] {title}")
        if len(pre_filtered_news) >= 15: break
        
    if not pre_filtered_news: return []

    prompt = f"Schlagzeilen für '{user_input}' ({ticker}):\n{chr(10).join(f'- {t}' for t in pre_filtered_news)}\nBehalte NUR harte Fakten. Liste ohne Herausgeber. Analysiere das Sentiment (BULLISH/BEARISH/NEUTRAL) für jede Schlagzeile. Falls nichts relevant: []."
    try:
        response = client.models.generate_content(
            model=MODEL_NAME, 
            contents=prompt, 
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=FilteredHeadlines, temperature=0.0)
        )
        formatted_news = []
        for item in response.parsed.relevant_headlines[:5]:
            emoji = "🟢" if item.sentiment.upper() == "BULLISH" else "🔴" if item.sentiment.upper() == "BEARISH" else "⚪"
            formatted_news.append(f"{emoji} [{item.sentiment.upper()}] {item.headline}")
        return formatted_news
    except: 
        return []

def generate_analysis(client: genai.Client, ticker: str, metrics: dict, news_list: list[str], macro_data: dict = None, euro_macro_data: dict = None, sec_filings: list[dict] = None, finnhub_data: dict = None) -> str:
    news_text = "\n".join([f"- {t}" for t in news_list]) if news_list else "KEINE RELEVANTEN NACHRICHTEN."
    direction = "gestiegen" if metrics["change_pct"] >= 0 else "gefallen"
    
    is_european = ".DE" in ticker.upper() or ".F" in ticker.upper() or ".PA" in ticker.upper()

    macro_text = ""
    if is_european:
        macro_text = "\n=== MAKROÖKONOMISCHES UMFELD FÜR EUROPA ===\n"
        if euro_macro_data:
            for key, val in euro_macro_data.items(): macro_text += f"- {key}: {val['value']}% (Update: {val['date']})\n"
        macro_text += "\nANWEISUNG: Analysiere nur dieses europäische Framework. Erwähne US-Zinsen nicht.\n"
    else:
        if macro_data:
            macro_text = "\n=== AKTUELLES MAKROÖKONOMISCHES UMFELD (USA) ===\n"
            for key, val in macro_data.items(): macro_text += f"- {key}: {val['value']}% (Update: {val['date']})\n"
            macro_text += "\nANWEISUNG: Analysiere, wie dieses US-Wirtschaftsumfeld die Bewertung beeinflusst.\n"

    sec_text = ""
    if sec_filings and not is_european:
        sec_text = "\n=== JÜNGSTE OFFIZIELLE SEC FILINGS ===\n"
        for f in sec_filings: sec_text += f"- Form {f['form']} am {f['filingDate']} (Betreff: {f['description']})\n"

    finnhub_text = ""
    if finnhub_data:
        rec = finnhub_data.get("recommendations")
        ins = finnhub_data.get("insider")
        if rec or ins:
            finnhub_text = "\n=== FINNHUB ANALYSTEN & INSIDER DATEN ===\n"
            if rec:
                finnhub_text += f"- Analysten-Konsens: {rec.get('strongBuy', 0) + rec.get('buy', 0)} Buy, {rec.get('hold', 0)} Hold, {rec.get('sell', 0) + rec.get('strongSell', 0)} Sell\n"
            if ins:
                sentiment_str = "Positiv (Insider kaufen)" if ins['mspr'] > 0 else "Negativ (Insider verkaufen)" if ins['mspr'] < 0 else "Neutral"
                finnhub_text += f"- Insider Sentiment (letzte 6 Monate): {sentiment_str} (MSPR Score: {ins['mspr']})\n"
            finnhub_text += "\nANWEISUNG: Beziehe diese Analysten-Ratings und das Verhalten der Insider zwingend in deine fundamentale Einordnung ein.\n"

    prompt = (
        f"Aktie {ticker} ist zum {metrics['ref_label']} um {abs(metrics['change_pct']):.2f}% {direction} (Letzter Kurs: {metrics['close']:.2f}).\n"
        f"Technischer Trend (SMA20): {metrics['trend_signal']} | Volatilität (20 Tage): {metrics['volatility']}\n"
        f"News mit Sentiment-Rating:\n{news_text}\n"
        f"{sec_text}\n"
        f"{finnhub_text}\n"
        f"{macro_text}\n"
        "Erkläre präzise WARUM der Kurs sich heute bewegt.\n"
        "WICHTIG: Beginne DIREKT mit der ersten Überschrift. Schreibe KEINE Einleitungssätze. Nutze zwingend diese exakte Struktur:\n"
        "### 🗞️ Was bewegt den Kurs?\n"
        "### 📊 Fundamentale Einordnung\n"
        "### 🏢 Sektor-Kontext\n"
        "### 🌍 Makro-Kontext"
    )
    try: 
        return client.models.generate_content(
            model=MODEL_NAME, 
            contents=prompt, 
            config=types.GenerateContentConfig(temperature=0.2)
        ).text
    except Exception as e: 
        return f"⚠️ Fehler: {e}"