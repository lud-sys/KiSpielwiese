import os
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from core_logic import (
    get_ticker_from_name,
    load_stock_data,
    calculate_metrics,
    filter_news_with_ai,
    generate_analysis,
    get_macro_data,
    get_euro_macro_data,
    get_sec_filings,
    get_finnhub_data
)

st.set_page_config(page_title="TrueFin Terminal", layout="wide", page_icon="📈")

# ==========================================
# CACHING & SETUP
# ==========================================
def load_api_keys() -> tuple[str, str, str, str]:
    gemini_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    fred_key = os.environ.get("FRED_API_KEY") or st.secrets.get("FRED_API_KEY")
    sec_email = os.environ.get("SEC_API_EMAIL") or st.secrets.get("SEC_API_EMAIL")
    finnhub_key = os.environ.get("FINNHUB_API_KEY") or st.secrets.get("FINNHUB_API_KEY")
    
    if not gemini_key or not finnhub_key:
        st.error("🚨 Gemini-API-Key ODER Finnhub-API-Key fehlt! Bitte in den Secrets hinterlegen.")
        st.stop()
    return gemini_key, fred_key, sec_email, finnhub_key

@st.cache_data(ttl=3600, show_spinner=False)
def cached_get_ticker(query: str, api_key: str):
    return get_ticker_from_name(query, api_key)

@st.cache_data(ttl=900, show_spinner=False)
def cached_load_data(ticker: str, api_key: str):
    return load_stock_data(ticker, api_key)

@st.cache_data(ttl=43200, show_spinner=False)
def cached_macro_data(api_key: str):
    return get_macro_data(api_key)

@st.cache_data(ttl=43200, show_spinner=False)
def cached_euro_macro_data_v3():
    return get_euro_macro_data()

@st.cache_data(ttl=3600, show_spinner=False)
def cached_sec_filings(ticker: str, email: str):
    return get_sec_filings(ticker, email)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_finnhub_data(ticker: str, api_key: str):
    return get_finnhub_data(ticker, api_key)

# ==========================================
# UI DESIGN (CSS)
# ==========================================
def inject_custom_css():
    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container { padding-top: 2rem !important; max-width: 1150px; }
    
    .feature-card {
        background-color: #1E2530;
        border: 1px solid #2D3748;
        border-radius: 12px;
        padding: 20px;
        transition: all 0.3s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    @media (min-width: 768px) {
        .feature-card {
            min-height: 250px;
        }
    }
    
    .feature-card:hover {
        border-color: #00FFA3;
        box-shadow: 0 4px 20px rgba(0, 255, 163, 0.05);
        transform: translateY(-2px);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
        border: 1px solid #2D3748 !important;
        background-color: #1E2530 !important; 
        padding: 12px !important; 
        transition: all 0.3s ease !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #00FFA3 !important; 
        box-shadow: 0 4px 20px rgba(0, 255, 163, 0.05) !important;
        transform: translateY(-2px);
    }

    button[kind="primary"] {
        border: none !important;
        transition: transform 0.2s ease !important;
    }
    button[kind="primary"] p {
        color: #0E1117 !important; 
        font-weight: 700 !important;
        font-size: 1.05rem !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-2px);
    }

    [data-testid="stSidebar"] h3 {
        font-size: 0.8rem !important;
        color: #A0AEC0 !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 0.5rem;
        padding-left: 5px;
    }

    [data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebar"] .stButton > button div[data-testid="stMarkdownContainer"] > p {
        text-align: left !important;
        width: 100% !important;
        margin: 0 !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
    }

    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background-color: transparent !important;
        color: #A0AEC0 !important; 
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        background-color: rgba(255, 255, 255, 0.05) !important; 
        color: #FAFAFA !important;
    }

    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: rgba(0, 255, 163, 0.1) !important;
        color: #00FFA3 !important; 
        border-left: 3px solid #00FFA3 !important; 
        border-radius: 0px 6px 6px 0px !important; 
        padding-left: 10px !important; 
    }
    </style>""", unsafe_allow_html=True)

# ==========================================
# UI KOMPONENTEN
# ==========================================
def render_dashboard(ticker: str, info: dict, metrics: dict, news_list: list[str], analysis: str, raw_news_count: int, macro_data: dict, euro_macro_data: dict, sec_filings: list[dict], finnhub_data: dict):
    company_name = info.get("shortName", ticker)
    st.divider()
    st.subheader(f"{company_name} ({ticker})")
    
    # 4 saubere KPIs in einer Reihe!
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    kpi_col1.metric("Aktueller Kurs", f"${metrics.get('close', 0):.2f}", f"{metrics.get('change_pct', 0):.2f}%")
    kpi_col2.metric("Marktkapitalisierung", metrics.get("market_cap", "N/A"))
    kpi_col3.metric("KGV (P/E)", metrics.get("pe_ratio", "N/A"))
    kpi_col4.metric("Dividendenrendite", metrics.get("dividend_yield", "N/A"))

    st.markdown("<br>", unsafe_allow_html=True)

    country = info.get("country", "")
    is_us_stock = country in ["US", "United States"] if country else "." not in ticker

    tab_names = ["🧠 KI-Analyse", "📈 Interaktiver Chart", "📰 Signal vs. Rauschen"]
    if is_us_stock:
        tab_names.append("📑 SEC Filings")
        
    tabs = st.tabs(tab_names)

    with tabs[0]:
        col_text, col_stats = st.columns([2.5, 1])
        with col_text: 
            st.markdown(analysis)
        with col_stats: 
            with st.container(border=True):
                st.markdown(f"**🛡️ KI-Filter aktiv**\n\nVon **{raw_news_count}** Artikeln wurden nur **{len(news_list)}** zugelassen.")
            
            if finnhub_data and (finnhub_data.get("recommendations") or finnhub_data.get("insider")):
                with st.container(border=True):
                    st.markdown("**Wall Street & Insider**")
                    rec = finnhub_data.get("recommendations")
                    if rec:
                        buys = rec.get('strongBuy', 0) + rec.get('buy', 0)
                        holds = rec.get('hold', 0)
                        sells = rec.get('sell', 0) + rec.get('strongSell', 0)
                        st.markdown(f"<span style='font-size:0.85em; color:#A0AEC0;'>Analysten-Konsens</span><br><b>🟢 {buys} | ⚪ {holds} | 🔴 {sells}</b>", unsafe_allow_html=True)
                    
                    st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
                    ins = finnhub_data.get("insider")
                    if ins:
                        mspr = ins['mspr']
                        color = "#00FFA3" if mspr > 0 else "#FF4B4B" if mspr < 0 else "#A0AEC0"
                        trend = "Netto-Käufe" if mspr > 0 else "Netto-Verkäufe" if mspr < 0 else "Neutral"
                        st.markdown(f"<span style='font-size:0.85em; color:#A0AEC0;'>Insider (letzte 6M)</span><br><b style='color:{color};'>{trend} (Score: {mspr})</b>", unsafe_allow_html=True)

            if macro_data or euro_macro_data:
                with st.container(border=True):
                    st.markdown("**🏦 Makro-Umfeld**")
                    mac_tab1, mac_tab2 = st.tabs(["🇺🇸 USA", "🇪🇺 Europa"])
                    with mac_tab1:
                        if macro_data:
                            for name, m_info in macro_data.items():
                                st.markdown(f"<span style='font-size:0.85em; color:#A0AEC0;'>{name}</span><br><b>{m_info['value']}%</b>", unsafe_allow_html=True)
                        else:
                            st.caption("Keine US-Daten geladen.")
                    with mac_tab2:
                        if euro_macro_data:
                            for name, m_info in euro_macro_data.items():
                                st.markdown(f"<span style='font-size:0.85em; color:#A0AEC0;'>{name}</span><br><b>{m_info['value']}%</b>", unsafe_allow_html=True)
                        else:
                            st.caption("Keine EU-Daten geladen.")

    with tabs[1]:
        components.html(f"""
        <div class="tradingview-widget-container" style="height:550px;width:100%">
            <div id="tv_chart_main" style="height:100%;width:100%"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
            <script type="text/javascript">
            new TradingView.widget({{"autosize": true, "symbol": "{ticker}", "interval": "D", "range": "3M", "timezone": "Europe/Berlin", "theme": "dark", "style": "2", "locale": "de_DE", "enable_publishing": false, "hide_top_toolbar": false, "hide_side_toolbar": false, "allow_symbol_change": true, "container_id": "tv_chart_main"}});
            </script>
        </div>
        """, height=550)

    with tabs[2]:
        if news_list:
            st.success("✅ Diese Fakten haben unseren Sentiment-Filter passiert:")
            for title in news_list: st.markdown(f"- **{title}**")
        else:
            st.warning("📭 Keine harten Fakten gefunden. Das Markt-Rauschen besteht aktuell nur aus Spekulationen.")

    if is_us_stock:
        with tabs[3]:
           if sec_filings:
            st.success(f"📂 Offizielle, marktrelevante SEC-Daten für {ticker} geladen:")
            for f in sec_filings:
                with st.container(border=True):
                    col_info, col_link = st.columns([3, 1])
                    with col_info:
                        st.markdown(f"### Formular: **{f['form']}**")
                        st.markdown(f"📅 **Filing Date:** {f['filingDate']} | 📊 **Period of Report:** {f['reportDate']}")
                        st.caption(f"Gegenstand: {f['description']}")
                    with col_link:
                        st.link_button("Zum Bericht", f['url'], use_container_width=True)
           else:
                st.warning("ℹ️ Keine wichtigen SEC-Daten gefunden oder Asset ist nicht an der US-Börse registriert.")

# ==========================================
# HAUPT-PROGRAMM (Routing & State)
# ==========================================
def main():
    inject_custom_css()
    
    gemini_key, fred_key, sec_email, finnhub_key = load_api_keys()
    client = genai.Client(api_key=gemini_key)

    macro_data = cached_macro_data(fred_key) if fred_key else {}
    euro_macro_data = cached_euro_macro_data_v3() 

    if "target_ticker" not in st.session_state: st.session_state.target_ticker = None
    if "search_input" not in st.session_state: st.session_state.search_input = ""
    if "search_history" not in st.session_state: st.session_state.search_history = []
    
    if "current_analysis_ticker" not in st.session_state: st.session_state.current_analysis_ticker = None
    if "current_dashboard_data" not in st.session_state: st.session_state.current_dashboard_data = {}

    def set_target(ticker):
        st.session_state.target_ticker = ticker
        if ticker:
            if ticker in st.session_state.search_history:
                st.session_state.search_history.remove(ticker)
            st.session_state.search_history.append(ticker)
            if len(st.session_state.search_history) > 5:
                st.session_state.search_history.pop(0)
                
    def handle_search():
        val = st.session_state.get("search_input", "").strip()
        set_target(val if val else None)

    with st.sidebar:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Navigation")
        
        is_home = st.session_state.target_ticker is None
        st.button("🏠 Startseite", use_container_width=True, type="primary" if is_home else "secondary", on_click=set_target, args=(None,))
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Historie")
        
        if not st.session_state.search_history:
            st.caption("Noch keine Analysen.")
        else:
            for past_ticker in reversed(st.session_state.search_history):
                is_active = (st.session_state.target_ticker == past_ticker)
                st.button(f"📊 {past_ticker}", key=f"hist_{past_ticker}", use_container_width=True, type="primary" if is_active else "secondary", on_click=set_target, args=(past_ticker,))
                
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("### ⚙️ System Status")
        st.caption("🟢 Gemini API aktiv")
        st.caption("🟢 FRED API (Makro US) aktiv" if (fred_key and macro_data) else "⚪ FRED API inaktiv")
        st.caption("🟢 DBnomics (Makro EU) aktiv" if euro_macro_data else "⚪ DBnomics API inaktiv")
        st.caption("🟢 SEC EDGAR Gate aktiv" if sec_email else "⚪ SEC EDGAR Gate unverschlüsselt")
        st.caption("🟢 Finnhub API aktiv" if finnhub_key else "⚪ Finnhub API inaktiv")

    st.markdown("""
        <h1 style='background: -webkit-linear-gradient(45deg, #00FFA3, #00B8FF); 
                   -webkit-background-clip: text; 
                   -webkit-text-fill-color: transparent; 
                   font-size: 3.5rem; 
                   font-weight: 800; 
                   letter-spacing: -1px;
                   margin-bottom: 0px;'>
            TrueFin Terminal
        </h1>
        <p style='color: #A0AEC0; font-size: 1.15rem; margin-top: 5px; margin-bottom: 25px;'>
            Institutionelle Aktienanalyse. Befreit von Marktrauschen und Clickbait.
        </p>
    """, unsafe_allow_html=True)

    search_col1, search_col2 = st.columns([4, 1])
    with search_col1:
        st.text_input("Suche", key="search_input", label_visibility="collapsed", placeholder="Unternehmen oder Ticker eingeben (z.B. NVIDIA, AAPL)...", on_change=handle_search)
    with search_col2:
        st.button("Analysieren", use_container_width=True, type="primary", on_click=handle_search)

    # ==========================================
    # ANSICHT: ANALYSE STARTEN
    # ==========================================
    if st.session_state.target_ticker:
        ticker_query = st.session_state.target_ticker
        
        current_data = st.session_state.current_dashboard_data
        needs_reload = not current_data or current_data.get("original_query") != ticker_query

        if needs_reload:
            loading_container = st.empty()
            success = False 
            actual_ticker = ticker_query 
            
            with loading_container.status(f"Analysiere Marktdaten für '{ticker_query}'...", expanded=True) as status:
                st.write("🔍 Identifiziere Ticker-Symbol...")
                actual_ticker = cached_get_ticker(ticker_query, finnhub_key) 
                
                st.write(f"📡 Lade Realtime-Daten für {actual_ticker}...")
                hist, raw_news, info = cached_load_data(actual_ticker, finnhub_key) 
                
                if hist.empty:
                    status.update(label="Fehler bei der Datenabfrage", state="error", expanded=True)
                    st.error(f"Keine Daten für '{actual_ticker}' gefunden. Bitte überprüfe die Schreibweise.")
                else:
                    country = info.get("country", "")
                    is_us_stock = country in ["US", "United States"] if country else "." not in actual_ticker
                    sec_filings = []
                    
                    if is_us_stock and sec_email:
                        st.write("📂 Scanne offizielle SEC Regulierungsberichte...")
                        sec_filings = cached_sec_filings(actual_ticker, sec_email)
                    
                    st.write("🕵️‍♂️ Frage Analysten & Insider-Daten ab...")
                    finnhub_data = cached_finnhub_data(actua