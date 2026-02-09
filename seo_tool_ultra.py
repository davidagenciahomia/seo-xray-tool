import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from collections import Counter
import trafilatura
import urllib3
from datetime import datetime
import json
from textstat import flesch_reading_ease
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from urllib.parse import urlparse

# ==========================================
# 🔧 CONFIGURACIÓN INICIAL
# ==========================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(
    page_title="SERP X-RAY 360™ PRO", 
    layout="wide", 
    page_icon="🎯",
    initial_sidebar_state="expanded"
)

# ==========================================
# 🎨 CSS PERSONALIZADO - DISEÑO PROFESIONAL
# ==========================================
st.markdown("""
<style>
    /* Importar fuente profesional */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    /* Variables de color profesionales */
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --secondary: #8b5cf6;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --dark: #0f172a;
        --dark-light: #1e293b;
        --text: #f1f5f9;
        --text-muted: #94a3b8;
    }
    
    /* Fondo general */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar profesional */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: var(--text);
    }
    
    /* Títulos principales */
    h1 {
        font-weight: 800 !important;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
        margin-bottom: 1rem !important;
        letter-spacing: -0.02em;
    }
    
    h2 {
        color: var(--text) !important;
        font-weight: 700 !important;
        font-size: 1.75rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        color: var(--text) !important;
        font-weight: 600 !important;
        font-size: 1.25rem !important;
    }
    
    /* Métricas (cards superiores) */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stMetric"] label {
        color: var(--text-muted) !important;
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Tabs profesionales */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 41, 59, 0.5);
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background: transparent;
        border-radius: 8px;
        color: var(--text-muted);
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(99, 102, 241, 0.1);
        color: var(--text);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    
    /* Botones premium */
    .stButton button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.02em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        text-transform: uppercase;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);
    }
    
    /* Expanders profesionales */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 10px;
        padding: 1rem 1.5rem;
        font-weight: 600;
        color: var(--text);
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    .streamlit-expanderContent {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(99, 102, 241, 0.1);
        border-radius: 0 0 10px 10px;
        padding: 1.5rem;
    }
    
    /* DataFrames */
    .dataframe {
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 1rem !important;
        border: none !important;
    }
    
    .dataframe tbody tr {
        background: rgba(30, 41, 59, 0.5) !important;
        transition: all 0.2s ease;
    }
    
    .dataframe tbody tr:hover {
        background: rgba(99, 102, 241, 0.1) !important;
    }
    
    /* Info/Success/Warning boxes */
    .stAlert {
        border-radius: 10px;
        border: 1px solid;
        padding: 1rem 1.5rem;
        backdrop-filter: blur(10px);
    }
    
    [data-baseweb="notification"] {
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    /* Inputs */
    .stTextInput input, .stSelectbox select, .stSlider {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        padding: 0.75rem !important;
    }
    
    .stTextInput input:focus, .stSelectbox select:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        border-radius: 10px;
    }
    
    /* Dividers elegantes */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(99, 102, 241, 0.3) 50%, transparent 100%);
        margin: 2rem 0;
    }
    
    /* Status container */
    [data-testid="stStatus"] {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    /* Code blocks */
    code {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 6px !important;
        padding: 0.25rem 0.5rem !important;
        color: #a5b4fc !important;
    }
    
    /* Charts */
    .stPlotlyChart {
        border-radius: 12px;
        padding: 1rem;
        background: rgba(30, 41, 59, 0.3);
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    /* Scrollbar personalizado */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-dark);
    }
    
    /* Animaciones suaves */
    * {
        transition: background-color 0.3s ease, border-color 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🔑 API KEYS
# ==========================================
SERPER_API_KEY = st.secrets.get("SERPER_API_KEY", "cb2f958314203c51f8835f854b6a5038df46fb11")

# ==========================================
# 📋 LISTAS DE FILTRADO
# ==========================================
DOMINIOS_EXCLUIDOS = [
    "youtube.com", "facebook.com", "instagram.com", "twitter.com", 
    "tiktok.com", "pinterest", "linkedin", "amazon", "ebay", "aliexpress", 
    "mercadolibre", "wallapop", "milanuncios", "pccomponentes", "mediamarkt"
]

H2_BLACKLIST = [
    "suscríbete", "newsletter", "contacto", "ayuda", "política", "privacidad", 
    "cookies", "derechos", "copyright", "menú", "buscar", "categorías", 
    "enlaces", "siguenos", "redes", "login", "registro", "carrito", "cesta",
    "productos relacionados", "te puede interesar", "deja un comentario"
]

FAQ_BLACKLIST = [
    "cookie", "política", "privacidad", "aceptar", "configurar", "derechos", 
    "boletín", "suscripción", "créditos", "copyright", "iniciar sesión"
]

STOPWORDS = set([
    "de", "la", "que", "el", "en", "y", "a", "los", "se", "del", "las", "un", 
    "por", "con", "no", "una", "su", "para", "es", "al", "lo", "como", "mas", 
    "más", "o", "pero", "sus", "le", "ha", "me", "si", "sin", "sobre", "este", 
    "ya", "todo", "esta", "entre", "cuando", "muy", "años", "ser", "nos"
])

# ==========================================
# 🧠 ESTADO DE SESIÓN
# ==========================================
if 'data_seo' not in st.session_state: st.session_state.data_seo = None
if 'global_words' not in st.session_state: st.session_state.global_words = []
if 'gap_analysis' not in st.session_state: st.session_state.gap_analysis = None
if 'clusters' not in st.session_state: st.session_state.clusters = None

# ==========================================
# 🔍 FUNCIONES CORE (SIN CAMBIOS)
# ==========================================
def get_serper_results(query, n_needed):
    url = "https://google.serper.dev/search"
    n_safe = min(n_needed, 100)
    payload = {"q": query, "num": n_safe, "gl": "es", "hl": "es"}
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, json=payload)
        return response.json().get('organic', [])
    except: return []

def clean_h2s(soup_h2s):
    valid_h2s = []
    for h in soup_h2s:
        text = h.get_text(" ", strip=True)
        if len(text) < 5: continue
        if any(bad in text.lower() for bad in H2_BLACKLIST): continue
        valid_h2s.append(text)
    return valid_h2s

def get_ngrams(words, n):
    return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]

def detectar_intencion(soup, text):
    text_lower = text.lower()
    title_lower = soup.title.string.lower() if soup.title else ""
    transaccional = ["precio", "comprar", "venta", "carrito", "oferta", "barato", "tienda", "envío", "stock", "€", "$"]
    informacional = ["guía", "tutorial", "opinión", "review", "qué es", "cómo", "cuándo", "consejos", "mejores", "comparativa"]
    score_trans = sum(1 for w in transaccional if w in text_lower or w in title_lower)
    score_info = sum(1 for w in informacional if w in text_lower or w in title_lower)
    if score_trans > score_info: return "🛒 Transaccional"
    if score_info > score_trans: return "📚 Informacional"
    return "⚖️ Mixto"

def extraer_fecha(soup):
    date = "N/A"
    meta_date = soup.find("meta", attrs={"property": "article:published_time"}) or \
                soup.find("meta", attrs={"name": "date"})
    if meta_date:
        try: date = meta_date['content'][:10]
        except: pass
    return date

def extraer_preguntas_validas(text):
    preguntas_raw = re.findall(r'[¿][^?]+[?]', text)
    if not preguntas_raw:
        patrones = [r'(?:qué|cómo|cuándo|dónde|por qué|cuánto)\s+\w+\s+\w+[^,.:;]+']
        for p in patrones:
            match = re.findall(p, text.lower(), re.IGNORECASE)
            preguntas_raw.extend(match)
    preguntas_limpias = []
    for p in preguntas_raw:
        p_clean = p.strip().capitalize()
        if len(p_clean) < 15 or len(p_clean) > 150: continue
        if any(bad in p_clean.lower() for bad in FAQ_BLACKLIST): continue
        if not p_clean.endswith('?'): p_clean += '?'
        preguntas_limpias.append(p_clean)
    return list(set(preguntas_limpias))[:5]

def analizar_enlaces_internos(soup, url):
    try:
        domain = urlparse(url).netloc
        links = soup.find_all('a', href=True)
        internal = [l for l in links if domain in l['href'] or l['href'].startswith('/')]
        return len(internal)
    except: return 0

def detectar_schema_markup(soup):
    schemas = []
    scripts = soup.find_all('script', type='application/ld+json')
    for script in scripts:
        try:
            data = json.loads(script.string)
            if isinstance(data, dict) and '@type' in data:
                schemas.append(data['@type'])
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and '@type' in item:
                        schemas.append(item['@type'])
        except: pass
    return list(set(schemas))

def analizar_media_richness(soup):
    imagenes = len(soup.find_all('img'))
    videos = len(soup.find_all(['video', 'iframe']))
    return {"images": imagenes, "videos": videos, "total": imagenes + videos}

def calcular_readability(text):
    try:
        score = flesch_reading_ease(text)
        if score >= 80: return f"✅ Muy fácil ({score:.0f})"
        elif score >= 60: return f"👍 Fácil ({score:.0f})"
        elif score >= 40: return f"⚠️ Medio ({score:.0f})"
        else: return f"❌ Difícil ({score:.0f})"
    except: return "N/A"

def extraer_entidades(text):
    entidades = re.findall(r'\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*\b', text)
    entidades_filtradas = [e for e in entidades if e.lower() not in ['el', 'la', 'los', 'las']]
    return list(set(entidades_filtradas))[:10]

def estimar_domain_authority(url):
    domain = urlparse(url).netloc
    score = 50
    if domain.endswith('.edu'): score += 20
    elif domain.endswith('.gov'): score += 25
    elif domain.endswith('.org'): score += 10
    if domain.count('.') > 1: score -= 10
    if len(domain) < 15: score += 5
    return min(max(score, 0), 100)

def analyze_url_final(url, target_kw, snippet_serp):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }
    try:
        downloaded = trafilatura.fetch_url(url)
        main_text = None
        if downloaded:
            main_text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        soup = None
        if not main_text:
            res = requests.get(url, timeout=12, headers=headers, verify=False)
            if res.status_code in [403, 429, 503]: return {"error": f"Bloqueo ({res.status_code})"}
            soup = BeautifulSoup(res.text, 'html.parser')
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]): tag.decompose()
            main_text = soup.get_text(separator=' ', strip=True)
        else:
            soup = BeautifulSoup(downloaded, 'html.parser')
        if len(main_text) < 100: return {"error": "Contenido insuficiente"}
        title = soup.title.string.strip() if soup.title else "N/A"
        if title == "N/A" or len(title) < 3: return {"error": "Título no detectado"}
        meta_tag = soup.find("meta", attrs={"name": "description"})
        meta_desc = meta_tag["content"] if meta_tag else snippet_serp
        h1s = [h.get_text(" ", strip=True) for h in soup.find_all('h1')]
        h2s_clean = clean_h2s(soup.find_all('h2'))
        h3s_clean = [h.get_text(" ", strip=True) for h in soup.find_all('h3')][:10]
        clean_words = [w for w in re.sub(r'[^\w\s]', '', main_text.lower()).split() 
                      if w not in STOPWORDS and len(w) > 2]
        enlaces_internos = analizar_enlaces_internos(soup, url)
        schemas = detectar_schema_markup(soup)
        media = analizar_media_richness(soup)
        readability = calcular_readability(main_text)
        entidades = extraer_entidades(main_text)
        da_proxy = estimar_domain_authority(url)
        toc_detected = bool(soup.find('div', class_=re.compile('table.*content|toc', re.I)))
        return {
            "title": title, "meta_desc": meta_desc, "date": extraer_fecha(soup),
            "h1": h1s, "h2": h2s_clean, "h3": h3s_clean,
            "word_count": len(main_text.split()),
            "kw_density": len(re.findall(rf'\b{re.escape(target_kw)}\b', main_text, re.IGNORECASE)),
            "content_sample": main_text[:1000], "all_words": clean_words,
            "intencion": detectar_intencion(soup, main_text),
            "preguntas": extraer_preguntas_validas(main_text),
            "enlaces_internos": enlaces_internos, "schemas": schemas,
            "media": media, "readability": readability, "entidades": entidades,
            "da_proxy": da_proxy, "toc": toc_detected, "full_text": main_text
        }
    except Exception as e:
        return {"error": str(e)}

def calcular_gap_analysis(data_list, keyword):
    all_h2s = []
    for item in data_list:
        all_h2s.extend(item.get('h2', []))
    h2_counter = Counter([h.lower() for h in all_h2s])
    threshold = len(data_list) * 0.5
    h2s_comunes = {h: count for h, count in h2_counter.items() if count >= threshold}
    all_words = []
    for item in data_list:
        all_words.extend(item.get('all_words', []))
    word_counter = Counter(all_words)
    top_words = word_counter.most_common(30)
    all_faqs = []
    for item in data_list:
        all_faqs.extend(item.get('preguntas', []))
    faq_counter = Counter([q.lower() for q in all_faqs])
    return {
        "h2s_criticos": h2s_comunes,
        "palabras_clave_secundarias": top_words,
        "faqs_comunes": faq_counter.most_common(10),
        "coverage_benchmark": {
            "h2_avg": np.mean([len(item.get('h2', [])) for item in data_list]),
            "word_avg": np.mean([item.get('word_count', 0) for item in data_list]),
            "media_avg": np.mean([item.get('media', {}).get('total', 0) for item in data_list])
        }
    }

def realizar_clustering(data_list):
    if len(data_list) < 3: return None
    corpus = []
    urls = []
    for item in data_list:
        text = " ".join(item.get('h2', [])) + " " + " ".join(item.get('all_words', [])[:100])
        corpus.append(text)
        urls.append(item.get('URL', ''))
    try:
        vectorizer = TfidfVectorizer(max_features=50, stop_words=list(STOPWORDS))
        X = vectorizer.fit_transform(corpus)
        n_clusters = min(3, len(data_list))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        cluster_groups = {}
        for idx, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append({
                "url": urls[idx],
                "title": data_list[idx].get('title', 'N/A')
            })
        return cluster_groups
    except: return None

# ==========================================
# 🎨 SIDEBAR CON NUEVO DISEÑO
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0 1rem 0;'>
        <div style='font-size: 3rem; margin-bottom: 0.5rem;'>🎯</div>
        <h1 style='font-size: 1.75rem; margin: 0; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800;'>SERP X-RAY 360™</h1>
        <p style='color: #94a3b8; font-size: 0.875rem; margin-top: 0.5rem; font-weight: 600;'>PROFESSIONAL EDITION</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("#### 🎯 Configuración de Análisis")
    keyword = st.text_input("Keyword Principal:", value="que es el repep", help="Introduce la keyword que quieres analizar")
    num_target = st.slider("Competidores a analizar:", 3, 10, 5, help="Más competidores = análisis más profundo")
    
    st.markdown("---")
    st.markdown("#### ⚙️ Opciones Avanzadas")
    debug_mode = st.checkbox("🛠️ Modo Debug", value=False, help="Muestra errores detallados")
    
    st.markdown("---")
    analyze_button = st.button("🚀 INICIAR ANÁLISIS", use_container_width=True, type="primary")
    
    if st.button("🗑️ Limpiar Caché", use_container_width=True):
        st.session_state.data_seo = None
        st.session_state.gap_analysis = None
        st.session_state.clusters = None
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    <div style='padding: 1rem; background: rgba(99, 102, 241, 0.1); border-radius: 10px; border: 1px solid rgba(99, 102, 241, 0.2);'>
        <p style='color: #94a3b8; font-size: 0.75rem; margin: 0; text-align: center;'>
            <strong>v12.0 PRO</strong><br>
            Powered by SERP Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 🎯 HEADER PRINCIPAL
# ==========================================
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>🎯 SERP X-RAY 360™ PRO</h1>
    <p style='color: #94a3b8; font-size: 1.25rem; font-weight: 600;'>Inteligencia Competitiva SEO de Nueva Generación</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 🎯 LÓGICA PRINCIPAL DE EJECUCIÓN
# ==========================================
if analyze_button and keyword:
    if len(SERPER_API_KEY) < 20:
        st.error("⚠️ ERROR: Configura tu API Key de Serper")
    else:
        with st.status("🔄 Analizando competidores...", expanded=True) as status:
            buffer_size = 40
            raw_results = get_serper_results(keyword, num_target + buffer_size)
            final_data = []
            global_words = []
            count_valid = 0
            
            if not raw_results:
                st.error("❌ Error de conexión con Serper API")
            else:
                progress_bar = st.progress(0)
                
                for i, res in enumerate(raw_results):
                    if count_valid >= num_target: break
                    url = res['link']
                    if any(bad in url for bad in DOMINIOS_EXCLUIDOS): continue

                    status.update(label=f"🔍 Analizando #{count_valid+1}: {res.get('title', 'Web')[:30]}...")
                    data = analyze_url_final(url, keyword, res.get('snippet', ''))
                    
                    if data and "error" not in data:
                        global_words.extend(data['all_words'])
                        final_data.append({
                            "Pos": count_valid + 1, "URL": url, "Título": data['title'],
                            "Fecha": data['date'], "Meta Desc": data['meta_desc'],
                            "Palabras": data['word_count'], "Menciones KW": data['kw_density'],
                            "H1_list": data['h1'], "H2_list": data['h2'], "H3_list": data['h3'],
                            "Contenido": data['content_sample'], "Intención": data['intencion'],
                            "Preguntas": data['preguntas'], "Enlaces_Int": data['enlaces_internos'],
                            "Schemas": data['schemas'], "Media": data['media'],
                            "Readability": data['readability'], "Entidades": data['entidades'],
                            "DA_Proxy": data['da_proxy'], "TOC": data['toc'],
                            "full_text": data['full_text'], **data
                        })
                        count_valid += 1
                        progress_bar.progress(count_valid / num_target)
                    elif debug_mode and data:
                        st.write(f"❌ {url} -> {data['error']}")
                
                if not final_data:
                    st.error("❌ No se encontraron webs válidas")
                    st.session_state.data_seo = None
                else:
                    st.session_state.data_seo = final_data
                    st.session_state.global_words = global_words
                    st.session_state.gap_analysis = calcular_gap_analysis(final_data, keyword)
                    st.session_state.clusters = realizar_clustering(final_data)
                    status.update(label="✅ Análisis Completado", state="complete")
                    st.balloons()

# ==========================================
# 📊 VISUALIZACIÓN DE RESULTADOS
# ==========================================
if st.session_state.data_seo:
    data = st.session_state.data_seo
    df = pd.DataFrame(data)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview General", 
        "🎯 Gap Analysis", 
        "🧬 Topic Clustering", 
        "📋 Análisis Detallado",
        "💾 Exportar Datos"
    ])
    
    with tab1:
        st.markdown("### 📊 Métricas Clave del Análisis")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("📝 Media Palabras", int(df['Palabras'].mean()))
        c2.metric("🎯 Intención", df['Intención'].mode()[0] if not df['Intención'].empty else "N/A")
        c3.metric("🔗 Links Internos", int(df['Enlaces_Int'].mean()))
        c4.metric("🖼️ Media Avg", f"{df['Media'].apply(lambda x: x.get('total', 0)).mean():.1f}")
        c5.metric("📊 DA Promedio", int(df['DA_Proxy'].mean()))
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔥 Top Bigramas Semánticos")
            bigrams = get_ngrams(st.session_state.global_words, 2)
            if bigrams:
                chart_data = pd.DataFrame(Counter(bigrams).most_common(12), columns=['Frase', 'Freq']).set_index('Frase')
                st.bar_chart(chart_data, color="#6366f1")
        
        with col2:
            st.markdown("#### 📈 Distribución Word Count")
            st.bar_chart(df[['Pos', 'Palabras']].set_index('Pos'), color="#8b5cf6")
        
        st.markdown("---")
        st.markdown("#### 🏗️ Análisis Schema Markup")
        schema_stats = {}
        for item in data:
            for schema in item['Schemas']:
                schema_stats[schema] = schema_stats.get(schema, 0) + 1
        if schema_stats:
            schema_df = pd.DataFrame(list(schema_stats.items()), columns=['Schema Type', 'Count']).sort_values('Count', ascending=False)
            st.dataframe(schema_df, use_container_width=True)
        else:
            st.info("ℹ️ No se detectaron schemas en los competidores")
    
    with tab2:
        st.markdown("### 🎯 Análisis de Brechas Competitivas")
        if st.session_state.gap_analysis:
            gap = st.session_state.gap_analysis
            st.markdown("#### 🔴 H2s Críticos (Aparecen en >50% competidores)")
            if gap['h2s_criticos']:
                h2_df = pd.DataFrame([(h, count) for h, count in gap['h2s_criticos'].items()],
                                    columns=['H2', 'Apariciones']).sort_values('Apariciones', ascending=False)
                st.dataframe(h2_df, use_container_width=True)
                st.success("💡 **Recomendación:** Incluye TODOS estos H2s en tu contenido")
            else:
                st.warning("No hay H2s que se repitan consistentemente")
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📝 Top Palabras Clave Secundarias")
                words_df = pd.DataFrame(gap['palabras_clave_secundarias'][:15], columns=['Palabra', 'Frecuencia'])
                st.dataframe(words_df, use_container_width=True)
            with col2:
                st.markdown("#### ❓ FAQs Más Comunes")
                if gap['faqs_comunes']:
                    faq_df = pd.DataFrame(gap['faqs_comunes'], columns=['Pregunta', 'Apariciones'])
                    st.dataframe(faq_df, use_container_width=True)
                else:
                    st.info("Sin FAQs repetidas")
            
            st.markdown("---")
            st.markdown("#### 📊 Benchmarks de Competencia")
            bench = gap['coverage_benchmark']
            c1, c2, c3 = st.columns(3)
            c1.metric("📌 H2s Promedio", f"{bench['h2_avg']:.1f}")
            c2.metric("📝 Palabras Promedio", f"{bench['word_avg']:.0f}")
            c3.metric("🖼️ Media Promedio", f"{bench['media_avg']:.1f}")
    
    with tab3:
        st.markdown("### 🧬 Topic Clustering Semántico")
        if st.session_state.clusters:
            clusters = st.session_state.clusters
            st.info(f"📦 Se identificaron {len(clusters)} grupos temáticos distintos")
            for cluster_id, urls in clusters.items():
                with st.expander(f"📦 Cluster {cluster_id + 1} ({len(urls)} URLs)", expanded=True):
                    for item in urls:
                        st.markdown(f"• [{item['title'][:70]}...]({item['url']})")
        else:
            st.warning("Se necesitan al menos 3 URLs para clustering")
    
    with tab4:
        st.markdown("### 📋 Análisis Detallado por Competidor")
        for item in data:
            titulo_corto = (item['Título'][:70] + '..') if len(item['Título']) > 70 else item['Título']
            with st.expander(f"#{item['Pos']} | {titulo_corto}", expanded=False):
                c_badges = st.columns(5)
                if "Transaccional" in item['Intención']: 
                    c_badges[0].markdown(f"🛒 {item['Intención']}")
                else: 
                    c_badges[0].markdown(f"📚 {item['Intención']}")
                if item['Fecha'] != "N/A": 
                    c_badges[1].caption(f"📅 {item['Fecha']}")
                da_color = "🟢" if item['DA_Proxy'] >= 60 else "🟡" if item['DA_Proxy'] >= 40 else "🔴"
                c_badges[2].caption(f"{da_color} DA: {item['DA_Proxy']}")
                c_badges[3].caption(f"📖 {item['Readability']}")
                if item['TOC']:
                    c_badges[4].caption("✅ TOC")
                st.caption(f"🔗 {item['URL']}")
                st.info(f"**Meta:** {item['Meta Desc']}")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### 🏗️ Estructura")
                    st.markdown(f"**📊 {item['Palabras']} palabras** | **🎯 {item['Menciones KW']} menciones KW**")
                    st.markdown(f"**🔗 {item['Enlaces_Int']} enlaces internos**")
                    if item['H1_list']: 
                        st.markdown(f"**H1:** {item['H1_list'][0]}")
                    if item['H2_list']:
                        st.markdown(f"**H2s ({len(item['H2_list'])}):**")
                        for h in item['H2_list'][:8]: 
                            st.markdown(f"• {h}")
                with c2:
                    st.markdown("#### 🎨 Multimedia & Técnico")
                    media = item['Media']
                    st.markdown(f"**🖼️ Imágenes:** {media['images']}")
                    st.markdown(f"**🎥 Videos:** {media['videos']}")
                    if item['Schemas']:
                        st.markdown(f"**🏗️ Schemas:** {', '.join(item['Schemas'])}")
                    if item['Entidades']:
                        st.markdown(f"**🏷️ Entidades:** {', '.join(item['Entidades'][:5])}")
                st.markdown("#### ❓ FAQs")
                if item['Preguntas']:
                    faq_text = "\n".join([f"- {p}" for p in item['Preguntas']])
                    st.code(faq_text, language="text")
    
    with tab5:
        st.markdown("### 💾 Exportar Análisis")
        export_df = df[['Pos', 'URL', 'Título', 'Fecha', 'Palabras', 'Menciones KW',
                        'Intención', 'Enlaces_Int', 'DA_Proxy', 'Readability', 'TOC']].copy()
        export_df['Schemas'] = df['Schemas'].apply(lambda x: ', '.join(x) if x else 'None')
        export_df['Media_Total'] = df['Media'].apply(lambda x: x.get('total', 0))
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar Análisis CSV", csv, 
                          f"SERP_Analysis_{keyword.replace(' ', '_')}.csv",
                          "text/csv", type="primary", use_container_width=True)

# ==========================================
# 🎯 FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem 0; color: #94a3b8;'>
    <p style='margin: 0; font-size: 0.875rem;'><strong>SERP X-RAY 360™ PRO</strong> v12.0 | Professional Edition</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.75rem;'>Powered by Advanced SERP Intelligence © 2024</p>
</div>
""", unsafe_allow_html=True)
