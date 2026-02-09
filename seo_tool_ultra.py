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
import time

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
# 🎨 CSS + SISTEMA DE TOOLTIPS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
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
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    h1 {
        font-weight: 800 !important;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    h2, h3 {
        color: var(--text) !important;
        font-weight: 700 !important;
    }
    
    /* TOOLTIPS PERSONALIZADOS */
    .help-tooltip {
        display: inline-block;
        width: 18px;
        height: 18px;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 18px;
        font-size: 12px;
        font-weight: bold;
        cursor: help;
        margin-left: 5px;
        vertical-align: middle;
    }
    
    /* CAJAS DE EXPLICACIÓN */
    .explainer-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border-left: 4px solid var(--primary);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    .explainer-box strong {
        color: var(--primary);
    }
    
    /* SEMÁFOROS DE CALIDAD */
    .quality-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .quality-excellent {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
    }
    
    .quality-good {
        background: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid #f59e0b;
    }
    
    .quality-poor {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
    }
    
    /* MÉTRICAS CON EXPLICACIÓN */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        position: relative;
    }
    
    [data-testid="stMetric"] label {
        color: var(--text-muted) !important;
        font-size: 0.875rem !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 41, 59, 0.5);
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background: transparent;
        border-radius: 8px;
        color: var(--text-muted);
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%) !important;
        color: white !important;
    }
    
    .stButton button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 10px;
        padding: 1rem 1.5rem;
        font-weight: 600;
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%) !important;
        color: white !important;
        font-weight: 700 !important;
    }
    
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(99, 102, 241, 0.3) 50%, transparent 100%);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🎓 SISTEMA DE AYUDA INTEGRADO
# ==========================================
HELP_TEXTS = {
    "word_count": {
        "title": "📝 Conteo de Palabras",
        "simple": "Cuántas palabras tiene el artículo completo",
        "why": "Google prefiere contenido largo y detallado. Artículos más largos suelen rankear mejor.",
        "how_to_use": "Si el promedio es 1,500 palabras, tu artículo debe tener mínimo 1,800 (20% más).",
        "good": "Por encima del promedio +20%",
        "bad": "Por debajo del promedio -20%"
    },
    "title_length": {
        "title": "📏 Longitud del Título",
        "simple": "Cuántos caracteres tiene el título SEO (Title tag)",
        "why": "Google muestra solo 50-60 caracteres. Si es más largo, lo corta con '...' = pierde clicks.",
        "how_to_use": "Tu título debe tener entre 50-60 caracteres para verse completo en Google.",
        "good": "50-60 caracteres",
        "bad": "Más de 70 o menos de 30"
    },
    "meta_length": {
        "title": "📄 Longitud Meta Description",
        "simple": "Cuántos caracteres tiene la descripción que aparece en Google",
        "why": "Google muestra 150-160 caracteres. Si es más larga, la corta.",
        "how_to_use": "Escribe meta descriptions de 150-160 caracteres con tu keyword principal.",
        "good": "150-160 caracteres",
        "bad": "Más de 170 o menos de 120"
    },
    "h2_count": {
        "title": "📑 Número de H2s",
        "simple": "Cuántas secciones principales tiene el artículo",
        "why": "Los H2 organizan el contenido. Google los lee para entender de qué trata cada sección.",
        "how_to_use": "Usa al menos tantos H2s como el promedio del TOP 10. Incluye keywords en ellos.",
        "good": "6-12 H2s bien estructurados",
        "bad": "Menos de 3 o más de 20"
    },
    "internal_links": {
        "title": "🔗 Enlaces Internos",
        "simple": "Cuántos links apuntan a otras páginas de tu propio sitio",
        "why": "Enlaces internos distribuyen autoridad y ayudan a Google a entender tu estructura.",
        "how_to_use": "Incluye 15-25 enlaces a artículos relacionados, categorías o páginas importantes.",
        "good": "15-30 enlaces internos",
        "bad": "Menos de 10"
    },
    "external_links": {
        "title": "🌐 Enlaces Externos",
        "simple": "Cuántos links apuntan a otros sitios web",
        "why": "Citar fuentes de autoridad (.edu, .gov, sitios reconocidos) aumenta tu credibilidad.",
        "how_to_use": "Incluye 3-5 enlaces a fuentes autorizadas y relevantes.",
        "good": "3-8 enlaces de calidad",
        "bad": "0 enlaces o más de 15"
    },
    "images": {
        "title": "🖼️ Imágenes",
        "simple": "Cuántas imágenes tiene el artículo",
        "why": "Contenido visual mejora la experiencia. Google valora páginas con buenas imágenes.",
        "how_to_use": "Incluye 1 imagen cada 300 palabras aprox. Todas con ALT text optimizado.",
        "good": "8-15 imágenes",
        "bad": "Menos de 5"
    },
    "da_proxy": {
        "title": "📊 Domain Authority",
        "simple": "Qué tan 'importante' es el sitio web (escala 0-100)",
        "why": "Sitios con DA alto tienen más autoridad y es más difícil competir contra ellos.",
        "how_to_use": "Si todos tienen DA 70+ y tú tienes DA 20, necesitas contenido EXCEPCIONAL para competir.",
        "good": "DA 40+",
        "bad": "DA menor a 20"
    },
    "readability": {
        "title": "📖 Legibilidad",
        "simple": "Qué tan fácil es leer el texto (Flesch Reading Ease)",
        "why": "Google prefiere contenido que todos puedan entender. Texto simple = mejor UX.",
        "how_to_use": "Usa frases cortas, palabras simples, evita jerga técnica innecesaria.",
        "good": "Score 60-80 (Fácil)",
        "bad": "Score menor a 40 (Difícil)"
    },
    "schema": {
        "title": "🏗️ Schema Markup",
        "simple": "Código especial que le 'explica' tu contenido a Google",
        "why": "Schema Markup puede dar 'Rich Snippets' = resultados destacados con estrellas, imágenes, FAQs.",
        "how_to_use": "Si tus competidores tienen Schema, tú TAMBIÉN debes tenerlo. Usa plugins (Rank Math, Yoast).",
        "good": "Tiene Schema relevante (FAQ, Article, HowTo)",
        "bad": "Sin Schema cuando competidores lo tienen"
    }
}

def show_help_icon(key):
    """Muestra ícono de ayuda con tooltip"""
    if key in HELP_TEXTS:
        help_data = HELP_TEXTS[key]
        st.markdown(f"""
        <span class="help-tooltip" title="{help_data['simple']}">?</span>
        """, unsafe_allow_html=True)

def show_explainer(key, value=None, benchmark=None):
    """Muestra caja explicativa con semáforo de calidad"""
    if key not in HELP_TEXTS:
        return
    
    help_data = HELP_TEXTS[key]
    
    # Determinar calidad si hay valor
    quality_class = ""
    quality_text = ""
    
    if value is not None and benchmark is not None:
        if key == "word_count":
            if value >= benchmark * 1.2:
                quality_class = "quality-excellent"
                quality_text = "✅ Excelente - Por encima del promedio"
            elif value >= benchmark * 0.8:
                quality_class = "quality-good"
                quality_text = "⚠️ Aceptable - Cerca del promedio"
            else:
                quality_class = "quality-poor"
                quality_text = "❌ Insuficiente - Por debajo del promedio"
    
    st.markdown(f"""
    <div class="explainer-box">
        <strong>💡 {help_data['title']}</strong><br>
        <strong>¿Qué es?</strong> {help_data['simple']}<br>
        <strong>¿Por qué importa?</strong> {help_data['why']}<br>
        <strong>¿Cómo lo uso?</strong> {help_data['how_to_use']}<br>
        {f'<div class="quality-badge {quality_class}" style="margin-top: 0.5rem;">{quality_text}</div>' if quality_text else ''}
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 🔑 API KEYS & CONFIGURACIÓN
# ==========================================
SERPER_API_KEY = st.secrets.get("SERPER_API_KEY", "cb2f958314203c51f8835f854b6a5038df46fb11")

DOMINIOS_EXCLUIDOS = [
    "youtube.com", "facebook.com", "instagram.com", "twitter.com", 
    "tiktok.com", "pinterest", "linkedin", "amazon", "ebay"
]

H2_BLACKLIST = [
    "suscríbete", "newsletter", "contacto", "ayuda", "política", "privacidad", 
    "cookies", "menú", "login", "carrito"
]

FAQ_BLACKLIST = ["cookie", "política", "privacidad", "suscripción"]

STOPWORDS = set([
    "de", "la", "que", "el", "en", "y", "a", "los", "se", "del", "las", "un", 
    "por", "con", "no", "una", "su", "para", "es", "al", "lo", "como", "mas", "más"
])

# ==========================================
# 🧠 ESTADO DE SESIÓN
# ==========================================
if 'data_seo' not in st.session_state: st.session_state.data_seo = None
if 'global_words' not in st.session_state: st.session_state.global_words = []
if 'gap_analysis' not in st.session_state: st.session_state.gap_analysis = None
if 'show_help' not in st.session_state: st.session_state.show_help = True

# ==========================================
# 🔍 FUNCIONES CORE MEJORADAS
# ==========================================
def get_serper_results(query, n_needed):
    url = "https://google.serper.dev/search"
    payload = {"q": query, "num": min(n_needed, 100), "gl": "es", "hl": "es"}
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, json=payload)
        return response.json().get('organic', [])
    except: return []

def clean_h2s(soup_h2s):
    valid_h2s = []
    for h in soup_h2s:
        text = h.get_text(" ", strip=True)
        if len(text) < 5 or any(bad in text.lower() for bad in H2_BLACKLIST): continue
        valid_h2s.append(text)
    return valid_h2s

def get_ngrams(words, n):
    return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]

def detectar_intencion(soup, text):
    text_lower = text.lower()
    title_lower = soup.title.string.lower() if soup.title else ""
    transaccional = ["precio", "comprar", "venta", "oferta", "tienda"]
    informacional = ["guía", "tutorial", "qué es", "cómo", "consejos"]
    score_trans = sum(1 for w in transaccional if w in text_lower or w in title_lower)
    score_info = sum(1 for w in informacional if w in text_lower or w in title_lower)
    if score_trans > score_info: return "🛒 Transaccional"
    if score_info > score_trans: return "📚 Informacional"
    return "⚖️ Mixto"

def analizar_enlaces(soup, url):
    """MEJORADO: Analiza enlaces internos Y externos"""
    try:
        domain = urlparse(url).netloc
        links = soup.find_all('a', href=True)
        
        internal = []
        external = []
        external_nofollow = 0
        
        for link in links:
            href = link.get('href', '')
            rel = link.get('rel', [])
            
            if domain in href or href.startswith('/'):
                internal.append(href)
            elif href.startswith('http'):
                external.append(href)
                if 'nofollow' in rel:
                    external_nofollow += 1
        
        return {
            "internal": len(internal),
            "external": len(external),
            "external_nofollow": external_nofollow,
            "external_dofollow": len(external) - external_nofollow
        }
    except:
        return {"internal": 0, "external": 0, "external_nofollow": 0, "external_dofollow": 0}

def analizar_seo_onpage(soup, keyword):
    """NUEVO: Análisis SEO on-page detallado"""
    try:
        title = soup.title.string.strip() if soup.title else ""
        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag and "content" in meta_tag.attrs:
            meta_desc = meta_tag["content"]
        
        # Longitudes
        title_length = len(title)
        meta_length = len(meta_desc)
        
        # Keyword en lugares estratégicos
        kw_in_title = keyword.lower() in title.lower()
        kw_in_meta = keyword.lower() in meta_desc.lower()
        kw_in_url = keyword.lower() in soup.find('link', rel='canonical')['href'].lower() if soup.find('link', rel='canonical') else False
        
        # H1 con keyword
        h1s = soup.find_all('h1')
        kw_in_h1 = any(keyword.lower() in h.get_text().lower() for h in h1s)
        
        # Uso de negritas/strong en keyword
        strongs = soup.find_all(['strong', 'b'])
        kw_in_strong = any(keyword.lower() in s.get_text().lower() for s in strongs)
        
        return {
            "title_length": title_length,
            "meta_length": meta_length,
            "kw_in_title": kw_in_title,
            "kw_in_meta": kw_in_meta,
            "kw_in_url": kw_in_url,
            "kw_in_h1": kw_in_h1,
            "kw_in_strong": kw_in_strong
        }
    except:
        return {
            "title_length": 0, "meta_length": 0,
            "kw_in_title": False, "kw_in_meta": False,
            "kw_in_url": False, "kw_in_h1": False, "kw_in_strong": False
        }

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
    
    # Contar imágenes CON alt text
    imgs_with_alt = len([img for img in soup.find_all('img') if img.get('alt')])
    
    return {
        "images": imagenes,
        "videos": videos,
        "total": imagenes + videos,
        "images_with_alt": imgs_with_alt,
        "alt_ratio": (imgs_with_alt / imagenes * 100) if imagenes > 0 else 0
    }

def calcular_readability(text):
    try:
        score = flesch_reading_ease(text)
        if score >= 80: return f"✅ Muy fácil ({score:.0f})"
        elif score >= 60: return f"👍 Fácil ({score:.0f})"
        elif score >= 40: return f"⚠️ Medio ({score:.0f})"
        else: return f"❌ Difícil ({score:.0f})"
    except: return "N/A"

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
    """ANÁLISIS COMPLETO MEJORADO"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml'
    }
    try:
        downloaded = trafilatura.fetch_url(url)
        main_text = None
        if downloaded:
            main_text = trafilatura.extract(downloaded, include_comments=False)
        
        soup = None
        if not main_text:
            res = requests.get(url, timeout=12, headers=headers, verify=False)
            if res.status_code in [403, 429, 503]: 
                return {"error": f"Bloqueo ({res.status_code})"}
            soup = BeautifulSoup(res.text, 'html.parser')
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]): 
                tag.decompose()
            main_text = soup.get_text(separator=' ', strip=True)
        else:
            soup = BeautifulSoup(downloaded, 'html.parser')
        
        if len(main_text) < 100: 
            return {"error": "Contenido insuficiente"}
        
        title = soup.title.string.strip() if soup.title else "N/A"
        if title == "N/A" or len(title) < 3: 
            return {"error": "Título no detectado"}
        
        meta_tag = soup.find("meta", attrs={"name": "description"})
        meta_desc = meta_tag["content"] if meta_tag else snippet_serp
        
        h1s = [h.get_text(" ", strip=True) for h in soup.find_all('h1')]
        h2s_clean = clean_h2s(soup.find_all('h2'))
        
        clean_words = [w for w in re.sub(r'[^\w\s]', '', main_text.lower()).split() 
                      if w not in STOPWORDS and len(w) > 2]
        
        # ANÁLISIS MEJORADO
        enlaces = analizar_enlaces(soup, url)
        seo_onpage = analizar_seo_onpage(soup, target_kw)
        schemas = detectar_schema_markup(soup)
        media = analizar_media_richness(soup)
        readability = calcular_readability(main_text)
        da_proxy = estimar_domain_authority(url)
        
        return {
            "title": title,
            "meta_desc": meta_desc,
            "h1": h1s,
            "h2": h2s_clean,
            "word_count": len(main_text.split()),
            "kw_density": len(re.findall(rf'\b{re.escape(target_kw)}\b', main_text, re.IGNORECASE)),
            "all_words": clean_words,
            "intencion": detectar_intencion(soup, main_text),
            "enlaces": enlaces,
            "seo_onpage": seo_onpage,
            "schemas": schemas,
            "media": media,
            "readability": readability,
            "da_proxy": da_proxy,
            "full_text": main_text
        }
    except Exception as e:
        return {"error": str(e)}

def calcular_gap_analysis(data_list, keyword):
    all_h2s = []
    for item in data_list:
        h2_list = item.get('H2_list', item.get('h2', []))
        all_h2s.extend(h2_list)
    h2_counter = Counter([h.lower() for h in all_h2s])
    threshold = len(data_list) * 0.5
    h2s_comunes = {h: count for h, count in h2_counter.items() if count >= threshold}
    
    all_words = []
    for item in data_list:
        all_words.extend(item.get('all_words', []))
    word_counter = Counter(all_words)
    
    # Calcular promedios con manejo seguro de diccionarios
    def safe_get_enlaces(item, key):
        enlaces = item.get('Enlaces', item.get('enlaces', {}))
        return enlaces.get(key, 0) if isinstance(enlaces, dict) else 0
    
    def safe_get_media(item, key):
        media = item.get('Media', item.get('media', {}))
        return media.get(key, 0) if isinstance(media, dict) else 0
    
    return {
        "h2s_criticos": h2s_comunes,
        "palabras_clave_secundarias": word_counter.most_common(30),
        "coverage_benchmark": {
            "h2_avg": np.mean([len(item.get('H2_list', item.get('h2', []))) for item in data_list]),
            "word_avg": np.mean([item.get('Palabras', item.get('word_count', 0)) for item in data_list]),
            "media_avg": np.mean([safe_get_media(item, 'total') for item in data_list]),
            "internal_links_avg": np.mean([safe_get_enlaces(item, 'internal') for item in data_list]),
            "external_links_avg": np.mean([safe_get_enlaces(item, 'external') for item in data_list])
        }
    }

# ==========================================
# 🎨 SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0 1rem 0;'>
        <div style='font-size: 3rem;'>🎯</div>
        <h1 style='font-size: 1.75rem;'>SERP X-RAY 360™</h1>
        <p style='color: #94a3b8; font-size: 0.875rem;'>BEGINNER FRIENDLY EDITION</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### 🎯 Configuración")
    keyword = st.text_input("Keyword:", value="que es el repep")
    num_target = st.slider("Competidores:", 3, 10, 5)
    
    st.markdown("---")
    st.markdown("#### 🎓 Modo Aprendizaje")
    st.session_state.show_help = st.checkbox("Mostrar explicaciones", value=True, 
                                              help="Activa para ver ayudas y tooltips")
    
    st.markdown("---")
    analyze_button = st.button("🚀 ANALIZAR", use_container_width=True, type="primary")
    
    if st.button("🗑️ Limpiar", use_container_width=True):
        st.session_state.data_seo = None
        st.rerun()

# ==========================================
# 🎯 HEADER
# ==========================================
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1 style='font-size: 3rem;'>🎓 SERP X-RAY 360™</h1>
    <p style='color: #94a3b8; font-size: 1.25rem;'>Edición para Principiantes en SEO</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 🎯 EJECUCIÓN
# ==========================================
if analyze_button and keyword:
    with st.status("🔄 Analizando...", expanded=True) as status:
        raw_results = get_serper_results(keyword, num_target + 40)
        final_data = []
        global_words = []
        count_valid = 0
        
        if raw_results:
            progress_bar = st.progress(0)
            for res in raw_results:
                if count_valid >= num_target: break
                url = res['link']
                if any(bad in url for bad in DOMINIOS_EXCLUIDOS): continue
                
                status.update(label=f"🔍 #{count_valid+1}: {res.get('title', '')[:30]}...")
                data = analyze_url_final(url, keyword, res.get('snippet', ''))
                
                if data and "error" not in data:
                    global_words.extend(data['all_words'])
                    final_data.append({
                        "Pos": count_valid + 1,
                        "URL": url,
                        "Título": data['title'],
                        "Meta Desc": data['meta_desc'],
                        "Palabras": data['word_count'],
                        "Intención": data['intencion'],
                        "Menciones KW": data['kw_density'],
                        "H1_list": data.get('h1', []),
                        "H2_list": data.get('h2', []),
                        "Contenido": data.get('full_text', '')[:1000],
                        "Enlaces": data.get('enlaces', {}),
                        "SEO_Onpage": data.get('seo_onpage', {}),
                        "Schemas": data.get('schemas', []),
                        "Media": data.get('media', {}),
                        "Readability": data.get('readability', 'N/A'),
                        "DA_Proxy": data.get('da_proxy', 0),
                        **data
                    })
                    count_valid += 1
                    progress_bar.progress(count_valid / num_target)
            
            if final_data:
                st.session_state.data_seo = final_data
                st.session_state.global_words = global_words
                st.session_state.gap_analysis = calcular_gap_analysis(final_data, keyword)
                status.update(label="✅ Completado", state="complete")
                st.balloons()

# ==========================================
# 📊 RESULTADOS CON EXPLICACIONES
# ==========================================
if st.session_state.data_seo:
    data = st.session_state.data_seo
    df = pd.DataFrame(data)
    
    tab1, tab2, tab3 = st.tabs([
        "📊 Overview (Con Guía)", 
        "🎯 Gap Analysis (Qué Hacer)", 
        "📋 Detalle Competidores"
    ])
    
    with tab1:
        st.markdown("### 📊 Métricas Principales")
        
        if st.session_state.show_help:
            st.info("💡 **Guía rápida:** Estas son las métricas promedio del TOP 10. Tu objetivo es IGUALAR o SUPERAR estos números.")
        
        c1, c2, c3, c4, c5 = st.columns(5)
        
        word_avg = int(df['Palabras'].mean())
        c1.metric("📝 Palabras Promedio", word_avg)
        if st.session_state.show_help:
            with c1:
                show_explainer("word_count")
        
        c2.metric("🎯 Intención", df['Intención'].mode()[0] if not df['Intención'].empty else "N/A")
        
        internal_avg = int(df.apply(lambda x: x.get('Enlaces', x.get('enlaces', {})).get('internal', 0) if isinstance(x.get('Enlaces', x.get('enlaces', {})), dict) else 0, axis=1).mean())
        c3.metric("🔗 Links Internos", internal_avg)
        if st.session_state.show_help:
            with c3:
                show_explainer("internal_links")
        
        media_avg = df.apply(lambda x: x.get('Media', x.get('media', {})).get('total', 0) if isinstance(x.get('Media', x.get('media', {})), dict) else 0, axis=1).mean()
        c4.metric("🖼️ Imágenes/Videos", f"{media_avg:.1f}")
        if st.session_state.show_help:
            with c4:
                show_explainer("images")
        
        c5.metric("📊 DA Promedio", int(df['DA_Proxy'].mean()))
        if st.session_state.show_help:
            with c5:
                show_explainer("da_proxy")
        
        st.markdown("---")
        
        # Gráfico con explicación
        st.markdown("### 🔥 Palabras Clave Secundarias Más Usadas")
        if st.session_state.show_help:
            st.markdown("""
            <div class="explainer-box">
                <strong>💡 ¿Qué significa este gráfico?</strong><br>
                Estas son las frases de 2 palabras que MÁS aparecen en el TOP 10.<br>
                <strong>¿Qué hacer?</strong> Incluye TODAS estas frases en tu artículo de forma natural.
            </div>
            """, unsafe_allow_html=True)
        
        bigrams = get_ngrams(st.session_state.global_words, 2)
        if bigrams:
            chart_data = pd.DataFrame(Counter(bigrams).most_common(12), 
                                     columns=['Frase', 'Frecuencia']).set_index('Frase')
            st.bar_chart(chart_data, color="#6366f1")
    
    with tab2:
        st.markdown("### 🎯 Plan de Acción: Qué Debes Hacer")
        
        if st.session_state.gap_analysis:
            gap = st.session_state.gap_analysis
            bench = gap['coverage_benchmark']
            
            # Checklist accionable
            st.markdown("#### ✅ Checklist para Tu Artículo")
            
            checklist_items = [
                f"📝 Escribir mínimo **{int(bench['word_avg'] * 1.2)} palabras** (20% más que el promedio de {int(bench['word_avg'])})",
                f"📑 Incluir al menos **{int(bench['h2_avg'])} secciones H2**",
                f"🔗 Agregar **{int(bench['internal_links_avg'])} enlaces internos** a otras páginas de tu sitio",
                f"🌐 Citar **{int(bench['external_links_avg'])} fuentes externas** de autoridad",
                f"🖼️ Incluir **{int(bench['media_avg'])} imágenes** (todas con ALT text optimizado)",
            ]
            
            for item in checklist_items:
                st.markdown(f"- {item}")
            
            st.markdown("---")
            
            # H2s críticos
            st.markdown("#### 🔴 Secciones OBLIGATORIAS (H2s que usan tus competidores)")
            if gap['h2s_criticos']:
                if st.session_state.show_help:
                    st.markdown("""
                    <div class="explainer-box">
                        <strong>💡 ¿Qué son los H2s críticos?</strong><br>
                        Son las secciones que aparecen en MÁS DE LA MITAD de tus competidores.<br>
                        Si todos hablan de estos temas, Google ESPERA verlos en tu artículo también.
                    </div>
                    """, unsafe_allow_html=True)
                
                h2_df = pd.DataFrame(
                    [(h, count) for h, count in gap['h2s_criticos'].items()],
                    columns=['Sección H2', 'Aparece en X competidores']
                ).sort_values('Aparece en X competidores', ascending=False)
                st.dataframe(h2_df, use_container_width=True)
                
                st.success("✅ **Acción:** Copia estas secciones H2 y escribe contenido original para cada una")
            else:
                st.info("No hay H2s que se repitan consistentemente (cada competidor usa estructura diferente)")
    
    with tab3:
        st.markdown("### 📋 Análisis Detallado de Cada Competidor")
        
        for item in data:
            titulo_corto = (item['Título'][:70] + '..') if len(item['Título']) > 70 else item['Título']
            
            with st.expander(f"#{item['Pos']} | {titulo_corto}", expanded=False):
                
                # Métricas principales
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### 📊 Métricas Básicas")
                    st.markdown(f"**Palabras:** {item['Palabras']}")
                    st.markdown(f"**Intención:** {item['Intención']}")
                    st.markdown(f"**DA:** {item['DA_Proxy']}")
                    st.markdown(f"**Legibilidad:** {item['Readability']}")
                
                with col2:
                    st.markdown("#### 🔗 Enlaces")
                    enlaces = item.get('enlaces', {})
                    st.markdown(f"**Internos:** {enlaces.get('internal', 0)}")
                    st.markdown(f"**Externos:** {enlaces.get('external', 0)}")
                    st.markdown(f"**Externos Dofollow:** {enlaces.get('external_dofollow', 0)}")
                
                with col3:
                    st.markdown("#### 🖼️ Multimedia")
                    media = item.get('media', {})
                    st.markdown(f"**Imágenes:** {media.get('images', 0)}")
                    st.markdown(f"**Videos:** {media.get('videos', 0)}")
                    st.markdown(f"**Imágenes con ALT:** {media.get('images_with_alt', 0)}")
                
                # SEO On-Page
                seo = item.get('SEO_Onpage', item.get('seo_onpage', {}))
                if seo:
                    st.markdown("#### ✅ SEO On-Page")
                    
                    checks = [
                        ("Keyword en Título", seo.get('kw_in_title', False)),
                        ("Keyword en Meta Description", seo.get('kw_in_meta', False)),
                        ("Keyword en H1", seo.get('kw_in_h1', False)),
                        ("Keyword en Negritas", seo.get('kw_in_strong', False))
                    ]
                    
                    for check_name, has_it in checks:
                        icon = "✅" if has_it else "❌"
                        st.markdown(f"{icon} {check_name}")
                    
                    st.markdown(f"**Longitud Título:** {seo.get('title_length', 0)} caracteres")
                    st.markdown(f"**Longitud Meta:** {seo.get('meta_length', 0)} caracteres")
                
                # H2s
                h2_list = item.get('H2_list', item.get('h2', []))
                if h2_list:
                    st.markdown("#### 📑 Estructura H2")
                    for h2 in h2_list[:10]:
                        st.markdown(f"• {h2}")
                
                # Schemas
                schemas = item.get('Schemas', item.get('schemas', []))
                if schemas:
                    st.markdown(f"#### 🏗️ Schemas: {', '.join(schemas)}")

# ==========================================
# 🎯 FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem 0; color: #94a3b8;'>
    <p><strong>SERP X-RAY 360™ PRO</strong> - Beginner Friendly Edition</p>
    <p style='font-size: 0.75rem;'>Diseñado para que CUALQUIERA pueda entender SEO © 2024</p>
</div>
""", unsafe_allow_html=True)