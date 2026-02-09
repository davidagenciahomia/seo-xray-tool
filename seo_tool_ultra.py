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
st.set_page_config(page_title="SERP X-RAY 360™ ULTRA", layout="wide", page_icon="🚀")

# ==========================================
# 🔑 API KEYS
# ==========================================
SERPER_API_KEY = st.secrets.get("SERPER_API_KEY", "cb2f958314203c51f8835f854b6a5038df46fb11")
MOZ_ACCESS_ID = "TU_MOZ_ACCESS_ID"  # Opcional - para Domain Authority
MOZ_SECRET_KEY = "TU_MOZ_SECRET_KEY"  # Opcional
PAGESPEED_API_KEY = "TU_PAGESPEED_KEY"  # Opcional - Google PageSpeed Insights

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
# 🔍 FUNCIÓN 1: BÚSQUEDA EN GOOGLE
# ==========================================
def get_serper_results(query, n_needed):
    """Obtiene resultados orgánicos de Google"""
    url = "https://google.serper.dev/search"
    n_safe = min(n_needed, 100)
    payload = {"q": query, "num": n_safe, "gl": "es", "hl": "es"}
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, json=payload)
        return response.json().get('organic', [])
    except: return []

# ==========================================
# 🧹 FUNCIÓN 2: LIMPIEZA DE DATOS
# ==========================================
def clean_h2s(soup_h2s):
    """Limpia encabezados H2"""
    valid_h2s = []
    for h in soup_h2s:
        text = h.get_text(" ", strip=True)
        if len(text) < 5: continue
        if any(bad in text.lower() for bad in H2_BLACKLIST): continue
        valid_h2s.append(text)
    return valid_h2s

def get_ngrams(words, n):
    """Genera n-gramas"""
    return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]

# ==========================================
# 🎯 FUNCIÓN 3: ANÁLISIS DE INTENCIÓN
# ==========================================
def detectar_intencion(soup, text):
    """Clasifica intención: Transaccional, Informacional, Mixto"""
    text_lower = text.lower()
    title_lower = soup.title.string.lower() if soup.title else ""
    
    transaccional = ["precio", "comprar", "venta", "carrito", "oferta", "barato", "tienda", "envío", "stock", "€", "$", "pedir", "catálogo"]
    informacional = ["guía", "tutorial", "opinión", "review", "qué es", "cómo", "cuándo", "consejos", "mejores", "comparativa", "análisis", "blog"]
    
    score_trans = sum(1 for w in transaccional if w in text_lower or w in title_lower)
    score_info = sum(1 for w in informacional if w in text_lower or w in title_lower)
    
    if score_trans > score_info: return "🛒 Transaccional"
    if score_info > score_trans: return "📚 Informacional"
    return "⚖️ Mixto"

# ==========================================
# 📅 FUNCIÓN 4: EXTRACCIÓN DE FECHA
# ==========================================
def extraer_fecha(soup):
    """Extrae fecha de publicación"""
    date = "N/A"
    meta_date = soup.find("meta", attrs={"property": "article:published_time"}) or \
                soup.find("meta", attrs={"name": "date"}) or \
                soup.find("meta", attrs={"name": "pubdate"})
    if meta_date:
        try: date = meta_date['content'][:10]
        except: pass
    return date

# ==========================================
# ❓ FUNCIÓN 5: EXTRACCIÓN DE FAQS
# ==========================================
def extraer_preguntas_validas(text):
    """Extrae preguntas reales del contenido"""
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

# ==========================================
# 🔗 FUNCIÓN 6: ANÁLISIS DE ENLACES INTERNOS
# ==========================================
def analizar_enlaces_internos(soup, url):
    """Cuenta enlaces internos"""
    try:
        domain = urlparse(url).netloc
        links = soup.find_all('a', href=True)
        internal = [l for l in links if domain in l['href'] or l['href'].startswith('/')]
        return len(internal)
    except:
        return 0

# ==========================================
# 📊 FUNCIÓN 7: SCHEMA MARKUP DETECTION
# ==========================================
def detectar_schema_markup(soup):
    """Detecta datos estructurados (JSON-LD)"""
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

# ==========================================
# 🖼️ FUNCIÓN 8: MEDIA RICHNESS
# ==========================================
def analizar_media_richness(soup):
    """Cuenta elementos multimedia"""
    imagenes = len(soup.find_all('img'))
    videos = len(soup.find_all(['video', 'iframe']))
    return {"images": imagenes, "videos": videos, "total": imagenes + videos}

# ==========================================
# 📖 FUNCIÓN 9: READABILITY SCORE
# ==========================================
def calcular_readability(text):
    """Calcula índice Flesch Reading Ease"""
    try:
        score = flesch_reading_ease(text)
        if score >= 80: return f"✅ Muy fácil ({score:.0f})"
        elif score >= 60: return f"👍 Fácil ({score:.0f})"
        elif score >= 40: return f"⚠️ Medio ({score:.0f})"
        else: return f"❌ Difícil ({score:.0f})"
    except:
        return "N/A"

# ==========================================
# 🔍 FUNCIÓN 10: NER (ENTIDADES NOMBRADAS)
# ==========================================
def extraer_entidades(text):
    """Extrae entidades nombradas (simplificado con regex)"""
    # Patrón para nombres propios (palabras en mayúscula)
    entidades = re.findall(r'\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*\b', text)
    # Filtrar palabras comunes que empiezan con mayúscula
    entidades_filtradas = [e for e in entidades if e.lower() not in ['el', 'la', 'los', 'las', 'un', 'una']]
    return list(set(entidades_filtradas))[:10]

# ==========================================
# 🏆 FUNCIÓN 11: DOMAIN AUTHORITY (PROXY)
# ==========================================
def estimar_domain_authority(url):
    """Estima DA basándose en TLD y características del dominio"""
    domain = urlparse(url).netloc
    score = 50  # Base
    
    # Bonificación por TLD premium
    if domain.endswith('.edu'): score += 20
    elif domain.endswith('.gov'): score += 25
    elif domain.endswith('.org'): score += 10
    
    # Penalización por subdominios
    if domain.count('.') > 1: score -= 10
    
    # Bonificación por dominio corto
    if len(domain) < 15: score += 5
    
    return min(max(score, 0), 100)

# ==========================================
# 📊 FUNCIÓN 12: ANÁLISIS DE URL COMPLETO
# ==========================================
def analyze_url_final(url, target_kw, snippet_serp):
    """Análisis completo de una URL"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }
    
    try:
        # Extracción con Trafilatura
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
        
        # NUEVOS ANÁLISIS
        enlaces_internos = analizar_enlaces_internos(soup, url)
        schemas = detectar_schema_markup(soup)
        media = analizar_media_richness(soup)
        readability = calcular_readability(main_text)
        entidades = extraer_entidades(main_text)
        da_proxy = estimar_domain_authority(url)
        
        # Detectar Table of Contents
        toc_detected = bool(soup.find('div', class_=re.compile('table.*content|toc', re.I)))
        
        return {
            "title": title,
            "meta_desc": meta_desc,
            "date": extraer_fecha(soup),
            "h1": h1s,
            "h2": h2s_clean,
            "h3": h3s_clean,
            "word_count": len(main_text.split()),
            "kw_density": len(re.findall(rf'\b{re.escape(target_kw)}\b', main_text, re.IGNORECASE)),
            "content_sample": main_text[:1000],
            "all_words": clean_words,
            "intencion": detectar_intencion(soup, main_text),
            "preguntas": extraer_preguntas_validas(main_text),
            "enlaces_internos": enlaces_internos,
            "schemas": schemas,
            "media": media,
            "readability": readability,
            "entidades": entidades,
            "da_proxy": da_proxy,
            "toc": toc_detected,
            "full_text": main_text
        }
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# 🎯 FUNCIÓN 13: GAP ANALYSIS
# ==========================================
def calcular_gap_analysis(data_list, keyword):
    """Analiza brechas de contenido vs competencia"""
    # Recopilar todos los H2s
    all_h2s = []
    for item in data_list:
        all_h2s.extend(item.get('h2', []))
    
    # Contar frecuencia de H2s
    h2_counter = Counter([h.lower() for h in all_h2s])
    
    # H2s que aparecen en >50% de competidores
    threshold = len(data_list) * 0.5
    h2s_comunes = {h: count for h, count in h2_counter.items() if count >= threshold}
    
    # Palabras clave secundarias más comunes
    all_words = []
    for item in data_list:
        all_words.extend(item.get('all_words', []))
    
    word_counter = Counter(all_words)
    top_words = word_counter.most_common(30)
    
    # Preguntas más frecuentes
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

# ==========================================
# 🔮 FUNCIÓN 14: TOPIC CLUSTERING
# ==========================================
def realizar_clustering(data_list):
    """Agrupa URLs por similitud semántica"""
    if len(data_list) < 3:
        return None
    
    # Crear corpus de textos
    corpus = []
    urls = []
    for item in data_list:
        text = " ".join(item.get('h2', [])) + " " + " ".join(item.get('all_words', [])[:100])
        corpus.append(text)
        urls.append(item.get('URL', ''))
    
    # Vectorización TF-IDF
    try:
        vectorizer = TfidfVectorizer(max_features=50, stop_words=list(STOPWORDS))
        X = vectorizer.fit_transform(corpus)
        
        # K-Means clustering
        n_clusters = min(3, len(data_list))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        # Organizar resultados
        cluster_groups = {}
        for idx, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append({
                "url": urls[idx],
                "title": data_list[idx].get('title', 'N/A')
            })
        
        return cluster_groups
    except:
        return None

# ==========================================
# 🎨 INTERFAZ GRÁFICA
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1066/1066373.png", width=50)
    st.title("🚀 SERP X-RAY 360™")
    st.caption("ULTRA Edition | v12.0")
    st.divider()
    
    keyword = st.text_input("🎯 Keyword:", value="que es el repep")
    num_target = st.slider("📊 Competidores:", 3, 10, 5)
    
    st.markdown("### ⚙️ Módulos Activos")
    col1, col2 = st.columns(2)
    with col1:
        st.success("✅ Gap Analysis")
        st.success("✅ Clustering")
        st.success("✅ Schema Detection")
        st.success("✅ Media Analysis")
    with col2:
        st.success("✅ Readability")
        st.success("✅ NER Entities")
        st.success("✅ DA Proxy")
        st.success("✅ TOC Detection")
    
    st.divider()
    debug_mode = st.checkbox("🛠️ Modo Debug", value=False)
    analyze_button = st.button("🚀 ESCANEAR SERPS", type="primary", use_container_width=True)
    
    if st.button("🗑️ Limpiar Cache"):
        st.session_state.data_seo = None
        st.session_state.gap_analysis = None
        st.session_state.clusters = None
        st.rerun()

# ==========================================
# 🎯 LÓGICA PRINCIPAL
# ==========================================
st.title("🚀 SERP X-RAY 360™ ULTRA: Inteligencia SEO Competitiva")

if analyze_button and keyword:
    if "TU_API_KEY" in SERPER_API_KEY or len(SERPER_API_KEY) < 20:
        st.error("⚠️ ERROR: Configura tu API Key de Serper en línea 17")
    else:
        with st.status("🔄 Escaneando SERPs...") as status:
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
                            "Pos": count_valid + 1,
                            "URL": url,
                            "Título": data['title'],
                            "Fecha": data['date'],
                            "Meta Desc": data['meta_desc'],
                            "Palabras": data['word_count'],
                            "Menciones KW": data['kw_density'],
                            "H1_list": data['h1'],
                            "H2_list": data['h2'],
                            "H3_list": data['h3'],
                            "Contenido": data['content_sample'],
                            "Intención": data['intencion'],
                            "Preguntas": data['preguntas'],
                            "Enlaces_Int": data['enlaces_internos'],
                            "Schemas": data['schemas'],
                            "Media": data['media'],
                            "Readability": data['readability'],
                            "Entidades": data['entidades'],
                            "DA_Proxy": data['da_proxy'],
                            "TOC": data['toc'],
                            "full_text": data['full_text'],
                            **data
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
                    
                    # CALCULAR GAP ANALYSIS
                    st.session_state.gap_analysis = calcular_gap_analysis(final_data, keyword)
                    
                    # REALIZAR CLUSTERING
                    st.session_state.clusters = realizar_clustering(final_data)
                    
                    status.update(label="✅ Análisis Completado", state="complete")

# ==========================================
# 📊 VISUALIZACIÓN DE RESULTADOS
# ==========================================
if st.session_state.data_seo:
    data = st.session_state.data_seo
    df = pd.DataFrame(data)
    
    # TABS PRINCIPALES
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🎯 Gap Analysis", 
        "🧬 Clustering", 
        "📋 Detalle Competidores",
        "💾 Exportar"
    ])
    
    # ==========================================
    # TAB 1: OVERVIEW
    # ==========================================
    with tab1:
        st.subheader("📊 Métricas Clave")
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("📝 Palabras Promedio", int(df['Palabras'].mean()))
        c2.metric("🎯 Intención Dominante", df['Intención'].mode()[0] if not df['Intención'].empty else "N/A")
        c3.metric("🔗 Links Int. Avg", int(df['Enlaces_Int'].mean()))
        c4.metric("🖼️ Media Avg", f"{df['Media'].apply(lambda x: x.get('total', 0)).mean():.1f}")
        c5.metric("📊 DA Promedio", int(df['DA_Proxy'].mean()))
        
        st.divider()
        
        # Gráfico de Palabras Clave Secundarias
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔥 Top Bigramas Semánticos")
            bigrams = get_ngrams(st.session_state.global_words, 2)
            if bigrams:
                chart_data = pd.DataFrame(
                    Counter(bigrams).most_common(12), 
                    columns=['Frase', 'Freq']
                ).set_index('Frase')
                st.bar_chart(chart_data, color="#FF4B4B")
        
        with col2:
            st.markdown("#### 📈 Distribución de Word Count")
            st.bar_chart(df[['Pos', 'Palabras']].set_index('Pos'))
        
        # Schema Markup Analysis
        st.divider()
        st.markdown("#### 🏗️ Análisis de Schema Markup")
        
        schema_stats = {}
        for item in data:
            for schema in item['Schemas']:
                schema_stats[schema] = schema_stats.get(schema, 0) + 1
        
        if schema_stats:
            schema_df = pd.DataFrame(
                list(schema_stats.items()), 
                columns=['Schema Type', 'Count']
            ).sort_values('Count', ascending=False)
            st.dataframe(schema_df, use_container_width=True)
        else:
            st.info("No se detectaron schemas en los competidores")
    
    # ==========================================
    # TAB 2: GAP ANALYSIS
    # ==========================================
    with tab2:
        st.subheader("🎯 Análisis de Brechas de Contenido")
        
        if st.session_state.gap_analysis:
            gap = st.session_state.gap_analysis
            
            # H2s Críticos
            st.markdown("### 🔴 H2s Críticos (Aparecen en >50% de competidores)")
            if gap['h2s_criticos']:
                h2_df = pd.DataFrame(
                    [(h, count) for h, count in gap['h2s_criticos'].items()],
                    columns=['H2', 'Apariciones']
                ).sort_values('Apariciones', ascending=False)
                
                st.dataframe(h2_df, use_container_width=True)
                
                st.info("💡 **Recomendación:** Incluye TODOS estos H2s en tu contenido para competir efectivamente")
            else:
                st.warning("No hay H2s que se repitan consistentemente")
            
            st.divider()
            
            # Palabras Clave Secundarias
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📝 Top 15 Palabras Clave Secundarias")
                words_df = pd.DataFrame(
                    gap['palabras_clave_secundarias'][:15],
                    columns=['Palabra', 'Frecuencia']
                )
                st.dataframe(words_df, use_container_width=True)
            
            with col2:
                st.markdown("### ❓ FAQs Más Comunes")
                if gap['faqs_comunes']:
                    faq_df = pd.DataFrame(
                        gap['faqs_comunes'],
                        columns=['Pregunta', 'Apariciones']
                    )
                    st.dataframe(faq_df, use_container_width=True)
                else:
                    st.info("No hay FAQs que se repitan")
            
            st.divider()
            
            # Benchmarks
            st.markdown("### 📊 Benchmarks de la Competencia")
            
            bench = gap['coverage_benchmark']
            
            c1, c2, c3 = st.columns(3)
            c1.metric("📌 H2s Promedio", f"{bench['h2_avg']:.1f}")
            c2.metric("📝 Palabras Promedio", f"{bench['word_avg']:.0f}")
            c3.metric("🖼️ Media Promedio", f"{bench['media_avg']:.1f}")
            
            st.success("""
            **💡 Cómo usar estos datos:**
            1. Tu contenido debe tener al menos el número promedio de H2s
            2. Apunta a superar el word count promedio en un 20%
            3. Incluye más elementos multimedia que el promedio
            4. Cubre TODOS los H2s críticos identificados arriba
            """)
    
    # ==========================================
    # TAB 3: CLUSTERING
    # ==========================================
    with tab3:
        st.subheader("🧬 Clustering por Similitud Semántica")
        
        if st.session_state.clusters:
            clusters = st.session_state.clusters
            
            st.info(f"Se identificaron {len(clusters)} grupos temáticos distintos")
            
            for cluster_id, urls in clusters.items():
                with st.expander(f"📦 Cluster {cluster_id + 1} ({len(urls)} URLs)", expanded=True):
                    for item in urls:
                        st.markdown(f"- [{item['title'][:70]}...]({item['url']})")
            
            st.success("""
            **💡 Insight Estratégico:**
            - URLs en el mismo cluster atacan ángulos similares del tema
            - Identifica qué cluster domina (más URLs) para entender la intención principal
            - Considera crear contenido que cubra MÚLTIPLES clusters para mayor autoridad
            """)
        else:
            st.warning("Se necesitan al menos 3 URLs para clustering")
    
    # ==========================================
    # TAB 4: DETALLE COMPETIDORES
    # ==========================================
    with tab4:
        st.subheader("📋 Análisis Detallado por Competidor")
        
        for item in data:
            titulo_corto = (item['Título'][:70] + '..') if len(item['Título']) > 70 else item['Título']
            
            with st.expander(f"#{item['Pos']} | {titulo_corto}", expanded=False):
                # Badges superiores
                c_badges = st.columns(5)
                
                # Intención
                if "Transaccional" in item['Intención']: 
                    c_badges[0].warning(item['Intención'])
                else: 
                    c_badges[0].success(item['Intención'])
                
                # Fecha
                if item['Fecha'] != "N/A": 
                    c_badges[1].caption(f"📅 {item['Fecha']}")
                
                # DA Proxy
                da_color = "🟢" if item['DA_Proxy'] >= 60 else "🟡" if item['DA_Proxy'] >= 40 else "🔴"
                c_badges[2].caption(f"{da_color} DA: {item['DA_Proxy']}")
                
                # Readability
                c_badges[3].caption(f"📖 {item['Readability']}")
                
                # TOC
                if item['TOC']:
                    c_badges[4].caption("✅ TOC")
                
                st.caption(f"🔗 {item['URL']}")
                st.info(f"**Meta:** {item['Meta Desc']}")
                
                # Estructura vs Multimedia
                c1, c2 = st.columns(2)
                
                with c1:
                    st.markdown("#### 🏗️ Estructura de Contenido")
                    st.markdown(f"**📊 {item['Palabras']} palabras** | **🎯 {item['Menciones KW']} menciones KW**")
                    st.markdown(f"**🔗 {item['Enlaces_Int']} enlaces internos**")
                    
                    if item['H1_list']: 
                        st.markdown(f"**H1:** {item['H1_list'][0]}")
                    
                    if item['H2_list']:
                        st.markdown(f"**H2s ({len(item['H2_list'])}):**")
                        for h in item['H2_list'][:8]: 
                            st.markdown(f"• {h}")
                    else: 
                        st.caption("❌ Sin H2s")
                
                with c2:
                    st.markdown("#### 🎨 Multimedia & Técnico")
                    
                    media = item['Media']
                    st.markdown(f"**🖼️ Imágenes:** {media['images']}")
                    st.markdown(f"**🎥 Videos:** {media['videos']}")
                    st.markdown(f"**📊 Total Media:** {media['total']}")
                    
                    if item['Schemas']:
                        st.markdown(f"**🏗️ Schemas:** {', '.join(item['Schemas'])}")
                    else:
                        st.caption("❌ Sin Schema Markup")
                    
                    if item['Entidades']:
                        st.markdown(f"**🏷️ Entidades:** {', '.join(item['Entidades'][:5])}")
                
                # FAQs
                st.markdown("#### ❓ FAQs Detectadas")
                if item['Preguntas']:
                    faq_text = "\n".join([f"- {p}" for p in item['Preguntas']])
                    st.code(faq_text, language="text")
                else:
                    st.caption("Sin preguntas detectadas")
                
                # Muestra de contenido
                with st.expander("📄 Ver muestra de contenido"):
                    st.text_area("", value=item['Contenido'], height=150, label_visibility="collapsed")
    
    # ==========================================
    # TAB 5: EXPORTAR
    # ==========================================
    with tab5:
        st.subheader("💾 Exportar Análisis")
        
        # Preparar CSV
        export_df = df[[
            'Pos', 'URL', 'Título', 'Fecha', 'Palabras', 'Menciones KW',
            'Intención', 'Enlaces_Int', 'DA_Proxy', 'Readability', 'TOC'
        ]].copy()
        
        export_df['Schemas'] = df['Schemas'].apply(lambda x: ', '.join(x) if x else 'None')
        export_df['Media_Total'] = df['Media'].apply(lambda x: x.get('total', 0))
        
        csv = export_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            "📥 Descargar Análisis Completo (CSV)",
            csv,
            f"SERP_XRAY_Ultra_{keyword.replace(' ', '_')}.csv",
            "text/csv",
            type="primary",
            use_container_width=True
        )
        
        # Exportar Gap Analysis
        if st.session_state.gap_analysis:
            gap = st.session_state.gap_analysis
            
            gap_text = "=== H2S CRÍTICOS ===\n"
            gap_text += "\n".join([f"- {h} ({c} veces)" for h, c in gap['h2s_criticos'].items()])
            gap_text += "\n\n=== PALABRAS CLAVE SECUNDARIAS ===\n"
            gap_text += "\n".join([f"- {w} ({c} veces)" for w, c in gap['palabras_clave_secundarias'][:30]])
            
            st.download_button(
                "📥 Descargar Gap Analysis (TXT)",
                gap_text.encode('utf-8'),
                f"Gap_Analysis_{keyword.replace(' ', '_')}.txt",
                "text/plain",
                use_container_width=True
            )
        
        st.success("✅ Análisis listo para exportar")

# ==========================================
# 🎯 FOOTER
# ==========================================
st.divider()
st.caption("🚀 SERP X-RAY 360™ ULTRA - Versión 12.0 | Powered by Agencia Homia")
st.caption("Módulos activos: Gap Analysis, Clustering, Schema Detection, Media Analysis, Readability, NER, DA Proxy, TOC Detection")
