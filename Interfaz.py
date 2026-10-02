import streamlit as st
import streamlit.components.v1 as components
from groq import Groq
import os
import base64
import unicodedata
import numpy as np
from pypdf import PdfReader
import chromadb
import cv2
import fitz  # PyMuPDF para renderizado en RAM a 300 DPI
from sentence_transformers import SentenceTransformer

try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    from streamlit_autorefresh import st_autorefresh
    WEBRTC_DISPONIBLE = True
except ImportError:
    WEBRTC_DISPONIBLE = False

# --- CONFIGURACIÓN DE RUTAS DINÁMICAS (Compatibles con Windows y Render/Linux) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "img")
DATA_DIR = os.path.join(BASE_DIR, "CONOCIMIENTO_VPO")
DOCS_DIR = os.path.join(BASE_DIR, "DOCUMENTOS")
INDICE_DIR = os.path.join(BASE_DIR, "INDICE_VECTORIAL")

# --- CONFIGURACIÓN DE MODELOS ---
MODELO_CHAT = "llama-3.3-70b-versatile"

# Inicializar cliente de Groq leyendo la variable de entorno o clave directa
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
client_groq = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

@st.cache_resource
def cargar_modelo_embedding():
    # Modelo de embeddings súper liviano (ideal para servidores gratis como Render)
    return SentenceTransformer("all-MiniLM-L6-v2")

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="MELENA - VPO Support System", layout="wide")

# --- DICCIONARIO MAESTRO ---
ESTRUCTURA_FALLAS = {
    "Depaletizadora": {"foto": "depaletizadora", "componentes": {"Cabezal de agarre": ["Mecánica", "Neumática", "Electrónica"], "Mesa de descarga": ["Mecánica"], "Sistema de plano móvil": ["Mecánica"], "Sistema de elevación": ["Mecánica", "Electrónica"], "Sistema de transportación": ["Mecánica"]}},
    "Desempacadora": {"foto": "desempacadora", "componentes": {"Transporte": ["Electrónica"], "Preabridora de solapa": ["Mecánica", "Neumática", "Electrónica"], "Mesa de descarga": ["Mecánica"], "Cabezales": ["Mecánica", "Neumática", "Electrónica"], "Volteadora": ["Mecánica", "Electrónica"]}},
    "Lavadora": {"foto": "Lavadora", "componentes": {"Transportación de lavadora": ["Mecánica"], "Cargador": ["Mecánica", "Electrónica"], "Descarga": ["Mecánica", "Electrónica"], "Accionamiento principal": ["Electrónica"], "Sistema en bombas": ["Neumática", "Consumos"]}},
    "Sekamat": {"foto": "Sekamat", "componentes": {"Checkmat de salida": ["Mecánica", "Electrónica"], "Manejo de botellas": ["Mecánica"], "Sistema de visión": ["Electrónica"]}},
    "Linatronic": {"foto": "Linatronic", "componentes": {"Cabezal de inspección": ["Mecánica"], "Manejo de botellas": ["Mecánica"], "Módulo de pared externa": ["Mecánica"], "Inspector": ["Electrónica"]}},
    "Llenadora": {"foto": "Llenadora", "componentes": {"Coronador": ["Mecánica"], "Tazón Calderín": ["Mecánica", "Calidad"], "Entrada/Salida Tornillo Sinfín": ["Mecánica"], "Pistones elevadores": ["Mecánica"], "Sistema de Autoflush": ["Neumática"], "Válvulas": ["Neumática"], "Alimentador de tapas": ["Mecánica"]}},
    "Pasteurizador": {"foto": "Pasteurizador", "componentes": {"Malla transportadora": ["Mecánica", "Consumos"], "Sistema de transportación": ["Mecánica"], "Sistema de espreado": ["Mecánica"], "Sistema de seguridad": ["Electrónica"], "Sistema de vapor/calentamiento": ["Eléctrico"]}},
    "Etiquetadora": {"foto": "etiquetadora", "componentes": {"Agregados": ["Mecánica"], "Carrusel": ["Electrónica"], "Codificador": ["Electrónica"], "Bomba de adhesivo": ["Neumática"], "Transporte": ["Mecánica"]}},
    "Empacadora": {"foto": "empacadora", "componentes": {"Accionamiento principal": ["Mecánica"], "Cerradora": ["Mecánica"], "Multidivideer": ["Mecánica"], "Sensores entrada/salida": ["Electrónica"]}},
    "Paletizadora": {"foto": "Paletizadora", "componentes": {"Cabezal": ["Mecánica", "Electrónica"], "Mesa giratoria": ["Mecánica"], "Filtec": ["Electrónica"], "Envolvedora": ["Mecánica", "Electrónica"], "Sistema de seguridad": ["Electrónica"]}}
}

TIPO_CARPETA = {
    "Manual de Mantenimiento": "Manuales",
    "Plan de Reacción (PDR)": "PDR",
    "Procedimiento Estándar (SOP)": "SOP",
    "Registro de Averías (RDA)": "RDA",
    "Recomendaciones de Seguridad": "Seguridad",
    "Información General": "General"
}

OPCIONES_TIPO_FALLA = [
    "Falla mecánica",
    "Falla eléctrica",
    "Fallas de proceso",
    "Falla de calidad",
    "Fallas humanas",
    "Fallas en sistemas de control y sensores"
]

# --- FUNCIONES AUXILIARES ---
def encontrar_imagen(nombre_base):
    extensiones = ['.jpg', '.png', '.jpeg', '.gif']
    if not os.path.exists(IMG_DIR): return None
    archivos = os.listdir(IMG_DIR)
    for ext in extensiones:
        nombre_con_ext = f"{nombre_base}{ext}"
        for archivo in archivos:
            if archivo.lower() == nombre_con_ext.lower():
                return os.path.join(IMG_DIR, archivo)
    return None

def get_base64(nombre_base):
    path = encontrar_imagen(nombre_base)
    if path:
        with open(path, 'rb') as f: return base64.b64encode(f.read()).decode()
    return None

def normalizar_texto(texto):
    if not texto: return ""
    s = unicodedata.normalize('NFD', str(texto))
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return s.lower().strip()

def parsear_codigo_qr(qr_raw):
    qr_norm = normalizar_texto(qr_raw)
    linea_detectada = None
    area_detectada = None

    if "linea 5" in qr_norm or "linea_5" in qr_norm or "l5" in qr_norm:
        linea_detectada = "Línea 5"
    elif "linea 6" in qr_norm or "linea_6" in qr_norm or "l6" in qr_norm:
        linea_detectada = "Línea 6"
    elif "5" in qr_norm and "6" not in qr_norm:
        linea_detectada = "Línea 5"
    elif "6" in qr_norm and "5" not in qr_norm:
        linea_detectada = "Línea 6"

    for area_real in ESTRUCTURA_FALLAS.keys():
        area_norm = normalizar_texto(area_real)
        if area_norm in qr_norm:
            area_detectada = area_real
            break

    return linea_detectada, area_detectada

if WEBRTC_DISPONIBLE:
    class ProcesadorQR(VideoProcessorBase):
        def __init__(self):
            self.resultado = None
            self.detector = cv2.QRCodeDetector()

        def recv(self, frame):
            imagen = frame.to_ndarray(format="bgr24")
            if self.resultado is None:
                contenido, _puntos, _ = self.detector.detectAndDecode(imagen)
                if contenido:
                    self.resultado = contenido
            return av.VideoFrame.from_ndarray(imagen, format="bgr24")

# --- LÓGICA DE CACHÉ SEMÁNTICO ---
def calcular_similitud_coseno(vec1, vec2):
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def consultar_cache_semantico(embedding_pregunta, clave_filtro, umbral=0.92):
    if "semantic_cache" not in st.session_state:
        st.session_state["semantic_cache"] = []
    
    mejor_coincidencia = None
    max_similitud = 0.0
    
    for item in st.session_state["semantic_cache"]:
        if item["clave_filtro"] == clave_filtro:
            sim = calcular_similitud_coseno(embedding_pregunta, item["embedding"])
            if sim >= umbral and sim > max_similitud:
                max_similitud = sim
                mejor_coincidencia = item
                
    if mejor_coincidencia:
        return mejor_coincidencia, max_similitud
    return None, 0.0

def guardar_en_cache_semantico(embedding_pregunta, clave_filtro, respuesta, es_oficial, fragmentos, evidencias):
    if "semantic_cache" not in st.session_state:
        st.session_state["semantic_cache"] = []
        
    st.session_state["semantic_cache"].append({
        "clave_filtro": clave_filtro,
        "embedding": embedding_pregunta,
        "respuesta": respuesta,
        "es_oficial": es_oficial,
        "fragmentos": fragmentos,
        "evidencias": evidencias
    })

# --- INICIALIZACIÓN DE ESTADOS ---
areas_lista = list(ESTRUCTURA_FALLAS.keys())

if "linea_sel" not in st.session_state:
    st.session_state["linea_sel"] = "Línea 5"

if "tab_activa" not in st.session_state:
    st.session_state["tab_activa"] = "plan"

if "area_sel" not in st.session_state or st.session_state["area_sel"] not in areas_lista:
    st.session_state["area_sel"] = areas_lista[0]

if "_pendiente_linea" in st.session_state:
    st.session_state["linea_sel"] = st.session_state.pop("_pendiente_linea")
if "_pendiente_area" in st.session_state:
    st.session_state["area_sel"] = st.session_state.pop("_pendiente_area")

def procesar_qr_manual_callback():
    texto = st.session_state.get("qr_input_manual", "").strip()
    if texto:
        linea_det, area_det = parsear_codigo_qr(texto)
        if linea_det:
            st.session_state["linea_sel"] = linea_det
        if area_det:
            st.session_state["area_sel"] = area_det
        st.session_state["tab_activa"] = "docs"

if "qr_scanned" in st.query_params:
    qr_val = str(st.query_params.get("qr_scanned", "")).strip()
    if qr_val:
        linea_det, area_det = parsear_codigo_qr(qr_val)
        if linea_det:
            st.session_state["linea_sel"] = linea_det
        if area_det:
            st.session_state["area_sel"] = area_det
        st.session_state["tab_activa"] = "docs"
    st.query_params.clear()

def encontrar_subcarpeta(ruta_base, nombre_buscado):
    if not ruta_base or not os.path.exists(ruta_base): return None
    for item in os.listdir(ruta_base):
        ruta_item = os.path.join(ruta_base, item)
        if os.path.isdir(ruta_item) and item.lower() == nombre_buscado.lower():
            return ruta_item
    return None

def encontrar_valor_valido(opciones_validas, texto_leido):
    if not texto_leido: return None
    texto_normalizado = texto_leido.strip().lower()
    for opcion in opciones_validas:
        if opcion.strip().lower() == texto_normalizado:
            return opcion
    return None

def listar_pdfs(linea, area, tipo_doc):
    carpeta_linea = encontrar_subcarpeta(DOCS_DIR, linea)
    carpeta_area = encontrar_subcarpeta(carpeta_linea, area) if carpeta_linea else None
    carpeta_tipo = encontrar_subcarpeta(carpeta_area, TIPO_CARPETA[tipo_doc]) if carpeta_area else None
    if not carpeta_tipo: return []
    return [os.path.join(carpeta_tipo, f) for f in os.listdir(carpeta_tipo) if f.lower().endswith(".pdf")]

def listar_archivos_pdf_filtrados(linea=None, area=None):
    archivos = set()
    base = DOCS_DIR
    if linea:
        c_linea = encontrar_subcarpeta(base, linea)
        if c_linea: base = c_linea
        else: return []
    if area:
        c_area = encontrar_subcarpeta(base, area)
        if c_area: base = c_area
        else: return []

    for root, dirs, files in os.walk(base):
        for f in files:
            if f.lower().endswith(".pdf"):
                archivos.add(f)
    return sorted(list(archivos))

@st.cache_data(show_spinner=False)
def leer_pdf_bytes_cacheado(ruta_pdf, fecha_modificacion):
    with open(ruta_pdf, "rb") as f: return f.read()

# --- RAG, RENDERIZADO RAM E ÍNDICE VECTORIAL ---
@st.cache_resource
def obtener_coleccion_vectorial():
    if not os.path.exists(INDICE_DIR):
        os.makedirs(INDICE_DIR, exist_ok=True)
    cliente = chromadb.PersistentClient(path=INDICE_DIR)
    return cliente.get_or_create_collection(name="documentos_melena")

def resolver_ruta_y_pagina(top_f):
    if not top_f: return None, 1
    ruta_pdf = top_f.get("ruta_pdf")
    pagina = top_f.get("pagina")
    archivo_nombre = top_f.get("archivo")

    if not ruta_pdf or not os.path.exists(ruta_pdf):
        if archivo_nombre:
            for root, dirs, files in os.walk(DOCS_DIR):
                for f in files:
                    if f.lower() == archivo_nombre.lower():
                        ruta_pdf = os.path.join(root, f)
                        break
                if ruta_pdf and os.path.exists(ruta_pdf): break

    if not ruta_pdf or not os.path.exists(ruta_pdf): return None, 1

    if not pagina:
        try:
            doc = fitz.open(ruta_pdf)
            texto_fragmento = top_f.get("texto", "").strip()
            palabras = texto_fragmento.split()
            muestra = " ".join(palabras[:5]) if len(palabras) >= 5 else texto_fragmento
            pagina_encontrada = 1
            if muestra:
                for idx_p, page in enumerate(doc):
                    text_p = page.get_text() or ""
                    if muestra.lower() in text_p.lower():
                        pagina_encontrada = idx_p + 1
                        break
            pagina = pagina_encontrada
        except Exception:
            pagina = 1

    return ruta_pdf, int(pagina)

def obtener_imagen_pagina_ram(pdf_path, page_number, dpi=300):
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("png")

def obtener_evidencias_priorizadas(fragmentos):
    evidencias = []
    vistos = set()
    
    for f in fragmentos:
        pdf_path, pag_num = resolver_ruta_y_pagina(f)
        if pdf_path and os.path.exists(pdf_path):
            clave = (pdf_path, pag_num)
            if clave not in vistos:
                vistos.add(clave)
                tiene_img = False
                try:
                    doc = fitz.open(pdf_path)
                    page = doc[pag_num - 1]
                    tiene_img = (len(page.get_images()) > 0) or (len(page.get_drawings()) > 8)
                except Exception:
                    pass
                
                evidencias.append({
                    "pdf_path": pdf_path,
                    "pagina": pag_num,
                    "archivo": f.get("archivo", os.path.basename(pdf_path)),
                    "tipo": f.get("tipo", "Documento"),
                    "texto": f.get("texto", ""),
                    "tiene_imagen": tiene_img
                })
    
    evidencias.sort(key=lambda x: x["tiene_imagen"], reverse=True)
    return evidencias

def trocear_texto(texto, tam_chunk=100, solape=20):
    palabras = texto.split()
    chunks = []
    i = 0
    paso = max(1, tam_chunk - solape)
    while i < len(palabras):
        fragmento = " ".join(palabras[i:i + tam_chunk])
        if fragmento.strip():
            chunks.append(fragmento)
        i += paso
    return chunks

def obtener_embedding(texto):
    model = cargar_modelo_embedding()
    texto_seguro = texto[:1000]
    return model.encode(texto_seguro).tolist()

def indexar_todos_los_documentos():
    coleccion = obtener_coleccion_vectorial()
    if not os.path.exists(DOCS_DIR): return 0, 0
    total_chunks, total_archivos = 0, 0
    
    for linea in os.listdir(DOCS_DIR):
        ruta_linea = os.path.join(DOCS_DIR, linea)
        if not os.path.isdir(ruta_linea): continue
        for area in os.listdir(ruta_linea):
            ruta_area = os.path.join(ruta_linea, area)
            if not os.path.isdir(ruta_area): continue
            for tipo_carpeta in os.listdir(ruta_area):
                ruta_tipo = os.path.join(ruta_area, tipo_carpeta)
                if not os.path.isdir(ruta_tipo): continue
                linea_normalizada = encontrar_valor_valido(["Línea 5", "Línea 6"], linea) or linea
                area_normalizada = encontrar_valor_valido(list(ESTRUCTURA_FALLAS.keys()), area) or area
                
                for archivo in os.listdir(ruta_tipo):
                    if not archivo.lower().endswith(".pdf"): continue
                    ruta_pdf = os.path.join(ruta_tipo, archivo)
                    
                    try: doc = fitz.open(ruta_pdf)
                    except Exception: continue
                        
                    total_archivos += 1
                    for page_num in range(len(doc)):
                        page_text = doc[page_num].get_text() or ""
                        if not page_text.strip(): continue
                        
                        chunks = trocear_texto(page_text)
                        for idx, chunk in enumerate(chunks):
                            embedding = obtener_embedding(chunk)
                            id_unico = f"{linea_normalizada}|{area_normalizada}|{tipo_carpeta}|{archivo}|p{page_num + 1}|{idx}"
                            
                            coleccion.upsert(
                                ids=[id_unico],
                                embeddings=[embedding],
                                documents=[chunk],
                                metadatas=[{
                                    "linea": linea_normalizada,
                                    "area": area_normalizada,
                                    "tipo": tipo_carpeta,
                                    "archivo": archivo,
                                    "ruta_pdf": ruta_pdf,
                                    "pagina": page_num + 1
                                }],
                            )
                            total_chunks += 1
    return total_archivos, total_chunks

def buscar_contexto(pregunta, linea=None, area=None, archivo=None, top_k=8, permitir_fallback=True):
    coleccion = obtener_coleccion_vectorial()
    if coleccion.count() == 0: return []
    embedding_pregunta = obtener_embedding(pregunta)
    
    condiciones = []
    if archivo:
        condiciones.append({"archivo": archivo})
    else:
        if linea: condiciones.append({"linea": linea})
        if area: condiciones.append({"area": area})
    
    filtro = {"$and": condiciones} if len(condiciones) > 1 else (condiciones[0] if len(condiciones) == 1 else None)
    resultados = coleccion.query(query_embeddings=[embedding_pregunta], n_results=top_k, where=filtro)
    
    fragmentos = []
    if resultados and resultados.get('documents') and resultados['documents'][0]:
        for doc, meta in zip(resultados['documents'][0], resultados['metadatas'][0]):
            fragmentos.append({
                "texto": doc, 
                "archivo": meta.get('archivo'), 
                "tipo": meta.get('tipo'), 
                "ruta_pdf": meta.get('ruta_pdf'),
                "pagina": meta.get('pagina'),
                "es_fallback": False
            })
            
    if not fragmentos and filtro is not None and permitir_fallback:
        resultados_fb = coleccion.query(query_embeddings=[embedding_pregunta], n_results=top_k)
        if resultados_fb and resultados_fb.get('documents') and resultados_fb['documents'][0]:
            for doc, meta in zip(resultados_fb['documents'][0], resultados_fb['metadatas'][0]):
                fragmentos.append({
                    "texto": doc, 
                    "archivo": meta.get('archivo'), 
                    "tipo": meta.get('tipo'), 
                    "ruta_pdf": meta.get('ruta_pdf'),
                    "pagina": meta.get('pagina'),
                    "es_fallback": True
                })
                
    return fragmentos

# --- BASE64 DE ICONOS Y FONDOS ---
fondo_sidebar = get_base64("fondo")
sidebar_css = f'background-image: url("data:image/png;base64,{fondo_sidebar}"); background-size: cover;' if fondo_sidebar else ""

_icono_generador_b64 = get_base64("icono_generador")
_icono_biblioteca_b64 = get_base64("icono_biblioteca")
_icono_preguntar_b64 = get_base64("icono_preguntar")
_icono_escanear_b64 = get_base64("escanear")
_icono_advertencia_b64 = get_base64("advertencia")

_icono_ver_b64 = get_base64("icono_ver")
_icono_descargar_b64 = get_base64("icono_descargar")

css_icono_ver = f"""
div[class*="st-key-btn_ver_doc_"] > button {{
    background-image: url("data:image/png;base64,{_icono_ver_b64}") !important;
    background-repeat: no-repeat !important;
    background-position: 14px center !important;
    background-size: 22px 22px !important;
    padding-left: 44px !important;
}}
""" if _icono_ver_b64 else ""

css_icono_descargar = f"""
div[data-testid="stDownloadButton"] > button {{
    background-image: url("data:image/png;base64,{_icono_descargar_b64}") !important;
    background-repeat: no-repeat !important;
    background-position: 14px center !important;
    background-size: 22px 22px !important;
    padding-left: 44px !important;
}}
""" if _icono_descargar_b64 else ""

# --- ESTILOS CSS GLOBALES ---
st.markdown(f"""
    <style>
    .stApp {{ background-color: #000000 !important; }}
    h1, h2, h3, p, span, label, .stMarkdown {{ color: #FFFFFF !important; font-family: 'Segoe UI', sans-serif; }}

    [data-testid="stSidebar"] {{ {sidebar_css} border-right: 2px solid #FFD700; }}
    [data-testid="stSidebar"] > div:first-child {{ background-color: rgba(0, 0, 0, 0.75) !important; }}
    [data-testid="stSidebar"] * {{ color: #FFFFFF !important; }}

    div[data-testid="column"] {{
        background-color: #002D72 !important;
        padding: 25px !important;
        border-radius: 15px !important;
        border: 2px solid #FFD700 !important;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.2) !important;
    }}
    div[data-testid="column"] h3 {{ color: #FFD700 !important; border-bottom: 1px solid #FFFFFF; padding-bottom: 10px; }}

    .stSelectbox div[data-baseweb="select"], .stTextArea textarea, .stTextInput input {{
        background-color: #1E1E1E !important; color: #FFFFFF !important; border: 1px solid #FFD700 !important;
    }}

    .block-container {{ padding-top: 1rem !important; }}

    div[data-testid="stButton"] > button[kind="secondary"], 
    div[data-testid="stDownloadButton"] > button {{
        background-color: #222222 !important;
        border: 2px solid #FFD700 !important;
        color: #FFFFFF !important;
        font-weight: bold !important;
        height: 48px !important;
        border-radius: 8px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 0.9em !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }}

    div[data-testid="stButton"] > button[kind="secondary"] *, 
    div[data-testid="stDownloadButton"] > button * {{
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }}

    div[data-testid="stButton"] > button[kind="primary"] {{
        background-color: #FFD700 !important;
        border: 2px solid #FFD700 !important;
        color: #000000 !important;
        font-weight: bold !important;
        height: 48px !important;
        border-radius: 8px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 0.9em !important;
        box-shadow: 0 0 14px rgba(255, 215, 0, 0.9) !important;
        width: 100% !important;
    }}

    div[data-testid="stButton"] > button[kind="primary"] * {{
        color: #000000 !important;
        fill: #000000 !important;
        font-weight: bold !important;
    }}

    div[data-testid="stButton"] > button:hover, 
    div[data-testid="stDownloadButton"] > button:hover {{
        background-color: #FFD700 !important;
        color: #000000 !important;
        box-shadow: 0 0 10px rgba(255, 215, 0, 0.8) !important;
    }}

    div[data-testid="stButton"] > button:hover *, 
    div[data-testid="stDownloadButton"] > button:hover * {{
        color: #000000 !important;
        fill: #000000 !important;
    }}

    {css_icono_ver}
    {css_icono_descargar}

    hr {{ border-top: 2px solid #FFD700 !important; }}
    
    .caja-componentes {{
        background-color: #0A192F;
        border: 1px dashed #FFD700;
        border-radius: 10px;
        padding: 12px 15px;
        margin-top: 5px;
    }}
    </style>
""", unsafe_allow_html=True)

# --- ENCABEZADO FIJO ---
_logo_ab_b64 = get_base64("abinbev")
_logo_modelo_b64 = get_base64("modelo")
_logo_melena_b64 = get_base64("melena")

def _tag_logo(b64):
    if not b64: return ""
    return f'<img src="data:image/png;base64,{b64}" style="height:42px; margin-right:22px;">'

st.markdown(f"""
    <style>
    .melena-header-fijo {{
        position: sticky; top: 0; z-index: 999; background-color: #000000;
        border-bottom: 2px solid #FFD700; padding: 12px 20px;
        display: flex; justify-content: space-between; align-items: center; margin: 0 -1rem 1.5rem -1rem;
    }}
    .melena-header-logos {{ display: flex; align-items: center; }}
    .melena-header-titulo {{ text-align: right; }}
    .melena-header-titulo h1 {{ color: #FFD700 !important; font-size: 1.5em; margin: 0; line-height: 1.2; }}
    .melena-header-titulo p {{ color: #FFFFFF !important; font-size: 0.7em; margin: 0; font-weight: bold; line-height: 1.3; }}
    </style>

    <div class="melena-header-fijo">
        <div class="melena-header-logos">
            {_tag_logo(_logo_ab_b64)}
            {_tag_logo(_logo_modelo_b64)}
            {_tag_logo(_logo_melena_b64)}
        </div>
        <div class="melena-header-titulo">
            <h1>MELENA</h1>
            <p>Módulo Especializado de Lectura y Ejecución para Normas y Averías</p>
            <p>Líneas de Envasado 5 y 6 | Gestión de Excelencia Operativa VPO</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color: #FFD700 !important; margin-bottom:15px;'>Parametrización</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-weight:bold; margin-bottom:8px; color:#FFD700; font-size: 0.95em;'>LÍNEA DE PRODUCCIÓN</p>", unsafe_allow_html=True)

    _l5_activa = (st.session_state["linea_sel"] == "Línea 5")
    _l6_activa = (st.session_state["linea_sel"] == "Línea 6")

    col_l5, col_l6 = st.columns(2)
    with col_l5:
        if st.button("LÍNEA 5", key="btn_linea5", type="primary" if _l5_activa else "secondary", use_container_width=True):
            st.session_state["linea_sel"] = "Línea 5"
            st.rerun()
    with col_l6:
        if st.button("LÍNEA 6", key="btn_linea6", type="primary" if _l6_activa else "secondary", use_container_width=True):
            st.session_state["linea_sel"] = "Línea 6"
            st.rerun()

    linea_sel = st.session_state["linea_sel"]
    st.divider()

    area_sel = st.selectbox("Área Operativa:", areas_lista, key="area_sel")
    
    path_foto = encontrar_imagen(ESTRUCTURA_FALLAS[area_sel]["foto"])
    if path_foto:
        st.image(path_foto, use_container_width=True)
        st.caption(f"Activo seleccionado: {area_sel}")
    st.divider()

    st.markdown("<h3 style='color: #FFD700 !important; font-size: 1.1em;'>Índice de Documentos (IA)</h3>", unsafe_allow_html=True)
    try: _total_indexado = obtener_coleccion_vectorial().count()
    except Exception: _total_indexado = None

    if _total_indexado is None: st.caption("⚠️ No se pudo abrir el índice todavía.")
    elif _total_indexado == 0: st.caption("⚠️ El índice está vacío. Dale clic al botón de abajo.")
    else: st.caption(f"✅ {_total_indexado} fragmento(s) indexado(s) actualmente.")

    if st.button("Actualizar índice de documentos", key="btn_actualizar_indice", use_container_width=True):
        with st.spinner("Leyendo PDFs e indexando texto por páginas..."):
            archivos, chunks = indexar_todos_los_documentos()
        st.success(f"Índice actualizado: {archivos} documento(s), {chunks} fragmento(s).")

# --- NAVEGACIÓN Y PESTAÑAS PRINCIPALES ---
def _css_icono_boton(clave_boton, icono_b64):
    if not icono_b64: return ""
    return f"""
    .st-key-{clave_boton} button {{
        background-image: url(data:image/png;base64,{icono_b64});
        background-repeat: no-repeat; background-position: 16px center; background-size: 26px 26px;
        text-align: left !important; padding-left: 54px !important;
    }}
    """

st.markdown(f"""
    <style>
    {_css_icono_boton("btn_tab_plan", _icono_generador_b64)}
    {_css_icono_boton("btn_tab_docs", _icono_biblioteca_b64)}
    {_css_icono_boton("btn_tab_preguntas", _icono_preguntar_b64)}
    {_css_icono_boton("btn_tab_qr", _icono_escanear_b64)}
    </style>
""", unsafe_allow_html=True)

_tab = st.session_state["tab_activa"]
p_plan = (_tab == "plan")
p_docs = (_tab == "docs")
p_preg = (_tab == "preguntas")
p_qr = (_tab == "qr")

menu_col1, menu_col2, menu_col3, menu_col4 = st.columns(4)

with menu_col1:
    if st.button("Generador de Plan de Reacción", key="btn_tab_plan", type="primary" if p_plan else "secondary", use_container_width=True):
        st.session_state.tab_activa = "plan"
        st.rerun()
with menu_col2:
    if st.button("Biblioteca de Documentos", key="btn_tab_docs", type="primary" if p_docs else "secondary", use_container_width=True):
        st.session_state.tab_activa = "docs"
        st.rerun()
with menu_col3:
    if st.button("Preguntar a los Documentos", key="btn_tab_preguntas", type="primary" if p_preg else "secondary", use_container_width=True):
        st.session_state.tab_activa = "preguntas"
        st.rerun()
with menu_col4:
    if st.button("Escanear Código QR", key="btn_tab_qr", type="primary" if p_qr else "secondary", use_container_width=True):
        st.session_state.tab_activa = "qr"
        st.rerun()

st.divider()

# --- SECCIÓN 1: PLAN DE REACCIÓN ---
if st.session_state.tab_activa == "plan":
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        st.markdown(f"<h3>Registro de Falla: {area_sel}</h3>", unsafe_allow_html=True)
        
        col_form, col_componentes = st.columns([1.4, 1], gap="medium")
        
        with col_form:
            tipo_falla_sel = st.selectbox("Tipo de falla:", OPCIONES_TIPO_FALLA)
            desc = st.text_area("Descripción técnica de la desviación:", placeholder="Detalles observados...", height=110)
            btn = st.button("GENERAR PLAN DE ACCIÓN", type="primary")

        with col_componentes:
            tag_adv = f'<img src="data:image/png;base64,{_icono_advertencia_b64}" style="height:22px; width:22px; vertical-align:middle; margin-right:6px;">' if _icono_advertencia_b64 else ''
            st.markdown(f"<p style='font-weight: bold; color: #FFD700; margin-bottom: 6px; font-size: 0.9em; display: flex; align-items: center;'>{tag_adv}Componentes críticos de la máquina:</p>", unsafe_allow_html=True)
            
            comp_list = list(ESTRUCTURA_FALLAS[area_sel]["componentes"].keys())
            
            html_componentes = "<div class='caja-componentes'>"
            for comp in comp_list:
                html_componentes += f"<p style='margin: 0 0 4px 0; font-size: 0.85em; color: #FFFFFF;'>• {comp}</p>"
            html_componentes += "</div>"
            
            st.markdown(html_componentes, unsafe_allow_html=True)

    with c2:
        st.markdown("<h3>Plan de Reacción Validado</h3>", unsafe_allow_html=True)
        if btn:
            if desc:
                placeholder = st.empty()
                
                # --- VERIFICACIÓN DE CACHÉ SEMÁNTICO EN PLAN DE REACCIÓN ---
                consulta_texto = f"{tipo_falla_sel} {desc}"
                clave_filtro_plan = f"PLAN|{linea_sel}|{area_sel}"
                emb_consulta = obtener_embedding(consulta_texto)
                
                item_cached, sim_score = consultar_cache_semantico(emb_consulta, clave_filtro_plan, umbral=0.92)
                
                if item_cached:
                    placeholder.empty()
                    st.info(f"⚡ RESPUESTA RECUPERADA EN <0.5s DESDE CACHÉ SEMÁNTICO (Similitud: {sim_score*100:.1f}%)")
                    if item_cached["es_oficial"]:
                        st.success(f"✅ INFORMACIÓN SUSTENTADA EN MANUALES DE {linea_sel.upper()} - {area_sel.upper()}")
                    else:
                        st.warning(f"⚠️ RESPUESTA BASADA EN CONOCIMIENTO GENERAL (Sin documentos cargados para {linea_sel} - {area_sel})")
                    
                    st.markdown(f"<div style='color: white; font-size: 1.1em;'>{item_cached['respuesta']}</div>", unsafe_allow_html=True)
                    
                    if item_cached.get("evidencias"):
                        st.divider()
                        st.markdown(f"<h4 style='color: #FFD700;'>🔍 Evidencias Visuales</h4>", unsafe_allow_html=True)
                        titulos_tabs = [f"{'🖼️' if e['tiene_imagen'] else '📄'} Pág. {e['pagina']}" for e in item_cached["evidencias"]]
                        tabs = st.tabs(titulos_tabs)
                        for idx, tab in enumerate(tabs):
                            e = item_cached["evidencias"][idx]
                            with tab:
                                img_bytes = obtener_imagen_pagina_ram(e["pdf_path"], e["pagina"], dpi=300)
                                st.image(img_bytes, caption=f"Documento: {e['archivo']} — Pág. {e['pagina']}", use_container_width=True)
                else:
                    gif_b64 = get_base64("procesando")
                    if gif_b64:
                        placeholder.markdown(f'''
                            <div style="text-align: center; padding: 20px;">
                                <img src="data:image/gif;base64,{gif_b64}" style="width: 150px; border-radius: 50%; border: 3px solid #FFD700;">
                                <p style="color: #FFD700; font-weight: bold; margin-top: 15px;">Consultando manuales oficiales VPO...</p>
                            </div>''', unsafe_allow_html=True)

                    fragmentos_pdf = buscar_contexto(
                        consulta_texto, 
                        linea=linea_sel, 
                        area=area_sel, 
                        top_k=8, 
                        permitir_fallback=False
                    )

                    manual_file = os.path.join(DATA_DIR, f"{area_sel}.txt")
                    es_oficial = False
                    contexto = ""

                    if fragmentos_pdf:
                        es_oficial = True
                        contexto += "\n\n".join(f"[Fuente: {f['tipo']} - {f['archivo']}]\n{f['texto']}" for f in fragmentos_pdf)

                    if os.path.exists(manual_file) and os.path.getsize(manual_file) > 10:
                        with open(manual_file, "r", encoding="utf-8") as f: 
                            contexto = f.read() + "\n\n" + contexto
                        es_oficial = True

                    if es_oficial and contexto.strip():
                        instruccion_fuente = f"DOCUMENTACIÓN OFICIAL REGISTRADA PARA {linea_sel} - {area_sel}:\n{contexto}"
                    else:
                        instruccion_fuente = (
                            f"ATENCIÓN: No existen documentos cargados ni manuales registrados para {linea_sel} - {area_sel}.\n"
                            "Genera el plan de acción basándote exclusivamente en tu conocimiento técnico general "
                            "como especialista en envasado de cerveza en botella de vidrio."
                        )

                    sys_prompt_plan = (
                        "Eres el Ingeniero Especialista en VPO (Global Management System) de Grupo Modelo / AB InBev, "
                        "experto exclusivo en plantas industriales de envasado de CERVEZA EN BOTELLA DE VIDRIO (Líneas 5 y 6).\n\n"
                        "REGLAS OBLIGATORIAS:\n"
                        "1. Tu dominio técnico es EXCLUSIVAMENTE la producción y envasado de cerveza en botella de vidrio.\n"
                        "2. QUEDA PROHIBIDO mencionar o usar ejemplos de leche, jugos, refrescos, latas, plásticos PET o alimentos ajenos a la industria cervecera.\n"
                        "3. Toda explicación sobre maquinaria (pasteurizadores túnel, lavadoras de botellas, llenadoras isobarométricas, motobombas, etc.) "
                        "debe basarse en el proceso cervecero (duchas de agua, choque térmico en vidrio, unidades de pasteurización UP, etc.).\n"
                        "4. Responde con lenguaje técnico, profesional y en pasos numerados claros.\n\n"
                        f"{instruccion_fuente}"
                    )

                    try:
                        if not client_groq:
                            raise ValueError("No se configuró la variable GROQ_API_KEY")

                        res = client_groq.chat.completions.create(
                            model=MODELO_CHAT,
                            messages=[
                                {'role': 'system', 'content': sys_prompt_plan},
                                {'role': 'user', 'content': f'Línea {linea_sel}, Área {area_sel}, Tipo de Falla: {tipo_falla_sel}. Detalles: {desc}'},
                            ]
                        )
                        placeholder.empty()
                        
                        texto_res = res.choices[0].message.content
                        evidencias_calc = obtener_evidencias_priorizadas(fragmentos_pdf) if fragmentos_pdf else []
                        
                        # Guardar en Caché Semántico
                        guardar_en_cache_semantico(emb_consulta, clave_filtro_plan, texto_res, es_oficial, fragmentos_pdf, evidencias_calc)
                        
                        if es_oficial: 
                            st.success(f"✅ INFORMACIÓN SUSTENTADA EN MANUALES DE {linea_sel.upper()} - {area_sel.upper()}")
                        else: 
                            st.warning(f"⚠️ RESPUESTA BASADA EN CONOCIMIENTO GENERAL (Sin documentos cargados para {linea_sel} - {area_sel})")
                        
                        st.markdown(f"<div style='color: white; font-size: 1.1em;'>{texto_res}</div>", unsafe_allow_html=True)
                        
                        if evidencias_calc:
                            st.divider()
                            st.markdown(f"<h4 style='color: #FFD700;'>🔍 Evidencias Visuales</h4>", unsafe_allow_html=True)
                            titulos_tabs = [f"{'🖼️' if e['tiene_imagen'] else '📄'} Pág. {e['pagina']}" for e in evidencias_calc]
                            tabs = st.tabs(titulos_tabs)
                            for idx, tab in enumerate(tabs):
                                e = evidencias_calc[idx]
                                with tab:
                                    img_bytes = obtener_imagen_pagina_ram(e["pdf_path"], e["pagina"], dpi=300)
                                    st.image(img_bytes, caption=f"Documento: {e['archivo']} — Pág. {e['pagina']}", use_container_width=True)
                    except Exception as e:
                        placeholder.empty()
                        st.error(f"Error al conectar con la API de Groq: {e}")
            else:
                st.error("Por favor, ingrese una descripción técnica.")

# --- SECCIÓN 2: BIBLIOTECA DE DOCUMENTOS ---
elif st.session_state.tab_activa == "docs":
    st.markdown(f"<h3>Documentos disponibles: {area_sel} — {linea_sel}</h3>", unsafe_allow_html=True)
    tipo_doc_sel = st.selectbox("Tipo de documento:", list(TIPO_CARPETA.keys()))
    pdfs_encontrados = listar_pdfs(linea_sel, area_sel, tipo_doc_sel)
    
    LIMITE_PREVIEW_MB = 8

    if pdfs_encontrados:
        st.success(f"Se encontraron {len(pdfs_encontrados)} documento(s).")
        for idx, ruta_pdf in enumerate(pdfs_encontrados):
            nombre_archivo = os.path.basename(ruta_pdf)
            pdf_bytes = leer_pdf_bytes_cacheado(ruta_pdf, os.path.getmtime(ruta_pdf))
            tamano_mb = len(pdf_bytes) / (1024 * 1024)

            clave_ver = f"ver_doc_{TIPO_CARPETA[tipo_doc_sel]}_{idx}"
            clave_btn_ver = f"btn_{clave_ver}"
            clave_descarga = f"descarga_{TIPO_CARPETA[tipo_doc_sel]}_{idx}"
            if clave_ver not in st.session_state: st.session_state[clave_ver] = False

            col_nombre, col_ver, col_descarga = st.columns([3, 1, 1])
            with col_nombre:
                st.markdown(f"**{nombre_archivo}** &nbsp; <span style='color:#AAAAAA; font-size:0.8em;'>({tamano_mb:.1f} MB)</span>", unsafe_allow_html=True)
            with col_ver:
                if st.button("VER", key=clave_btn_ver, use_container_width=True):
                    st.session_state[clave_ver] = not st.session_state[clave_ver]
            with col_descarga:
                st.download_button(
                    label="DESCARGAR",
                    data=pdf_bytes,
                    file_name=nombre_archivo,
                    mime="application/pdf",
                    key=clave_descarga,
                    use_container_width=True,
                )

            if st.session_state[clave_ver]:
                if tamano_mb > LIMITE_PREVIEW_MB:
                    st.warning(f"Este archivo pesa {tamano_mb:.1f} MB, descarga directa para visualización completa.")
                else:
                    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
                    st.markdown(f'<iframe src="data:application/pdf;base64,{pdf_b64}" width="100%" height="650" style="border: 1px solid #FFD700; border-radius: 8px;"></iframe>', unsafe_allow_html=True)
            st.divider()
    else:
        st.info("No hay documentos disponibles para la selección actual.")

# --- SECCIÓN 3: PREGUNTAR A LOS DOCUMENTOS ---
elif st.session_state.tab_activa == "preguntas":
    st.markdown("<h3>Pregúntale a los manuales, PDR, SOP y RDA</h3>", unsafe_allow_html=True)
    
    col_f1, col_f2 = st.columns(2)
    with col_f1: usar_linea_actual = st.checkbox(f"Buscar solo en {linea_sel}", value=True)
    with col_f2: usar_area_actual = st.checkbox(f"Buscar solo en {area_sel}", value=True)

    filtro_linea = linea_sel if usar_linea_actual else None
    filtro_area = area_sel if usar_area_actual else None

    alcance_busqueda = st.radio(
        "Modo de filtrado de documentos:",
        ["Todos los documentos del filtro", "Un documento en específico"],
        horizontal=True
    )

    archivo_especifico = None
    if alcance_busqueda == "Un documento en específico":
        archivos_disponibles = listar_archivos_pdf_filtrados(filtro_linea, filtro_area)
        if archivos_disponibles:
            archivo_especifico = st.selectbox("Selecciona el documento específico:", archivos_disponibles)
        else:
            st.warning("⚠️ No se encontraron documentos PDF con los filtros de Línea y Área aplicados.")

    pregunta_libre = st.text_area("Escribe tu pregunta:", placeholder="Ej. ¿De qué marca es la lavadora?", height=100)
    btn_preguntar = st.button("PREGUNTAR A LOS DOCUMENTOS", type="primary")

    if btn_preguntar:
        if pregunta_libre.strip():
            if alcance_busqueda == "Un documento en específico" and not archivo_especifico:
                st.error("Por favor selecciona un documento válido de la lista.")
            else:
                # --- VERIFICACIÓN DE CACHÉ SEMÁNTICO EN PREGUNTAS LIBRES ---
                clave_filtro_preg = f"PREG|{filtro_linea}|{filtro_area}|{archivo_especifico}"
                emb_pregunta = obtener_embedding(pregunta_libre)
                
                item_cached, sim_score = consultar_cache_semantico(emb_pregunta, clave_filtro_preg, umbral=0.92)
                
                if item_cached:
                    st.info(f"⚡ RESPUESTA RECUPERADA EN <0.5s DESDE CACHÉ SEMÁNTICO (Similitud semántica: {sim_score*100:.1f}%)")
                    col_ans1, col_ans2 = st.columns([1.1, 1], gap="medium")
                    
                    with col_ans1:
                        st.markdown(f"<div style='color: white; font-size: 1.05em;'>{item_cached['respuesta']}</div>", unsafe_allow_html=True)
                        if item_cached.get("fragmentos"):
                            with st.expander("📎 Fuentes consultadas"):
                                for f in item_cached["fragmentos"]:
                                    pdf_p, p_num = resolver_ruta_y_pagina(f)
                                    st.markdown(f"**{f['tipo']} — {f['archivo']} (Pág. {p_num})**")
                                    st.caption(f['texto'][:250] + "...")
                    
                    with col_ans2:
                        evidencias = item_cached.get("evidencias", [])
                        if evidencias:
                            st.markdown(f"<h4 style='color: #FFD700;'>🔍 Evidencias Visuales ({len(evidencias)})</h4>", unsafe_allow_html=True)
                            titulos_tabs = [f"{'🖼️️' if e['tiene_imagen'] else '📄'} Pág. {e['pagina']}" for e in evidencias]
                            tabs = st.tabs(titulos_tabs)
                            
                            for idx, tab in enumerate(tabs):
                                e = evidencias[idx]
                                with tab:
                                    if e['tiene_imagen']:
                                        st.caption("⭐ **Prioridad:** Página con diagramas/fotografías.")
                                    else:
                                        st.caption("📄 Página con referencia en texto.")
                                    
                                    try:
                                        img_bytes = obtener_imagen_pagina_ram(e["pdf_path"], e["pagina"], dpi=300)
                                        st.image(img_bytes, caption=f"{e['tipo']} - {e['archivo']} (Pág. {e['pagina']})", use_container_width=True)
                                    except Exception as img_err:
                                        st.caption(f"Error al renderizar vista previa: {img_err}")
                        else:
                            st.caption("No se encontraron páginas asociadas en los archivos PDF originales.")
                else:
                    with st.spinner("Consultando en la base de datos de documentos..."):
                        permitir_fb = (alcance_busqueda == "Todos los documentos del filtro")
                        fragmentos = buscar_contexto(
                            pregunta_libre, 
                            linea=filtro_linea, 
                            area=filtro_area, 
                            archivo=archivo_especifico,
                            top_k=8, 
                            permitir_fallback=permitir_fb
                        )

                        if not fragmentos:
                            st.warning("⚠️ No se encontró información relevante para la consulta solicitada.")
                        else:
                            if fragmentos[0].get("es_fallback"):
                                st.info("ℹ️ Se obtuvieron coincidencias en documentos generales del sistema:")

                            contexto_docs = "\n\n".join(f"[Fuente: {f['tipo']} - {f['archivo']}]\n{f['texto']}" for f in fragmentos)
                            
                            sys_prompt_preguntas = (
                                "Eres el Asistente Técnico Especializado del Sistema VPO en Grupo Modelo / AB InBev, "
                                "diseñado exclusivamente para líneas de envasado industrial de CERVEZA EN BOTELLA DE VIDRIO.\n\n"
                                "REGLAS DE RESPUESTA STRICTAS:\n"
                                "1. Trabajas ÚNICAMENTE en la industria cervecera. JAMÁS bajo ningún motivo menciones productos como leche, jugos, salsas, refrescos, latas o envases PET.\n"
                                "2. Contextualiza los componentes según la cerveza embotellada. Por ejemplo: las motobombas del pasteurizador sirven para bombear y recircular el agua de las duchas de recargado/calentamiento del túnel de pasteurización para asegurar la estabilización microbiológica de la cerveza en botellas de vidrio.\n"
                                "3. Sé conciso, directo, sumamente técnico y transparente. Basate de forma directa en el contexto provisto.\n\n"
                                f"DOCUMENTACIÓN Y MANUALES OFICIALES VPO:\n{contexto_docs}"
                            )

                            try:
                                if not client_groq:
                                    raise ValueError("No se configuró la variable GROQ_API_KEY")

                                res = client_groq.chat.completions.create(
                                    model=MODELO_CHAT,
                                    messages=[
                                        {'role': 'system', 'content': sys_prompt_preguntas},
                                        {'role': 'user', 'content': pregunta_libre},
                                    ]
                                )
                                
                                texto_res = res.choices[0].message.content
                                evidencias_calc = obtener_evidencias_priorizadas(fragmentos)
                                
                                # Guardar en Caché Semántico
                                guardar_en_cache_semantico(emb_pregunta, clave_filtro_preg, texto_res, True, fragmentos, evidencias_calc)
                                
                                col_ans1, col_ans2 = st.columns([1.1, 1], gap="medium")
                                
                                with col_ans1:
                                    st.markdown(f"<div style='color: white; font-size: 1.05em;'>{texto_res}</div>", unsafe_allow_html=True)
                                    with st.expander("📎 Fuentes consultadas"):
                                        for f in fragmentos:
                                            pdf_p, p_num = resolver_ruta_y_pagina(f)
                                            st.markdown(f"**{f['tipo']} — {f['archivo']} (Pág. {p_num})**")
                                            st.caption(f['texto'][:250] + "...")
                                
                                with col_ans2:
                                    if evidencias_calc:
                                        st.markdown(f"<h4 style='color: #FFD700;'>🔍 Evidencias Visuales ({len(evidencias_calc)})</h4>", unsafe_allow_html=True)
                                        
                                        titulos_tabs = [f"{'🖼️' if e['tiene_imagen'] else '📄'} Pág. {e['pagina']}" for e in evidencias_calc]
                                        tabs = st.tabs(titulos_tabs)
                                        
                                        for idx, tab in enumerate(tabs):
                                            e = evidencias_calc[idx]
                                            with tab:
                                                if e['tiene_imagen']:
                                                    st.caption("⭐ **Prioridad:** Página con diagramas/fotografías.")
                                                else:
                                                    st.caption("📄 Página con referencia en texto.")
                                                
                                                try:
                                                    img_bytes = obtener_imagen_pagina_ram(e["pdf_path"], e["pagina"], dpi=300)
                                                    st.image(img_bytes, caption=f"{e['tipo']} - {e['archivo']} (Pág. {e['pagina']})", use_container_width=True)
                                                except Exception as img_err:
                                                    st.caption(f"Error al renderizar vista previa: {img_err}")
                                    else:
                                        st.caption("No se encontraron páginas asociadas en los archivos PDF originales.")
                            except Exception as e:
                                st.error(f"Error al conectar con la API de Groq: {e}")
        else:
            st.error("Por favor, escribe una pregunta.")

# --- SECCIÓN 4: ESCÁNER QR ---
elif st.session_state.tab_activa == "qr":
    st.markdown("<h3>Escáner QR</h3>", unsafe_allow_html=True)

    if WEBRTC_DISPONIBLE:
        st.caption("Apunta la cámara al código QR de la máquina. En cuanto lo detecte, te lleva automáticamente a la Biblioteca de Documentos — no hace falta tomar foto.")

        ctx = webrtc_streamer(
            key="lector_qr_vivo",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=ProcesadorQR,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": []},
        )

        if ctx.state.playing:
            st_autorefresh(interval=700, key="autorefresco_qr")
            if ctx.video_processor and ctx.video_processor.resultado:
                contenido = ctx.video_processor.resultado
                linea_det, area_det = parsear_codigo_qr(contenido)
                if linea_det and area_det:
                    st.success(f"Código leído: {linea_det} — {area_det}. Abriendo la Biblioteca de Documentos...")
                    st.session_state["_pendiente_linea"] = linea_det
                    st.session_state["_pendiente_area"] = area_det
                    st.session_state["tab_activa"] = "docs"
                    st.rerun()
                else:
                    st.warning(f"Se detectó un código ('{contenido}') pero no coincide con ninguna línea/área conocida. Sigue apuntando a otro QR.")
                    ctx.video_processor.resultado = None
    else:
        st.warning("La cámara en vivo necesita instalar librerías adicionales en el servidor: `pip install streamlit-webrtc streamlit-autorefresh av`. Mientras tanto, usa el texto manual de abajo.")

    st.divider()
    st.caption("¿La cámara no abre? Escribe el texto del código manualmente:")

    col_qr1, col_qr2 = st.columns([3, 1])
    with col_qr1:
        st.text_input("Ingresar o simular código QR manualmente:", placeholder="Ej. Línea 5 Lavadora", key="qr_input_manual", label_visibility="collapsed")
    with col_qr2:
        st.button("PROCESAR QR", type="primary", use_container_width=True, on_click=procesar_qr_manual_callback)

st.markdown("<br><br><hr>", unsafe_allow_html=True)
st.markdown("<center><p style='color: #FFD700; font-weight: bold;'>MELENA | CONFIDENCIAL - Uso exclusivo personal autorizado Grupo Modelo</p></center>", unsafe_allow_html=True)