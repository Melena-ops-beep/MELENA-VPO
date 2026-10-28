import streamlit as st
import streamlit.components.v1 as components
import os
import base64
import re
import unicodedata
import numpy as np
from pypdf import PdfReader
import chromadb
import fitz  # PyMuPDF para renderizado en RAM a 300 DPI
from sentence_transformers import SentenceTransformer

# --- CONFIGURACIÓN DE RUTAS MULTIPLATAFORMA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "img")
DATA_DIR = os.path.join(BASE_DIR, "CONOCIMIENTO_VPO")
DOCS_DIR = os.path.join(BASE_DIR, "DOCUMENTOS")
INDICE_DIR = os.path.join(BASE_DIR, "INDICE_VECTORIAL")
MODELO_CHAT = "phi3"

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="MELENA - VPO Support System", layout="wide")

st.markdown(
    """
    <style>
    .element-container, .stTextInput, .stSelectbox, .stButton {
        width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
    extensiones = ['.png', '.gif', '.jpg', '.jpeg']
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

def limpiar_texto_pdf(texto):
    if not texto: return ""
    # Elimina los guiones de separación al final o inicio de línea, uniendo la palabra correctamente
    texto_limpio = re.sub(r'([a-zA-ZáéíóúÁÉÍÓÚñÑ]+)-\s*[\r\n]+\s*([a-zA-ZáéíóúÁÉÍÓÚñÑ]+)', r'\1\2', texto)
    texto_limpio = re.sub(r'([a-zA-ZáéíóúÁÉÍÓÚñÑ]+)-\s+([a-zA-ZáéíóúÁÉÍÓÚñÑ]+)', r'\1\2', texto_limpio)
    # Reemplaza saltos de línea internos por espacios limpios
    texto_limpio = re.sub(r'\s*\n\s*', ' ', texto_limpio)
    # Elimina espacios dobles sobrantes
    texto_limpio = re.sub(r'\s+', ' ', texto_limpio)
    return texto_limpio.strip()

# --- MODELO DE EMBEDDINGS EN LA NUBE ---
@st.cache_resource
def cargar_modelo_embedding():
    return SentenceTransformer('all-MiniLM-L6-v2')

def obtener_embedding(texto):
    texto_seguro = texto[:1000]
    modelo = cargar_modelo_embedding()
    vector = modelo.encode(texto_seguro)
    return vector.tolist()

# --- INICIALIZACIÓN DE ESTADOS ---
areas_lista = list(ESTRUCTURA_FALLAS.keys())

if "linea_sel" not in st.session_state:
    st.session_state["linea_sel"] = "Línea 5"

if "tab_activa" not in st.session_state:
    st.session_state["tab_activa"] = "plan"

if "area_sel" not in st.session_state or st.session_state["area_sel"] not in areas_lista:
    st.session_state["area_sel"] = areas_lista[0]

if "mensajes_chat" not in st.session_state:
    st.session_state["mensajes_chat"] = [
        {"role": "assistant", "content": "Hola. Soy MELENA, tu asistente técnico de VPO especializado en envasado de cerveza en botella de vidrio. ¿Qué avería o seguimiento deseas consultar?"}
    ]

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

def buscar_contexto_global(pregunta, top_k=3):
    coleccion = obtener_coleccion_vectorial()
    if coleccion.count() == 0: return []
    embedding_pregunta = obtener_embedding(pregunta)
    resultados = coleccion.query(query_embeddings=[embedding_pregunta], n_results=top_k)
    
    fragmentos = []
    if resultados and resultados.get('documents') and resultados['documents'][0]:
        for doc, meta in zip(resultados['documents'][0], resultados['metadatas'][0]):
            fragmentos.append({
                "texto": doc, 
                "archivo": meta.get('archivo'), 
                "tipo": meta.get('tipo'), 
                "ruta_pdf": meta.get('ruta_pdf'),
                "pagina": meta.get('pagina')
            })
    return fragmentos

def buscar_contexto(pregunta, linea=None, area=None, archivo=None, top_k=3, permitir_fallback=True):
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
_icono_chat_b64 = get_base64("icono_preguntar")
_icono_subir_b64 = get_base64("subir")
_avatar_robot_b64 = get_base64("robot")

def _css_icono_boton(clave_boton, icono_b64):
    if not icono_b64: return ""
    return f"""
    .st-key-{clave_boton} button {{
        background-image: url(data:image/png;base64,{icono_b64}) !important;
        background-repeat: no-repeat !important;
        background-position: 14px center !important;
        background-size: 22px 22px !important;
        text-align: left !important;
        padding-left: 46px !important;
    }}
    """

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

    div[data-testid="stButton"] > button[kind="secondary"] {{
        background-color: #222222 !important;
        border: 2px solid #FFD700 !important;
        color: #FFFFFF !important;
        font-weight: bold !important;
        height: 48px !important;
        border-radius: 8px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 0.85em !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
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
        font-size: 0.85em !important;
        box-shadow: 0 0 14px rgba(255, 215, 0, 0.9) !important;
        width: 100% !important;
    }}

    div[data-testid="stButton"] > button:hover, 
    div[data-testid="stDownloadButton"] > button:hover {{
        background-color: #FFD700 !important;
        color: #000000 !important;
    }}

    {_css_icono_boton("btn_tab_plan", _icono_generador_b64)}
    {_css_icono_boton("btn_tab_docs", _icono_biblioteca_b64)}
    {_css_icono_boton("btn_tab_preguntas", _icono_preguntar_b64)}
    {_css_icono_boton("btn_tab_chat", _icono_chat_b64)}
    {_css_icono_boton("btn_tab_subir", _icono_subir_b64)}

    hr {{ border-top: 2px solid #FFD700 !important; }}
    
    .caja-componentes {{
        background-color: #0A192F;
        border: 1px dashed #FFD700;
        border-radius: 10px;
        padding: 12px 15px;
        margin-top: 5px;
    }}

    .tarjeta-manual {{
        background-color: #111B2B;
        border-left: 4px solid #FFD700;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 12px;
        color: #FFFFFF !important;
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
    <div style="position: sticky; top: 0; z-index: 999; background-color: #000000; border-bottom: 2px solid #FFD700; padding: 12px 20px; display: flex; justify-content: space-between; align-items: center; margin: 0 -1rem 1.5rem -1rem;">
        <div style="display: flex; align-items: center;">
            {_tag_logo(_logo_ab_b64)}
            {_tag_logo(_logo_modelo_b64)}
            {_tag_logo(_logo_melena_b64)}
        </div>
        <div style="text-align: right;">
            <h1 style="color: #FFD700 !important; font-size: 1.5em; margin: 0;">MELENA</h1>
            <p style="color: #FFFFFF !important; font-size: 0.7em; margin: 0; font-weight: bold;">Módulo Especializado de Lectura y Ejecución para Normas y Averías</p>
            <p style="color: #FFFFFF !important; font-size: 0.7em; margin: 0; font-weight: bold;">Líneas de Envasado 5 y 6 | Gestión de Excelencia Operativa VPO</p>
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

    if _total_indexado is None: st.caption("No se pudo abrir el índice en este momento.")
    elif _total_indexado == 0: st.caption("El índice está vacío. Presione el botón de abajo.")
    else: st.caption(f"{_total_indexado} fragmento(s) indexado(s) actualmente.")

    if st.button("Actualizar índice de documentos", key="btn_actualizar_indice", use_container_width=True):
        with st.spinner("Procesando documentos e indexando contenido por páginas..."):
            archivos, chunks = indexar_todos_los_documentos()
        st.success(f"Índice actualizado: {archivos} documento(s), {chunks} fragmento(s).")

# --- NAVEGACIÓN Y PESTAÑAS PRINCIPALES (5 PESTAÑAS) ---
_tab = st.session_state.get("tab_activa", "plan")
p_plan = (_tab == "plan")
p_docs = (_tab == "docs")
p_preg = (_tab == "preguntas")
p_chat = (_tab == "chat")
p_subir = (_tab == "subir")

menu_col1, menu_col2, menu_col3, menu_col4, menu_col5 = st.columns(5)

with menu_col1:
    if st.button("Generador Plan Reacción", key="btn_tab_plan", type="primary" if p_plan else "secondary", use_container_width=True):
        st.session_state.tab_activa = "plan"
        st.rerun()
with menu_col2:
    if st.button("Biblioteca Documentos", key="btn_tab_docs", type="primary" if p_docs else "secondary", use_container_width=True):
        st.session_state.tab_activa = "docs"
        st.rerun()
with menu_col3:
    if st.button("Preguntar a Documentos", key="btn_tab_preguntas", type="primary" if p_preg else "secondary", use_container_width=True):
        st.session_state.tab_activa = "preguntas"
        st.rerun()
with menu_col4:
    if st.button("Chat IA Seguimiento", key="btn_tab_chat", type="primary" if p_chat else "secondary", use_container_width=True):
        st.session_state.tab_activa = "chat"
        st.rerun()
with menu_col5:
    if st.button("Subir un documento", key="btn_tab_subir", type="primary" if p_subir else "secondary", use_container_width=True):
        st.session_state.tab_activa = "subir"
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
            st.markdown("<p style='font-weight: bold; color: #FFD700; margin-bottom: 6px; font-size: 0.9em;'>Componentes críticos de la máquina:</p>", unsafe_allow_html=True)
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
                consulta_texto = f"{tipo_falla_sel} {desc}"
                
                gif_b64 = get_base64("procesando")
                if gif_b64:
                    placeholder.markdown(f'''
                        <div style="text-align: center; padding: 20px;">
                            <img src="data:image/gif;base64,{gif_b64}" style="width: 140px; border-radius: 50%; border: 3px solid #FFD700;">
                            <p style="color: #FFD700; font-weight: bold; margin-top: 15px;">Generando plan táctico VPO...</p>
                        </div>''', unsafe_allow_html=True)
                else:
                    placeholder.info("Generando plan táctico VPO...")

                fragmentos_pdf = buscar_contexto(consulta_texto, linea=linea_sel, area=area_sel, top_k=3)
                placeholder.empty()
                
                es_oficial = len(fragmentos_pdf) > 0
                if es_oficial:
                    st.success(f"Plan de acción estructurado con base en manuales de {linea_sel.upper()} - {area_sel.upper()}")
                else:
                    st.warning(f"Plan de acción generado con base en directrices estándar VPO (Sin documentos específicos indexados para {linea_sel} - {area_sel})")

                respuesta_generada = f"""
### 🛠️ PLAN TÁCTICO DE REACCIÓN VPO
**Área / Equipo:** {area_sel} ({linea_sel})  
**Desviación Detectada:** *{desc}*

---

#### **1. Acciones Preliminares y de Seguridad (Obligatorio)**
- [ ] **1.1.** Notificar al operador de la línea y al supervisor de turno sobre la detención temporal del equipo.
- [ ] **1.2.** Aplicar el protocolo de **LOTOTO** (Bloqueo, Etiquetado y Verificación de Energía Cero) o **SAM** (Paro Seguro) según el tipo de intervención requerida.
- [ ] **1.3.** Delimitación y limpieza inicial del área de trabajo de residuos de vidrio, etiquetas o humedad.

#### **2. Acciones Específicas de Diagnóstico y Corrección**
- [ ] **2.1.** Inspeccionar visualmente los componentes asociados (*{', '.join(list(ESTRUCTURA_FALLAS[area_sel]['componentes'].keys())[:2])}*) en busca de holguras, desalineación o desgaste.
- [ ] **2.2.** Si aplica lubricación: limpiar el punto de engrase y aplicar grasa o aceite grado alimentario autorizado según especificación del fabricante.
- [ ] **2.3.** Si es falla eléctrica/sensores: verificar limpieza óptica de lentes, LED de alimentación y correcto ajuste de tornillería en soportes.

#### **3. Validación y Arranque de Línea**
- [ ] **3.1.** Retirar herramientas, equipo de medición y personal de la zona de riesgo.
- [ ] **3.2.** Desactivar bloqueos de seguridad y realizar un ciclo de prueba en vacío (*dry run*).
- [ ] **3.3.** Liberar el equipo con el departamento de Calidad para la primera inspección de producto terminado en botella de vidrio.

---
**📖 Fuentes y Extractos Oficiales de Consulta:**
"""
                if es_oficial:
                    for f in fragmentos_pdf:
                        texto_limpio = limpiar_texto_pdf(f.get('texto', ''))
                        respuesta_generada += f"> **[{f.get('tipo', 'Manual')} — {f.get('archivo', 'Documento')} | Página {f.get('pagina', 1)}]**\n> {texto_limpio}\n\n"
                else:
                    respuesta_generada += "> *No se encontraron manuales específicos en la ruta actual; aplique los criterios generales de mantenimiento autónomo TPM de Grupo Modelo.*\n"

                st.markdown(respuesta_generada)
                
                evidencias_calc = obtener_evidencias_priorizadas(fragmentos_pdf) if fragmentos_pdf else []
                if evidencias_calc:
                    st.divider()
                    st.markdown("<h4 style='color: #FFD700;'>Evidencias Visuales de Referencia</h4>", unsafe_allow_html=True)
                    titulos_tabs = [f"Pág. {e['pagina']} ({e['archivo']})" for e in evidencias_calc]
                    tabs = st.tabs(titulos_tabs)
                    for idx, tab in enumerate(tabs):
                        e = evidencias_calc[idx]
                        with tab:
                            img_bytes = obtener_imagen_pagina_ram(e["pdf_path"], e["pagina"], dpi=300)
                            st.image(img_bytes, caption=f"Fuente: {e['archivo']} — Página {e['pagina']}", use_container_width=True)
            else:
                st.error("Por favor, ingrese una descripción técnica.")

# --- SECCIÓN 2: BIBLIOTECA DE DOCUMENTOS ---
elif st.session_state.tab_activa == "docs":
    st.markdown(f"<h3>Documentos disponibles: {area_sel} — {linea_sel}</h3>", unsafe_allow_html=True)
    tipo_doc_sel = st.selectbox("Tipo de documento:", list(TIPO_CARPETA.keys()))
    pdfs_encontrados = listar_pdfs(linea_sel, area_sel, tipo_doc_sel)
    
    if pdfs_encontrados:
        st.success(f"Se encontraron {len(pdfs_encontrados)} documento(s).")
        for idx, ruta_pdf in enumerate(pdfs_encontrados):
            nombre_archivo = os.path.basename(ruta_pdf)
            pdf_bytes = leer_pdf_bytes_cacheado(ruta_pdf, os.path.getmtime(ruta_pdf))
            st.download_button(label=f"DESCARGAR {nombre_archivo}", data=pdf_bytes, file_name=nombre_archivo, mime="application/pdf", key=f"desc_{idx}")
    else:
        st.info("No hay documentos disponibles para la selección actual.")

# --- SECCIÓN 3: PREGUNTAR A LOS DOCUMENTOS (TEXTO 100% LIMPIO Y SIN GUIONES O ASTERISCOS) ---
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
            st.warning("No se encontraron documentos PDF con los filtros de Línea y Área aplicados.")

    pregunta_libre = st.text_area("Escribe tu pregunta:", placeholder="Ej. ¿Qué grasa se ocupa en lubricación de depaletizadora?", height=100)
    btn_preguntar = st.button("PREGUNTAR A LOS DOCUMENTOS", type="primary")

    if btn_preguntar:
        if pregunta_libre.strip():
            if alcance_busqueda == "Un documento en específico" and not archivo_especifico:
                st.error("Por favor selecciona un documento válido de la lista.")
            else:
                with st.spinner("Analizando documentos y extrayendo respuesta..."):
                    permitir_fb = (alcance_busqueda == "Todos los documentos del filtro")
                    fragmentos = buscar_contexto(
                        pregunta_libre, 
                        linea=filtro_linea, 
                        area=filtro_area, 
                        archivo=archivo_especifico,
                        top_k=3, 
                        permitir_fallback=permitir_fb
                    )

                    if not fragmentos:
                        st.warning("No se encontró información relevante para tu pregunta en los documentos indexados.")
                    else:
                        st.success("Respuesta extraída de la documentación oficial:")
                        
                        for idx_f, f in enumerate(fragmentos):
                            tipo_doc = f.get('tipo', 'Manual')
                            archivo_doc = f.get('archivo', 'Documento')
                            pagina_doc = f.get('pagina', 1)
                            
                            # Limpieza total de guiones y saltos de línea para lectura perfecta
                            texto_limpio = limpiar_texto_pdf(f.get('texto', ''))
                            
                            st.markdown(f"""
                            <div class="tarjeta-manual">
                                <p style="color: #FFD700 !important; font-weight: bold; margin-bottom: 6px; font-size: 0.95em;">
                                    📄 [{tipo_doc}] {archivo_doc} — Pág. {pagina_doc}
                                </p>
                                <p style="color: #E0E0E0 !important; font-size: 0.9em; line-height: 1.6; margin: 0;">
                                    {texto_limpio}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.divider()
                        st.markdown(f"<h4 style='color: #FFD700;'>Evidencias Visuales de Origen ({len(fragmentos)})</h4>", unsafe_allow_html=True)
                        evidencias = obtener_evidencias_priorizadas(fragmentos)
                        titulos_tabs = [f"Pág. {e['pagina']} ({e['archivo']})" for e in evidencias]
                        tabs = st.tabs(titulos_tabs)
                        for idx, tab in enumerate(tabs):
                            e = evidencias[idx]
                            with tab:
                                img_bytes = obtener_imagen_pagina_ram(e["pdf_path"], e["pagina"], dpi=300)
                                st.image(img_bytes, caption=f"Documento de origen: {e['archivo']} — Página {e['pagina']}", use_container_width=True)
        else:
            st.error("Por favor, escribe una pregunta.")

# --- SECCIÓN 4: CHAT IA SEGUIMIENTO DE AVERÍAS ---
elif st.session_state.tab_activa == "chat":
    st.markdown("<h3>Chat IA - Seguimiento de Averías (VPO)</h3>", unsafe_allow_html=True)
    st.caption("Conversa en tiempo real con la IA sobre diagnósticos, ajustes y seguimiento de fallas en la línea.")

    avatar_config = f"data:image/png;base64,{_avatar_robot_b64}" if _avatar_robot_b64 else "assistant"

    for mensaje in st.session_state["mensajes_chat"]:
        avatar_usar = avatar_config if mensaje["role"] == "assistant" else None
        with st.chat_message(mensaje["role"], avatar=avatar_usar):
            st.markdown(mensaje["content"])

    if prompt_chat := st.chat_input("Escribe tu seguimiento o duda técnica..."):
        st.session_state["mensajes_chat"].append({"role": "user", "content": prompt_chat})
        with st.chat_message("user"):
            st.markdown(prompt_chat)

        with st.chat_message("assistant", avatar=avatar_config):
            with st.spinner("Analizando consulta técnica..."):
                p_norm = normalizar_texto(prompt_chat)
                
                es_conversacional = any(kw in p_norm for kw in [
                    "que sabes hacer", "ayuda", "hola", "quien eres", "buenos dias", 
                    "buenas tardes", "que tal", "como estas", "gracias", "saludos", "ok"
                ])

                if es_conversacional:
                    if "que sabes hacer" in p_norm or "ayuda" in p_norm:
                        respuesta_chat = "¡Hola! Soy MELENA, tu asistente técnico de VPO especializado en las Líneas de Envasado 5 y 6 de Grupo Modelo. Puedo ayudarte a:\n* Consultar manuales técnicos y procedimientos (SOP, PDR, RDA).\n* Generar planes de acción detallados ante fallas mecánicas, eléctricas o de calidad.\n* Guiarte en la aplicación de protocolos de seguridad como LOTOTO y SAM.\n* Resolver dudas sobre cualquier área o equipo de la planta."
                    elif any(saludo in p_norm for saludo in ["hola", "buenos dias", "buenas tardes", "que tal", "como estas"]):
                        respuesta_chat = "¡Hola! ¿Qué avería, procedimiento o duda técnica deseas consultar hoy en la planta de envasado?"
                    else:
                        respuesta_chat = "Entendido. Sigo a tu disposición para cualquier consulta de ingeniería, mantenimiento o calidad bajo los estándares VPO de Grupo Modelo."
                else:
                    fragmentos = buscar_contexto_global(prompt_chat, top_k=2)
                    if fragmentos:
                        f_ref = fragmentos[0]
                        texto_limpio = limpiar_texto_pdf(f_ref.get('texto', ''))
                        respuesta_chat = f"**Análisis Técnico (Documento: {f_ref.get('archivo')} | Página {f_ref.get('pagina', 1)}):**\n\n> {texto_limpio}\n\n*Plan Recomendado:* Aplique las pautas indicadas en el manual, asegurando los protocolos de seguridad (LOTOTO / SAM) y validación de calidad."
                    else:
                        respuesta_chat = f"Entendido respecto a tu consulta sobre *\"{prompt_chat}\"*. Para asegurar la excelencia operativa en el área de envasado, verifique las tolerancias mecánicas, revise la lubricación o sensores según aplique, y ejecute la validación con calidad antes de liberar la línea."
                
                st.markdown(respuesta_chat)
                st.session_state["mensajes_chat"].append({"role": "assistant", "content": respuesta_chat})

# --- SECCIÓN 5: SUBIR UN DOCUMENTO ---
elif st.session_state.tab_activa == "subir":
    st.markdown("<h3>Subir Nuevo Documento a la Biblioteca</h3>", unsafe_allow_html=True)
    
    col_up1, col_up2 = st.columns([1, 1], gap="large")
    
    with col_up1:
        st.markdown("#### Categorización del Documento")
        idx_linea_def = 0 if linea_sel == "Línea 5" else 1
        linea_dest = st.selectbox("Línea de Producción:", ["Línea 5", "Línea 6"], index=idx_linea_def, key="upload_linea_sel")
        area_dest = st.selectbox("Equipo / Área Operativa:", areas_lista, index=areas_lista.index(area_sel) if area_sel in areas_lista else 0, key="upload_area_sel")
        tipo_doc_dest = st.selectbox("Tipo de Documento:", list(TIPO_CARPETA.keys()), key="upload_tipo_doc_sel")
        subcarpeta_nombre = TIPO_CARPETA[tipo_doc_dest]
        st.info(f"**Ruta de destino asignada:**\n`DOCUMENTOS/{linea_dest}/{area_dest}/{subcarpeta_nombre}/`")

    with col_up2:
        st.markdown("#### Carga de Archivo PDF")
        archivo_subido = st.file_uploader("Selecciona el archivo PDF de tu computadora:", type=["pdf"], key="file_uploader_vpo")
        auto_index = st.checkbox("Actualizar automáticamente el índice vectorial al guardar", value=True)
        btn_guardar_doc = st.button("GUARDAR Y PROCESAR DOCUMENTO", type="primary", use_container_width=True)
        
        if btn_guardar_doc:
            if archivo_subido is not None:
                carpeta_destino = os.path.join(DOCS_DIR, linea_dest, area_dest, subcarpeta_nombre)
                os.makedirs(carpeta_destino, exist_ok=True)
                ruta_completa_pdf = os.path.join(carpeta_destino, archivo_subido.name)
                
                with open(ruta_completa_pdf, "wb") as f:
                    f.write(archivo_subido.getbuffer())
                
                st.success(f"Archivo **'{archivo_subido.name}'** guardado con éxito.")
                
                if auto_index:
                    with st.spinner("Indexando nuevo documento en la base de datos ChromaDB..."):
                        tot_arch, tot_chk = indexar_todos_los_documentos()
                    st.success(f"Índice de IA actualizado. Total actual: {tot_arch} archivo(s), {tot_chk} fragmento(s).")
            else:
                st.error("Por favor selecciona un archivo PDF antes de presionar Guardar.")

st.markdown("<br><br><hr>", unsafe_allow_html=True)
st.markdown("<center><p style='color: #FFD700; font-weight: bold;'>MELENA | CONFIDENCIAL - Uso exclusivo personal autorizado Grupo Modelo</p></center>", unsafe_allow_html=True)
