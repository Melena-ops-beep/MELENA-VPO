import streamlit as st
import streamlit.components.v1 as components
import os
import base64
import re
import unicodedata
import time
import io
import requests
import numpy as np
import pandas as pd
from pypdf import PdfReader
import chromadb
import fitz  # PyMuPDF para renderizado en RAM a 300 DPI
from sentence_transformers import SentenceTransformer
from google import genai
from gtts import gTTS  # Librería para Text-to-Speech (Modo Manos Libres)

# --- CONFIGURACIÓN DE RUTAS MULTIPLATAFORMA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "img")
DATA_DIR = os.path.join(BASE_DIR, "CONOCIMIENTO_VPO")
DOCS_DIR = os.path.join(BASE_DIR, "DOCUMENTOS")
INDICE_DIR = os.path.join(BASE_DIR, "INDICE_VECTORIAL")

# --- CONFIGURACIÓN DE GEMINI API CON AUTODETECCIÓN DE MODELOS ---
def obtener_cliente_gemini():
    try:
        api_key_str = os.environ.get("GEMINI_API_KEY") or (st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else None)
        if api_key_str:
            return genai.Client(api_key=api_key_str.strip()), None
        return None, "No se encontró GEMINI_API_KEY en Settings > Secrets de Streamlit Cloud."
    except Exception as e:
        return None, f"Error al inicializar cliente Gemini: {e}"

@st.cache_data(ttl=3600, show_spinner=False)
def obtener_modelos_activos_api():
    cliente, _ = obtener_cliente_gemini()
    preferidos = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]
    if not cliente:
        return preferidos
    try:
        disponibles = []
        for m in cliente.models.list():
            nombre = getattr(m, "name", "")
            metodos = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None) or []
            if "gemini" in nombre.lower() and ("generateContent" in metodos or not metodos):
                limpio = nombre.replace("models/", "")
                if not any(x in limpio for x in ["embedding", "imagen", "veo", "aqa", "tts"]):
                    disponibles.append(limpio)
        ordenados = []
        for p in preferidos:
            for d in disponibles:
                if p in d and d not in ordenados:
                    ordenados.append(d)
        for d in disponibles:
            if "flash" in d and d not in ordenados:
                ordenados.append(d)
        for d in disponibles:
            if d not in ordenados:
                ordenados.append(d)
        return ordenados[:5] if ordenados else preferidos
    except Exception:
        return preferidos

def ejecutar_generacion_gemini(cliente, contents, system_instruction, temperature=0.2):
    modelos = obtener_modelos_activos_api()
    errores = []
    for modelo in modelos:
        for intento in range(2):
            try:
                response = cliente.models.generate_content(
                    model=modelo,
                    contents=contents,
                    config={
                        "system_instruction": system_instruction,
                        "temperature": temperature,
                    }
                )
                if response and response.text:
                    return response.text, None
            except Exception as e:
                err_str = str(e)
                errores.append(f"[{modelo}]: {err_str}")
                if "503" in err_str or "429" in err_str:
                    time.sleep(2.5)
                    continue
                break
    return None, " | ".join(errores[:3])

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

# --- FUNCIONES AUXILIARES Y GOOGLE DRIVE (APPS SCRIPT BRIDGE) ---
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

def subir_archivo_a_google_drive(file_obj, nombre_archivo, mime_type, linea="General", area="General", tipo="Documentos"):
    try:
        if "APPS_SCRIPT_URL" not in st.secrets:
            st.error("No se encontró APPS_SCRIPT_URL en Settings > Secrets de Streamlit Cloud.")
            return False
        
        url_script = st.secrets["APPS_SCRIPT_URL"].strip()
        archivo_b64 = base64.b64encode(file_obj.getbuffer()).decode("utf-8")
        
        payload = {
            "fileName": nombre_archivo,
            "mimeType": mime_type,
            "fileBase64": archivo_b64,
            "linea": linea,
            "area": area,
            "tipo": tipo
        }
        
        respuesta = requests.post(url_script, json=payload, timeout=90)
        respuesta.raise_for_status()
        resultado = respuesta.json()
        
        if resultado.get("status") == "success":
            return True
        else:
            st.error(f"Error en Apps Script: {resultado.get('message')}")
            return False
    except Exception as e:
        st.error(f"Error al conectar con Google Drive: {e}")
        return False

def generar_audio_tts(texto):
    try:
        texto_limpio_voz = re.sub(r'[#*`>\-\[\]]', '', texto)
        tts = gTTS(text=texto_limpio_voz[:1500], lang='es', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception:
        return None

def normalizar_texto(texto):
    if not texto: return ""
    s = unicodedata.normalize('NFD', str(texto))
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return s.lower().strip()

def limpiar_texto_pdf(texto):
    if not texto: return ""
    texto_limpio = re.sub(r'([a-zA-ZáéíóúÁÉÍÓÚñÑ]{2,})-\s*[\r\n]+\s*([a-zA-ZáéíóúÁÉÍÓÚñÑ]{2,})', r'\1\2', texto)
    texto_limpio = re.sub(r'([a-zA-ZáéíóúÁÉÍÓÚñÑ]{2,})-\s+([a-zA-ZáéíóúÁÉÍÓÚñÑ]{2,})', r'\1\2', texto_limpio)
    texto_limpio = re.sub(r'\s*\n\s*', ' ', texto_limpio)
    texto_limpio = re.sub(r'\s+', ' ', texto_limpio)
    return texto_limpio.strip()

# --- MODELO DE EMBEDDINGS EN LA NUBE (CARGA BAJO DEMANDA) ---
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
        {"role": "assistant", "content": "¡Hola! Soy **MELENA**, tu asistente técnico de VPO especializado en las Líneas de Envasado 5 y 6 de Grupo Modelo. ¿Qué avería, procedimiento o duda técnica deseas consultar hoy?"}
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

def listar_archivos(linea, area, tipo_doc, extension=".pdf"):
    carpeta_linea = encontrar_subcarpeta(DOCS_DIR, linea)
    carpeta_area = encontrar_subcarpeta(carpeta_linea, area) if carpeta_linea else None
    carpeta_tipo = encontrar_subcarpeta(carpeta_area, TIPO_CARPETA[tipo_doc]) if carpeta_area else None
    if not carpeta_tipo: return []
    return [os.path.join(carpeta_tipo, f) for f in os.listdir(carpeta_tipo) if f.lower().endswith(extension)]

def listar_archivos_filtrados(linea=None, area=None, extension=None):
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

    if not os.path.exists(base): return []
    for root, dirs, files in os.walk(base):
        for f in files:
            if extension:
                if f.lower().endswith(extension): archivos.add(f)
            else:
                if f.lower().endswith((".pdf", ".xlsx")): archivos.add(f)
    return sorted(list(archivos))

@st.cache_data(show_spinner=False)
def leer_archivo_bytes_cacheado(ruta_archivo, fecha_modificacion):
    with open(ruta_archivo, "rb") as f: return f.read()

# --- RAG, RENDERIZADO RAM E ÍNDICE VECTORIAL ---
@st.cache_resource
def obtener_coleccion_vectorial():
    os.makedirs(INDICE_DIR, exist_ok=True)
    cliente = chromadb.PersistentClient(path=INDICE_DIR)
    return cliente.get_or_create_collection(name="documentos_melena")

def indice_tiene_datos():
    if not os.path.exists(INDICE_DIR) or not os.listdir(INDICE_DIR):
        return False
    try:
        coleccion = obtener_coleccion_vectorial()
        return coleccion.count() > 0
    except Exception:
        return False

def resolver_ruta_y_pagina(top_f):
    if not top_f: return None, 1
    ruta_archivo = top_f.get("ruta_pdf")
    pagina = top_f.get("pagina", 1)
    archivo_nombre = top_f.get("archivo")

    if not ruta_archivo or not os.path.exists(ruta_archivo):
        if archivo_nombre and os.path.exists(DOCS_DIR):
            for root, dirs, files in os.walk(DOCS_DIR):
                for f in files:
                    if f.lower() == archivo_nombre.lower():
                        ruta_archivo = os.path.join(root, f)
                        break
                if ruta_archivo and os.path.exists(ruta_archivo): break

    if not ruta_archivo or not os.path.exists(ruta_archivo): return None, 1
    return ruta_archivo, int(pagina)

def obtener_imagen_pagina_ram(archivo_path, page_number, dpi=300):
    if archivo_path.lower().endswith(".xlsx"):
        doc = fitz.open()
        page = doc.new_page()
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    
    doc = fitz.open(archivo_path)
    page = doc[page_number - 1]
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("png")

def obtener_evidencias_priorizadas(fragmentos):
    evidencias = []
    vistos = set()
    
    for f in fragmentos:
        archivo_path, pag_num = resolver_ruta_y_pagina(f)
        if archivo_path and os.path.exists(archivo_path):
            clave = (archivo_path, pag_num)
            if clave not in vistos:
                vistos.add(clave)
                tiene_img = False
                if archivo_path.lower().endswith(".pdf"):
                    try:
                        doc = fitz.open(archivo_path)
                        page = doc[pag_num - 1]
                        tiene_img = (len(page.get_images()) > 0) or (len(page.get_drawings()) > 8)
                    except Exception:
                        pass
                
                evidencias.append({
                    "pdf_path": archivo_path,
                    "pagina": pag_num,
                    "archivo": f.get("archivo", os.path.basename(archivo_path)),
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
                    ruta_archivo = os.path.join(ruta_tipo, archivo)
                    
                    if archivo.lower().endswith(".pdf"):
                        try: doc = fitz.open(ruta_archivo)
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
                                        "ruta_pdf": ruta_archivo,
                                        "pagina": page_num + 1
                                    }],
                                )
                                total_chunks += 1

                    elif archivo.lower().endswith(".xlsx"):
                        try:
                            df = pd.read_excel(ruta_archivo, sheet_name=None)
                        except Exception:
                            continue
                        
                        total_archivos += 1
                        for nombre_hoja, sheet_df in df.items():
                            texto_hoja = sheet_df.to_string(index=False)
                            if not texto_hoja.strip(): continue
                            
                            chunks = trocear_texto(texto_hoja, tam_chunk=150, solape=30)
                            for idx, chunk in enumerate(chunks):
                                embedding = obtener_embedding(chunk)
                                id_unico = f"{linea_normalizada}|{area_normalizada}|{tipo_carpeta}|{archivo}|hoja_{nombre_hoja}|{idx}"
                                
                                coleccion.upsert(
                                    ids=[id_unico],
                                    embeddings=[embedding],
                                    documents=[chunk],
                                    metadatas=[{
                                        "linea": linea_normalizada,
                                        "area": area_normalizada,
                                        "tipo": tipo_carpeta,
                                        "archivo": archivo,
                                        "ruta_pdf": ruta_archivo,
                                        "pagina": 1
                                    }],
                                )
                                total_chunks += 1
                                
    return total_archivos, total_chunks

def buscar_contexto(pregunta, linea=None, area=None, archivo=None, top_k=4, permitir_fallback=True):
    if not indice_tiene_datos():
        return []
        
    coleccion = obtener_coleccion_vectorial()
    embedding_pregunta = obtener_embedding(pregunta)
    where_filter = {}
    if linea and area:
        where_filter = {"$and": [{"linea": linea}, {"area": area}]}
    elif linea:
        where_filter = {"linea": linea}
    elif area:
        where_filter = {"area": area}
        
    try:
        if where_filter:
            resultados = coleccion.query(query_embeddings=[embedding_pregunta], n_results=top_k, where=where_filter)
        else:
            resultados = coleccion.query(query_embeddings=[embedding_pregunta], n_results=top_k)
    except Exception:
        resultados = coleccion.query(query_embeddings=[embedding_pregunta], n_results=top_k)
        
    fragmentos = []
    if resultados and resultados.get('documents') and resultados['documents'][0]:
        for doc, meta in zip(resultados['documents'][0], resultados['metadatas'][0]):
            if archivo and meta.get('archivo', '').lower() != archivo.lower():
                continue
            fragmentos.append({
                "texto": doc, 
                "archivo": meta.get('archivo'), 
                "tipo": meta.get('tipo'), 
                "ruta_pdf": meta.get('ruta_pdf'),
                "pagina": meta.get('pagina')
            })
            
    if not fragmentos and permitir_fallback:
        resultados_globales = coleccion.query(query_embeddings=[embedding_pregunta], n_results=top_k)
        if resultados_globales and resultados_globales.get('documents') and resultados_globales['documents'][0]:
            for doc, meta in zip(resultados_globales['documents'][0], resultados_globales['metadatas'][0]):
                fragmentos.append({
                    "texto": doc, 
                    "archivo": meta.get('archivo'), 
                    "tipo": meta.get('tipo'), 
                    "ruta_pdf": meta.get('ruta_pdf'),
                    "pagina": meta.get('pagina')
                })
    return fragmentos

def buscar_contexto_global(pregunta, top_k=3):
    if not indice_tiene_datos():
        return []
        
    coleccion = obtener_coleccion_vectorial()
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

# --- BASE64 DE ICONOS Y FONDOS ---
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

# --- ESTILOS CSS ADAPTATIVOS ---
st.markdown(f"""
    <style>
    #MainMenu, header, footer, [data-testid="stHeader"], [data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton {{
        visibility: hidden !important;
        display: none !important;
    }}

    .stApp {{ background-color: #000000 !important; overflow-x: hidden; }}
    h1, h2, h3, p, span, label, .stMarkdown {{ color: #FFFFFF !important; font-family: 'Segoe UI', sans-serif; }}

    div[data-testid="column"] {{
        background-color: #002D72 !important;
        padding: 22px !important;
        border-radius: 15px !important;
        border: 2px solid #FFD700 !important;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.2) !important;
    }}
    div[data-testid="column"] h3 {{ color: #FFD700 !important; border-bottom: 1px solid #FFFFFF; padding-bottom: 10px; }}

    .stSelectbox div[data-baseweb="select"], .stTextArea textarea, .stTextInput input {{
        background-color: #1E1E1E !important; color: #FFFFFF !important; border: 1px solid #FFD700 !important;
    }}

    .block-container {{ 
        padding-top: 0.5rem !important; 
        max-width: 100% !important; 
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }}

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
        width: 100% !important;
    }}

    div[data-testid="stButton"] > button[kind="primary"],
    button[kind="primary"],
    .stButton > button[data-baseweb="button"] {{
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
    
    div[data-testid="stButton"] > button[kind="primary"] * {{
        color: #000000 !important;
    }}

    div[data-testid="stButton"] > button:hover, 
    div[data-testid="stDownloadButton"] > button:hover {{
        background-color: #E6C200 !important;
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

    @media (max-width: 900px) {{
        [data-testid="stHorizontalBlock"] {{
            flex-direction: column !important;
            gap: 10px !important;
        }}
        [data-testid="column"] {{
            width: 100% !important;
            flex: 1 1 100% !important;
        }}
    }}
    </style>
""", unsafe_allow_html=True)

# --- ENCABEZADO FIJO ---
_logo_ab_b64 = get_base64("abinbev")
_logo_modelo_b64 = get_base64("modelo")
_logo_melena_b64 = get_base64("melena")

def _tag_logo(b64):
    if not b64: return ""
    return f'<img src="data:image/png;base64,{b64}" style="height:55px; margin-right:20px; object-fit: contain;">'

st.markdown(f"""
    <div style="position: sticky; top: 0; z-index: 999; background-color: #000000; border-bottom: 2px solid #FFD700; padding: 12px 20px; display: flex; justify-content: space-between; align-items: center; margin: 0 -1rem 1rem -1rem;">
        <div style="display: flex; align-items: center; flex-wrap: wrap;">
            {_tag_logo(_logo_ab_b64)}
            {_tag_logo(_logo_modelo_b64)}
            {_tag_logo(_logo_melena_b64)}
        </div>
        <div style="text-align: right;">
            <h1 style="color: #FFD700 !important; font-size: 1.4em; margin: 0;">MELENA</h1>
            <p style="color: #FFFFFF !important; font-size: 0.7em; margin: 0; font-weight: bold;">Módulo Especializado de Lectura y Ejecución para Normas y Averías</p>
            <p style="color: #FFFFFF !important; font-size: 0.7em; margin: 0; font-weight: bold;">Líneas de Envasado 5 y 6 | Gestión de Excelencia Operativa VPO</p>
        </div>
    </div>
""", unsafe_allow_html=True)

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
        st.markdown("<h3>Registro de Falla y Diagnóstico VPO</h3>", unsafe_allow_html=True)
        col_form, col_componentes = st.columns([1.4, 1], gap="medium")
        
        with col_form:
            linea_sel = st.selectbox("Línea de Producción:", ["Línea 5", "Línea 6"], index=0 if st.session_state["linea_sel"]=="Línea 5" else 1, key="form_linea_sel")
            st.session_state["linea_sel"] = linea_sel
            
            area_sel = st.selectbox("Área Operativa / Equipo:", areas_lista, key="form_area_sel")
            
            tipo_falla_sel = st.selectbox("Tipo de falla:", OPCIONES_TIPO_FALLA)
            desc = st.text_area("Descripción técnica de la desviación:", placeholder="Detalles observados...", height=100)
            btn = st.button("GENERAR PLAN DE ACCIÓN", type="primary")

        with col_componentes:
            st.markdown("<p style='font-weight: bold; color: #FFD700; margin-bottom: 6px; font-size: 0.9em;'>Componentes críticos de la máquina:</p>", unsafe_allow_html=True)
            comp_list = list(ESTRUCTURA_FALLAS[area_sel]["componentes"].keys())
            html_componentes = "<div class='caja-componentes'>"
            for comp in comp_list:
                html_componentes += f"<p style='margin: 0 0 4px 0; font-size: 0.85em; color: #FFFFFF;'>• {comp}</p>"
            html_componentes += "</div>"
            st.markdown(html_componentes, unsafe_allow_html=True)
            
            path_foto = encontrar_imagen(ESTRUCTURA_FALLAS[area_sel]["foto"])
            if path_foto:
                st.image(path_foto, use_container_width=True)

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
                            <p style="color: #FFD700; font-weight: bold; margin-top: 15px;">Analizando causa raíz y redactando Plan VPO con Gemini...</p>
                        </div>''', unsafe_allow_html=True)
                else:
                    placeholder.info("Analizando desviación técnica y redactando Plan VPO...")

                fragmentos_pdf = buscar_contexto(consulta_texto, linea=linea_sel, area=area_sel, top_k=4)
                contexto_documentos = ""
                if fragmentos_pdf:
                    for f in fragmentos_pdf:
                        contexto_documentos += f"- Documento: {f.get('archivo')} (Pág. {f.get('pagina', 1)}): {limpiar_texto_pdf(f.get('texto', ''))}\n"

                plan_generado_ia = ""
                error_api = ""
                cliente_actual, err_cli = obtener_cliente_gemini()
                
                if cliente_actual is not None:
                    prompt_sistema = f"""
Eres MELENA, Ingeniero Especialista en Mantenimiento, Procesos y Excelencia Operativa (VPO) de Cervecería Modelo (Líneas de Envasado 5 y 6).
Analiza técnicamente la siguiente desviación operativa y construye un Plan de Reacción paso a paso altamente específico, práctico y aplicable en piso de producción:

- Línea: {linea_sel}
- Equipo / Área: {area_sel}
- Clasificación: {tipo_falla_sel}
- Desviación reportada: {desc}

Documentación interna recuperada:
{contexto_documentos if contexto_documentos else "No se encontraron fragmentos específicos en manuales locales; aplica tu conocimiento experto en procesos de embotellado industrial, mecánica, química de lavado, instrumentación y automatización."}

Estructura obligatoria de tu respuesta:
1. **Análisis Técnico de Causa Raíz Probable:** Explica detalladamente qué sistemas, sensores, válvulas, componentes o parámetros del equipo ({area_sel}) provocan esta desviación.
2. **Acciones Inmediatas de Contención y Seguridad (LOTOTO/EPP):** Qué hacer para proteger al operador, aislar el riesgo y evitar producto no conforme.
3. **Plan de Reacción Paso a Paso:** Lista detallada con casillas (`- [ ] **Paso X:** ...`) con verificaciones de instrumentos, ajustes mecánicos/eléctricos o pruebas de proceso necesarias.
4. **Criterios de Liberación (Calidad VPO):** Cómo confirmar que el equipo o parámetro volvió dentro de especificación antes de arrancar.
"""
                    contents_plan = [{"role": "user", "parts": [{"text": f"Analiza y resuelve esta falla en {area_sel} ({linea_sel}): {desc}"}]}]
                    plan_generado_ia, error_api = ejecutar_generacion_gemini(cliente_actual, contents_plan, prompt_sistema, temperature=0.2)
                else:
                    error_api = err_cli or "No se pudo conectar con la API de Gemini."

                placeholder.empty()

                if plan_generado_ia:
                    st.success(f"Diagnóstico y Plan de Reacción analizado por IA para {linea_sel.upper()} - {area_sel.upper()}")

                    respuesta_generada = f"""
### 🛠️ PLAN DE REACCIÓN ANALIZADO (VPO)
**Área / Equipo:** {area_sel} ({linea_sel}) | **Tipo:** {tipo_falla_sel}  
**Desviación:** *{desc}*

---

{plan_generado_ia}

---
**📖 Referencias Documentales Consultadas:**
"""
                    if fragmentos_pdf:
                        for f in fragmentos_pdf[:3]:
                            texto_limpio = limpiar_texto_pdf(f.get('texto', ''))
                            respuesta_generada += f"> **[{f.get('tipo', 'Manual')} — {f.get('archivo', 'Documento')} | Pág. {f.get('pagina', 1)}]**\n> {texto_limpio}\n\n"
                    else:
                        respuesta_generada += "> *Análisis generado mediante razonamiento experto de ingeniería de procesos VPO (sin fragmentos coincidentes en manuales indexados).* \n\n"

                    st.markdown(respuesta_generada)
                    
                    # --- MODO MANOS LIBRES (TEXT-TO-SPEECH) ---
                    st.markdown("#### 🎧 Modo Manos Libres (Asistencia por Voz)")
                    texto_a_voz = f"Plan de reacción para {area_sel} en {linea_sel}. {plan_generado_ia}"
                    audio_bytes = generar_audio_tts(texto_a_voz)
                    if audio_bytes:
                        st.audio(audio_bytes, format='audio/mp3')

                    # --- TARJETAS DE EVIDENCIA CON ZOOM Y DESCARGA ---
                    evidencias_calc = obtener_evidencias_priorizadas(fragmentos_pdf) if fragmentos_pdf else []
                    if evidencias_calc:
                        st.divider()
                        st.markdown("<h4 style='color: #FFD700;'>Evidencias Visuales y Planos de Referencia</h4>", unsafe_allow_html=True)
                        titulos_tabs = [f"Pág. {e['pagina']} ({e['archivo']})" for e in evidencias_calc[:3]]
                        tabs = st.tabs(titulos_tabs)
                        for idx, tab in enumerate(tabs):
                            e = evidencias_calc[idx]
                            with tab:
                                if e["pdf_path"].lower().endswith(".pdf"):
                                    img_bytes = obtener_imagen_pagina_ram(e["pdf_path"], e["pagina"], dpi=300)
                                    st.image(img_bytes, caption=f"Documento de referencia: {e['archivo']} — Página {e['pagina']}", use_container_width=True)
                                    st.download_button(
                                        label=f"📥 Descargar hoja de referencia (Pág. {e['pagina']})",
                                        data=img_bytes,
                                        file_name=f"{os.path.splitext(e['archivo'])[0]}_pag_{e['pagina']}.png",
                                        mime="image/png",
                                        key=f"desc_evidencia_{idx}"
                                    )
                                else:
                                    st.info(f"El archivo '{e['archivo']}' es una hoja de cálculo Excel analizada por la IA.")
                else:
                    st.error(f"⚠️ La IA no pudo procesar el análisis. Detalle técnico: {error_api}")
            else:
                st.error("Por favor, ingrese una descripción técnica.")

# --- SECCIÓN 2: BIBLIOTECA DE DOCUMENTOS ---
elif st.session_state.tab_activa == "docs":
    st.markdown("<h3>Biblioteca de Documentos y Reportes</h3>", unsafe_allow_html=True)
    linea_doc_sel = st.selectbox("Selecciona Línea:", ["Línea 5", "Línea 6"], key="doc_linea_sel")
    area_doc_sel = st.selectbox("Selecciona Área:", areas_lista, key="doc_area_sel")
    tipo_doc_sel = st.selectbox("Tipo de documento:", list(TIPO_CARPETA.keys()))
    
    pdfs_encontrados = listar_archivos(linea_doc_sel, area_doc_sel, tipo_doc_sel, extension=".pdf")
    excels_encontrados = listar_archivos(linea_doc_sel, area_doc_sel, tipo_doc_sel, extension=".xlsx")
    todos_docs = pdfs_encontrados + excels_encontrados
    
    if todos_docs:
        st.success(f"Se encontraron {len(todos_docs)} documento(s).")
        for idx, ruta_archivo in enumerate(todos_docs):
            nombre_archivo = os.path.basename(ruta_archivo)
            archivo_bytes = leer_archivo_bytes_cacheado(ruta_archivo, os.path.getmtime(ruta_archivo))
            mime_tipo = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if nombre_archivo.lower().endswith(".xlsx") else "application/pdf"
            st.download_button(label=f"DESCARGAR {nombre_archivo}", data=archivo_bytes, file_name=nombre_archivo, mime=mime_tipo, key=f"desc_{idx}")
    else:
        st.info("No hay documentos disponibles para la selección actual.")

# --- SECCIÓN 3: PREGUNTAR A LOS DOCUMENTOS ---
elif st.session_state.tab_activa == "preguntas":
    st.markdown("<h3>Pregúntale a los manuales, PDR, SOP, RDA y Hojas Excel</h3>", unsafe_allow_html=True)
    
    col_f1, col_f2 = st.columns(2)
    with col_f1: usar_linea_actual = st.checkbox("Filtrar por línea", value=True)
    with col_f2: usar_area_actual = st.checkbox("Filtrar por área", value=False)

    filtro_linea = st.session_state["linea_sel"] if usar_linea_actual else None
    filtro_area = st.selectbox("Área de filtro:", areas_lista) if usar_area_actual else None

    alcance_busqueda = st.radio(
        "Modo de filtrado de documentos:",
        ["Todos los documentos del filtro", "Un documento en específico"],
        horizontal=True
    )

    archivo_especifico = None
    if alcance_busqueda == "Un documento en específico":
        archivos_disponibles = listar_archivos_filtrados(filtro_linea, filtro_area)
        if archivos_disponibles:
            archivo_especifico = st.selectbox("Selecciona el documento específico:", archivos_disponibles)
        else:
            st.warning("No se encontraron archivos con los filtros aplicados.")

    pregunta_libre = st.text_area("Escribe tu pregunta:", placeholder="Ej. ¿Qué procedimiento seguir ante un atasco de caja?", height=100)
    btn_preguntar = st.button("PREGUNTAR A LOS DOCUMENTOS", type="primary")

    if btn_preguntar:
        if pregunta_libre.strip():
            if alcance_busqueda == "Un documento en específico" and not archivo_especifico:
                st.error("Por favor selecciona un documento válido.")
            else:
                with st.spinner("Buscando respuesta exacta..."):
                    permitir_fb = (alcance_busqueda == "Todos los documentos del filtro")
                    fragmentos = buscar_contexto(
                        pregunta_libre, 
                        linea=filtro_linea, 
                        area=filtro_area, 
                        archivo=archivo_especifico,
                        top_k=6, 
                        permitir_fallback=permitir_fb
                    )

                    if not fragmentos:
                        st.warning("No se encontró información relevante.")
                    else:
                        st.success("Respuesta extraída de la documentación oficial:")
                        for idx_f, f in enumerate(fragmentos[:3]):
                            tipo_doc = f.get('tipo', 'Manual')
                            archivo_doc = f.get('archivo', 'Documento')
                            texto_limpio = limpiar_texto_pdf(f.get('texto', ''))
                            
                            st.markdown(f"""
                            <div class="tarjeta-manual">
                                <p style="color: #FFD700 !important; font-weight: bold; margin-bottom: 6px; font-size: 0.95em;">
                                    📄 [{tipo_doc}] {archivo_doc}
                                </p>
                                <p style="color: #E0E0E0 !important; font-size: 0.9em; line-height: 1.6; margin: 0;">
                                    {texto_limpio}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.divider()
                        st.markdown("<h4 style='color: #FFD700;'>Evidencias Visuales y Descarga</h4>", unsafe_allow_html=True)
                        evidencias = obtener_evidencias_priorizadas(fragmentos)
                        titulos_tabs = [f"{e['archivo']}" for e in evidencias]
                        tabs = st.tabs(titulos_tabs)
                        for idx, tab in enumerate(tabs):
                            e = evidencias[idx]
                            with tab:
                                if e["pdf_path"].lower().endswith(".pdf"):
                                    img_bytes = obtener_imagen_pagina_ram(e["pdf_path"], e["pagina"], dpi=300)
                                    st.image(img_bytes, caption=f"Documento de origen: {e['archivo']} — Página {e['pagina']}", use_container_width=True)
                                    st.download_button(
                                        label=f"📥 Descargar hoja de referencia (Pág. {e['pagina']})",
                                        data=img_bytes,
                                        file_name=f"{os.path.splitext(e['archivo'])[0]}_pag_{e['pagina']}.png",
                                        mime="image/png",
                                        key=f"desc_preg_{idx}"
                                    )
                                else:
                                    st.info(f"El archivo '{e['archivo']}' es un reporte en Excel (.xlsx).")
        else:
            st.error("Por favor, escribe una pregunta.")

# --- SECCIÓN 4: CHAT IA CON GEMINI API ---
elif st.session_state.tab_activa == "chat":
    st.markdown("<h3>Chat IA - Seguimiento de Averías (VPO)</h3>", unsafe_allow_html=True)
    st.caption("Conversa en tiempo real con la IA sobre diagnósticos, ajustes y seguimiento de fallas.")

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
            with st.spinner("MELENA está analizando tu consulta..."):
                
                fragmentos_rag = buscar_contexto_global(prompt_chat, top_k=3)
                contexto_documentos = ""
                if fragmentos_rag:
                    for frag in fragmentos_rag:
                        contexto_documentos += f"- Documento: {frag.get('archivo')} (Pág. {frag.get('pagina', 1)}): {limpiar_texto_pdf(frag.get('texto', ''))}\n"

                respuesta_chat = ""
                error_chat = ""
                cliente_chat, err_cli_chat = obtener_cliente_gemini()
                
                if cliente_chat is not None:
                    prompt_sistema = f"""
Eres MELENA, asistente técnico experto en Automatización y Mantenimiento Industrial (Líneas 5 y 6 Grupo Modelo).
Contexto: Línea: {st.session_state["linea_sel"]}. Manuales: {contexto_documentos if contexto_documentos else "Aplica conocimiento experto."}
Instrucciones: Responde de forma analítica y experta con una guía paso a paso.
"""
                    contents_historial = []
                    for m in st.session_state["mensajes_chat"][:-1]:
                        role_gemini = "user" if m["role"] == "user" else "model"
                        contents_historial.append({"role": role_gemini, "parts": [{"text": m["content"]}]})
                    contents_historial.append({"role": "user", "parts": [{"text": prompt_chat}]})

                    respuesta_chat, error_chat = ejecutar_generacion_gemini(cliente_chat, contents_historial, prompt_sistema, temperature=0.3)

                    if not respuesta_chat:
                        respuesta_chat = f"⚠️ Error al consultar la IA: {error_chat}"
                else:
                    respuesta_chat = f"⚠️ {err_cli_chat}"

                st.markdown(respuesta_chat)
                st.session_state["mensajes_chat"].append({"role": "assistant", "content": respuesta_chat})

# --- SECCIÓN 5: SUBIR O ACTUALIZAR UN DOCUMENTO (CON GOOGLE DRIVE VÍA APPS SCRIPT) ---
elif st.session_state.tab_activa == "subir":
    st.markdown("<h3>Subir o Actualizar Documento en la Biblioteca</h3>", unsafe_allow_html=True)
    st.caption("Los documentos subidos se guardarán de forma permanente en Google Drive y se indexarán automáticamente.")
    
    col_up1, col_up2 = st.columns([1, 1], gap="large")
    
    with col_up1:
        st.markdown("#### Categorización del Documento")
        linea_dest = st.selectbox("Línea de Producción:", ["Línea 5", "Línea 6"], key="upload_linea_sel")
        area_dest = st.selectbox("Equipo / Área Operativa:", areas_lista, key="upload_area_sel")
        tipo_doc_dest = st.selectbox("Tipo de Documento:", list(TIPO_CARPETA.keys()), key="upload_tipo_doc_sel")
        subcarpeta_nombre = TIPO_CARPETA[tipo_doc_dest]
        st.info(f"**Ruta de destino asignada:**\n`DOCUMENTOS/{linea_dest}/{area_dest}/{subcarpeta_nombre}/`")

    with col_up2:
        st.markdown("#### Carga de Archivo (PDF o XLSX)")
        archivo_subido = st.file_uploader("Selecciona el archivo:", type=["pdf", "xlsx"], key="file_uploader_vpo")
        auto_index = st.checkbox("Actualizar automáticamente el índice vectorial (ChromaDB)", value=True)
        btn_guardar_doc = st.button("GUARDAR / ACTUALIZAR VERSIÓN", type="primary", use_container_width=True)
        
        if btn_guardar_doc:
            if archivo_subido is not None:
                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if archivo_subido.name.endswith(".xlsx") else "application/pdf"
                
                with st.spinner("Subiendo documento de forma segura a Google Drive..."):
                    exito_drive = subir_archivo_a_google_drive(
                        archivo_subido, 
                        archivo_subido.name, 
                        mime, 
                        linea=linea_dest, 
                        area=area_dest, 
                        tipo=subcarpeta_nombre
                    )
                    
                if exito_drive:
                    carpeta_destino = os.path.join(DOCS_DIR, linea_dest, area_dest, subcarpeta_nombre)
                    os.makedirs(carpeta_destino, exist_ok=True)
                    ruta_completa_archivo = os.path.join(carpeta_destino, archivo_subido.name)
                    with open(ruta_completa_archivo, "wb") as f:
                        f.write(archivo_subido.getbuffer())
                        
                    st.success(f"Archivo **'{archivo_subido.name}'** subido a Google Drive y guardado con éxito.")
                    
                    if auto_index:
                        with st.spinner("Reindexando la base de datos vectorial..."):
                            tot_arch, tot_chk = indexar_todos_los_documentos()
                        st.success(f"Índice actualizado. Total: {tot_arch} archivo(s), {tot_chk} fragmento(s).")
                else:
                    st.error("No se pudo completar la subida a Google Drive.")
            else:
                st.error("Por favor selecciona un archivo PDF o Excel antes de presionar Guardar.")

st.markdown("<br><br><hr>", unsafe_allow_html=True)
st.markdown("<center><p style='color: #FFD700; font-weight: bold;'>MELENA | CONFIDENCIAL - Uso exclusivo personal autorizado Grupo Modelo</p></center>", unsafe_allow_html=True)
