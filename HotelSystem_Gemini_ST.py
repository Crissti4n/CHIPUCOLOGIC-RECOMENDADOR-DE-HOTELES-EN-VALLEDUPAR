import os
import re
import pickle
import logging
import pandas as pd
import numpy as np
from typing import List
from googletrans import Translator
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss  # 🔹 FAISS añadido

# ========== CONFIGURACIÓN ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGHotelSystem_Gemini_STF:
    def __init__(self, excel_file="100_Valledupar_hotels_English.xlsx"):
        self.excel_file = excel_file
        self.vector_path = "sentence_doc_vectors.pkl"
        self.index_path = "sentence_faiss.index"  # 🔹 FAISS index
        self.translator = Translator()
        # 🔹 Modelo de embeddings local
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Modelo Sentence Transformers cargado")
        self._setup_gemini()
        self.df, self.docs = self._load_hotels()
        self._setup_nomic_embeddings()  # Mantengo el nombre del método original

    def _setup_gemini(self):
        api_key = os.getenv("GEMINI_API_KEY", "AIzaSyC7rCfD94tZaMOIbWRq229_S11I0tS43Ns")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def _load_hotels(self):
        df = pd.read_excel(self.excel_file)
        df.fillna("No disponible", inplace=True)
        docs = (
            df["HotelName"].astype(str) + ". " +
            df["HotelRating"].astype(str) + " stars. " +
            df["Address"].astype(str) + ". " +
            df["Attractions"].astype(str) + ". " +
            df["HotelFacilities"].astype(str) + ". " +
            df["Description"].astype(str)
        ).tolist()
        return df, docs

    def _setup_nomic_embeddings(self):  # Mantengo el nombre original
        if os.path.exists(self.vector_path) and os.path.exists(self.index_path):
            with open(self.vector_path, "rb") as f:
                self.doc_vectors = pickle.load(f)
            self.doc_vectors = np.array(self.doc_vectors).astype("float32")
            faiss.normalize_L2(self.doc_vectors)
            self.index = faiss.read_index(self.index_path)
            logger.info("✅ Embeddings y FAISS index cargados.")
        else:
            logger.info("🔄 Generando embeddings con Sentence Transformers...")
            # 🔹 Usar Sentence Transformers en lugar de Nomic
            self.doc_vectors = self.embedding_model.encode(self.docs).tolist()

            with open(self.vector_path, "wb") as f:
                pickle.dump(self.doc_vectors, f)

            self.doc_vectors = np.array(self.doc_vectors).astype("float32")
            faiss.normalize_L2(self.doc_vectors)
            dimension = self.doc_vectors.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.doc_vectors)
            faiss.write_index(self.index, self.index_path)
            logger.info("✅ FAISS index creado y guardado.")

    def _translate(self, text: str, src='auto', dest='en') -> str:
        try:
            result = self.translator.translate(text, src=src, dest=dest)
            return result.text
        except Exception as e:
            logger.warning(f"⚠️ Error en traducción: {e}")
            return text

    def _is_english(self, text: str) -> bool:
        palabras_ingles = len(re.findall(r'\b(the|hotel|and|of|to|with|for|room|service|internet|breakfast)\b', text.lower()))
        return palabras_ingles / max(1, len(text.split())) > 0.3

    def buscar_contexto(self, pregunta: str, top_k=3) -> List[str]:
        pregunta_en = self._translate(pregunta, dest='en')
        # 🔹 Usar Sentence Transformers en lugar de Nomic
        pregunta_vec = self.embedding_model.encode([pregunta_en])[0]

        pregunta_vec = np.array([pregunta_vec]).astype("float32")
        faiss.normalize_L2(pregunta_vec)

        D, I = self.index.search(pregunta_vec, top_k)
        return [self.docs[i] for i in I[0]]

    def generar_respuesta(self, pregunta: str) -> str:
        if not pregunta.strip():
            return "❗ Por favor, ingresa una consulta válida."

        contexto = self.buscar_contexto(pregunta)
        prompt = f"""
Eres un experto en turismo en Valledupar, Colombia.

Usuario pregunta: "{pregunta}"

Información de hoteles:
{chr(10).join(contexto)}

Responde en español, usando solo la información proporcionada. Sé claro, amable y directo. Usa máximo 120 palabras.
"""
        try:
            respuesta = self.model.generate_content(prompt)
            texto = respuesta.text.strip()
            if self._is_english(texto):
                texto = self._translate(texto, src="en", dest="es")
            return texto
        except Exception as e:
            logger.error(f"❌ Error generando respuesta: {e}")
            return "❌ Hubo un problema generando la respuesta. Intenta de nuevo."

    def procesar_consulta(self, pregunta: str) -> str:
        return self.generar_respuesta(pregunta)

if __name__ == "__main__":
    sistema = RAGHotelSystem_Gemini_STF("100_Valledupar_hotels_English.xlsx")  # Asegúrate de usar la ruta real del Excel
    resultado = sistema.procesar_consulta("¿Cuáles son los hoteles con piscina?")
    print(resultado)