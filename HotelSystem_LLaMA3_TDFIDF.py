import os
import re
import pickle
import logging
import pandas as pd
import numpy as np
from typing import List
from googletrans import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama
import faiss  # 🔹 FAISS añadido


# Carga el modelo localmente desde archivo .gguf
llm = Llama(model_path="Open-Insurance-LLM-Llama3-8B-Q3_K_M.gguf")

# Prueba rápida
response = llm("¿Cuál es el hotel mas lujoso?", max_tokens=50)
print(response["choices"][0]["text"].strip())

# ========== CONFIGURACIÓN ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGHotelSystem_LLaMA3_TFIDF:
    def __init__(self, excel_file="100_Valledupar_hotels_English.xlsx", model_path="Open-Insurance-LLM-Llama3-8B-Q3_K_M.gguf"):
        self.excel_file = excel_file
        self.vectorizer_path = "vectorizer_llama3.pkl"
        self.vector_path = "doc_vectors_llama3.pkl"
        self.index_path = "tfidf_llama3_faiss.index"  # 🔹 nuevo: FAISS index
        self.translator = Translator()
        self.model_path = model_path
        self._setup_llama()
        self.df, self.docs = self._load_hotels()
        self._setup_tfidf()

    def _setup_llama(self):
        self.model = Llama(model_path=self.model_path, n_ctx=1056, n_threads=4, n_gpu_layers=35)

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

    def _setup_tfidf(self):
        if os.path.exists(self.vectorizer_path) and os.path.exists(self.vector_path) and os.path.exists(self.index_path):
            # Cargar vectorizer
            with open(self.vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            # Cargar vectores (mantenemos para compatibilidad)
            with open(self.vector_path, "rb") as f:
                self.doc_vectors = pickle.load(f)
            # 🔹 Cargar índice FAISS
            self.index = faiss.read_index(self.index_path)
            logger.info("✅ TF-IDF vectorizer, vectores y FAISS index cargados.")
        else:
            logger.info("🔄 Generando TF-IDF vectores con FAISS...")
            # Crear vectorizer y ajustar
            self.vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
            self.doc_vectors = self.vectorizer.fit_transform(self.docs)
            
            # Guardar vectorizer y vectores
            with open(self.vectorizer_path, "wb") as f:
                pickle.dump(self.vectorizer, f)
            with open(self.vector_path, "wb") as f:
                pickle.dump(self.doc_vectors, f)
            
            # 🔹 Crear índice FAISS
            # Convertir sparse matrix a dense y normalizar
            doc_vectors_dense = self.doc_vectors.toarray().astype("float32")
            faiss.normalize_L2(doc_vectors_dense)
            
            # Crear índice FAISS con cosine similarity (Inner Product después de normalización)
            dimension = doc_vectors_dense.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(doc_vectors_dense)
            
            # Guardar índice
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
        # 🔹 Usar FAISS en lugar de cosine_similarity manual
        pregunta_vec = self.vectorizer.transform([pregunta_en])
        pregunta_dense = pregunta_vec.toarray().astype("float32")
        faiss.normalize_L2(pregunta_dense)  # Normalizar para cosine similarity
        
        # Buscar con FAISS
        D, I = self.index.search(pregunta_dense, top_k)
        contexto = [self.docs[i] for i in I[0]]
        return contexto

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
            output = self.model(prompt, max_tokens=300, stop=["Usuario:", "Pregunta:"], echo=False)
            texto = output["choices"][0]["text"].strip()
            if self._is_english(texto):
                texto = self._translate(texto, src="en", dest="es")
            return texto
        except Exception as e:
            logger.error(f"❌ Error generando respuesta con LLaMA3: {e}")
            return "❌ Hubo un problema generando la respuesta. Intenta de nuevo."

    def procesar_consulta(self, pregunta: str) -> str:
        return self.generar_respuesta(pregunta)


# === EJEMPLO DE USO ===
if __name__ == "__main__":
    rag = RAGHotelSystem_LLaMA3_TFIDF()
    pruebas = [
        "Busco un hotel barato con wifi",
        "Quiero un hotel de lujo cerca del centro",
        "Hay hoteles con desayuno incluido y gimnasio?",
        "Hotel familiar cerca de atracciones"
    ]

    for consulta in pruebas:
        print(f"\n🟢 Consulta: {consulta}")
        print(rag.procesar_consulta(consulta))