import os
import re
import pickle
import logging
import pandas as pd
from typing import List
from googletrans import Translator
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import numpy as np
import faiss  # 🔹 FAISS agregado

# Carga el modelo localmente desde archivo .gguf
llm = Llama(model_path="Open-Insurance-LLM-Llama3-8B-Q3_K_M.gguf")

# Prueba rápida
response = llm("¿Cuál es el hotel más lujoso?", max_tokens=50)
print(response["choices"][0]["text"].strip())

# ========== CONFIGURACIÓN ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGHotelSystem_LLaMA3_STF:
    def __init__(self, excel_file="100_Valledupar_hotels_English.xlsx", model_path="Open-Insurance-LLM-Llama3-8B-Q3_K_M.gguf"):
        self.excel_file = excel_file
        self.embeddings_path = "doc_embeddings_sentence.pkl"  # 🔹 Cambio de nombre
        self.index_path = "faiss_index_sentence.index"  # 🔹 Cambio de nombre
        self.translator = Translator()
        self.model_path = model_path
        self._setup_llama()
        self.df, self.docs = self._load_hotels()
        self._setup_embeddings()

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

    def _setup_embeddings(self):
        # 🔹 Usar Sentence Transformers en lugar de Nomic
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("✅ Modelo Sentence Transformers cargado")

        # Cargar embeddings si existen
        if os.path.exists(self.embeddings_path) and os.path.exists(self.index_path):
            with open(self.embeddings_path, "rb") as f:
                self.doc_embeddings = pickle.load(f)
            self.index = faiss.read_index(self.index_path)
            logger.info("✅ Embeddings y FAISS index cargados.")
        else:
            logger.info("🔄 Generando embeddings con Sentence Transformers...")
            # 🔹 Usar Sentence Transformers
            self.doc_embeddings = self.embedding_model.encode(self.docs, show_progress_bar=True)
            with open(self.embeddings_path, "wb") as f:
                pickle.dump(self.doc_embeddings, f)

            # 🔹 Crear índice FAISS (con L2, tras normalizar)
            self.doc_embeddings = np.array(self.doc_embeddings).astype("float32")
            faiss.normalize_L2(self.doc_embeddings)
            dimension = self.doc_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.doc_embeddings)
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
        # 🔹 Usar Sentence Transformers
        embedding = self.embedding_model.encode([pregunta_en])[0]
        embedding = np.array([embedding]).astype("float32")
        faiss.normalize_L2(embedding)  # 🔹 Normaliza también la consulta
        D, I = self.index.search(embedding, top_k)
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
    rag = RAGHotelSystem_LLaMA3_STF()
    pruebas = [
        "Busco un hotel barato con wifi",
        "Quiero un hotel de lujo cerca del centro",
        "Hay hoteles con desayuno incluido y gimnasio?",
        "Hotel familiar cerca de atracciones"
    ]

    for consulta in pruebas:
        print(f"\n🟢 Consulta: {consulta}")
        print(rag.procesar_consulta(consulta))