import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Cargar dataset ---
df = pd.read_csv("Dataframe_para_métricas.csv")
df["contexts"] = df["contexts"].apply(eval)

# --- 2. Cargar modelo de embeddings ---
embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# --- 3. Cargar modelo LLaMA local (.gguf) ---
llm = Llama(
    model_path="Open-Insurance-LLM-Llama3-8B-Q3_K_M.gguf",  # Ajusta ruta si es necesario
    n_ctx=2048,
    n_threads=4
)

# --- 4. Función para similitud coseno ---
def cosine_sim(text1, text2):
    emb = embedder.encode([text1, text2])
    return cosine_similarity([emb[0]], [emb[1]])[0][0]

# --- 5. Evaluación ---
resultados = []

def preguntar_llama(prompt):
    output = llm(prompt, max_tokens=100)
    respuesta = output["choices"][0]["text"].strip().lower()
    return "sí" in respuesta

for _, row in df.iterrows():
    pregunta = row["question"]
    respuesta_real = row["answer"]
    respuesta_generada = row["generated_answer"]
    contexto = " ".join(row["contexts"])

    sim = cosine_sim(respuesta_real, respuesta_generada)

    # Fidelity
    prompt_fidelidad = f"""Dado este contexto:\n{contexto}\n\nY esta respuesta generada:\n{respuesta_generada}\n\n¿La respuesta es fiel al contexto? Responde solo 'sí' o 'no'."""
    fidelity = preguntar_llama(prompt_fidelidad)

    # Relevancia
    prompt_relevancia = f"""Pregunta: {pregunta}\nRespuesta: {respuesta_generada}\n\n¿La respuesta es relevante a la pregunta? Responde solo 'sí' o 'no'."""
    relevancia = preguntar_llama(prompt_relevancia)

    # Precisión
    prompt_precision = f"""Respuesta: {respuesta_generada}\n\nContexto:\n{contexto}\n\n¿La respuesta se basa exclusivamente en el contexto? Responde solo 'sí' o 'no'."""
    precision = preguntar_llama(prompt_precision)

    # Recall
    prompt_recall = f"""Pregunta: {pregunta}\nContexto:\n{contexto}\n\n¿El contexto contiene suficiente información para responder la pregunta? Responde solo 'sí' o 'no'."""
    recall = preguntar_llama(prompt_recall)

    resultados.append({
        "Pregunta": pregunta,
        "Similitud (0-1)": round(sim, 3),
        "Fiel al contexto": fidelity,
        "Relevante": relevancia,
        "Preciso": precision,
        "Suficiente (Recall)": recall
    })

# --- 6. Crear DataFrame final ---
resultados_df = pd.DataFrame(resultados)

# --- 7. Guardar en CSV ---
resultados_df.to_csv("metricas_llama3_local.csv", index=False)

# --- 8. Visualización ---
sns.set(style="whitegrid")

# Promedio similitud
plt.figure(figsize=(8, 4))
sns.barplot(x=["Similitud promedio"], y=[resultados_df["Similitud (0-1)"].mean()])
plt.ylim(0, 1)
plt.title("Promedio de similitud entre respuestas (coseno)")
plt.ylabel("Similitud")
plt.show()

# Porcentaje de respuestas 'sí' en booleanas
metricas_bool = ["Fiel al contexto", "Relevante", "Preciso", "Suficiente (Recall)"]
porcentajes = [(resultados_df[m].sum() / len(resultados_df)) * 100 for m in metricas_bool]

plt.figure(figsize=(10, 5))
sns.barplot(x=metricas_bool, y=porcentajes, palette="viridis")
plt.ylabel("Porcentaje (%)")
plt.title("Métricas booleanas por respuesta")
plt.ylim(0, 100)
plt.xticks(rotation=15)
plt.show()

# --- 9. Imprimir resultados en consola ---
print(resultados_df)
