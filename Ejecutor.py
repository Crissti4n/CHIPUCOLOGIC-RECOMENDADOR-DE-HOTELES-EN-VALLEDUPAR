from hotel_recommender_rag import RAGHotelSystem  # clase base
from HotelSystem_Gemini_ST import RAGHotelSystem_Gemini_STF
from HotelSystem_LLaMA3_ST import RAGHotelSystem_LLaMA3_STF
from HotelSystem_LLaMA3_TDFIDF import RAGHotelSystem_LLaMA3_TFIDF
from Interfaz import InterfazHotel


def seleccionar_sistema(nombre_modelo: str = ""):
    """Retorna una instancia del sistema RAG correspondiente según el nombre dado."""
    nombre_modelo = nombre_modelo.lower()

    if nombre_modelo == "gemini_nomic":
        return RAGHotelSystem_Gemini_STF()
    elif nombre_modelo == "llama3_":
        return RAGHotelSystem_LLaMA3_STF()
    elif nombre_modelo == "llama3_tfidf":
        return RAGHotelSystem_LLaMA3_TFIDF()
    else:
        print(f"⚠️ Modelo '{nombre_modelo}' no reconocido o no especificado. Usando modelo base (TF-IDF simple).")
        return RAGHotelSystem()  # sistema base por defecto


if __name__ == "__main__":
    # Escoge el modelo que quieres usar por defecto
    modelo = ""  # opciones: "", "gemini_STF", "llama3_STF", "llama3_tfidf"
    
    sistema_rag = seleccionar_sistema(modelo)

    # ⚠️ ¡Aquí pasamos el sistema RAG al constructor!
    interfaz = InterfazHotel(hotel_system=sistema_rag)

    # Lanza la interfaz
    interfaz.crear_interfaz().launch()
