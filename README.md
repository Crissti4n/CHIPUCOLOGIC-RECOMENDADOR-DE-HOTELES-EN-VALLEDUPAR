# CHIPUCOLOGIC
![](C3.jpeg) 

**CHIPUCOLOGIC** es un agente conversacional inteligente desarrollado para recomendar hoteles de forma personalizada en la ciudad de Valledupar, Colombia. El sistema utiliza técnicas de **Retrieval-Augmented Generation (RAG)** y procesamiento de lenguaje natural para interpretar las preferencias del usuario y ofrecer sugerencias adaptadas a sus necesidades específicas. Se integra una base de conocimiento detallada de 100 establecimientos de alojamiento, incorporando no solo sus características propias, sino también información contextual relevante como proximidad a atracciones turísticas, lugares de esparcimiento y de actividades recreativas, la terminal de transporte, el aeropuerto, servicios para mascotas y otros factores determinantes para diferentes perfiles de viajeros.

##  Características Principales

-  Interfaz conversacional en lenguaje natural (texto y voz) diseñada en gradio
-  Base de datos de 100 establecimientos hoteleros  
-  4 configuraciones RAG diferentes para optimizar respuestas  
-  Información contextual sobre ubicaciones, servicios y atracciones turísticas  
-  Respuestas en tiempo real (3-5 segundos promedio)  
-  Recomendaciones personalizadas basadas en preferencias del usuario  

##  Arquitectura del Sistema

El sistema implementa una arquitectura **RAG** que combina cuatro configuraciones diferentes:

- `Gemini + TF-IDF` (Configuración óptima - ~3s por respuesta)  
- `Gemini + Sentence Transformer` (~5s por respuesta)  
- `LLaMA3 + Sentence Transformer` (~1m 57s por respuesta)  
- `LLaMA3 + TF-IDF` (~2m 13s por respuesta)
## Uso del Sistema

### Ejemplos de Consultas

- "Quiero un hotel cerca del centro y que acepte mascotas"
- "¿Qué hoteles tienen piscina y están cerca del balneario Hurtado?"
- "Necesito alojamiento familiar con desayuno incluido"
- "Hoteles de lujo cerca del aeropuerto"

###  Selección de Configuración RAG

La interfaz permite seleccionar entre las 4 configuraciones disponibles:

- **Base (Recomendado):** Gemini + TF-IDF
- **Gemini + Sentence Transformers:** Mayor precisión semántica
- **LLaMA3 + Sentence Transformers:** Funcionamiento offline
- **LLaMA3 + TF-IDF:** Versión completamente local
 ## Requisitos del Entorno y Ejecución

Antes de ejecutar el sistema CHIPUCOLOGIC, es necesario configurar un entorno virtual con las dependencias requeridas. Esto asegura la correcta ejecución de la arquitectura RAG y la interfaz conversacional.

###  Creación del Entorno Virtual

Se recomienda utilizar `venv` para crear un entorno aislado:

python -m venv chipucologic-env
source chipucologic-env/bin/activate  # En Linux/macOS
chipucologic-env\Scripts\activate     # En Windows

### Instalación de Dependencias
Ejecuta el siguiente comando para instalar todas las dependencias necesarias:

pip install -r requirements.txt
También puedes instalar manualmente cada biblioteca según sea necesario.
Adicional a esto debes instalar el modelo LLaMA utilizado "Open-Insurance-LLM-Llama3-8B-Q3_K_M.gguf", anexarlo a la misma carpeta donde tienes tus demás códigos del Rag y luego si proceder con la ejecución mencionada.

### Ejecución del Sistema
Para iniciar el sistema, simplemente ejecuta el archivo principal:
python ejecutor.py
Este script inicializa la interfaz en Gradio y carga las cuatro configuraciones del sistema RAG

## Autor
**Cristian Enrique Hernandez Ospino**  
📧 Email: [chernandezos@unal.edu.co](mailto:chernandezos@unal.edu.co)  
🎓 Universidad Nacional de Colombia - Sede De La Paz
