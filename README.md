# CHIPUCOLOGIC

**CHIPUCOLOGIC** es un agente conversacional inteligente desarrollado para recomendar alojamientos personalizados en la ciudad de Valledupar, Colombia.  
El sistema utiliza técnicas de **Retrieval-Augmented Generation (RAG)** y procesamiento de lenguaje natural para interpretar las preferencias del usuario y ofrecer sugerencias adaptadas a sus necesidades específicas.

## ✨ Características Principales

- 🗣️ Interfaz conversacional en lenguaje natural (texto y voz)  
- 🏨 Base de datos de 100 establecimientos hoteleros  
- 🧠 4 configuraciones RAG diferentes para optimizar respuestas  
- 🌍 Información contextual sobre ubicaciones, servicios y atracciones turísticas  
- ⚡ Respuestas en tiempo real (3-5 segundos promedio)  
- 🎯 Recomendaciones personalizadas basadas en preferencias del usuario  

## 🏗️ Arquitectura del Sistema

El sistema implementa una arquitectura **RAG** que combina cuatro configuraciones diferentes:

- `Gemini + TF-IDF` (Configuración óptima - ~3s por respuesta)  
- `Gemini + Sentence Transformer` (~5s por respuesta)  
- `LLaMA3 + Sentence Transformer` (~1m 57s por respuesta)  
- `LLaMA3 + TF-IDF` (~2m 13s por respuesta)  

