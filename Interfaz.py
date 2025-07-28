import gradio as gr
import logging
from hotel_recommender_rag import RAGHotelSystem  # clase base
from HotelSystem_Gemini_ST import RAGHotelSystem_Gemini_STF
from HotelSystem_LLaMA3_ST import RAGHotelSystem_LLaMA3_STF
from HotelSystem_LLaMA3_TDFIDF import RAGHotelSystem_LLaMA3_TFIDF

# === Configurar logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InterfazHotel:
    def __init__(self, hotel_system):
        self.hotel_system = hotel_system

    def limpiar_campos(self):
        """Limpiar todos los campos de la interfaz"""
        return ("", "")

    def actualizar_modelo(self, modelo):
        """Actualiza el modelo RAG según la selección del usuario"""
        try:
            logger.info(f"🔄 Cambiando a modelo: {modelo}")
            
            if modelo == "Base":
                self.hotel_system = RAGHotelSystem()
            elif modelo == "Gemini + Sentence Transformers":
                self.hotel_system = RAGHotelSystem_Gemini_STF()
            elif modelo == "LLaMA3 + Sentence Transformers":
                self.hotel_system = RAGHotelSystem_LLaMA3_STF()
            elif modelo == "LLaMA3 + TF-IDF":
                self.hotel_system = RAGHotelSystem_LLaMA3_TFIDF()
            else:
                raise ValueError(f"Modelo no reconocido: {modelo}")
                
            logger.info(f"✅ Modelo {modelo} cargado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error al cargar el modelo '{modelo}': {e}")
            self.hotel_system = None
            return False

    def procesar_consulta_con_modelo(self, modelo, consulta):
        """Procesa la consulta con el modelo seleccionado"""
        try:
            if not consulta.strip():
                return "⚠️ Por favor, escribe una consulta."

            # Actualizar el modelo si es necesario
            if not self.actualizar_modelo(modelo):
                return f"❌ No se pudo cargar el modelo '{modelo}'."

            logger.info(f"🔍 Procesando consulta con {modelo}: {consulta[:50]}...")
            
            # Procesar la consulta
            if self.hotel_system:
                respuesta = self.hotel_system.procesar_consulta(consulta)
                logger.info(f"✅ Respuesta generada exitosamente")
                return respuesta
            else:
                return f"❌ Error: Sistema {modelo} no inicializado."
                
        except Exception as e:
            error_msg = f"❌ Error al procesar consulta con {modelo}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _get_custom_css(self):
        return """
         body { background-color: #f5fdf7; }

        h1, h2, h3 {
            font-weight: 700;
        }

        .custom-header {
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            background: linear-gradient(135deg, #0b6623, #198754);
            color: white;
            border-radius: 0 0 15px 15px;
            padding: 20px;
            margin-bottom: 20px;
        }

        .custom-header img {
            width: 120px;
            border-radius: 12px;
            margin-bottom: 10px;
        }

        .custom-header h1 {
            margin: 0;
            font-size: 2.2rem;
        }

        .custom-header h3 {
            margin: 5px 0 0;
            font-weight: 400;
        }
        """

    def _create_header(self):
        with gr.Row(elem_id="header-row"):
            with gr.Column():
                gr.HTML(f"""
                    <div class="custom-header">
                        <h1>🏨 Recomendador de Hoteles</h1>
                        <h3>Valledupar - Turismo, naturaleza y comodidad</h3>
                    </div>
                """)

    def _create_examples(self, entrada):
        gr.Examples(
            examples=[
                "¿Qué hoteles tienen piscina y están cerca del centro?",
                "Hoteles con parqueadero gratis y cerca del aeropuerto",
                "Busco un hotel económico con aire acondicionado",
                "¿Qué hotel boutique tiene buena calificación y Wi-Fi gratis?",
                "Quiero un hotel cerca del río con desayuno incluido",
            ],
            inputs=entrada
        )

    def _create_info_section(self):
        gr.Markdown(
            """
            ### ℹ️ Información
            Este sistema combina procesamiento de lenguaje natural con embeddings y recuperación semántica para sugerirte hoteles que se adapten a tus preferencias.
            """
        )

    def crear_interfaz(self):
        """Crear la interfaz Gradio"""
        
        with gr.Blocks(
            css=self._get_custom_css(),
            theme=gr.themes.Soft(
                primary_hue="emerald",
                secondary_hue="green",
                neutral_hue="slate",
                font=gr.themes.GoogleFont("Inter")
            )
        ) as demo:

            self._create_header()

            # 🔘 Selector de modelo
            with gr.Row():
                modelo_selector = gr.Radio(
                    label="🧠 Elige el motor de recomendación",
                    choices=[
                        "Base",
                        "Gemini + Sentence Transformers",
                        "LLaMA3 + Sentence Transformers",
                        "LLaMA3 + TF-IDF"
                    ],
                    value="Base"
                )

            # Entrada
            with gr.Row():
                with gr.Column(scale=3):
                    entrada = gr.Textbox(
                        label="🔍 Escribe tu pregunta:",
                        placeholder="Ej: ¿Qué hoteles tienen piscina y están cerca del centro?",
                        lines=3,
                        info="Describe qué tipo de hotel buscas, ubicación preferida, servicios deseados, etc."
                    )

            # Área de respuesta
            salida = gr.Textbox(
                label="💬 Respuesta del asistente",
                lines=12,
                max_lines=25,
                show_copy_button=True,
                info="Aquí aparecerán las recomendaciones personalizadas"
            )

            # Botones
            with gr.Row():
                with gr.Column(scale=2):
                    boton = gr.Button("🔍 Consultar Hoteles", size="lg", variant="primary")
                with gr.Column(scale=1):
                    limpiar = gr.Button("🧹 Limpiar", size="lg", variant="secondary")

            # Ejemplos
            self._create_examples(entrada)

            # Información
            self._create_info_section()

            # === EVENTOS CORREGIDOS ===
            boton.click(
                fn=self.procesar_consulta_con_modelo,
                inputs=[modelo_selector, entrada],
                outputs=[salida]
            )

            limpiar.click(
                fn=self.limpiar_campos,
                outputs=[entrada, salida]
            )

            entrada.submit(
                fn=self.procesar_consulta_con_modelo,
                inputs=[modelo_selector, entrada],
                outputs=[salida]
            )

            # Configuración avanzada (opcional, colapsable)
            with gr.Accordion("⚙️ Configuración avanzada (opcional)", open=False):
                with gr.Row():
                    temperatura = gr.Slider(
                        label="🎛️ Temperatura de generación (solo para modelos generativos)",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=0.7,
                        interactive=True,
                        info="Controla la creatividad de las respuestas"
                    )
                    top_k = gr.Slider(
                        label="🔝 Número de documentos a recuperar (Top-K)",
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=5,
                        interactive=True,
                        info="Cuántos documentos relevantes considerar en la respuesta"
                    )

            # Ejemplos destacados (zona de pruebas)
            gr.Markdown("### 🧪 Ejemplos Rápidos de Prueba")
            with gr.Row():
                gr.Examples(
                    examples=[
                        "¿Qué hoteles tienen parqueadero gratis?",
                        "Busco un hotel cerca del aeropuerto con desayuno incluido",
                        "Hoteles recomendados para viaje de negocios",
                        "Quiero un hotel ecológico y con buena puntuación",
                        "¿Hay hoteles con spa y piscina en Valledupar?"
                    ],
                    inputs=entrada,
                    label="Haz clic en un ejemplo para probarlo"
                )

        return demo