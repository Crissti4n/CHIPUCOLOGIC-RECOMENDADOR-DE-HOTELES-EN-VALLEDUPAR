import pandas as pd
import numpy as np

# Crear dataset completo para evaluación RAG con RAGAS
df = pd.DataFrame ( {
    'question': [
        # Preguntas facturas básicas
        "¿Cuántos hoteles hay en total en Valledupar según la lista?",
        "¿En qué dirección se encuentra el Hotel Tativan?",
        "¿Qué rating tiene el Sonesta Valledupar?",
        "¿Cuál es el número de teléfono del Hotel Sicarare?",
        
        # Preguntas sobre ubicaciones específicas
        "¿Cuáles son los hoteles que están cerca de la Plaza Alfonso López?",
        "¿Qué hoteles ofrecen vista al río Guatapurí?",
        "¿Cuáles hoteles están ubicados en el Centro Histórico de Valledupar?",
        "¿Qué hoteles están cerca del Balneario Hurtado?",
        
        # Preguntas sobre servicios y amenidades
        "¿Qué hoteles ofrecen piscina al aire libre?",
        "¿Cuáles hoteles tienen restaurante propio?",
        "¿Qué hoteles ofrecen servicio de Wi-Fi gratuito?",
        "¿Cuáles hoteles tienen aire acondicionado en todas las habitaciones?",
        "¿Qué hoteles ofrecen servicio de transporte al aeropuerto?",
        
        # Preguntas sobre categorías de hoteles
        "¿Cuántos hoteles tienen rating de cinco estrellas?",
        "¿Cuáles son todos los hoteles con rating de dos estrellas?",
        "¿Qué hoteles tienen rating de cuatro estrellas y están cerca del centro?",
        "¿Cuáles son los hoteles boutique disponibles en Valledupar?",
        "¿Qué hostales están disponibles en la ciudad?",
        
        # Preguntas sobre temática vallenata
        "¿Qué hoteles ofrecen serenatas de acordeón en las noches?",
        "¿Cuáles hoteles tienen nombres inspirados en canciones vallenatas?",
        "¿Qué hoteles tienen decoración inspirada en compositores locales?",
        "¿Cuál es el hotel llamado 'La Gota Fría' y dónde está ubicado?",
        "¿Qué hoteles tienen nombres relacionados con la cultura vallenata?",
        
        # Preguntas comparativas
        "¿Cuál es la diferencia entre Hotel Mizare I y Hotel Mizare II?",
        "¿Qué hoteles Hampton están disponibles en Valledupar?",
        "¿Cuáles son las diferencias entre los hoteles Ayenda en la ciudad?",
        "¿Qué hoteles Casa están disponibles y cuáles son sus características?",
        "¿Cuáles son los hoteles más económicos según su rating?",
        
        # Preguntas sobre servicios específicos
        "¿Qué hoteles ofrecen servicio de spa y wellness?",
        "¿Cuáles hoteles tienen facilidades para personas con discapacidad?",
        "¿Qué hoteles permiten mascotas?",
        "¿Cuáles hoteles ofrecen servicio de lavandería?",
        "¿Qué hoteles tienen salas de reuniones y conferencias?",
        
        # Preguntas sobre distancias y ubicación
        "¿Cuál es el hotel más cercano al aeropuerto Alfonso López Pumarejo?",
        "¿Qué hoteles están a menos de 1 km del Coliseo de Valledupar?",
        "¿Cuáles hoteles están cerca del Centro Comercial Guatapurí?",
        "¿Qué hoteles ofrecen fácil acceso al mercado artesanal de Nabusimake?",
        "¿Cuáles son los hoteles ubicados en Pueblo Bello?",
        
        # Preguntas sobre características especiales
        "¿Qué hoteles tienen piscina cubierta?",
        "¿Cuáles hoteles ofrecen desayuno buffet?",
        "¿Qué hoteles tienen bar o lounge?",
        "¿Cuáles hoteles ofrecen estacionamiento gratuito?",
        "¿Qué hoteles tienen recepción 24 horas?",
       
    ],
    
    'contexts': [
        # Contextos correspondientes a cada pregunta
        ["La lista contiene hoteles numerados desde el código 1005799 hasta 6073910, con un total de aproximadamente 100 hoteles listados en Valledupar y algunos en Pueblo Bello."],
        ["Hotel Tativan está ubicado en Calle 16 A # 9 50, Valledupar 200001, es un hotel de cuatro estrellas."],
        ["Sonesta Valledupar tiene un rating de ThreeStar y está ubicado en Diagonal 10 # 6N 15, Centro Comercial."],
        ["Hotel Sicarare - Código: 1357287, Rating: FourStar, Dirección: Carrera 9A No. 16-04, Valledupar. No se especifica número telefónico en la información disponible."],
        
        ["Hoteles cerca de Plaza Alfonso López: Hotel Arawak Upar (200 metros), Hotel Palatino (250 metros), Hostal La Patillalera, Hotel El Viejo Migue, Hotel Tierra de Cantores, Hotel El Cóndor Herido, Hotel Vallenato Inmortal, Hostal Los Papayeros."],
        ["Hoteles con vista/cerca del río Guatapurí: Hostal Luna Sanjuanera (near balneario hurtado on guatapurí river), Hotel Balneario Hurtado, Hotel Mis Mejores Días, Hotel El Chanchullito, Hostal Plaza Alfonso López (walking distance to guatapurí river beach), Hotel Guatapurí Riverside."],
        ["Hoteles en Centro Histórico: Hostal Hermanos Zuleta, Hotel El Testamento, ambos especifican 'close to centro histórico de valledupar'."],
        ["Hoteles cerca del Balneario Hurtado: Hostal Luna Sanjuanera, varios hoteles mencionan 'Near Balneario Hurtado on Guatapurí River' en sus descripciones de atracciones."],
        
        ["Hoteles con piscina exterior: Hotel Tativan, Hotel Boutique Casa Rosalía, Vajamar Hotel, Hotel Rancho Regis, Hotel Boutique Casa de Los Santos Reyes Valledupar, Hotel Campestre Villa Ocha, y muchos otros que mencionan 'outdoor pool' en sus facilidades."],
        ["Hoteles con restaurante: Hotel Tativan, Vajamar Hotel, Hotel Arawak Upar, Hotel Kurakatá, Hotel Palatino, Hotel Balop, Hotel Nuevo Tio, Hotel Progreso Plaza, Hotel Arce Plaza, Hotel Serrano Plaza, entre muchos otros."],
        ["Prácticamente todos los hoteles listados ofrecen Wi-Fi gratuito, incluyendo Hotel Tativan, Provincia Hostel, Hotel Boutique Casa Rosalía, Sonesta Valledupar, y la gran mayoría de establecimientos."],
        ["La mayoría de hoteles tienen aire acondicionado, especialmente los de rating superior. Hostal Mizare Valledupar ofrece 'rooms with and without air conditioning'."],
        ["Hoteles con servicio al aeropuerto: varios hoteles mencionan 'airport shuttle' o 'airport transportation' en sus servicios, incluyendo Hotel Casa Blanca, Sierra Nevada B&B, Hostal Caminos del Cesar, entre otros."],
        
        ["Hoteles cinco estrellas: Hotel La Gota Fría, Hotel Sirena Vallenata, Hostal La Parranda, Hostal Luna Sanjuanera, Hostal El Cantor de Fonseca, Hotel El Testamento, Hotel Tierra de Cantores, Hotel El Cóndor Herido, Hostal Cerro de La Popa."],
        ["Hoteles dos estrellas: Hotel 999, Hotel Sarachuy Valledupar, La Pilonera, Hostal Lola, Apartamentos Las Piloneras, Hotel Pedazo de Acordeón, Hotel El Ángel de mi Tambo, Hotel La Cañaguatera, Hotel Lluvia de Bendiciones, Hostal Mírame Fijamente, Hotel Mis Mejores Días, Hotel El Viejo Migue, Hotel El Alcaraván, Hostal Plaza Alfonso López."],
        ["Hoteles cuatro estrellas cerca del centro: Hotel Tativan, Sicarare, Pompatao Apartamentos & Suites, Hotel Ucla Center, Hotel Casa en el Aire, Hostal Hermanos Zuleta, Hotel Fiesta Vallenata, Hostal Nació Mi Poesía, Hotel El Chanchullito, Hostal Mi Muchacho, Hostal El Rey de Reyes, Hotel El Chanchullito, Hostal Nido de Palomas, Hostal La Casa en Reyes, Hotel Río Badillo."],
        ["Hoteles boutique: Hotel Boutique Casa Rosalia, Hotel Boutique Casa de Los Santos Reyes Valledupar."],
        ["Hostales: Provincia Hostel Valledupar, Hostal Mizare I, Hostal Mizare II, Hostal Caminos del Cesar, más todos los establecimientos con nombres que incluyen 'Hostal'."],
        
        ["Hoteles con acordeón: La mayoría de hoteles con nombres vallenatos ofrecen 'live accordion serenades every evening' según sus descripciones, especialmente los de códigos 6073868 en adelante."],
        ["Hoteles con nombres vallenatos: Hotel La Gota Fría, Hotel Sirena Vallenata, Hostal La Parranda, Hotel Fiesta Vallenata, Hotel Pedazo de Acordeón, Hostal Nació Mi Poesía, Hotel El Ángel de mi Tambo, Hostal Luna Sanjuanera, entre muchos otros."],
        ["Todos los hoteles de la serie 6073868-6073910 mencionan 'décor inspired by local composers and folklore' en sus descripciones."],
        ["Hotel La Gota Fría - Código: 6073872, Rating: FiveStar, Dirección: Carrera 6 #12-13, Valledupar, Cesar. Facing colegio nacional loperena."],
        ["Hoteles con nombres culturales vallenatos incluyen referencias a canciones, compositores y elementos del folklore: La Gota Fría, Sirena Vallenata, La Parranda, Fiesta Vallenata, Pedazo de Acordeón, Nació Mi Poesía, Luna Sanjuanera, El Testamento, entre otros."],
        
        ["Hotel Mizare I: Código 1542046, ubicación Calle 9 8-42 Barrio Novalito, rating All. Hotel Mizare II: Código 1659541, ubicación Carrera 5 9A-42, rating ThreeStar. Mizare II tiene mejor rating."],
        ["Hampton by Hilton Valledupar Colombia: Código 1392610, ThreeStar, Calle 30 # 6 a 133, ValleduparCesar. Es el único hotel Hampton en la lista."],
        ["Ayenda Consulado Santandereano: Código 5014454, dirección 1636 Calle 14. Ayenda Union: Código 5014456, dirección 1115 Calle 19b. Ambos con rating 'All'."],
        ["Hoteles Casa: Hotel Boutique Casa Rosalía, Hotel Boutique Casa de Los Santos Reyes, Hotel Casa Blanca, Casa Paulina Hostal, Casa Kundalini Conjunto Los Corales, Hotel Casa en el Aire, Hostal La Casa en Reyes."],
        ["Hoteles económicos (2 estrellas): Hotel 999, Hotel Sarachuy Valledupar, La Pilonera, Hostal Lola, varios otros con rating TwoStar."],
        
        ["Hoteles con spa: Hotel Serrano Plaza, Hotel Campestre Villa Ocha, Nabusimake Hospedaje Sra Mariangela mencionan 'spa and wellness centre' o servicios de masajes."],
        ["Hoteles accesibles: varios mencionan 'Wheelchair accessible' o 'Facilities for disabled guests' incluyendo Sonesta Valledupar, Sierra Nevada B&B, Hotel La Vieja Sara, entre otros."],
        ["Hoteles pet-friendly: muchos hoteles mencionan 'Pets allowed' en sus facilidades, incluyendo Hotel Kurakatá, Hotel Balop, Hotel Urak, Hotel Colibri, entre otros."],
        ["Servicios de lavandería disponibles en la mayoría de hoteles listados, mencionados como 'Laundry' o 'Laundry facilities' en las descripciones de servicios."],
        ["Hoteles con salas de reuniones: Hotel Tativan (7 conference rooms), Vajamar Hotel, Hotel Arawak Upar, Hotel Progreso Plaza, Hotel Serrano Plaza, entre otros que mencionan 'meeting/banquet facilities'."],
        
        ["Hotel más cercano al aeropuerto: Hotel San Diego Inc. Valledupar a 1 km del Alfonso López Pumarejo Airport."],
        ["Hoteles cerca del Coliseo: según las distancias mencionadas, Casa Cultural y de Huéspedes La Pilonera está a 0.7 km del Valledupar Coliseum."],
        ["Hoteles cerca de Guatapurí Shopping: Sonesta Hotel Valledupar está 'steps from Guatapuri Shopping Mall', Hotel Rancho Regis a 1.2 km."],
        ["Hoteles con acceso a Nabusimake: varios mencionan 'Short ride to Nabusimake artesanal market' incluyendo Hostal Mi Muchacho, Hostal La Casa en Reyes."],
        ["Hoteles en Pueblo Bello: Nabusimake Hospedaje Sra Mariangela, Villasthefany."],
        
        ["Hoteles con piscina cubierta: Hotel Sicarare menciona 'indoor swimming pool', Hotel Colibri menciona 'Indoor pool'."],
        ["Hoteles con desayuno buffet: Hotel Tativan, Hotel Serrano Plaza mencionan específicamente 'buffet breakfast'."],
        ["Hoteles con bar: Hotel Tativan, Hotel Arawak Upar, Hotel Nuevo Tio, Hotel La Vieja Sara, Hotel Serrano Plaza, Villasthefany, entre otros."],
        ["La mayoría de hoteles ofrecen estacionamiento gratuito, mencionado como 'Free parking' o 'Free self parking' en sus servicios."],
        ["Prácticamente todos los hoteles mencionan '24-hour front desk' en sus servicios."],

       ], 
    
    'answer': [
        # Respuestas esperadas
        "La lista contiene aproximadamente 100 hoteles en total, con códigos que van desde 1005799 hasta 6073910, ubicados principalmente en Valledupar y algunos en Pueblo Bello.",
        "El Hotel Tativan está ubicado en Calle 16 A # 9 50, Valledupar 200001.",
        "El Sonesta Valledupar tiene un rating de tres estrellas (ThreeStar).",
        "La información proporcionada no incluye el número telefónico específico del Hotel Sicarare. Se encuentra ubicado en Carrera 9A No. 16-04, Valledupar y tiene un rating de cuatro estrellas.",
        
        "Los hoteles cerca de la Plaza Alfonso López incluyen Hotel Arawak Upar (200 metros), Hotel Palatino (250 metros), Hostal La Patillalera, Hotel El Viejo Migue, Hotel Tierra de Cantores, Hotel El Cóndor Herido, Hotel Vallenato Inmortal y Hostal Los Papayeros.",
        "Los hoteles con vista o cerca del río Guatapurí incluyen Hostal Luna Sanjuanera (cerca del balneario Hurtado), Hotel Mis Mejores Días, Hotel El Chanchullito, Hostal Plaza Alfonso López (a distancia caminable de la playa del río) y Hotel Guatapurí Riverside.",
        "Los hoteles ubicados en el Centro Histórico de Valledupar son Hostal Hermanos Zuleta y Hotel El Testamento.",
        "Los hoteles cerca del Balneario Hurtado incluyen Hostal Luna Sanjuanera y varios otros que mencionan estar cerca del Balneario Hurtado en el río Guatapurí.",
        
        "Los hoteles con piscina al aire libre incluyen Hotel Tativan, Hotel Boutique Casa Rosalía, Vajamar Hotel, Hotel Rancho Regis, Hotel Boutique Casa de Los Santos Reyes, Hotel Campestre Villa Ocha, y muchos otros que especifican 'outdoor pool' en sus servicios.",
        "Los hoteles con restaurante propio incluyen Hotel Tativan, Vajamar Hotel, Hotel Arawak Upar, Hotel Kurakatá, Hotel Palatino, Hotel Balop, Hotel Nuevo Tio, Hotel Progreso Plaza, Hotel Arce Plaza, Hotel Serrano Plaza, entre muchos otros.",
        "Prácticamente todos los hoteles listados ofrecen Wi-Fi gratuito, incluyendo desde establecimientos básicos hasta hoteles de lujo como Hotel Tativan, Provincia Hostel, Hotel Boutique Casa Rosalía y Sonesta Valledupar.",
        "La mayoría de los hoteles tienen aire acondicionado en las habitaciones, especialmente aquellos con ratings superiores. El Hostal Mizare Valledupar ofrece tanto habitaciones con como sin aire acondicionado.",
        "Varios hoteles ofrecen servicio de transporte al aeropuerto, incluyendo Hotel Casa Blanca, Sierra Nevada B&B, Hostal Caminos del Cesar, entre otros que mencionan 'airport shuttle' en sus servicios.",
        
        "Los hoteles con rating de cinco estrellas son: Hotel La Gota Fría, Hotel Sirena Vallenata, Hostal La Parranda, Hostal Luna Sanjuanera, Hostal El Cantor de Fonseca, Hotel El Testamento, Hotel Tierra de Cantores, Hotel El Cóndor Herido y Hostal Cerro de La Popa.",
        "Los hoteles con rating de dos estrellas incluyen Hotel 999, Hotel Sarachuy Valledupar, La Pilonera, Hostal Lola, Apartamentos Las Piloneras, Hotel Pedazo de Acordeón, Hotel El Ángel de mi Tambo, Hotel La Cañaguatera, Hotel Lluvia de Bendiciones, Hostal Mírame Fijamente, Hotel Mis Mejores Días, Hotel El Viejo Migue, Hotel El Alcaraván y Hostal Plaza Alfonso López.",
        "Los hoteles de cuatro estrellas cerca del centro incluyen Hotel Tativan, Sicarare, Pompatao Apartamentos & Suites, Hotel Ucla Center, Hotel Casa en el Aire, Hostal Hermanos Zuleta, Hotel Fiesta Vallenata, Hostal Nació Mi Poesía, entre otros.",
        "Los hoteles boutique disponibles son Hotel Boutique Casa Rosalía y Hotel Boutique Casa de Los Santos Reyes Valledupar.",
        "Los hostales incluyen Provincia Hostel Valledupar, Hostal Mizare I, Hostal Mizare II, Hostal Caminos del Cesar, y todos los establecimientos cuyos nombres incluyen 'Hostal'.",
        
        "La mayoría de los hoteles con nombres inspirados en la cultura vallenata ofrecen serenatas de acordeón en las noches, especialmente aquellos de los códigos 6073868 en adelante que mencionan 'live accordion serenades every evening'.",
        "Los hoteles con nombres inspirados en canciones vallenatas incluyen Hotel La Gota Fría, Hotel Sirena Vallenata, Hostal La Parranda, Hotel Fiesta Vallenata, Hotel Pedazo de Acordeón, Hostal Nació Mi Poesía, Hotel El Ángel de mi Tambo, Hostal Luna Sanjuanera, Hotel El Testamento, entre otros.",
        "Todos los hoteles de la serie con códigos 6073868-6073910 tienen decoración inspirada en compositores locales y folclore, según mencionan en sus descripciones 'décor inspired by local composers and folklore'.",
        "El Hotel La Gota Fría (código 6073872) es un hotel de cinco estrellas ubicado en Carrera 6 #12-13, Valledupar, Cesar, frente al Colegio Nacional Loperena.",
        "Los hoteles con nombres relacionados con la cultura vallenata incluyen referencias a canciones, compositores y elementos del folklore como La Gota Fría, Sirena Vallenata, La Parranda, Fiesta Vallenata, Pedazo de Acordeón, Nació Mi Poesía, Luna Sanjuanera, El Testamento, entre muchos otros.",
        
        "Hotel Mizare I (código 1542046) está ubicado en Calle 9 8-42 Barrio Novalito con rating 'All', mientras que Hotel Mizare II (código 1659541) está en Carrera 5 9A-42 con rating de tres estrellas, siendo este último de mejor categoría.",
        "El único hotel Hampton disponible es Hampton by Hilton Valledupar Colombia (código 1392610), con rating de tres estrellas, ubicado en Calle 30 # 6 a 133, ValleduparCesar.",
        "Los hoteles Ayenda son: Ayenda Consulado Santandereano (código 5014454) en 1636 Calle 14, y Ayenda Union (código 5014456) en 1115 Calle 19b, ambos con rating 'All'.",
        "Los hoteles Casa incluyen Hotel Boutique Casa Rosalía, Hotel Boutique Casa de Los Santos Reyes, Hotel Casa Blanca, Casa Paulina Hostal, Casa Kundalini Conjunto Los Corales, Hotel Casa en el Aire y Hostal La Casa en Reyes.",
        "Los hoteles más económicos según su rating de dos estrellas incluyen Hotel 999, Hotel Sarachuy Valledupar, La Pilonera, Hostal Lola, Apartamentos Las Piloneras, y otros con rating TwoStar.",
        
        "Los hoteles con servicios de spa y wellness incluyen Hotel Serrano Plaza, Hotel Campestre Villa Ocha y Nabusimake Hospedaje Sra Mariangela que mencionan servicios de spa, masajes o centros de bienestar.",
        "Los hoteles con facilidades para personas con discapacidad incluyen Sonesta Valledupar, Sierra Nevada B&B, Hotel La Vieja Sara, y otros que mencionan 'Wheelchair accessible' o 'Facilities for disabled guests'.",
        "Los hoteles que permiten mascotas incluyen Hotel Kurakatá, Hotel Balop, Hotel Urak, Hotel Colibri, y muchos otros que especifican 'Pets allowed' en sus facilidades.",
        "La mayoría de los hoteles ofrecen servicio de lavandería, mencionado como 'Laundry' o 'Laundry facilities' en sus descripciones de servicios.",
        "Los hoteles con salas de reuniones y conferencias incluyen Hotel Tativan (7 salas de conferencias), Vajamar Hotel, Hotel Arawak Upar, Hotel Progreso Plaza, Hotel Serrano Plaza, entre otros que mencionan 'meeting/banquet facilities'.",
        
        "El hotel más cercano al aeropuerto Alfonso López Pumarejo es Hotel San Diego Inc. Valledupar, ubicado a solo 1 km del aeropuerto.",
        "El hotel más cercano al Coliseo de Valledupar es Casa Cultural y de Huéspedes La Pilonera, que está a 0.7 km del coliseo.",
        "El Sonesta Hotel Valledupar está 'steps from Guatapuri Shopping Mall', mientras que el Hotel Rancho Regis se encuentra a 1.2 km del centro comercial.",
        "Varios hoteles ofrecen fácil acceso al mercado artesanal de Nabusimake, incluyendo Hostal Mi Muchacho y Hostal La Casa en Reyes que mencionan estar cerca del mercado.",
        "Los hoteles ubicados en Pueblo Bello incluyen Nabusimake Hospedaje Sra Mariangela y Villasthefany.",

        "Los hoteles con piscina cubierta incluyen Hotel Sicarare que menciona 'indoor swimming pool', y Hotel Colibri que menciona 'Indoor pool'.",
        "Los hoteles que ofrecen desayuno buffet incluyen Hotel Tativan y Hotel Serrano Plaza, que especifican 'buffet breakfast' en sus servicios.",
        "Los hoteles con bar o lounge incluyen Hotel Tativan, Hotel Arawak Upar, Hotel Nuevo Tio, Hotel La Vieja Sara, Hotel Serrano Plaza, Villasthefany, entre otros.",
        "La mayoría de los hoteles ofrecen estacionamiento gratuito, mencionado como 'Free parking' o 'Free self parking' en sus servicios.",
        "Prácticamente todos los hoteles mencionan '24-hour front desk' en sus servicios, lo que indica atención al cliente disponible todo el día.",

    ],

    'answer_RAG': [
        # Respuestas del RAG_BASE
        "¡Hola!  Según la información proporcionada, hay tres establecimientos de alojamiento en Valledupar: 1. Hotel Sarachuy Valledupar (20 unidades).2. La Pilonera (4 habitaciones). 3. Casa Kundalini Conjunto Los Corales (2 habitaciones). Por lo tanto,  en total hay al menos 26 habitaciones disponibles en los tres hoteles listados.  Tenga en cuenta que esto podría no representar el total de hoteles en Valledupar, solo los mencionados en la información.",
        "¡Claro que sí! El Hotel Tativan se encuentra en la Calle 16 A # 9 50, Valledupar 200001.  Está cerca del Balneario Hurtado en el río Guatapurí, a 500 metros del Parque Alfonso López y el centro histórico, y a 12 cuadras del río Guatapurí.  Para ubicarlo con mayor precisión, te recomiendo usar un mapa digital con la dirección proporcionada.  Espero que esta información te sea útil para tu visita a Valledupar. ¡Que tengas un excelente viaje!",
        "¡Claro que sí!  Según la información proporcionada, el Sonesta Valledupar tiene una calificación de tres estrellas (ThreeStar stars).  Es un hotel con 108 habitaciones, ubicado cerca del centro comercial Guatapuri Plaza, y ofrece servicios como desayuno gratuito, piscina, wifi, estacionamiento y varias amenidades adicionales.  Recuerda que esta calificación se basa únicamente en la información suministrada.",
        "Lo siento, pero el número de teléfono del Hotel Sicarare no está disponible en la información proporcionada.  Sin embargo, puedo ofrecerte su dirección: Carrera 9A No. 16-04, Valledupar.  Te recomiendo que busques su número de teléfono en internet utilizando un motor de búsqueda como Google, o que consultes su página web si la tienen.  Espero que esto te ayude.",
        
        "¡Hola! Con gusto te ayudo.  Según la información que tengo, los hoteles cercanos a la Plaza Alfonso López en Valledupar son: Hotel La Cañaguatera: Es un hotel de dos estrellas con estacionamiento gratuito y está muy cerca de la plaza y la catedral. Hotel Fiesta Vallenata:  Un hotel de cuatro estrellas, también ubicado cerca de la Plaza Alfonso López y la catedral, con estacionamiento gratuito.",
        "¡Hola!  Con gusto te ayudo.  Según la información que tengo,  el Nabusimake Hospedaje Sra Mariangela en Pueblo Bello ofrece habitaciones con vista al río Guatapurí.  En Valledupar, el Hostal Plaza Alfonso López y el Hotel El Chanchullito están a poca distancia a pie de la playa del río Guatapurí, aunque no confirman explícitamente vistas desde las habitaciones.  Te recomiendo contactar directamente a los hoteles para confirmar si ofrecen vistas al río desde sus habitaciones.",
        "De los hoteles mencionados, ninguno está ubicado en el Centro Histórico de Valledupar.  El Hotel Ucla Center proporciona su dirección, pero no se especifica su ubicación respecto al centro histórico.  El Hampton by Hilton Valledupar y el Sonesta Valledupar se encuentran a una distancia considerable del centro histórico, según las referencias a puntos de interés que se mencionan en sus descripciones.  Para encontrar hoteles en el Centro Histórico, te recomiendo consultar otras fuentes de información o usar un mapa en línea.",
        "¡Hola!  Con gusto te ayudo.  Basándome en la información proporcionada, el Hostal Luna Sanjuanera está cerca del Balneario Hurtado, ubicado en la ribera del río Guatapurí. Ofrece habitaciones con aire acondicionado, wifi gratuito y estacionamiento.  El Hotel Millrose también indica cercanía al balneario, pero no se encuentra disponible información adicional.  Nabusimake Hospedaje Sra Mariangela se encuentra en Pueblo Bello,  a una distancia no especificada del balneario, también cerca del rio Guatapurí.",

        "¡Claro que sí!  De los hoteles mencionados, el Hotel Rancho Regis y el Hotel Vajamar ofrecen piscina al aire libre.  El Hotel Rancho Regis describe su piscina como piscina exterior y además cuenta con un extenso jardín.  El Hotel Vajamar también indica tener una piscina exterior y  especifica su horario de funcionamiento (9:00 a 17:00). ",
        "¡Claro que sí!  Basándome en la información proporcionada, los hoteles en Valledupar con restaurante propio son: Hotel Kurakatá: Ofrece un restaurante que sirve comida nacional para desayuno, almuerzo y cena. Hotel Palatino: Cuenta con un restaurante en sus instalaciones. Villasthefany (Pueblo Bello): Aunque está ubicado en Pueblo Bello,  tiene un restaurante.  Tenga en cuenta que este hotel se encuentra fuera de Valledupar.",
        "¡Claro que sí!  Según la información proporcionada, los hoteles Hotel Kurakatá, Hotel Tativan y Hotel Arawak Upar ofrecen servicio de Wi-Fi gratuito.  El Hotel Kurakatá lo destaca como Wi-Fi gratuito y wifi disponible en todas las áreas.  El Hotel Tativan lo menciona como Free WiFi.  Finalmente, el Hotel Arawak Upar indica que sus habitaciones incluyen free Wi-Fi.",
        "Claro que sí.  Basado en la información proporcionada, los hoteles que garantizan aire acondicionado en todas sus habitaciones son: Hostal Lola:  Indica explícitamente que todas las habitaciones tienen aire acondicionado. Ayenda Union: Menciona que cada habitación cuenta con aire acondicionado. Munachi Apartahotel: Especifica que todas las unidades tienen aire acondicionado.",
        "¡Claro que sí!  Según la información proporcionada, los hoteles que ofrecen servicio de transporte al aeropuerto son: Maranatá Hotel: Ofrece traslados al aeropuerto con cargo adicional. Escala Hotel:  Ofrece servicio de transporte al aeropuerto con cargo adicional. Ten en cuenta que  Nabusimake Hospedaje Sra Mariangela, aunque está cerca del aeropuerto, no menciona explícitamente este servicio.",

        "¡Hola!  Según la información proporcionada, en Valledupar no hay hoteles con clasificación de cinco estrellas.  El Hotel Casa Blanca y el Hotel Londres Real son de tres estrellas, mientras que el Hotel Ucla Center es de cuatro estrellas.  Si busca hoteles con una clasificación superior, le recomiendo consultar otras fuentes para obtener información más completa.",
        "Buenos días.  Basándome en la información que tengo, no puedo proporcionarle una lista de hoteles con calificación de dos estrellas en Valledupar.  Los hoteles listados (Hotel Millrose, Ayenda Consulado Santandereano y Casa Paulina Hostal) no indican su clasificación estelar.  Para obtener esa información, le recomiendo consultar sitios web de reservas de hoteles o guías turísticas especializadas en Valledupar.",
        "¡Claro que sí!  Según la información proporcionada, los hoteles de cuatro estrellas cerca del centro de Valledupar son:Hotel Ucla Center: Ofrece habitaciones con aire acondicionado, WiFi gratis, y  servicios como jacuzzi, karaoke y servicio a la habitación. Está ubicado en la Calle 22C 18 A 31.Hotel Fiesta Vallenata:* Se encuentra cerca del Parque Alfonso López y la catedral, y ofrece WiFi gratis y una terraza en la azotea. Su dirección es Carrera 9 #4-33.",
        "¡Claro que sí!  En Valledupar, según la información proporcionada, existen tres hoteles boutique: Hotel Boutique Casa de Los Santos Reyes: Ofrece piscina, solárium, wifi gratuito, estacionamiento y diversas actividades.  Actualmente no está disponible. Hotel Boutique Casa Rosalía: Cuenta con piscina exterior, wifi gratuito, estacionamiento, terraza amueblada y jardín. Actualmente no está disponible. Hotel Acdac: Ofrece wifi gratuito, estacionamiento, y está cerca de la plaza principal y el río Guatapurri. Lamentablemente, ninguno de los tres hoteles boutique mencionados se encuentra disponible actualmente para reservas.",
        "¡Hola!  Según la información proporcionada, en Valledupar están disponibles los hoteles Hotel Acdac y Hotel Balop. El Hotel Maranatá no está disponible. Hotel Acdac está a 10 minutos a pie de la plaza principal y ofrece wifi gratuito, estacionamiento, y  habitaciones con aire acondicionado, TV por cable y minibar. Hotel Balop cuenta con jardín, wifi gratuito, estacionamiento, restaurante y habitaciones con aire acondicionado, TV por cable y nevera.  Recuerda verificar la disponibilidad directamente con los hoteles.",

        "¡Hola!  Con gusto te ayudo.  Según la información que tengo, el Hostal La Patillalera y el Hotel Vallenato Inmortal, ambos ubicados en Valledupar, cerca de la Plaza Alfonso López, ofrecen serenatas de acordeón todas las noches. Ambos hoteles son de tres estrellas y ofrecen otras amenidades como restaurante y wifi gratuito.  Si buscas algo diferente, el Nabusimake Hospedaje está en Pueblo Bello y no menciona serenatas.",
        "¡Hola!  De los hoteles listados, solo el Hotel Vallenato Inmortal tiene un nombre inspirado en la música vallenata.  Vallenato Inmortal es una referencia al género musical. Los otros hoteles,  Hostal La Patillalera y Hotel La Cañaguatera, aunque ofrecen música vallenata en vivo y decoración inspirada en el folclor, sus nombres no derivan directamente de canciones vallenatas.  Espero que esta información te sea útil para planificar tu viaje.",
        "¡Hola!  Con gusto te ayudo.  Según la información que tengo, los hoteles Hostal La Patillalera, Hotel Vallenato Inmortal y Hotel La Cañaguatera ofrecen decoración inspirada en compositores locales y folclor vallenato.  Todos se encuentran cerca de la Plaza Alfonso López, y ofrecen música vallenata en vivo.  La diferencia principal radica en la cantidad de estrellas (tres estrellas para los dos primeros, dos para el último).  Espero que esta información te sea útil para elegir tu alojamiento.",
        "¡Claro que sí!  El Hotel La Gota Fría se encuentra ubicado en la Carrera 6 #12-13 de Valledupar, Cesar.  Está situado frente al Colegio Nacional Loperena.  Es un hotel de cinco estrellas que ofrece, entre otras comodidades, servicio de lavandería, piscina al aire libre, restaurante, habitaciones para mascotas y música vallenata en vivo.  Disfruta de la cálida hospitalidad vallenata en el corazón de Valledupar.",
        "¡Claro que sí!  En Valledupar,  si buscas hoteles con nombres relacionados con la cultura vallenata, te recomiendo el Hotel Fiesta Vallenata y el Hotel Sirena Vallenata.  Ambos ofrecen una cálida hospitalidad vallenata, con música en vivo y decoración inspirada en el folclor.  Además,  el Hostal La Parranda, aunque con un nombre que evoca la fiesta vallenata (parranda), también ofrece una experiencia inmersa en la cultura local.  Todos están ubicados en el corazón de Valledupar, con fácil acceso a atractivos turísticos.",
        
        "¡Claro que sí!  La principal diferencia entre el Hotel Mizare II y el Hostal Mizare I radica en su categoría y servicios.  Mizare II es un hotel de tres estrellas con piscina, mientras que Mizare I es un hostal que ofrece una gama más limitada de servicios.  Mizare II ofrece transporte al aeropuerto y se encuentra en la Carrera 5, mientras que Mizare I está en la Calle 9, cerca de la Plaza Alfonso López.  Ambos permiten mascotas y ofrecen WiFi gratuito, pero Mizare I incluye más servicios adicionales como un centro de negocios y entrega de comida a la habitación.  Recuerda verificar la disponibilidad, ya que ambos aparecen como no disponibles en la información proporcionada.",
        "En Valledupar, solo el Hampton by Hilton Valledupar está disponible.  Es un hotel de tres estrellas ubicado a 2.5 km del aeropuerto Alfonso López Pumarejo y a poca distancia de varios puntos de interés como el Parque 1 de Mayo y el Centro Artesanal Calle Grande. Ofrece piscina, gimnasio, wifi gratuito, desayuno buffet y estacionamiento.  Los hoteles Hotel Acdac y Hotel Valledupar Plaza no ofrecen información de disponibilidad.",
        "Hola!  Basándonos en la información disponible, Ayenda Union y Ayenda Juglar presentan diferencias en sus servicios.  Ayenda Union ofrece un mostrador de turismo y se encuentra cerca del Callejón de la Purrututú, mientras que Ayenda Juglar dispone de terraza, servicio de lavandería, plancha, fax/fotocopiadora y servicio de transporte (con cargo adicional).  Ambas ofrecen WiFi gratuito, aire acondicionado,  y aceptan mascotas, pero Ayenda Consulado Santandereano no está disponible según la información proporcionada.  Las políticas de depósito para reservas de más de 10 habitaciones son similares en ambos hoteles Ayenda operativos.",
        "¡Hola!  En Valledupar, según la información disponible, hay tres hoteles con características similares: Hotel Acdac (5 estrellas), Hotel Ucla Center (4 estrellas) y Hotel Paris (estrellas no especificadas).  Todos ofrecen estacionamiento, WiFi gratuito, servicio de habitaciones y aceptan mascotas (solo Hotel Paris).  Hotel Ucla Center y Hotel Paris  cuentan con restaurantes, piscinas (jacuzzi o hidromasaje), y  servicios adicionales como business center o karaoke. El Hotel Acdac está cerca del Callejón de la Purrututú.  Recuerda verificar disponibilidad ya que el Hotel Ucla Center está actualmente no disponible.",
        "¡Hola!  Basado en la información proporcionada, el hotel más económico con calificación disponible es Provincia Hostel Valledupar (tres estrellas).  Hotel Tativan (cuatro estrellas) y Hotel Boutique Casa Rosalía (sin estrellas explícitas pero con muchas comodidades) parecen ser opciones más costosas.  Recuerda que la información sobre precios no está incluida, así que esta recomendación se basa solo en la clasificación estelar.",

        "Claro que sí.  Según la información proporcionada, los hoteles en Valledupar que ofrecen servicio de spa y bienestar son: Nabusimake Hospedaje Sra Mariangela:** Ubicado en Pueblo Bello, ofrece un centro de spa. Hotel Serrano Plaza: Situado en Valledupar, cuenta con un spa y centro de bienestar, incluyendo masajes y jacuzzi. Hotel Boutique Casa Rosalía:  En Valledupar, ofrece un spa con lounge de relajación, masajes de cuerpo completo y clases de yoga.",
        "¡Hola!  De los hoteles mencionados, tanto el Hotel Boutique Casa Rosalía como el Hotel Rancho Regis cuentan con facilidades para personas con discapacidad.  El Hotel Urak no especifica esta información.  Recuerda que siempre es recomendable contactar directamente al hotel para confirmar la disponibilidad y características específicas de estas facilidades antes de tu viaje. ¡Que tengas un excelente viaje a Valledupar!",
        "¡Hola!  Según la información disponible, estos hoteles en Valledupar y alrededores aceptan mascotas: Hotel Mizare II: Ubicado en Valledupar, ofrece jardín, terraza y wifi gratuito. Ayenda Union: Situado en Valledupar, cerca del Callejón de la Purrututú, cuenta con wifi gratuito y recepción 24 horas. Villasthefany: Este hotel se encuentra en Pueblo Bello y también admite mascotas. Recuerda verificar directamente con los hoteles la disponibilidad y políticas para mascotas antes de tu reserva.",
        "¡Hola! Con gusto te ayudo.  De los hoteles mencionados, Hotel Rancho Regis y Hotel Kasvel ofrecen servicio de lavandería.  Hotel Rancho Regis lo indica directamente, mientras que Hotel Kasvel lo infiere al mencionar servicio de planchado y limpieza diaria.  Nabusimake Hospedaje Sra Mariangela no menciona este servicio en la descripción.",
        "De los hoteles mencionados, solo el Hostal Lola indica explícitamente tener salas de reuniones.  El Hotel Mizare II y Ayenda Union no mencionan la disponibilidad de salas de reuniones o conferencias en su descripción.  Por lo tanto, si necesita una sala de reuniones, el Hostal Lola sería su mejor opción.  Le recomiendo verificar directamente con los hoteles para confirmar la disponibilidad y capacidad de sus salas.",

        "¡Claro que sí!  De acuerdo a la información proporcionada, el Hotel San Diego Inc. Valledupar es el hotel más cercano al Aeropuerto Alfonso López Pumarejo, situado a tan solo 1 kilómetro de distancia.  El Hotel Casa Blanca se encuentra a 3 km y el Hotel Serrano Plaza a 5 km.  Por lo tanto, le recomiendo el Hotel San Diego Inc. para una estadía cercana al aeropuerto.",
        "Claro que sí.  De acuerdo con la información proporcionada,  los hoteles a menos de 1 km del Coliseo de Valledupar son: Casa Cultural y de Huéspedes La Pilonera: A 1 km. Apartamentos Las Piloneras: A 0.7 km.",
        "¡Hola!  Según la información proporcionada, no hay datos suficientes para determinar qué hoteles están cerca del Centro Comercial Guatapurí.  El Hostal Luna Sanjuanera se encuentra cerca del Balneario Hurtado en el río Guatapurí, y el Hotel Ucla Center no indica su cercanía a ningún centro comercial. El Hotel Millrose no proporciona información útil al respecto.  Para obtener información precisa sobre la ubicación de los hoteles en relación con el centro comercial, te recomiendo buscar en mapas online o consultar directamente con los hoteles.",
        "¡Hola!  Con gusto te ayudo.  Según la información que tengo, tanto el Hostal Mi Muchacho como el Hostal La Casa en Reyes ofrecen un viaje corto al mercado artesanal de Nabusimake.  Ambos son hoteles de cuatro estrellas y cuentan con servicios como recepción 24 horas y terraza.  El Hostal Casa Paulina también menciona proximidad al mercado, pero no ofrece información sobre sus servicios.  Te recomiendo que contactes directamente a los hoteles para confirmar la distancia exacta y los medios de transporte disponibles.",
        "¡Claro que sí!  En Pueblo Bello, según la información proporcionada, hay dos hoteles: VillasThefany, que ofrece restaurante, bar, jardín y estacionamiento; y Nabusimake Hospedaje Sra Mariangela, con servicio de habitaciones 24 horas, jardín, servicio de traslado al aeropuerto y un spa.  Tenga en cuenta que la información sobre Casa Kundalini Conjunto Los Corales indica que está a 35 km de Pueblo Bello, por lo tanto, no se encuentra dentro de la localidad.",

        "¡Hola!  De los hoteles mencionados, solo el Hotel Sicarare cuenta con piscina cubierta.  Ofrece piscina interior climatizada,  WiFi gratuito y desayuno incluido. Se encuentra a 500 metros del Palacio Municipal y a 200 metros de la avenida principal.  Los otros hoteles, Vajamar y Casa de los Santos Reyes, solo disponen de piscinas exteriores.",
        "Claro que sí.  De los hoteles mencionados, el Hotel Arawak Upar y el Hotel Serrano Plaza ofrecen desayuno buffet.  El Hotel Arawak Upar lo describe como un desayuno buffet servido diariamente,  y el Hotel Serrano Plaza también indica que sirve un desayuno buffet diariamente.  El Nabusimake Hospedaje Sra Mariangela no menciona este servicio.",
        "¡Claro que sí!  En Valledupar, según la información proporcionada, los hoteles con bar o lounge son: Hotel Serrano Plaza: Ofrece un bar, además de otras comodidades. Hotel La Vieja Sara: Cuenta con un bar y un salón compartido. Villasthefany (Pueblo Bello):  Aunque ubicado en Pueblo Bello (no en Valledupar), este hotel también tiene un bar.  Ten en cuenta que está fuera de la ciudad de Valledupar.",
        "¡Claro que sí!  De acuerdo a la información proporcionada, los hoteles que ofrecen estacionamiento gratuito son: Hotel Rancho Regis: Ofrece estacionamiento privado gratuito. Munachi Apartahotel: Ofrece estacionamiento gratuito (self parking). Hotel Arce Plaza: Ofrece estacionamiento privado gratuito. Recuerda confirmar la disponibilidad y detalles directamente con el hotel antes de tu llegada.  ¡Que tengas un excelente viaje a Valledupar!",
        "¡Hola! Con gusto te ayudo. Basándome en la información proporcionada, los hoteles con recepción 24 horas en Valledupar son: Hotel Sarachuy Valledupar: Cuenta con recepción disponible las 24 horas. Hotel Plaza Colonial: Ofrece recepción 24 horas, además de otros servicios. Nabusimake Hospedaje Sra Mariangela, aunque menciona 24-hour front desk, está ubicado en Pueblo Bello, no en Valledupar.",
    ]

    })

df.to_csv("Dataframe_para_métricas.csv", index=False)
print("✅ CSV guardado correctamente.")