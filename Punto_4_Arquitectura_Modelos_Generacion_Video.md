# Punto 4 - Arquitectura de los modelos actuales de generacion de video

## 1. Introduccion

Los modelos modernos de generacion de video son una extension natural de los modelos generativos de imagenes estudiados en el curso, pero con una dificultad adicional: no basta con producir fotogramas realistas de forma aislada, tambien deben mantener continuidad temporal, coherencia de objetos, movimiento plausible, iluminacion estable y correspondencia entre texto, escena y acciones. En una imagen, el modelo solo necesita explicar una matriz espacial de pixeles. En video, debe explicar un volumen espacio-temporal: alto, ancho, canales y tiempo.

La base conceptual sigue siendo la misma que en los modelos de difusion revisados en `Context`: se parte de datos limpios, se les agrega ruido de forma gradual durante un proceso directo conocido y luego se entrena una red neuronal para aprender el proceso inverso de eliminacion de ruido. En imagenes, esa red suele ser una U-Net o un Transformer de difusion. En video, la arquitectura agrega mecanismos temporales para que el modelo no trate cada fotograma como una imagen independiente.

En la actualidad conviven varias familias: modelos de difusion de video en pixeles, modelos de difusion latente de video, modelos basados en Transformers de difusion, modelos autoregresivos o hibridos y sistemas grandes de texto a video como Sora, Veo, Runway Gen-3, Stable Video Diffusion, Pika, Luma Dream Machine, CogVideoX y Wan. Aunque cada sistema tiene detalles propietarios, la mayoria comparte una tuberia arquitectonica comun: codificador de condicion, compresor visual o latente, red generativa espacio-temporal, scheduler de muestreo y decodificador final.

## 2. De imagen a video: que cambia arquitectonicamente

Un modelo de imagen aprende una distribucion sobre tensores con forma `(alto, ancho, canales)`. Un modelo de video aprende una distribucion sobre tensores con forma `(tiempo, alto, ancho, canales)`. Esta diferencia parece pequena, pero cambia toda la arquitectura:

- La memoria crece linealmente con el numero de fotogramas y cuadraticamente si se usa atencion global sobre todos los tokens espacio-temporales.
- El modelo debe representar movimiento, no solo apariencia.
- El condicionamiento textual debe alinearse con acciones distribuidas en el tiempo.
- El muestreo debe evitar parpadeos, saltos de identidad, cambios bruscos de camara y deformaciones acumuladas.
- La evaluacion requiere medir realismo visual y coherencia temporal.

Por eso, los modelos actuales no generan video como una lista de imagenes independientes. En cambio, procesan el video como un volumen continuo o como una secuencia de parches/tokens espacio-temporales.

## 3. Tuberia general de un modelo moderno de video

La arquitectura completa suele tener seis componentes.

### 3.1 Codificador de condicion

El usuario entrega un prompt, una imagen inicial, un video guia, una mascara, una pose, profundidad, audio u otra condicion. El sistema convierte esa condicion en embeddings numericos. Para texto se usan codificadores tipo CLIP, T5, UL2 u otros Transformers de lenguaje. Para imagen o video se usan encoders visuales. El resultado no es texto plano, sino una secuencia de vectores que luego se inyecta en la red generativa mediante cross-attention, modulacion adaptativa o concatenacion.

En un modelo texto-a-video, el prompt debe guiar tanto la apariencia como la accion. Por ejemplo, "un astronauta caminando bajo lluvia neon" contiene entidades, estilo, iluminacion y movimiento. La arquitectura necesita que esos elementos sobrevivan durante muchos pasos de denoising.

### 3.2 Compresor visual o autoencoder latente

Los modelos de difusion latente, como Stable Diffusion en imagenes, no operan directamente en pixeles. Primero comprimen la imagen en un espacio latente mas pequeno mediante un autoencoder variacional. En video ocurre algo parecido, pero el compresor puede ser 2D por fotograma o 3D/causal para preservar estructura temporal.

Si el video original tiene forma `(T, H, W, 3)`, el latente puede tener forma aproximada `(T', H/8, W/8, C)`. Esto reduce muchisimo el costo computacional. Sin embargo, introduce un reto: si el VAE no conserva bien los detalles temporales, el video final puede presentar parpadeo o inconsistencia entre fotogramas.

### 3.3 Proceso de ruido y scheduler

El proceso directo agrega ruido gaussiano a los latentes o pixeles. Durante entrenamiento se toma un video limpio, se elige un tiempo de difusion `t`, se agrega ruido y la red aprende a predecir el ruido, el dato limpio o una parametrizacion intermedia como `v-prediction`. En video, el ruido puede agregarse a todos los fotogramas y canales latentes a la vez.

El scheduler define como se recorre la trayectoria desde ruido puro hasta video limpio. Schedulers modernos como DDIM, Euler, DPM-Solver o variantes rectified-flow reducen pasos de muestreo y mejoran estabilidad. En video esto es importante porque cada paso es costoso.

### 3.4 Red generativa espacio-temporal

Este es el nucleo del modelo. Recibe latentes ruidosos, embeddings de tiempo de difusion y condiciones externas. Devuelve una prediccion que permite actualizar el latente hacia una version menos ruidosa. Las arquitecturas actuales se agrupan en dos grandes ramas:

**U-Net espacio-temporal.** Parte de la U-Net de imagenes y agrega capas temporales. La red reduce y aumenta resolucion espacial con skip connections, pero tambien mezcla informacion entre fotogramas usando convoluciones 3D, atencion temporal o bloques temporales intercalados. Stable Video Diffusion pertenece a esta familia de difusion latente de video.

**Transformer de difusion o DiT.** Convierte el video latente en tokens, normalmente parches espacio-temporales, y usa self-attention para modelar relaciones globales. Sora describe los videos e imagenes como colecciones de parches y entrena un Transformer de difusion sobre esos tokens. CogVideoX tambien usa un Transformer de difusion especializado para video. Esta familia escala muy bien con datos y computo, aunque requiere atencion eficiente y compresion cuidadosa.

### 3.5 Mecanismos de coherencia temporal

La coherencia temporal es la diferencia principal frente a imagen. Los modelos actuales usan varias tecnicas:

- Convoluciones 3D para capturar patrones locales en espacio y tiempo.
- Atencion temporal para que un fotograma consulte informacion de otros fotogramas.
- Atencion espacio-temporal factorada, separando atencion espacial y temporal para ahorrar memoria.
- Tokens de parches 3D, donde cada token representa una region durante varios fotogramas.
- Ventanas temporales o atencion local para videos largos.
- Condicionamiento por fotogramas clave, imagen inicial o trayectorias de camara.
- Entrenamiento con clips de video y augmentaciones que obligan a aprender movimiento.

Sin estos mecanismos, el modelo podria generar una secuencia visualmente atractiva pero inestable: caras que cambian, objetos que aparecen y desaparecen, fondos que vibran o movimientos fisicamente imposibles.

### 3.6 Decodificador y posprocesamiento

Cuando el muestreo termina, el latente se decodifica a pixeles. En modelos latentes, el VAE reconstruye los fotogramas. Luego puede haber etapas de super-resolucion, interpolacion de fotogramas, mejora de nitidez o upscaling. En sistemas comerciales, la tuberia completa puede incluir moderacion, seguridad, compresion de salida y filtros para reducir artefactos.

## 4. Familias de arquitecturas actuales

### 4.1 Video Diffusion Models en espacio de pixeles

Los primeros modelos de difusion de video extendieron directamente DDPM a tensores de video. La red de denoising procesa clips completos y predice ruido para cada fotograma. Estos modelos demostraron que la difusion podia producir video coherente, pero eran costosos porque operar en pixeles requiere muchisima memoria.

Arquitectonicamente, suelen combinar U-Net 2D con componentes temporales: convoluciones 3D, atencion temporal o bloques que alternan procesamiento espacial y temporal. La ventaja es que la red ve el video directamente. La desventaja es que escalar a alta resolucion y muchos fotogramas resulta caro.

### 4.2 Difusion latente de video

La difusion latente reduce el problema: en lugar de generar pixeles, genera latentes comprimidos. Es la idea central detras de Stable Diffusion para imagenes y de Stable Video Diffusion para video. El modelo aprende en un espacio mas pequeno, por ejemplo `H/8 x W/8`, y luego un decodificador reconstruye el video final.

Esta familia suele usar una U-Net condicionada. Para adaptarla de imagen a video se insertan capas temporales dentro de los bloques. Una estrategia comun es inicializar desde un modelo texto-a-imagen ya entrenado y luego entrenar componentes temporales con datos de video. Asi se reutiliza conocimiento visual de imagenes y se aprende movimiento con menor costo.

### 4.3 Transformers de difusion para video

Los modelos mas recientes tienden a representar el video como tokens. En lugar de una U-Net convolucional, usan un Transformer que recibe parches espacio-temporales ruidosos. Cada token puede representar una pequena region de varios fotogramas. El Transformer aprende relaciones largas: un objeto que aparece en el primer segundo debe conservar identidad y posicion relativa varios segundos despues.

Sora popularizo esta vision al describir los datos visuales como parches, de manera parecida a los tokens en modelos de lenguaje. Esto permite entrenar sobre videos e imagenes de diferentes duraciones, resoluciones y relaciones de aspecto. CogVideoX sigue una direccion similar con un Transformer de difusion para texto-a-video.

La ventaja del Transformer es la escalabilidad: al aumentar datos y parametros, mejora la capacidad de modelar escenas complejas. La desventaja es el costo de atencion. Por eso se usan compresores latentes, atencion eficiente, ventanas, factorizacion espacio-temporal y entrenamiento distribuido.

### 4.4 Modelos autoregresivos e hibridos

Otra familia genera video como una secuencia de tokens discretos o latentes, prediciendo el siguiente bloque a partir de los anteriores. Es similar a lenguaje: dado el pasado, se predice el futuro. Estos modelos pueden manejar videos largos, pero tienden a acumular errores. Muchos sistemas modernos combinan ideas autoregresivas con difusion: planifican a alto nivel y refinan con denoising, o generan por segmentos manteniendo memoria temporal.

### 4.5 Modelos condicionados por imagen, video o control

No todo modelo actual es texto-a-video puro. Muchos son imagen-a-video: reciben una imagen inicial y generan movimiento. Otros aceptan profundidad, pose humana, bordes, segmentacion, camara o video de referencia. Arquitectonicamente, esto agrega ramas de condicionamiento o adaptadores. La condicion puede entrar por concatenacion en los canales latentes, por cross-attention o por modulos tipo ControlNet/adapter.

Este enfoque mejora el control y reduce ambiguedad. Si se entrega una imagen inicial, el modelo no necesita inventar toda la apariencia; se concentra en animarla de forma coherente.

## 5. Componentes internos importantes

### 5.1 Embeddings de tiempo

Igual que en los modelos de imagen del curso, la red necesita saber en que paso de difusion esta. El tiempo `t` se codifica con senos/cosenos o embeddings aprendidos. Ese vector modula las capas de la red. En video, el modelo tambien puede necesitar posicion temporal del fotograma, distinta del tiempo de difusion. Son dos nociones diferentes: el tiempo de difusion mide nivel de ruido; el tiempo del video mide orden narrativo o movimiento.

### 5.2 Cross-attention texto-video

El prompt se inyecta mediante cross-attention: los tokens visuales consultan los tokens textuales. Asi, regiones del video pueden alinearse con palabras o frases. En modelos avanzados, esta atencion debe mantenerse consistente en el tiempo; por ejemplo, el token visual de un personaje debe seguir asociado con la misma descripcion durante varios fotogramas.

### 5.3 Classifier-free guidance

La guia libre de clasificador entrena el modelo con y sin condicion. En inferencia se combinan ambas predicciones para aumentar obediencia al prompt. En video, un guidance muy alto puede mejorar alineacion textual pero empeorar naturalidad temporal o crear artefactos. Por eso se ajusta con cuidado.

### 5.4 Positional encoding espacio-temporal

Los Transformers necesitan informacion de posicion. En video se codifica posicion espacial `(x, y)` y temporal `t_frame`. Se pueden usar embeddings absolutos, relativos, rotatorios o variantes adaptadas a video. Esto permite distinguir entre movimiento horizontal, cambio de escena y avance temporal.

### 5.5 Entrenamiento multi-resolucion y multi-duracion

Los modelos actuales se entrenan con clips de diferentes resoluciones, duraciones y relaciones de aspecto. Esto requiere arquitecturas flexibles. Los modelos basados en parches son especialmente utiles porque pueden procesar distintas formas como secuencias de tokens.

## 6. Ejemplo conceptual de flujo de inferencia

1. El usuario escribe un prompt: "un dron recorre una ciudad futurista al atardecer".
2. El codificador de texto produce embeddings.
3. Se inicializa un tensor de ruido en el espacio latente de video.
4. El scheduler selecciona el primer paso de denoising.
5. La red espacio-temporal predice ruido o velocidad condicionada por texto.
6. El latente se actualiza hacia una version menos ruidosa.
7. Se repiten decenas de pasos.
8. El VAE decodifica los latentes a fotogramas.
9. Una etapa opcional mejora resolucion, interpolacion o nitidez.
10. Se exporta el resultado como video.

## 7. Retos actuales

Aunque los resultados modernos son impresionantes, los modelos de video todavia enfrentan problemas:

- Mantener identidad de personajes y objetos durante clips largos.
- Representar fisica realista, manos, interacciones y causalidad.
- Generar texto legible dentro del video.
- Controlar camaras complejas sin deformaciones.
- Reducir costos de inferencia.
- Evitar sesgos, contenido inseguro o usos no autorizados.
- Evaluar calidad de forma objetiva, porque las metricas automaticas no capturan completamente coherencia narrativa.

## 8. Relacion con el notebook desarrollado

El notebook entregado acompana este documento con ejercicios en Colab. Primero muestra como un video se representa como tensor espacio-temporal. Luego simula el proceso de ruido sobre un video sintetico de un cuadrado en movimiento. Despues construye un mini-denoiser 3D para mostrar la diferencia entre procesar imagenes y procesar clips. Finalmente incluye celdas opcionales para cargar pipelines de `diffusers` y observar los componentes de un modelo moderno de video.

La idea no es entrenar un modelo competitivo, porque eso requiere enormes datasets y GPUs. El objetivo es que el codigo permita ver la arquitectura: latentes, ruido, scheduler, denoiser espacio-temporal, condicionamiento y decodificacion.

## 9. Conclusiones

Los modelos actuales de generacion de video son sistemas de difusion o flujo generativo que operan sobre representaciones espacio-temporales. Su avance principal frente a los modelos de imagen es la incorporacion explicita del tiempo: convoluciones 3D, atencion temporal, parches espacio-temporales, Transformers de difusion y condicionamiento persistente. La tendencia actual se mueve desde U-Nets latentes hacia Transformers de difusion escalables, capaces de tratar videos como secuencias de tokens visuales. Aun asi, la base conceptual sigue conectada con los modelos vistos en clase: agregar ruido, aprender a quitarlo y condicionar el proceso para generar contenido controlado.

## 10. Fuentes consultadas

- Material local del curso: `Context/Diffusion_Models_Gen_AI_2025_2S.pdf`.
- Material local del curso: diapositivas de Stable Diffusion incluidas en `Context`.
- Material local del libro: `Context/Chapter 17.txt` y `Context/chapter17_image_generation.ipynb`.
- Ho et al., "Denoising Diffusion Probabilistic Models", 2020. https://arxiv.org/abs/2006.11239
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", 2022. https://arxiv.org/abs/2112.10752
- Ho et al., "Video Diffusion Models", 2022. https://arxiv.org/abs/2204.03458
- Blattmann et al., "Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets", 2023. https://arxiv.org/abs/2311.15127
- OpenAI, "Video generation models as world simulators", 2024. https://openai.com/research/video-generation-models-as-world-simulators
- Yang et al., "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer", 2024. https://arxiv.org/abs/2408.06072
