# Modelos Generativos de Imagenes basados en Difusion

Prueba de concepto para la asignatura de IA Generativa de la Universidad Autonoma de Occidente (UAO).

Este repositorio contiene un notebook unico, `Difusion_Generacion_Imagenes_POC.ipynb`, que implementa y documenta modelos generativos de imagenes basados en difusion. El trabajo combina entrenamiento desde cero, evaluacion cuantitativa y una demostracion guiada del flujo interno de Stable Diffusion.

## Contenido

- DDPM desde cero sobre MNIST.
- DDPM adaptado a CIFAR-10.
- Modelo de difusion con bloques residuales y programacion angular inspirado en *Deep Learning with Python*.
- Generacion sobre Oxford Flowers.
- Calculo de metricas de calidad, incluyendo FID.
- Evaluacion auxiliar con clasificador para MNIST.
- Demostracion paso a paso de Stable Diffusion v1.5:
  - codificador de texto CLIP;
  - ruido inicial en espacio latente;
  - bucle de difusion con guia libre de clasificador;
  - decodificacion VAE desde latentes a pixeles;
  - comparacion entre representacion latente e imagen final.

## Archivo principal

| Archivo | Descripcion |
| --- | --- |
| `Difusion_Generacion_Imagenes_POC.ipynb` | Notebook principal con teoria, codigo, entrenamiento, visualizaciones, metricas y discusion de resultados. |

## Requisitos

El notebook esta pensado para ejecutarse en un entorno con GPU, preferiblemente Google Colab o un runtime local con TensorFlow/Keras.

Dependencias principales:

- Python 3.10+
- TensorFlow
- Keras / KerasCV
- NumPy
- Matplotlib
- TensorFlow Datasets
- scikit-learn
- SciPy

Algunas secciones, especialmente Stable Diffusion v1.5 y el entrenamiento sobre CIFAR-10 u Oxford Flowers, se benefician mucho de una GPU.

## Ejecucion

1. Abrir `Difusion_Generacion_Imagenes_POC.ipynb` en Jupyter, Google Colab o VS Code.
2. Ejecutar las celdas en orden.
3. Para la seccion de Stable Diffusion, reiniciar el entorno cuando el notebook lo indique.
4. Ajustar los valores de `EPOCHS_MNIST`, `EPOCHS_CIFAR` y `EPOCHS_FLOWERS` segun la capacidad de computo disponible.

## Estructura del trabajo

El notebook esta dividido en bloques progresivos:

1. Marco teorico de DDPM y proceso directo de adicion de ruido.
2. Arquitectura U-Net condicionada por tiempo.
3. Entrenamiento para predecir ruido.
4. Muestreo inverso para generar imagenes.
5. Evaluacion con FID y metricas auxiliares.
6. Extension a datasets mas complejos.
7. Inspeccion didactica de Stable Diffusion en espacio latente.

