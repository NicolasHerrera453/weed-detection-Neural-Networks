# Weed Detection using Neural Networks (Detección de Malezas)

Proyecto académico desarrollado para la asignatura de **Reconocimiento de Patrones** en la **Universidad Militar Nueva Granada (UMNG)**. Consiste en el diseño, extracción de características avanzadas y entrenamiento desde cero de una red neuronal artificial (Perceptrón Multicapa) para la detección automatizada de malezas en cultivos a partir de imágenes.

---

## 🚀 Descripción del Proyecto
El flujo de trabajo abarca desde el procesamiento digital de imágenes agrícolas hasta el modelado matemático y entrenamiento del algoritmo de aprendizaje automático:

1. **Preprocesamiento y Visión Artificial:** Carga, manipulación y conversión de imágenes utilizando `OpenCV` y `PIL`, aplicando técnicas de binarización y normalización de escalas con `MinMaxScaler` (`scikit-learn`).
2. **Extracción de Características (Features):** Uso de descriptores avanzados de textura y forma para enriquecer los datos de entrada:
   - **HOG (Histogram of Oriented Gradients)** (`skimage.feature`).
   - **LBP (Local Binary Patterns)** (`skimage.feature`).
3. **Estructuración de Datos:** Generación de una matriz $X$ con los descriptores extraídos y un vector $Y$ de etiquetas binarias para el entrenamiento supervisado.
4. **Red Neuronal desde Cero (NumPy):** Implementación de una red neuronal feedforward con una capa oculta y función de activación sigmoidal, entrenada mediante descenso por gradiente estocástico en lotes (*mini-batches*).

---

## 🛠️ Tecnologías y Librerías Utilizadas
* **Lenguaje:** Python
* **Procesamiento de Imágenes y Datos:** `OpenCV` (`cv2`), `Pillow` (`PIL`), `NumPy`, `Matplotlib`, `OS`
* **Visión por Computadora y Extracción de Características:** `scikit-image` (`hog`, `local_binary_pattern`)
* **Preprocesamiento y Machine Learning:** `scikit-learn` (`MinMaxScaler`)

---

## 🧠 Arquitectura y Entrenamiento de la Red Neuronal
La red fue construida puramente con **NumPy**, sin frameworks pesados de alto nivel, lo que demuestra un profundo entendimiento matemático del algoritmo:
* **Función de Activación:** Sigmoidal (`sigmoid`).
* **Arquitectura:** Capa de entrada dinámica (`X.shape[1]`), 1 capa oculta (4 neuronas) y 1 capa de salida.
* **Hiperparámetros:** 
  * Tasa de aprendizaje (*learning rate*): `0.1`
  * Épocas: `10000`
  * Tamaño de lote (*batch size*): `32`

---

## 📁 Estructura del Repositorio
```text
weed-detection-nn/
│
├── data/               # Directorio de imágenes y dataset
├── notebooks/          # Jupyter Notebooks con el preprocesamiento y entrenamiento
├── src/                # Scripts de extracción de características y modelo
└── README.md           # Documentación del proyecto
