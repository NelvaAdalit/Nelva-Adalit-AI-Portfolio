# 🤖 Portafolio de Inteligencia Artificial y Machine Learning
### Desarrollado por: **Nelva Adalit Mora Barrionuevo**

¡Bienvenido/a a mi portafolio de Inteligencia Artificial! Este repositorio contiene una colección organizada de cuadernos de Jupyter (`.ipynb`) donde implemento y explico diversos algoritmos de Machine Learning y Deep Learning, desde los fundamentos matemáticos hasta redes neuronales complejas en PyTorch.

La estructura del portafolio está dividida rigurosamente según los paradigmas de aprendizaje: **Supervisado**, **No Supervisado**, **Semi-supervisado** y **Por Refuerzo**.

---

## 📂 Estructura del Portafolio

```text
Nelva-Adalit-AI-Portfolio/
├── 1-aprendizaje-supervisado/
│   ├── regresion/
│   │   ├── 01_regresion_lineal_multiple.ipynb
│   │   └── 02_regresion_lineal_multivariable.ipynb
│   ├── clasificacion/
│   │   ├── 03_regresion_logistica_binaria.ipynb
│   │   ├── 04_clasificacion_multiclase_one_vs_all.ipynb
│   │   └── 05_clasificacion_imagenes_intel_scenes.ipynb
│   ├── redes-neuronales/
│   │   └── 06_redes_neuronales_pytorch_3_modelos.ipynb
│   └── deep-learning-nhanes/
│       └── 07_clasificacion_patologias_nhanes.ipynb
├── 2-aprendizaje-no-supervisado/
│   └── clustering-spotify/
│       └── 01_kmeans_spotify_clustering.ipynb
├── 3-aprendizaje-semi-supervisado/
│   └── clasificacion-frutas/
│       └── 01_clasificacion_frutas_semisupervisado.ipynb
├── 4-aprendizaje-por-refuerzo/
│   └── rompecabezas-deslizante/
│       ├── 01_aprendizaje_por_refuerzo_rompecabezas.ipynb
│       └── pesos_rompecabezas.pt
└── README.md
```

---

## 🛠️ Detalle de los Modelos y Notebooks

### 1. Aprendizaje Supervisado (Supervised Learning)
Modelos entrenados con datos etiquetados para realizar predicciones numéricas (regresión) o clasificar elementos.

*   **Regresión**
    *   [01_regresion_lineal_multiple.ipynb](./1-aprendizaje-supervisado/regresion/01_regresion_lineal_multiple.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NelvaAdalit/Nelva-Adalit-AI-Portfolio/blob/main/1-aprendizaje-supervisado/regresion/01_regresion_lineal_multiple.ipynb): Implementación de regresión lineal considerando múltiples variables independientes para predecir un valor continuo.
    *   [02_regresion_lineal_multivariable.ipynb](./1-aprendizaje-supervisado/regresion/02_regresion_lineal_multivariable.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NelvaAdalit/Nelva-Adalit-AI-Portfolio/blob/main/1-aprendizaje-supervisado/regresion/02_regresion_lineal_multivariable.ipynb): Análisis avanzado de regresión lineal con ajuste y optimización matemática multivariable.
*   **Clasificación**
    *   [03_regresion_logistica_binaria.ipynb](./1-aprendizaje-supervisado/clasificacion/03_regresion_logistica_binaria.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NelvaAdalit/Nelva-Adalit-AI-Portfolio/blob/main/1-aprendizaje-supervisado/clasificacion/03_regresion_logistica_binaria.ipynb): Modelo clásico para la clasificación binaria (dos clases) con análisis de la función sigmoide y entropía cruzada binaria.
    *   [04_clasificacion_multiclase_one_vs_all.ipynb](./1-aprendizaje-supervisado/clasificacion/04_clasificacion_multiclase_one_vs_all.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NelvaAdalit/Nelva-Adalit-AI-Portfolio/blob/main/1-aprendizaje-supervisado/clasificacion/04_clasificacion_multiclase_one_vs_all.ipynb): Extensión de clasificadores binarios a entornos multiclase utilizando la estrategia de *Uno contra Todos* (One-vs-All).
    *   [05_clasificacion_imagenes_intel_scenes.ipynb](./1-aprendizaje-supervisado/clasificacion/05_clasificacion_imagenes_intel_scenes.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NelvaAdalit/Nelva-Adalit-AI-Portfolio/blob/main/1-aprendizaje-supervisado/clasificacion/05_clasificacion_imagenes_intel_scenes.ipynb): Clasificación de paisajes utilizando visión por computadora tradicional y algoritmos de clasificación de imágenes.
*   **Redes Neuronales y Deep Learning**
    *   [06_redes_neuronales_pytorch_3_modelos.ipynb](./1-aprendizaje-supervisado/redes-neuronales/06_redes_neuronales_pytorch_3_modelos.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NelvaAdalit/Nelva-Adalit-AI-Portfolio/blob/main/1-aprendizaje-supervisado/redes-neuronales/06_redes_neuronales_pytorch_3_modelos.ipynb): Introducción práctica a PyTorch mediante la creación, entrenamiento y comparación de 3 arquitecturas diferentes de redes neuronales artificiales.
    *   [07_clasificacion_patologias_nhanes.ipynb](./1-aprendizaje-supervisado/deep-learning-nhanes/07_clasificacion_patologias_nhanes.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NelvaAdalit/Nelva-Adalit-AI-Portfolio/blob/main/1-aprendizaje-supervisado/deep-learning-nhanes/07_clasificacion_patologias_nhanes.ipynb): **[Notebook Destacado]** Clasificación multiclase de patologías crónicas (Obesidad, Hipertensión y Colesterol Alto) a partir del dataset NHANES. Implementa un Perceptrón Multicapa (MLP) en PyTorch con normalización `StandardScaler`, regularización L2 (Weight Decay), Dropout (0.4), Early Stopping y manejo del desbalanceo mediante pesos en la función de pérdida.

---

### 2. Aprendizaje No Supervisado (Unsupervised Learning)
Algoritmos que encuentran patrones ocultos o estructuras en datos sin etiquetas previas.

*   [01_kmeans_spotify_clustering.ipynb](./2-aprendizaje-no-supervisado/clustering-spotify/01_kmeans_spotify_clustering.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NelvaAdalit/Nelva-Adalit-AI-Portfolio/blob/main/2-aprendizaje-no-supervisado/clustering-spotify/01_kmeans_spotify_clustering.ipynb): Segmentación y agrupamiento (Clustering) de canciones de Spotify usando el algoritmo K-Means. Útil para recomendación de música basado en características de audio como bailabilidad, energía y tempo.

---

### 3. Aprendizaje Semi-supervisado (Semi-supervised Learning)
Modelos que combinan una pequeña cantidad de datos etiquetados con una gran cantidad de datos no etiquetados durante el entrenamiento.

*   [01_clasificacion_frutas_semisupervisado.ipynb](./3-aprendizaje-semi-supervisado/clasificacion-frutas/01_clasificacion_frutas_semisupervisado.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NelvaAdalit/Nelva-Adalit-AI-Portfolio/blob/main/3-aprendizaje-semi-supervisado/clasificacion-frutas/01_clasificacion_frutas_semisupervisado.ipynb): Clasificación de imágenes de frutas utilizando técnicas semi-supervisadas (como propagación de etiquetas). Permite entrenar clasificadores de alto rendimiento reduciendo drásticamente el costo de etiquetado manual de imágenes.

---

### 4. Aprendizaje por Refuerzo (Reinforcement Learning)
Modelos que aprenden a tomar decisiones en un entorno interactivo mediante prueba y error, utilizando un sistema de recompensas y castigos.

*   [01_aprendizaje_por_refuerzo_rompecabezas.ipynb](./4-aprendizaje-por-refuerzo/rompecabezas-deslizante/01_aprendizaje_por_refuerzo_rompecabezas.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NelvaAdalit/Nelva-Adalit-AI-Portfolio/blob/main/4-aprendizaje-por-refuerzo/rompecabezas-deslizante/01_aprendizaje_por_refuerzo_rompecabezas.ipynb): Implementación de un agente basado en gradiente de política tabular (Tabular Policy Gradient) que aprende a resolver de forma óptima el rompecabezas deslizante de 8 fichas en una cuadrícula de 3x3. Guarda las preferencias entrenadas en el archivo `pesos_rompecabezas.pt`.

---

## 🚀 Tecnologías y Librerías Utilizadas
El desarrollo de estos proyectos involucra principalmente las siguientes herramientas:

*   **Lenguaje**: Python 3
*   **Frameworks de Deep Learning**: PyTorch
*   **Bibliotecas de Machine Learning**: Scikit-Learn
*   **Manipulación y Análisis de Datos**: NumPy, Pandas
*   **Visualización**: Matplotlib, Seaborn
*   **Entorno de Desarrollo**: Jupyter Notebook / Google Colab

---

## ⚡ Cómo Empezar

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/TuUsuario/Nelva-Adalit-AI-Portfolio.git
    cd Nelva-Adalit-AI-Portfolio
    ```

2.  **Crear y activar un entorno virtual:**
    ```bash
    python -m venv env
    # En Windows:
    env\Scripts\activate
    ```

3.  **Instalar dependencias necesarias:**
    ```bash
    pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn jupyterlab tqdm
    ```

4.  **Iniciar Jupyter Lab:**
    ```bash
    jupyter lab
    ```