# Analisis-de-Metodos-de-Deteccion-de-SSVEP-en-Senales-EEG
Análisis de métodos de detección de potenciales visuales evocados en señales de electroencefalografía

Estructura del repositorio
El repositorio está organizado de la siguiente manera:

- Funciones auxiliares en Python:
Las funciones largas y repetitivas han sido implementadas por separado para facilitar su reutilización y mantener los notebooks más limpios. Estas funciones se encuentran en erchivo utils.py.

- Notebooks principales del proyecto:
La mayor parte del análisis y visualización de resultados se ha realizado en forma de cuadernos Jupyter. A continuación se describen los notebooks más importantes:

  - VisualizaciónBETA.ipynb:
Contiene los primeros gráficos y pruebas exploratorias sobre la base de datos. De esta forma, se obtiene una visión general de las señales EEG y nos familiarizamos con el contenido.

  - AnalisisInicial.ipynb:
Incluye el código correspondiente al análisis inicial que llevamos a cabo para seleccionar los datos de interés (métodos PSDA, CCA...) y generar señales sintéticas como referencia para entender mejor el comportamiento esperado.

  - AnalisisSecuencial.ipynb:
Este cuaderno contiene el código final utilizado para realizar la caracterización secuencial de los datos seleccionados, aplicando técnicas de análisis tiempo-frecuencia como espectrogramas y transformada wavelet continua.
