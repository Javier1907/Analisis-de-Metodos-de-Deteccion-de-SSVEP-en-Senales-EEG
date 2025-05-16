import h5py
import pywt
import scipy.io
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


## -------------------------------------- Funciones para guardar y cargar datos --------------------------------------------------------------------------

# Función que guarda datos en un archivo HDF5 (.h5)
def guardar_datos(nombre_archivo, datos):
    """
    Esta función guarda los datos de un array en un archivo con extensión .h5

    Parámetros:
    nombre_archivo (str): nombre del archivo donde se guardarán los datos
    datos (List[float]): array de datos a guardar (puede ser multidimensional)
    """

    # Comprobamos que el nombre del archivo termina en .h5
    if nombre_archivo.endswith('.h5'):

        # Abrimos el archivo en modo escritura ("w") y guardamos el array como un dataset llamado "data"
        with h5py.File(nombre_archivo, "w") as archivo:
            archivo.create_dataset("data", data=datos)

        # Confirmación por consola
        print("Datos guardados correctamente en", nombre_archivo)

    else:
        # Si el archivo no tiene extensión .h5, mostramos un mensaje de error
        print("El nombre del archivo debe tener la extensión .h5")




def cargar_datos(nombre_archivo):
    """
    Esta función carga los datos desde un archivo .h5 y los convierte en un array.

    Parámetros:
    nombre_archivo (str): nombre del archivo desde el que se cargarán los datos
    """

    # Comprobamos que el nombre del archivo termina en .h5
    if nombre_archivo.endswith('.h5'):

        # Abrimos el archivo HDF5 en modo lectura ("r")
        with h5py.File(nombre_archivo, "r") as archivo:
            # Cargamos el dataset llamado "data" y lo convertimos en array
            datos_cargados = archivo["data"][:]

        # Confirmamos que los datos se han cargado correctamente
        print("Datos cargados correctamente desde", nombre_archivo)

        return datos_cargados  # Devolvemos los datos cargados
    
    else:
        # Si el archivo no tiene la extensión correcta, mostramos un error y devolvemos una lista vacía
        print("El nombre del archivo debe tener la extensión .h5")
        return []

    



## -------------------------------------- Funciones de Formato en Pandas --------------------------------------------------------------------------
def colorear_valores_booleanos(valor):
    if valor == 1:
        return 'color: green'
    elif valor == 0:
        return 'color: red'
    else:
        return ''


## -------------------------------------- Funciones para el manejo del conjunto de datos --------------------------------------------------------

def load_ssvep_data(dataset_path, subject):
    """
    Esta función carga los datos del EEG desde los archivos .mat del conjunto BETA para un sujeto dado.
    Devuelve las grabaciones EEG originales, las grabaciones sin los periodos de reposo y la duración del ensayo.
    
    Parámetros:
    dataset_path (str): Ruta al conjunto de datos
    subject (float): Número del sujeto

    Retorna:
    Tuple[List[List[List[List[float]]]], List[List[List[List[float]]]], float]:
        Datos EEG originales (matriz 4D: canal x tiempo x ensayo x índice de frecuencia)
        Datos EEG sin los periodos de reposo (matriz 4D: canal x tiempo x ensayo x índice de frecuencia)
        Duración del ensayo
    """

    # Para los sujetos S1-S15, la ventana de tiempo es de 2 s y la duración del ensayo es de 3 s,
    # mientras que para los sujetos S16-S70, la ventana de tiempo es de 3 s y la duración del ensayo es de 4 s.

    ruta_completa = dataset_path

    if not dataset_path.endswith("/"):
        ruta_completa += "/"

    duracion_ensayo = 3

    ruta_completa += 'S' + str(subject) + '.mat'

    # NOTA: Las líneas comentadas a continuación corresponden a una posible forma de organizar por carpetas
    # a los sujetos según su número, pero no están activas.
    #
    #
    #if subject <= 10:
    #    file_name += 'S1-S10/S' + str(subject) + '.mat'
    #elif subject <= 20:
    #    file_name += 'S11-S20/S' + str(subject) + '.mat'
    #elif subject <= 30:
    #    file_name += 'S21-S30/S' + str(subject) + '.mat'
    #elif subject <= 40:
    #    file_name += 'S31-S40/S' + str(subject) + '.mat'
    #elif subject <= 50:
    #    file_name += 'S41-S50/S' + str(subject) + '.mat'
    #elif subject <= 60:
    #    file_name += 'S51-S60/S' + str(subject) + '.mat'
    #elif subject <= 70:
    #    file_name += 'S61-S70/S' + str(subject) + '.mat'

    if subject > 15:
        duracion_ensayo = 4

    contenido = scipy.io.loadmat(ruta_completa)
    frecuencia_muestreo = 250
    segundos_sin_estimulo = 0.5

    # Datos EEG (matriz 4D)
    registros_eeg = contenido['data'][0, 0]['EEG']

    # Eliminamos el tiempo sin estímulo:
    # 0,5 segundos sin estímulo y una frecuencia de muestreo de 250 -> 0,5 * 250 muestras sin estímulo
    muestras_a_omitir = int(segundos_sin_estimulo * frecuencia_muestreo)

    # Se eliminan x muestras del principio y del final. Quedan 500 muestras correspondientes al estímulo.
    registros_estimulo_solo = registros_eeg[:, muestras_a_omitir:750-muestras_a_omitir, :, :]

    return registros_eeg, registros_estimulo_solo, duracion_ensayo


def load_ssvep_additional_info(path):
    """
    Esta función devuelve información adicional del conjunto de datos:
    la frecuencia (en Hz) correspondiente a cada índice de frecuencia y los nombres de los canales.
    
    Parámetros:
    path (str): ruta de uno de los archivos del conjunto de datos

    Retorna:
    Tuple[List[float], List[str]]:
    Frecuencias en Hz para cada índice de frecuencia
    Nombres de los canales
    """

    # Esta información es igual en todos los archivos
    contenido = scipy.io.loadmat(path)

    # Se obtienen las 40 frecuencias utilizadas en el experimento
    frecuencias_utilizadas = contenido['data'][0, 0]['suppl_info']['freqs'][0, 0][0]

    # Se extraen los nombres de todos los canales (64 canales)
    nombres_canales = []
    for j in range(64):
        nombres_canales.append(contenido['data'][0, 0]['suppl_info']['chan'][0, 0][:, 3][j][0])
        
    nombres_canales = list(nombres_canales)

    return frecuencias_utilizadas, nombres_canales


def select_occipital_electrodes(electrodos):
    """
    Esta función selecciona todos los electrodos que contienen una 'O' en su nombre (electrodos occipitales).
    Devuelve esta información de tres formas distintas.

    Parámetros:
    electrodos (List[str]): lista de nombres de canales

    Retorna: 
    Tuple[List[List[int, str]], List[int], List[str]]:
    Lista de listas que contienen el índice y el nombre de los electrodos occipitales
    Lista de índices de los electrodos occipitales
    Lista de nombres de los electrodos occipitales
    """

    # Seleccionamos los electrodos que contienen una 'O'
    lista_occipital = []
    indices_occipitales = []
    nombres_occipitales = []
    for i in range(len(electrodos)):
        etiqueta = electrodos[i]
        if "O" in etiqueta:
            lista_occipital.append([i, etiqueta])
            indices_occipitales.append(i)
            nombres_occipitales.append(etiqueta)

    return lista_occipital, indices_occipitales, nombres_occipitales


## ---------------------------------- FUNCIONES DE GENERACIÓN DE SEÑALES PARA CCA --------------------------------------------------------------------------

# Constante PI
PI = np.pi 

# Función lambda que genera el componente seno de una señal con frecuencia f, armónico h, tiempo t y fase p
sin = lambda f, h, t, p: np.sin(2 * PI * f * h * t + p)  

# Función lambda que genera el componente coseno de una señal con frecuencia f, armónico h, tiempo t y fase p
cos = lambda f, h, t, p: np.cos(2 * PI * f * h * t + p)  

# Función lambda que genera una señal de referencia combinando seno y coseno para una frecuencia, armónico, tiempo y fase
reference_signal = lambda f, h, t, p: [sin(f, h, t, p), cos(f, h, t, p)]  

def construir_vector_referencia(f, t, cantidad_armonicos, fase):
    """
    Genera un punto en el tiempo de una señal de referencia compuesta por funciones seno y coseno, 
    incluyendo un número dado de armónicos.

    Parámetros:
    f (float): frecuencia de oscilación de la señal generada (Hz)
    t (float): instante de tiempo en el que se evalúa la señal (s)
    cantidad_armonicos (int): número de armónicos que se incluirán (a partir del 1)
    fase (float): fase a añadir al generar cada componente seno y coseno (radianes)

    Retorna:
    List[float]: valores de la señal de referencia en el instante t, incluyendo los componentes 
                 seno y coseno de todos los armónicos.
    """
    vector_referencia = []

    # Recorremos los armónicos desde 1 hasta num_harmonics (inclusive)
    # El armónico 1 es el fundamental, y luego se añaden múltiplos (2f, 3f, ..., n·f)
    for k in range(1, cantidad_armonicos + 1):
        vector_referencia += reference_signal(f, k, t, fase)

    return vector_referencia



def construir_senal_referencia(f, tasa_muestreo, duracion, armonicos, fase):
    """
    Genera una señal de referencia que oscila a la frecuencia f, con una duración y frecuencia de muestreo dadas.
    Se incluyen armónicos de la frecuencia y una fase inicial.

    Parámetros:
    f (float): frecuencia de oscilación de la señal generada (Hz)
    tasa_muestreo (int): frecuencia de muestreo (Hz)
    duracion (float): duración de la señal generada (en segundos)
    armonicos (int): número de armónicos que se incluirán (desde 1 hasta num_harmonics)
    fase (float): fase inicial a añadir (radianes)

    Retorna:
    List[List[float]]: señal de referencia generada, donde cada muestra es una lista con los valores 
                       de los componentes seno y coseno de todos los armónicos
    """
    
    senal_generada = []  # Lista donde se almacenarán las muestras de la señal de referencia

    # Número total de muestras a generar
    total_muestras = duracion * tasa_muestreo

    # Generamos una muestra por cada instante de tiempo, según la frecuencia de muestreo
    for n in range(int(total_muestras)):
        tiempo = n / tasa_muestreo  # Convertimos el índice de muestra en tiempo (en segundos)
        senal_generada.append(construir_vector_referencia(f, tiempo, armonicos, fase))  # Generamos los componentes seno y coseno de cada armónico
    
    return senal_generada




## -------------------------------------- FUNCIONES DE ANÁLISIS SECUENCIAL ----------------------------------------------------------------------

def generar_visualizacion_wavelet_espectrograma(senal_eeg, fs, idx_frecuencia, frecs_estimulo, mostrar, guardar, ruta_salida, zonas_sombreadas, duracion_sombra):
    """ 
    Genera una figura con tres subgráficas:
    - Señal EEG filtrada.
    - Espectrograma.
    - Coeficientes de la Transformada Wavelet Continua (CWT).

    Parámetros:
    senal_eeg (List[float]): señal EEG a representar
    fs (int): frecuencia de muestreo
    idx_frecuencia (int): índice de la frecuencia de estímulo en la lista
    frecs_estimulo (List[float]): lista completa de frecuencias de estímulo
    mostrar (bool): si True, se muestra la figura
    guardar (bool): si True, se guarda la figura en archivo
    ruta_salida (str): ruta del archivo de salida
    zonas_sombreadas (List[int]): muestras detectadas para sombrear (una por método)
    duracion_sombra (int): duración de la región sombreada en muestras
    """

    wavelet = "morl"

    # Filtro Butterworth (paso banda 5–40 Hz)
    f_min, f_max = 5, 40
    b, a = signal.butter(4, [f_min, f_max], fs=fs, btype='band')
    senal_filtrada = signal.lfilter(b, a, senal_eeg)

    # Frecuencias para análisis wavelet
    frecs_norm = np.arange(1, 40, 0.01) / fs
    escalas = pywt.frequency2scale(wavelet, frecs_norm)
    cwt_coef, _ = pywt.cwt(senal_filtrada, escalas, wavelet)
    frecs_abs = frecs_norm * fs

    # Eje temporal
    eje_tiempo = np.arange(0, len(senal_eeg) / fs, 1 / fs)

    # Inicialización de figura
    fig, (g1, g2, g3) = plt.subplots(nrows=3, figsize=(10, 6), sharex=True)

    # === Panel 1: Señal EEG filtrada ===
    g1.plot(eje_tiempo, senal_filtrada, 'k')
    g1.set_xlim(0, len(senal_eeg) / fs)
    g1.set_xticks([0.5, 1, 1.5, 2, 2.5] if len(senal_eeg) < 750 else [0.5, 1, 1.5, 2, 2.5, 3, 3.5])
    g1.set_ylabel("Amplitude V")
    g1.set_xlabel("Time (s)")

    colores = ["lightblue", "lightyellow"]
    etiquetas = ["m_PSDA", "CCA"]
    leyenda = [mpatches.Patch(color=c, label=e) for c, e in zip(colores, etiquetas)]
    g1.legend(handles=leyenda, loc="upper right")

    for i, inicio in enumerate(zonas_sombreadas):
        if inicio != -1:
            t_ini = inicio / fs
            t_fin = (inicio + duracion_sombra) / fs
            g1.axvspan(t_ini, t_fin, facecolor=colores[i], alpha=0.75)

    # === Panel 2: Espectrograma ===
    vmin_e, vmax_e = -10, 10
    Pxx, frecs, bins, im = g2.specgram(senal_filtrada, NFFT=300, Fs=fs,
                                       noverlap=int(300 / 4), cmap='jet',
                                       vmin=vmin_e, vmax=vmax_e)

    f_est = frecs_estimulo[idx_frecuencia]
    g2.set_yticks(np.arange(0, 50, step=f_est))
    g2.set_xticks([0.5, 1, 1.5, 2, 2.5] if len(senal_eeg) < 750 else [0.5, 1, 1.5, 2, 2.5, 3, 3.5])
    g2.set_xlabel("Time (s)")
    g2.set_ylabel("Frequency (Hz)")
    g2.set_ylim(0, 50)

    barra_espectro = fig.add_axes([1.01, 0.45, 0.02, 0.2])
    plt.colorbar(im, cax=barra_espectro, label='Power (dB/Hz)')

    # === Panel 3: Transformada Wavelet ===
    vmin_w, vmax_w = 0, 40
    g3.imshow(np.abs(cwt_coef), aspect='auto', cmap='jet',
              extent=[0, len(senal_filtrada) / fs, frecs_abs[-1], frecs_abs[0]],
              origin='upper', vmin=vmin_w, vmax=vmax_w)
    g3.set_xlabel("Time (s)")
    g3.set_ylabel("Frequency (Hz)")
    g3.set_ylim(2, 35)
    g3.set_yticks(np.arange(0, 35, step=f_est))
    g3.set_xlim(0, len(senal_eeg) / fs)

    barra_wavelet = fig.add_axes([1.01, 0.11, 0.02, 0.2])
    plt.colorbar(g3.images[0], cax=barra_wavelet, label="Magnitude")

    plt.tight_layout()

    if guardar:
        plt.savefig(ruta_salida, dpi=fig.dpi, bbox_inches='tight')

    if mostrar:
        plt.show()
    else:
        plt.close()



    


def generar_espectro_fft(senal, fs, frec_estimulacion, mostrar, guardar, ruta_salida):
    """
    Genera un gráfico del espectro de potencia de una señal EEG filtrada.
    Se destaca el pico de potencia más alto y se indica la frecuencia de estímulo.

    Parámetros:
    senal (List[float]): señal EEG original
    fs (int): frecuencia de muestreo (Hz)
    frec_estimulacion (float): frecuencia real de estímulo (Hz)
    mostrar (bool): si True, muestra la figura
    guardar (bool): si True, guarda la figura en disco
    ruta_salida (str): ruta del archivo a guardar
    """

    # === Filtro paso banda 5–40 Hz para limpiar la señal ===
    f_baja, f_alta = 5, 40
    b, a = signal.butter(4, [f_baja, f_alta], fs=fs, btype='band')
    senal_filtrada = signal.lfilter(b, a, senal)

    # === Cálculo de la FFT y el espectro de potencia ===
    fft_vals = np.fft.fft(senal_filtrada)
    frecs = np.fft.fftfreq(len(senal_filtrada), d=1/fs)
    potencia = np.abs(fft_vals) ** 2

    # === Representación del espectro ===
    plt.plot(frecs, potencia)
    plt.xlim(0, 400)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power V²")
    plt.title("Power Spectra (" + str(frec_estimulacion) + " Hz)")

    # Marcamos la frecuencia con mayor potencia
    idx_pico = np.argmax(potencia[0:500])
    freq_pico = frecs[idx_pico]
    pot_pico = potencia[idx_pico]

    plt.plot(freq_pico, pot_pico, "o", color="red")
    plt.text(freq_pico + 2, pot_pico, str(round(freq_pico, 2)) + " Hz")

    # Ajustamos los límites del gráfico
    plt.xlim(0, 55)
    plt.ylim(0)

    # Guardado opcional
    if guardar:
        plt.savefig(ruta_salida, dpi='figure', format=None)

    # Mostrar o cerrar según el parámetro
    if mostrar:
        plt.show()
    else:
        plt.close()



        

def representar_frecuencias_dominantes(senal_eeg, frec_estimulacion, fs, ruta_salida):
    """
    Genera un gráfico con la frecuencia dominante (de mayor magnitud) a lo largo del tiempo
    según la Transformada Wavelet Continua (CWT). Se descartan magnitudes menores a 25
    para facilitar la visualización, y se indica la frecuencia de estímulo con una línea roja.

    Parámetros:
    senal_eeg (List[float]): señal EEG a analizar
    frec_estimulacion (float): frecuencia de estímulo en Hz
    fs (int): frecuencia de muestreo
    ruta_salida (str): ruta donde se guarda la imagen resultante (si se especifica)
    """

    # === FILTRADO DE LA SEÑAL ===
    f_min, f_max = 5, 40
    b, a = signal.butter(4, [f_min, f_max], fs=fs, btype='band')
    senal_filtrada = signal.lfilter(b, a, senal_eeg)

    # === TRANSFORMADA WAVELET CONTINUA (CWT) ===
    frecs_norm = np.arange(1, 40, 0.01) / fs
    wavelet = "morl"
    escalas = pywt.frequency2scale(wavelet, frecs_norm)
    cwt_coef, _ = pywt.cwt(senal_filtrada, escalas, wavelet)

    # === FILTRADO DE MAGNITUDES BAJAS ===
    cwt_filtrado = np.where(cwt_coef < 25, 0, cwt_coef)

    # Obtenemos el índice de la frecuencia con mayor magnitud en cada instante
    idx_maximos = np.argmax(cwt_filtrado, axis=0)
    frecs_reales = np.arange(1, 40, 0.01)
    frecs_dominantes = frecs_reales[idx_maximos]

    # Eje de tiempo
    tiempo = np.arange(0, cwt_coef.shape[1] / fs, 1 / fs)

    # === REPRESENTACIÓN ===
    plt.plot(tiempo, frecs_dominantes, label='Frequency with Highest Magnitude (>25)')
    plt.axhline(frec_estimulacion, color="red", label="Stimulus frequency")

    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Highest magnitudes over time")
    plt.legend()
    plt.yticks(np.arange(1, 30, step=2))

    if cwt_coef.shape[1] > 800:
        plt.xlim(0, 4)
    else:
        plt.xlim(0, 3)

    # Guardamos o mostramos según la ruta especificada
    if ruta_salida != "":
        plt.savefig(ruta_salida, dpi='figure', format=None)
    else:
        plt.show()

    plt.close()
