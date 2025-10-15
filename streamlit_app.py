import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from datetime import datetime

st.set_page_config(page_title='Semana de Ciencias — Markov · Streamlit', layout='wide')

# --- Sample data (from your notebook) ---
SAMPLE_DATA = """Ruta,Subida,Bajada,Fecha,Sentado,Parada
Indeco,7:22,7:48,1/9/2025,no,Auto
Sauzal,7:57,8:18,1/9/2025,no,Burger
Indeco,5:58,6:18,2/9/2025,no,Auto
Sauzal,6:50,7:09,2/9/2025,no,Burger
Cortez,6:14,6:36,3/9/2025,no,Auto
Sauzal,6:42,7:07,3/9/2025,si,Plaza
Indeco,6:21,6:40,4/9/2025,no,Auto
Sauzal,6:44,7:08,4/9/2025,no,Burger
Cortez,6:17,6:41,5/9/2025,no,Auto
Sauzal,6:53,7:18,5/9/2025,si,Plaza
V. Sol,6:27,6:47,8/9/2025,no,Auto
Sauzal,6:56,7:15,8/9/2025,no,Burger
V. Sol,6:24,6:45,9/9/2025,no,Auto
Sauzal,6:53,7:11,9/9/2025,no,Burger
Indeco,5:48,6:06,02/10/2025,no,Auto
Sauzal,6:11,6:30,02/10/2025,no,Burger
V. Sol,6:04,6:20,03/10/2025,no,Auto
Sauzal,6:26,6:58,03/10/2025,si,Burger
Indeco,6:06,6:22,06/10/2025,no,Auto
Sauzal,6:26,7:42,06/10/2025,no,Burger
Indeco,6:22,6:44,07/10/2025,si,Side
Sauzal,6:48,7:11,07/10/2025,si,Burger
Cortez,6:35,7:00,08/10/2025,no,Auto
Sauzal,7:03,7:28,08/10/2025,si,Plaza
Indeco,6:36,6:59,09/10/2025,si,Auto
Sauzal,7:11,7:33,09/10/2025,no,Burger
Indeco,6:39,6:59,10/10/2025,no,Auto
Sauzal,7:06,7:28,10/10/2025,si,Burger
"""

# --- Helpers ---

def load_df(uploaded_file):
    if uploaded_file is None:
        df = pd.read_csv(io.StringIO(SAMPLE_DATA))
    else:
        df = pd.read_csv(uploaded_file)
    return df


def parse_times(df):
    # Try to parse 'Subida' and 'Bajada' as times and compute duration in minutes
    df = df.copy()
    def parse_time(t):
        try:
            return datetime.strptime(str(t), '%H:%M')
        except Exception:
            try:
                return datetime.strptime(str(t), '%H:%M:%S')
            except Exception:
                return None

    df['Subida_dt'] = df['Subida'].apply(parse_time)
    df['Bajada_dt'] = df['Bajada'].apply(parse_time)

    def duration_min(row):
        a = row['Subida_dt']
        b = row['Bajada_dt']
        if pd.isnull(a) or pd.isnull(b):
            return np.nan
        # if b < a assume next day
        delta = (b - a).seconds/60 if b>=a else ((b.replace(day=a.day+1) - a).seconds/60 if hasattr(b, 'replace') else np.nan)
        return delta

    df['Duracion_min'] = df.apply(duration_min, axis=1)
    return df


# --- Layout ---
st.title('Week Science — Markov · Streamlit')
st.sidebar.header('Controles')

uploaded = st.sidebar.file_uploader('Sube un CSV (opcional). Si no, uso muestra del notebook.', type=['csv'])

df = load_df(uploaded)

# Show dataset
st.header('1) Dataset — vista completa')
with st.expander('Ver / exportar datos'):
    st.write('Columnas detectadas:')
    st.write(list(df.columns))
    rows_to_show = st.number_input('Filas a mostrar', min_value=5, max_value=500, value=25)
    st.dataframe(df.head(rows_to_show))
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button('Descargar CSV', data=csv, file_name='dataset_export.csv', mime='text/csv')

# Prepare parsed dataframe
st.header('2) Preparación y estadísticas rápidas')
if st.button('Parsear tiempos y calcular duración (Subida/Bajada)'):
    df = parse_times(df)
    st.success('Columnas Subida_dt, Bajada_dt y Duracion_min añadidas (si fue posible parsear).')
    st.write(df[['Subida','Bajada','Duracion_min']].head(10))
else:
    st.info('Pulsa el botón para parsear Subida/Bajada y obtener Duracion_min (si aplica).')

# If Duracion_min exists, compute baseline for simulations
if 'Duracion_min' in df.columns and df['Duracion_min'].notna().sum()>0:
    duraciones = df['Duracion_min'].dropna().astype(float)
else:
    # fallback: synthesize durations from sample
    duraciones = None

# --- Visualizaciones interactivas ---
st.header('3) Visualizaciones interactivas')
col1, col2 = st.columns([2,1])

with col1:
    st.subheader('ECDF — marca un percentil (como en tu notebook)')
    n_bins = st.slider('Número de bins (histograma auxiliar)', 10, 200, 50)
    percentile = st.slider('Percentil a marcar (0-100)', 0, 100, 95)

    # Generate simulation data if no durations
    if duraciones is None:
        mu = st.number_input('Media (min) para simulación exponencial (si no hay Duracion_min)', value=20.0)
        tiempos = np.random.exponential(scale=mu, size=2000)
    else:
        # Use empirical durations with a small bootstrap noise to make ECDF smooth
        tiempos = duraciones.sample(n=2000, replace=True).values

    # ECDF calculation
    x = np.sort(tiempos)
    y = np.arange(1, len(x)+1)/len(x)
    pval = np.percentile(tiempos, percentile)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x, y, marker='.', linestyle='none')
    ax.axvline(pval, color='green', linestyle='--', linewidth=2, label=f'{percentile}% = {pval:.2f} min')
    ax.axhline(percentile/100.0, color='green', linestyle='--')
    ax.set_xlabel('Tiempo (min)')
    ax.set_ylabel('ECDF')
    ax.legend()
    st.pyplot(fig)

    st.markdown(f'**Interpretación rápida:** El {percentile}º percentil es **{pval:.2f} minutos**. Eso significa que el {percentile}% de las observaciones están por debajo de ese valor.')

with col2:
    st.subheader('Histograma resumen')
    fig2, ax2 = plt.subplots(figsize=(4,3))
    ax2.hist(tiempos, bins=n_bins, density=True)
    ax2.axvline(np.mean(tiempos), color='red', linestyle='--', label=f'Media {np.mean(tiempos):.2f}')
    ax2.legend()
    st.pyplot(fig2)

# --- Simulación paso a paso (visual y didáctica) ---
st.header('4) Simulación visual — construcción incremental')
st.write('Genera N muestras de una distribución (o muestreo bootstrap) y observa cómo cambia la ECDF / histograma conforme agregamos datos.')

n_total = st.slider('Número total de muestras a simular', 100, 10000, 2000)
chunk = st.slider('Tamaño del chunk (paso visual)', 10, 2000, 200)
run_mode = st.radio('Modo de interacción', ['Automático (animado)', 'Manual (pasos)'])

# Prepare source distribution
if duraciones is None:
    mu = st.number_input('Media (min) para simulación exponencial (modo simulación)', value=20.0, key='mu2')
    source = np.random.exponential(scale=mu, size=n_total)
else:
    source = duraciones.sample(n=n_total, replace=True).values

placeholder = st.empty()

if run_mode == 'Manual':
    st.write('Pulsa el botón "Añadir chunk" para ver más datos añadidos a la visualización')
    if 'i_idx' not in st.session_state:
        st.session_state['i_idx'] = 0
    if st.button('Añadir chunk'):
        st.session_state['i_idx'] += chunk
    current_n = min(st.session_state['i_idx'], n_total)
    if current_n<=0:
        st.info('Aún no hay muestras. Pulsa "Añadir chunk"')
    else:
        data_view = source[:current_n]
        fig3, ax3 = plt.subplots(figsize=(8,4))
        ax3.hist(data_view, bins=50, density=True, alpha=0.6)
        ax3.set_title(f'Histogram — {current_n} muestras')
        placeholder.pyplot(fig3)

else:
    # Automatic animated mode — update in-place
    import time
    current_n = 0
    for i in range(0, n_total, chunk):
        current_n = min(i+chunk, n_total)
        data_view = source[:current_n]
        fig3, ax3 = plt.subplots(figsize=(8,4))
        ax3.hist(data_view, bins=50, density=True, alpha=0.6)
        ax3.set_title(f'Histogram — {current_n} muestras')
        placeholder.pyplot(fig3)
        time.sleep(0.15)

st.caption('Sugerencia: si la animación es lenta en tu entorno, usa el modo Manual.')

# --- Sugerencias / mejoras posibles (sorpresas) ---
# st.header('Sugerencias adicionales (te pude sorprender)')
# st.markdown('''
# - Añadir una página de "Análisis de Markov" donde se muestre la matriz de transición entre estados (si tus datos tienen estados por paso).
# - Agregar export de resultados de simulación (CSV / figura).
# - Reemplazar matplotlib por Plotly para interactividad (zoom, hover) en el navegador.
# - Incluir tests unitarios y un pequeño README con comandos para desplegar.
# - Implementar cache para datasets grandes usando @st.cache_data.
# ''')

# st.markdown('Si quieres, lo convierto a un único archivo `app.py` listo para ejecutar y te doy los pasos para desplegarlo en Streamlit Cloud o en tu máquina.')
