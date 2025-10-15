import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime
import time

st.set_page_config(page_title='Week Science — Markov · Streamlit', layout='wide')

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
        a, b = row['Subida_dt'], row['Bajada_dt']
        if pd.isnull(a) or pd.isnull(b):
            return np.nan
        delta = (b - a).seconds / 60 if b >= a else ((b.replace(day=a.day + 1) - a).seconds / 60)
        return delta
    df['Duracion_min'] = df.apply(duration_min, axis=1)
    return df

# --- Layout ---
st.title('Week Science — Markov · Streamlit')
st.sidebar.header('Controles')

uploaded = st.sidebar.file_uploader('Sube un CSV (opcional). Si no, uso muestra del notebook.', type=['csv'])
df = load_df(uploaded)

st.header('1) Dataset — vista completa')
with st.expander('Ver / exportar datos'):
    st.write(list(df.columns))
    rows_to_show = st.number_input('Filas a mostrar', min_value=5, max_value=500, value=25)
    st.dataframe(df.head(rows_to_show))
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button('Descargar CSV', data=csv, file_name='dataset_export.csv', mime='text/csv')

st.header('2) Preparación y estadísticas rápidas')
if st.button('Parsear tiempos y calcular duración (Subida/Bajada)'):
    df = parse_times(df)
    st.success('Columnas Subida_dt, Bajada_dt y Duracion_min añadidas (si fue posible parsear).')
    st.write(df[['Subida', 'Bajada', 'Duracion_min']].head(10))
else:
    st.info('Pulsa el botón para parsear Subida/Bajada y obtener Duracion_min (si aplica).')

duraciones = df['Duracion_min'].dropna().astype(float) if 'Duracion_min' in df.columns and df['Duracion_min'].notna().sum() > 0 else None

st.header('3) Visualizaciones interactivas')
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader('ECDF — marca un percentil')
    n_bins = st.slider('Número de bins (histograma auxiliar)', 10, 200, 50)
    percentile = st.slider('Percentil a marcar (0-100)', 0, 100, 95)

    if duraciones is None:
        mu = st.number_input('Media (min) para simulación exponencial', value=20.0)
        tiempos = np.random.exponential(scale=mu, size=2000)
    else:
        tiempos = duraciones.sample(n=2000, replace=True).values

    x = np.sort(tiempos)
    y = np.arange(1, len(x) + 1) / len(x)
    pval = np.percentile(tiempos, percentile)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='ECDF'))
    fig.add_vline(x=pval, line_dash='dash', line_color='green', annotation_text=f'{percentile}% = {pval:.2f} min')
    fig.update_layout(title='ECDF Interactiva', xaxis_title='Tiempo (min)', yaxis_title='ECDF')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader('Histograma resumen')
    fig2 = px.histogram(x=tiempos, nbins=n_bins, title='Histograma de tiempos', labels={'x': 'Tiempo (min)'}).update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# --- 4) Simulación visual ---
st.header('4) Simulación visual — construcción incremental (Plotly)')
st.write('Genera N muestras y observa cómo evoluciona el histograma en modo manual o automático.')

n_total = st.slider('Número total de muestras', 100, 10000, 2000)
chunk = st.slider('Tamaño del paso', 10, 2000, 200)
run_mode = st.radio('Modo de interacción', ['Automático (animado)', 'Manual (pasos)'])

source = duraciones.sample(n=n_total, replace=True).values if duraciones is not None else np.random.exponential(scale=20.0, size=n_total)
placeholder = st.empty()

if run_mode == 'Manual':
    if 'i_idx' not in st.session_state:
        st.session_state['i_idx'] = 0
    if st.button('Añadir chunk'):
        st.session_state['i_idx'] += chunk
    current_n = min(st.session_state['i_idx'], n_total)
    if current_n <= 0:
        st.info('Pulsa "Añadir chunk" para empezar la simulación.')
    else:
        data_view = source[:current_n]
        fig3 = px.histogram(x=data_view, nbins=50, title=f'Histograma — {current_n} muestras')
        placeholder.plotly_chart(fig3, use_container_width=True)
else:
    for i in range(0, n_total, chunk):
        data_view = source[:i+chunk]
        fig3 = px.histogram(x=data_view, nbins=50, title=f'Histograma — {i+chunk} muestras')
        placeholder.plotly_chart(fig3, use_container_width=True)
        time.sleep(0.15)

# --- 5) Cadena de Markov (mapa de calor) ---
st.header('5) Cadena de Markov — Mapa de calor de transiciones')
if 'Ruta' in df.columns:
    states = df['Ruta'].astype(str).values
    trans_matrix = pd.crosstab(pd.Series(states[:-1], name='From'), pd.Series(states[1:], name='To'), normalize='index')
    fig_heat = px.imshow(trans_matrix, text_auto=True, color_continuous_scale='Viridis', title='Matriz de transición (probabilidades)')
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info('No se encontró la columna "Ruta" para calcular la cadena de Markov.')
