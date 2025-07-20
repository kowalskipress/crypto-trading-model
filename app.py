import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import time

# Importar módulos propios
from data.crypto_data import CryptoDataFetcher
from models.trading_model import TradingModel
from config.settings import CRYPTO_SYMBOLS, MODEL_CONFIG, UI_CONFIG

# Configuración de la página
st.set_page_config(
    page_title="Crypto Trading Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .signal-buy {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .signal-sell {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .signal-neutral {
        color: #ffc107;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=21600)  # Cache por 6 horas
def load_crypto_data():
    """Cargar datos de criptomonedas con cache"""
    fetcher = CryptoDataFetcher()
    data = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, symbol in enumerate(CRYPTO_SYMBOLS):
        status_text.text(f'Cargando datos de {symbol.upper()}...')
        try:
            df = fetcher.get_historical_data(symbol, days=MODEL_CONFIG['historical_days'])
            if df is not None and not df.empty:
                data[symbol] = df
        except Exception as e:
            st.warning(f"Error cargando {symbol}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(CRYPTO_SYMBOLS))
    
    status_text.empty()
    progress_bar.empty()
    
    return data

@st.cache_data(ttl=21600)
def calculate_signals(data):
    """Calcular señales de trading con cache"""
    model = TradingModel()
    signals = {}
    
    for symbol, df in data.items():
        if df is not None and len(df) >= MODEL_CONFIG['min_periods']:
            try:
                signal_data = model.generate_signal(df)
                signals[symbol] = signal_data
            except Exception as e:
                st.error(f"Error calculando señales para {symbol}: {str(e)}")
    
    return signals

def display_signal_card(symbol, signal_data):
    """Mostrar tarjeta de señal individual"""
    if signal_data is None:
        st.error(f"No hay datos disponibles para {symbol.upper()}")
        return
    
    signal = signal_data['signal']
    index_value = signal_data['index']
    confidence = signal_data['confidence']
    
    # Color según señal
    if signal == 'COMPRA':
        signal_class = 'signal-buy'
        emoji = "🟢"
    elif signal == 'VENTA':
        signal_class = 'signal-sell'
        emoji = "🔴"
    else:
        signal_class = 'signal-neutral'
        emoji = "🟡"
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown(f"### {emoji} {symbol.upper()}")
    
    with col2:
        st.markdown(f'<p class="{signal_class}">{signal}</p>', unsafe_allow_html=True)
        st.metric("Índice", f"{index_value:.1f}", f"{confidence:.1f}%")
    
    with col3:
        # Mini indicadores
        metrics = signal_data.get('metrics', {})
        if metrics:
            st.write("**Métricas Clave:**")
            st.write(f"RSI: {metrics.get('rsi', 0):.1f}")
            st.write(f"MACD: {metrics.get('macd_signal', 0):.3f}")

def create_index_chart(signals):
    """Crear gráfico de índices"""
    fig = go.Figure()
    
    symbols = list(signals.keys())
    indices = [signals[s]['index'] if signals[s] else 0 for s in symbols]
    colors = []
    
    for s in symbols:
        if signals[s]:
            signal = signals[s]['signal']
            if signal == 'COMPRA':
                colors.append('#28a745')
            elif signal == 'VENTA':
                colors.append('#dc3545')
            else:
                colors.append('#ffc107')
        else:
            colors.append('#6c757d')
    
    fig.add_trace(go.Bar(
        x=[s.upper() for s in symbols],
        y=indices,
        marker_color=colors,
        text=[f"{idx:.1f}" for idx in indices],
        textposition='auto',
    ))
    
    # Líneas de referencia
    fig.add_hline(y=65, line_dash="dash", line_color="green", annotation_text="Umbral Compra (65)")
    fig.add_hline(y=35, line_dash="dash", line_color="red", annotation_text="Umbral Venta (35)")
    fig.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutral (50)")
    
    fig.update_layout(
        title="Índices de Trading por Criptomoneda",
        yaxis_title="Índice (0-100)",
        xaxis_title="Criptomoneda",
        height=400,
        showlegend=False
    )
    
    return fig

def display_metrics_breakdown(symbol, signal_data):
    """Mostrar desglose de métricas"""
    if not signal_data or 'metrics' not in signal_data:
        return
    
    metrics = signal_data['metrics']
    weights = MODEL_CONFIG['weights']
    
    # Crear DataFrame para mostrar
    metrics_df = pd.DataFrame([
        {'Métrica': 'RSI (14)', 'Valor': f"{metrics.get('rsi', 0):.2f}", 'Peso': f"{weights['rsi']*100:.1f}%"},
        {'Métrica': 'MACD', 'Valor': f"{metrics.get('macd_signal', 0):.4f}", 'Peso': f"{weights['macd']*100:.1f}%"},
        {'Métrica': 'Bollinger Bands', 'Valor': f"{metrics.get('bb_position', 0):.2f}", 'Peso': f"{weights['bb_position']*100:.1f}%"},
        {'Métrica': 'VWAP', 'Valor': f"{metrics.get('vwap_signal', 0):.2f}", 'Peso': f"{weights['vwap']*100:.1f}%"},
        {'Métrica': 'Stochastic %K', 'Valor': f"{metrics.get('stoch_k', 0):.2f}", 'Peso': f"{weights['stoch_k']*100:.1f}%"},
        {'Métrica': 'OBV', 'Valor': f"{metrics.get('obv_signal', 0):.2f}", 'Peso': f"{weights['obv']*100:.1f}%"},
        {'Métrica': 'ATR', 'Valor': f"{metrics.get('atr_signal', 0):.2f}", 'Peso': f"{weights['atr']*100:.1f}%"},
        {'Métrica': 'ROC (14)', 'Valor': f"{metrics.get('roc', 0):.2f}", 'Peso': f"{weights['roc']*100:.1f}%"},
    ])
    
    st.dataframe(metrics_df, use_container_width=True)

def main():
    """Función principal de la aplicación"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Crypto Trading Model</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Información del modelo
        st.subheader("Información del Modelo")
        st.write(f"**Timeframe:** {MODEL_CONFIG['timeframe']}")
        st.write(f"**Análisis cada:** {MODEL_CONFIG['analysis_frequency']}")
        st.write(f"**Histórico:** {MODEL_CONFIG['historical_days']} días")
        
        # Umbrales
        st.subheader("Umbrales de Señal")
        st.write("🟢 **Compra:** Índice ≥ 65")
        st.write("🟡 **Neutro:** 35 < Índice < 65")
        st.write("🔴 **Venta:** Índice ≤ 35")
        
        # Botón de actualización manual
        if st.button("🔄 Actualizar Datos", type="primary"):
            st.cache_data.clear()
            st.experimental_rerun()
        
        # Última actualización
        st.write(f"**Última actualización:** {datetime.now().strftime('%H:%M:%S')}")
    
    # Contenido principal
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Análisis Detallado", "🔍 Backtesting", "ℹ️ Información"])
    
    with tab1:
        st.header("Dashboard de Señales")
        
        # Cargar datos
        with st.spinner("Cargando datos de mercado..."):
            data = load_crypto_data()
        
        if not data:
            st.error("No se pudieron cargar los datos de mercado. Verifica la conectividad.")
            return
        
        # Calcular señales
        with st.spinner("Calculando señales de trading..."):
            signals = calculate_signals(data)
        
        # Mostrar resumen
        col1, col2, col3, col4 = st.columns(4)
        
        buy_signals = sum(1 for s in signals.values() if s and s['signal'] == 'COMPRA')
        sell_signals = sum(1 for s in signals.values() if s and s['signal'] == 'VENTA')
        neutral_signals = sum(1 for s in signals.values() if s and s['signal'] == 'NEUTRO')
        
        col1.metric("🟢 Señales de Compra", buy_signals)
        col2.metric("🔴 Señales de Venta", sell_signals)
        col3.metric("🟡 Señales Neutras", neutral_signals)
        col4.metric("📊 Total Analizadas", len(signals))
        
        # Gráfico de índices
        st.plotly_chart(create_index_chart(signals), use_container_width=True)
        
        # Tarjetas de señales
        st.subheader("Señales Detalladas")
        
        cols = st.columns(2)
        for i, (symbol, signal_data) in enumerate(signals.items()):
            with cols[i % 2]:
                with st.container():
                    display_signal_card(symbol, signal_data)
                    
                    # Expandir para ver métricas
                    with st.expander(f"Ver métricas de {symbol.upper()}"):
                        display_metrics_breakdown(symbol, signal_data)
    
    with tab2:
        st.header("Análisis Detallado")
        
        selected_crypto = st.selectbox(
            "Selecciona una criptomoneda para análisis detallado:",
            options=list(CRYPTO_SYMBOLS),
            format_func=lambda x: x.upper()
        )
        
        if selected_crypto in data:
            df = data[selected_crypto]
            signal_data = signals.get(selected_crypto)
            
            if signal_data:
                # Información de la señal
                col1, col2, col3 = st.columns(3)
                col1.metric("Señal Actual", signal_data['signal'])
                col2.metric("Índice", f"{signal_data['index']:.2f}")
                col3.metric("Confianza", f"{signal_data['confidence']:.1f}%")
                
                # Gráfico de precio con indicadores
                st.subheader("Gráfico de Precio e Indicadores")
                
                # Aquí podrías agregar gráficos más detallados
                st.info("Gráficos detallados se implementarán en la siguiente iteración")
    
    with tab3:
        st.header("Backtesting")
        st.info("Funcionalidad de backtesting en desarrollo")
        
        # Placeholder para backtesting
        st.write("Esta sección mostrará:")
        st.write("- Rendimiento histórico del modelo")
        st.write("- Estadísticas de aciertos")
        st.write("- Curva de equity")
        st.write("- Métricas de riesgo")
    
    with tab4:
        st.header("Información del Modelo")
        
        st.subheader("🎯 Objetivo")
        st.write("""
        Este modelo de trading combina múltiples indicadores técnicos para generar señales 
        de compra, venta o mantener posición neutra en criptomonedas cada 6 horas.
        """)
        
        st.subheader("📊 Métricas Utilizadas")
        
        weights = MODEL_CONFIG['weights']
        
        st.write("**Tier 1 - Alto Impacto:**")
        st.write(f"- RSI (14): {weights['rsi']*100:.1f}% - Identificación de reversiones")
        st.write(f"- MACD: {weights['macd']*100:.1f}% - Análisis de tendencias")
        
        st.write("**Tier 2 - Impacto Medio:**")
        st.write(f"- Bollinger Bands: {weights['bb_position']*100:.1f}% - Volatilidad normalizada")
        st.write(f"- VWAP: {weights['vwap']*100:.1f}% - Soporte/resistencia institucional")
        st.write(f"- Stochastic %K: {weights['stoch_k']*100:.1f}% - Confirmación de momentum")
        
        st.write("**Tier 3 - Impacto Bajo:**")
        st.write(f"- OBV: {weights['obv']*100:.1f}% - Flujo de dinero")
        st.write(f"- ATR: {weights['atr']*100:.1f}% - Volatilidad")
        st.write(f"- ROC: {weights['roc']*100:.1f}% - Momentum puro")
        
        st.subheader("⚡ Especificaciones Técnicas")
        st.write(f"- **Frecuencia de análisis:** Cada 6 horas")
        st.write(f"- **Criptomonedas:** {', '.join([c.upper() for c in CRYPTO_SYMBOLS])}")
        st.write(f"- **Datos históricos:** {MODEL_CONFIG['historical_days']} días")
        st.write(f"- **Fuente de datos:** CoinGecko API")

if __name__ == "__main__":
    main()
