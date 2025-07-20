import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from datetime import datetime, timedelta
import time
import sqlite3
import json

# Configuración de la página
st.set_page_config(
    page_title="Crypto Trading Signal Model",
    page_icon="📈",
    layout="wide"
)

class CryptoTradingModel:
    def __init__(self):
        self.coins = ['bitcoin', 'ethereum', 'binancecoin', 'ripple']
        self.coin_symbols = {
            'bitcoin': 'BTC',
            'ethereum': 'ETH', 
            'binancecoin': 'BNB',
            'ripple': 'XRP'
        }
        self.weights = {
            'rsi': 0.20,
            'macd': 0.25,
            'bb': 0.15,
            'ema': 0.20,
            'volume': 0.10,
            'volatility': 0.10
        }
        
    @st.cache_data(ttl=300)  # Cache por 5 minutos
    def fetch_price_data(_self, coin_id, days=30):
        """Obtiene datos históricos de precios"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Convertir a DataFrame
                prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
                volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
                
                prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
                volumes['timestamp'] = pd.to_datetime(volumes['timestamp'], unit='ms')
                
                df = pd.merge(prices, volumes, on='timestamp')
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                return df
            else:
                st.error(f"Error al obtener datos para {coin_id}: {response.status_code}")
                return None
                
        except Exception as e:
            st.error(f"Error de conexión para {coin_id}: {str(e)}")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calcula todos los indicadores técnicos"""
        if df is None or len(df) < 20:
            return None
            
        try:
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['price'], window=14).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['price'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['price'], window=20)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # EMAs
            df['ema_12'] = ta.trend.EMAIndicator(df['price'], window=12).ema_indicator()
            df['ema_26'] = ta.trend.EMAIndicator(df['price'], window=26).ema_indicator()
            
            # Volume
            df['volume_sma'] = df['volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Volatilidad
            df['returns'] = df['price'].pct_change()
            df['volatility'] = df['returns'].rolling(window=10).std() * np.sqrt(365)
            
            return df
            
        except Exception as e:
            st.error(f"Error calculando indicadores: {str(e)}")
            return None
    
    def generate_signals(self, df):
        """Genera señales de trading basadas en indicadores ponderados"""
        if df is None:
            return None
            
        try:
            # Normalizar indicadores a escala -100 a +100
            signals = pd.DataFrame(index=df.index)
            
            # RSI Signal (30-70 range optimal)
            rsi_signal = np.where(df['rsi'] < 30, 100,  # Oversold = Buy
                         np.where(df['rsi'] > 70, -100,  # Overbought = Sell
                                 (50 - df['rsi']) * 2))  # Linear scaling
            signals['rsi_signal'] = rsi_signal
            
            # MACD Signal
            macd_signal = np.where(df['macd'] > df['macd_signal'], 
                                  np.minimum(df['macd_histogram'] * 1000, 100),
                                  np.maximum(df['macd_histogram'] * 1000, -100))
            signals['macd_signal'] = macd_signal
            
            # Bollinger Bands Signal
            bb_position = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            bb_signal = np.where(bb_position < 0.2, 100,  # Near lower band = Buy
                        np.where(bb_position > 0.8, -100,  # Near upper band = Sell
                                (0.5 - bb_position) * 200))  # Linear scaling
            signals['bb_signal'] = bb_signal
            
            # EMA Trend Signal
            ema_signal = np.where(df['ema_12'] > df['ema_26'], 
                                 np.minimum((df['ema_12'] / df['ema_26'] - 1) * 10000, 100),
                                 np.maximum((df['ema_12'] / df['ema_26'] - 1) * 10000, -100))
            signals['ema_signal'] = ema_signal
            
            # Volume Signal
            volume_signal = np.where(df['volume_ratio'] > 1.5, 50,
                           np.where(df['volume_ratio'] < 0.5, -50, 0))
            signals['volume_signal'] = volume_signal
            
            # Volatility Signal (alta volatilidad = precaución)
            vol_percentile = df['volatility'].rolling(30).rank(pct=True)
            volatility_signal = np.where(vol_percentile > 0.8, -30,
                               np.where(vol_percentile < 0.2, 30, 0))
            signals['volatility_signal'] = volatility_signal
            
            # Señal ponderada final
            final_signal = (
                signals['rsi_signal'] * self.weights['rsi'] +
                signals['macd_signal'] * self.weights['macd'] +
                signals['bb_signal'] * self.weights['bb'] +
                signals['ema_signal'] * self.weights['ema'] +
                signals['volume_signal'] * self.weights['volume'] +
                signals['volatility_signal'] * self.weights['volatility']
            )
            
            # Clasificar señales
            signal_class = np.where(final_signal > 25, 'COMPRA',
                           np.where(final_signal < -25, 'VENTA', 'NEUTRO'))
            
            # Nivel de confianza
            confidence = np.minimum(np.abs(final_signal), 100)
            
            # Agregar al DataFrame original
            df['final_signal'] = final_signal
            df['signal_class'] = signal_class
            df['confidence'] = confidence
            
            # Agregar señales individuales para análisis
            for col in signals.columns:
                df[col] = signals[col]
                
            return df
            
        except Exception as e:
            st.error(f"Error generando señales: {str(e)}")
            return None

    def create_summary_table(self, all_data):
        """Crea tabla resumen con señales actuales"""
        summary_data = []
        
        for coin_id in self.coins:
            if coin_id in all_data and all_data[coin_id] is not None:
                df = all_data[coin_id]
                latest = df.iloc[-1]
                
                # Emoji para señal
                signal_emoji = {
                    'COMPRA': '🟢',
                    'VENTA': '🔴',
                    'NEUTRO': '⚪'
                }.get(latest['signal_class'], '⚪')
                
                summary_data.append({
                    'Crypto': self.coin_symbols[coin_id],
                    'Precio': f"${latest['price']:,.2f}",
                    'Señal': f"{signal_emoji} {latest['signal_class']}",
                    'Score': f"{latest['final_signal']:.1f}",
                    'Confianza': f"{latest['confidence']:.1f}%",
                    'RSI': f"{latest['rsi']:.1f}",
                    'MACD': f"{latest['macd']:.4f}",
                    'BB Pos': f"{((latest['price'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower']) * 100):.1f}%",
                    'Vol Ratio': f"{latest['volume_ratio']:.2f}",
                    'Volatilidad': f"{latest['volatility']:.1f}%"
                })
        
        return pd.DataFrame(summary_data)

def main():
    st.title("🚀 Crypto Trading Signal Model")
    st.markdown("**Modelo avanzado de señales de trading para criptomonedas**")
    
    # Sidebar para configuración
    st.sidebar.header("⚙️ Configuración")
    
    # Selector de días históricos
    days = st.sidebar.slider("Días de datos históricos", 7, 90, 30)
    
    # Botón de actualización
    if st.sidebar.button("🔄 Actualizar Datos", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    # Instanciar modelo
    model = CryptoTradingModel()
    
    # Obtener datos para todas las criptomonedas
    with st.spinner("Obteniendo datos de mercado..."):
        all_data = {}
        
        for coin_id in model.coins:
            with st.spinner(f"Procesando {model.coin_symbols[coin_id]}..."):
                # Obtener datos
                df = model.fetch_price_data(coin_id, days)
                
                if df is not None:
                    # Calcular indicadores
                    df = model.calculate_technical_indicators(df)
                    
                    if df is not None:
                        # Generar señales
                        df = model.generate_signals(df)
                        all_data[coin_id] = df
    
    if not all_data:
        st.error("No se pudieron obtener datos de ninguna criptomoneda. Revisa tu conexión a internet.")
        return
    
    # Mostrar tabla resumen
    st.header("📊 Señales de Trading Actuales")
    summary_df = model.create_summary_table(all_data)
    
    # Aplicar estilos a la tabla
    def style_signal_table(df):
        def color_signals(val):
            if '🟢' in val:
                return 'background-color: #d4edda; color: #155724'
            elif '🔴' in val:
                return 'background-color: #f8d7da; color: #721c24'
            else:
                return 'background-color: #f8f9fa; color: #495057'
        
        def color_score(val):
            try:
                score = float(val)
                if score > 25:
                    return 'background-color: #d4edda; color: #155724; font-weight: bold'
                elif score < -25:
                    return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
                else:
                    return 'color: #495057'
            except:
                return ''
        
        return df.style.applymap(color_signals, subset=['Señal']).applymap(color_score, subset=['Score'])
    
    st.dataframe(style_signal_table(summary_df), use_container_width=True)
    
    # Mostrar última actualización
    st.info(f"📅 Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Gráficos detallados
    st.header("📈 Análisis Técnico Detallado")
    
    # Selector de criptomoneda para gráfico
    selected_coin = st.selectbox(
        "Selecciona criptomoneda para análisis detallado:",
        options=list(model.coin_symbols.keys()),
        format_func=lambda x: model.coin_symbols[x]
    )
    
    if selected_coin in all_data:
        df = all_data[selected_coin]
        
        # Crear subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=(
                f'{model.coin_symbols[selected_coin]} - Precio y Bollinger Bands',
                'RSI',
                'MACD', 
                'Señal Final'
            ),
            vertical_spacing=0.08,
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # Precio y Bollinger Bands
        fig.add_trace(go.Scatter(
            x=df.index, y=df['price'],
            name='Precio', line=dict(color='blue', width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['bb_upper'],
            name='BB Superior', line=dict(color='red', dash='dash')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['bb_lower'],
            name='BB Inferior', line=dict(color='red', dash='dash'),
            fill='tonexty', fillcolor='rgba(255,0,0,0.1)'
        ), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(
            x=df.index, y=df['rsi'],
            name='RSI', line=dict(color='purple')
        ), row=2, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(
            x=df.index, y=df['macd'],
            name='MACD', line=dict(color='blue')
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['macd_signal'],
            name='Signal', line=dict(color='red')
        ), row=3, col=1)
        
        # Señal Final
        colors = ['red' if x < -25 else 'green' if x > 25 else 'gray' for x in df['final_signal']]
        fig.add_trace(go.Scatter(
            x=df.index, y=df['final_signal'],
            name='Señal Final', 
            mode='markers+lines',
            marker=dict(color=colors, size=6),
            line=dict(color='black')
        ), row=4, col=1)
        
        fig.add_hline(y=25, line_dash="dash", line_color="green", row=4, col=1)
        fig.add_hline(y=-25, line_dash="dash", line_color="red", row=4, col=1)
        
        fig.update_layout(height=800, showlegend=True)
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas adicionales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Precio Actual", 
                f"${df['price'].iloc[-1]:,.2f}",
                delta=f"{((df['price'].iloc[-1] / df['price'].iloc[-2] - 1) * 100):.2f}%"
            )
        
        with col2:
            st.metric(
                "Señal Score",
                f"{df['final_signal'].iloc[-1]:.1f}",
                delta=f"{(df['final_signal'].iloc[-1] - df['final_signal'].iloc[-2]):.1f}"
            )
        
        with col3:
            st.metric(
                "RSI",
                f"{df['rsi'].iloc[-1]:.1f}",
                delta=f"{(df['rsi'].iloc[-1] - df['rsi'].iloc[-2]):.1f}"
            )
        
        with col4:
            st.metric(
                "Volatilidad",
                f"{df['volatility'].iloc[-1]:.1f}%",
                delta=f"{(df['volatility'].iloc[-1] - df['volatility'].iloc[-2]):.1f}%"
            )
    
    # Footer con información
    st.markdown("---")
    st.markdown("""
    **📋 Metodología:**
    - **RSI (20%)**: Identifica sobrecompra/sobreventa
    - **MACD (25%)**: Detecta cambios de momentum
    - **Bollinger Bands (15%)**: Señales de reversión por volatilidad
    - **EMA Trend (20%)**: Confirma dirección de tendencia
    - **Volumen (10%)**: Valida la fuerza del movimiento
    - **Volatilidad (10%)**: Ajusta riesgo según condiciones de mercado
    
    **⚠️ Disclaimer:** Este modelo es solo para fines educativos. No constituye asesoramiento financiero.
    """)

if __name__ == "__main__":
    main()
