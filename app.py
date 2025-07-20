import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime
import time

# Configuración de la página
st.set_page_config(
    page_title="Crypto Trading Signals",
    page_icon="📈",
    layout="wide"
)

# Suprimir warnings innecesarios
import warnings
warnings.filterwarnings('ignore')

class SimpleCryptoModel:
    def __init__(self):
        self.coins = {
            'bitcoin': 'BTC',
            'ethereum': 'ETH', 
            'binancecoin': 'BNB',
            'ripple': 'XRP'
        }
        
    @st.cache_data(ttl=300)
    def fetch_data(_self, coin_id):
        """Obtiene datos de precio actual y básicos"""
        try:
            # API simple de CoinGecko
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()[coin_id]
            else:
                return None
                
        except Exception as e:
            st.error(f"Error obteniendo datos para {coin_id}: {str(e)}")
            return None
    
    @st.cache_data(ttl=600)  
    def fetch_historical(_self, coin_id, days=7):
        """Obtiene datos históricos simples"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily'
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                prices = [price[1] for price in data['prices']]
                timestamps = [price[0] for price in data['prices']]
                return prices, timestamps
            else:
                return None, None
                
        except Exception as e:
            return None, None
    
    def calculate_simple_rsi(self, prices, period=14):
        """Cálculo simple de RSI"""
        if len(prices) < period + 1:
            return 50
            
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_simple_signal(self, current_data, prices):
        """Genera señal simple basada en precio y RSI"""
        if not current_data or not prices:
            return "NEUTRO", 0, 50
            
        price = current_data['usd']
        change_24h = current_data.get('usd_24h_change', 0)
        
        # Calcular RSI simple
        rsi = self.calculate_simple_rsi(prices)
        
        # Señal basada en RSI y cambio 24h
        signal_score = 0
        
        # RSI signals
        if rsi < 30:
            signal_score += 40  # Oversold = Buy signal
        elif rsi > 70:
            signal_score -= 40  # Overbought = Sell signal
        else:
            signal_score += (50 - rsi) * 0.8  # Linear scaling
            
        # Momentum signals
        if change_24h > 5:
            signal_score += 20
        elif change_24h < -5:
            signal_score -= 20
        else:
            signal_score += change_24h * 2
            
        # Price trend (simple moving average)
        if len(prices) >= 3:
            recent_avg = sum(prices[-3:]) / 3
            older_avg = sum(prices[-7:-4]) / 3 if len(prices) >= 7 else recent_avg
            
            if recent_avg > older_avg * 1.02:
                signal_score += 15
            elif recent_avg < older_avg * 0.98:
                signal_score -= 15
        
        # Clasificar señal
        if signal_score > 25:
            signal_class = "COMPRA"
        elif signal_score < -25:
            signal_class = "VENTA"
        else:
            signal_class = "NEUTRO"
            
        confidence = min(abs(signal_score), 100)
        
        return signal_class, signal_score, rsi

def main():
    st.title("🚀 Crypto Trading Signals")
    st.markdown("**Modelo simplificado de señales de trading**")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuración")
    
    if st.sidebar.button("🔄 Actualizar", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    # Instanciar modelo
    model = SimpleCryptoModel()
    
    # Obtener datos
    st.header("📊 Señales Actuales")
    
    results = []
    
    with st.spinner("Obteniendo datos..."):
        for coin_id, symbol in model.coins.items():
            # Datos actuales
            current_data = model.fetch_data(coin_id)
            
            # Datos históricos
            prices, timestamps = model.fetch_historical(coin_id, 14)
            
            if current_data:
                # Generar señal
                signal_class, score, rsi = model.generate_simple_signal(current_data, prices)
                
                # Emoji para señal
                signal_emoji = {
                    'COMPRA': '🟢',
                    'VENTA': '🔴',
                    'NEUTRO': '⚪'
                }.get(signal_class, '⚪')
                
                results.append({
                    'Crypto': symbol,
                    'Precio': f"${current_data['usd']:,.2f}",
                    'Cambio 24h': f"{current_data.get('usd_24h_change', 0):.2f}%",
                    'Señal': f"{signal_emoji} {signal_class}",
                    'Score': f"{score:.1f}",
                    'RSI': f"{rsi:.1f}",
                    'Volumen 24h': f"${current_data.get('usd_24h_vol', 0):,.0f}"
                })
    
    if results:
        # Mostrar tabla
        df = pd.DataFrame(results)
        
        # Aplicar colores
        def style_table(df):
            def color_signals(val):
                if '🟢' in val:
                    return 'background-color: #d4edda; color: #155724'
                elif '🔴' in val:
                    return 'background-color: #f8d7da; color: #721c24'
                else:
                    return 'background-color: #f8f9fa; color: #495057'
            
            def color_change(val):
                try:
                    change = float(val.replace('%', ''))
                    if change > 0:
                        return 'color: green; font-weight: bold'
                    elif change < 0:
                        return 'color: red; font-weight: bold'
                    else:
                        return 'color: gray'
                except:
                    return ''
            
            return df.style.applymap(color_signals, subset=['Señal']).applymap(color_change, subset=['Cambio 24h'])
        
        st.dataframe(style_table(df), use_container_width=True)
        
        # Gráfico simple
        st.header("📈 Análisis de Precios")
        
        selected_coin = st.selectbox(
            "Selecciona criptomoneda:",
            options=list(model.coins.keys()),
            format_func=lambda x: model.coins[x]
        )
        
        # Obtener datos históricos para gráfico
        prices, timestamps = model.fetch_historical(selected_coin, 30)
        
        if prices and timestamps:
            # Convertir timestamps
            dates = [datetime.fromtimestamp(ts/1000) for ts in timestamps]
            
            # Crear gráfico
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, 
                y=prices,
                name=model.coins[selected_coin],
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title=f"{model.coins[selected_coin]} - Últimos 30 días",
                xaxis_title="Fecha",
                yaxis_title="Precio (USD)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Métricas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            buy_signals = sum(1 for r in results if 'COMPRA' in r['Señal'])
            st.metric("Señales de Compra", buy_signals)
        
        with col2:
            sell_signals = sum(1 for r in results if 'VENTA' in r['Señal'])
            st.metric("Señales de Venta", sell_signals)
            
        with col3:
            neutral_signals = sum(1 for r in results if 'NEUTRO' in r['Señal'])
            st.metric("Señales Neutras", neutral_signals)
        
    else:
        st.error("No se pudieron obtener datos. Intenta actualizar en unos segundos.")
    
    # Info
    st.info(f"📅 Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **📊 Metodología Simplificada:**
    - RSI < 30: Señal de compra (sobreventa)
    - RSI > 70: Señal de venta (sobrecompra)  
    - Cambio 24h: Momentum del precio
    - Tendencia: Promedio móvil simple
    
    **⚠️ Disclaimer:** Solo para fines educativos. No constituye asesoramiento financiero.
    """)

if __name__ == "__main__":
    main()
