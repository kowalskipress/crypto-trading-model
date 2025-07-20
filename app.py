import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

# Configuración de la página
st.set_page_config(
    page_title="Crypto Trading Signals - Complete Model",
    page_icon="📈",
    layout="wide"
)

# Suprimir warnings innecesarios
import warnings
warnings.filterwarnings('ignore')

class CompleteCryptoModel:
    def __init__(self):
        self.coins = {
            'bitcoin': 'BTC',
            'ethereum': 'ETH', 
            'binancecoin': 'BNB',
            'ripple': 'XRP'
        }
        
        # Pesos originales del modelo completo
        self.weights = {
            'rsi': 0.20,      # 20%
            'macd': 0.25,     # 25%
            'bb': 0.15,       # 15%
            'ema': 0.20,      # 20%
            'volume': 0.10,   # 10%
            'volatility': 0.10 # 10%
        }
        
    @st.cache_data(ttl=300)
    def fetch_current_data(_self, coin_id):
        """Obtiene datos de precio actual"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if coin_id in data:
                    return data[coin_id]
            elif response.status_code == 429:
                st.warning("⏳ Rate limit alcanzado")
                time.sleep(5)
            
            return None
                
        except Exception as e:
            return None
    
    @st.cache_data(ttl=600)  
    def fetch_historical_data(_self, coin_id, days=30):
        """Obtiene datos históricos completos"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
            }
            
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                if 'prices' in data and 'total_volumes' in data:
                    # Crear DataFrame completo
                    prices_data = data['prices']
                    volumes_data = data['total_volumes']
                    
                    df = pd.DataFrame({
                        'timestamp': [p[0] for p in prices_data],
                        'price': [p[1] for p in prices_data],
                        'volume': [v[1] for v in volumes_data]
                    })
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    df.sort_index(inplace=True)
                    
                    return df
            
            return None
                
        except Exception as e:
            return None
    
    def calculate_rsi(self, prices, period=14):
        """Calcula RSI"""
        if len(prices) < period + 1:
            return [50] * len(prices)
            
        deltas = prices.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calcula MACD"""
        if len(prices) < slow:
            return pd.Series([0] * len(prices), index=prices.index), pd.Series([0] * len(prices), index=prices.index)
            
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        
        return macd_line.fillna(0), signal_line.fillna(0)
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calcula Bollinger Bands"""
        if len(prices) < period:
            return prices, prices, prices
            
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper.fillna(prices), lower.fillna(prices), sma.fillna(prices)
    
    def calculate_ema(self, prices, period=12):
        """Calcula EMA"""
        if len(prices) < period:
            return prices
        return prices.ewm(span=period).mean()
    
    def calculate_all_indicators(self, df):
        """Calcula todos los indicadores técnicos"""
        if df is None or len(df) < 20:
            return None
            
        try:
            # RSI
            df['rsi'] = self.calculate_rsi(df['price'])
            
            # MACD
            df['macd'], df['macd_signal'] = self.calculate_macd(df['price'])
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_upper'], df['bb_lower'], df['bb_middle'] = self.calculate_bollinger_bands(df['price'])
            df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # EMAs
            df['ema_12'] = self.calculate_ema(df['price'], 12)
            df['ema_26'] = self.calculate_ema(df['price'], 26)
            
            # Volume analysis
            df['volume_sma'] = df['volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Volatility
            df['returns'] = df['price'].pct_change()
            df['volatility'] = df['returns'].rolling(window=10).std() * np.sqrt(365) * 100
            
            return df
            
        except Exception as e:
            st.error(f"Error calculando indicadores: {str(e)}")
            return None
    
    def generate_signals(self, df, current_data):
        """Genera señales usando las 6 métricas originales"""
        if df is None or len(df) == 0:
            return 0, 50, 0, 0, 0, 0, 0  # neutral signals
            
        try:
            latest = df.iloc[-1]
            
            # 1. RSI Signal (20% weight)
            rsi_val = latest['rsi']
            if rsi_val < 30:
                rsi_signal = 100  # Strong buy
            elif rsi_val > 70:
                rsi_signal = -100  # Strong sell
            else:
                rsi_signal = (50 - rsi_val) * 2  # Linear scaling
            
            # 2. MACD Signal (25% weight)
            macd_val = latest['macd']
            macd_signal_val = latest['macd_signal']
            if macd_val > macd_signal_val:
                macd_signal = min(latest['macd_histogram'] * 1000, 100)
            else:
                macd_signal = max(latest['macd_histogram'] * 1000, -100)
            
            # 3. Bollinger Bands Signal (15% weight)
            bb_pos = latest['bb_position']
            if bb_pos < 0.2:
                bb_signal = 100  # Near lower band = Buy
            elif bb_pos > 0.8:
                bb_signal = -100  # Near upper band = Sell
            else:
                bb_signal = (0.5 - bb_pos) * 200  # Linear scaling
            
            # 4. EMA Trend Signal (20% weight)
            ema_12 = latest['ema_12']
            ema_26 = latest['ema_26']
            if ema_12 > ema_26:
                ema_signal = min((ema_12 / ema_26 - 1) * 10000, 100)
            else:
                ema_signal = max((ema_12 / ema_26 - 1) * 10000, -100)
            
            # 5. Volume Signal (10% weight)
            vol_ratio = latest['volume_ratio']
            if vol_ratio > 1.5:
                volume_signal = 50  # High volume confirmation
            elif vol_ratio < 0.5:
                volume_signal = -50  # Low volume warning
            else:
                volume_signal = 0
            
            # 6. Volatility Signal (10% weight)
            volatility = latest['volatility']
            vol_percentile = (df['volatility'].tail(30).rank(pct=True).iloc[-1]) if len(df) >= 30 else 0.5
            if vol_percentile > 0.8:
                volatility_signal = -30  # High volatility = caution
            elif vol_percentile < 0.2:
                volatility_signal = 30  # Low volatility = opportunity
            else:
                volatility_signal = 0
            
            # Calculate weighted final signal
            final_signal = (
                rsi_signal * self.weights['rsi'] +
                macd_signal * self.weights['macd'] +
                bb_signal * self.weights['bb'] +
                ema_signal * self.weights['ema'] +
                volume_signal * self.weights['volume'] +
                volatility_signal * self.weights['volatility']
            )
            
            return final_signal, rsi_val, macd_signal, bb_signal, ema_signal, volume_signal, volatility_signal
            
        except Exception as e:
            st.error(f"Error generando señales: {str(e)}")
            return 0, 50, 0, 0, 0, 0, 0

def main():
    st.title("🚀 Crypto Trading Model - Complete Analysis")
    st.markdown("**Modelo completo con 6 métricas técnicas ponderadas**")
    
    # Mostrar pesos del modelo
    st.sidebar.header("⚙️ Configuración del Modelo")
    
    model = CompleteCryptoModel()
    
    # Mostrar pesos
    st.sidebar.subheader("📊 Pesos de Indicadores")
    for indicator, weight in model.weights.items():
        st.sidebar.write(f"**{indicator.upper()}**: {weight*100:.0f}%")
    
    if st.sidebar.button("🔄 Actualizar Datos", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    # Obtener datos
    st.header("📊 Señales de Trading Completas")
    
    results = []
    error_count = 0
    
    with st.spinner("Obteniendo datos completos..."):
        for i, (coin_id, symbol) in enumerate(model.coins.items()):
            with st.spinner(f"Analizando {symbol}... ({i+1}/4)"):
                # Pausa entre requests
                if i > 0:
                    time.sleep(3 + i)
                
                # Datos actuales
                current_data = None
                for attempt in range(3):
                    current_data = model.fetch_current_data(coin_id)
                    if current_data is not None:
                        break
                    elif attempt < 2:
                        st.info(f"🔄 Reintentando {symbol}... ({attempt + 2}/3)")
                        time.sleep(3)
                
                if current_data is None:
                    error_count += 1
                    st.warning(f"❌ No se pudieron obtener datos para {symbol}")
                    continue
                
                # Datos históricos
                historical_df = model.fetch_historical_data(coin_id, 30)
                
                if historical_df is not None:
                    # Calcular indicadores
                    historical_df = model.calculate_all_indicators(historical_df)
                    
                    if historical_df is not None:
                        # Generar señales
                        final_signal, rsi_val, macd_sig, bb_sig, ema_sig, vol_sig, vol_penalty = model.generate_signals(historical_df, current_data)
                        
                        # Clasificar señal
                        if final_signal > 25:
                            signal_class = "COMPRA"
                            signal_emoji = "🟢"
                        elif final_signal < -25:
                            signal_class = "VENTA" 
                            signal_emoji = "🔴"
                        else:
                            signal_class = "NEUTRO"
                            signal_emoji = "⚪"
                        
                        # Calcular contribuciones individuales
                        rsi_contrib = rsi_val * model.weights['rsi'] if abs(rsi_val) < 200 else 0
                        macd_contrib = macd_sig * model.weights['macd']
                        bb_contrib = bb_sig * model.weights['bb'] 
                        ema_contrib = ema_sig * model.weights['ema']
                        vol_contrib = vol_sig * model.weights['volume']
                        volatility_contrib = vol_penalty * model.weights['volatility']
                        
                        results.append({
                            'Crypto': symbol,
                            'Precio': f"${current_data['usd']:,.2f}",
                            'Cambio 24h': f"{current_data.get('usd_24h_change', 0):.2f}%",
                            'Señal': f"{signal_emoji} {signal_class}",
                            'Score Final': f"{final_signal:.1f}",
                            'RSI': f"{rsi_val:.1f}",
                            'RSI Contrib': f"{rsi_contrib:.1f}",
                            'MACD Contrib': f"{macd_contrib:.1f}",
                            'BB Contrib': f"{bb_contrib:.1f}",
                            'EMA Contrib': f"{ema_contrib:.1f}",
                            'Vol Contrib': f"{vol_contrib:.1f}",
                            'Volat Contrib': f"{volatility_contrib:.1f}",
                            'Confianza': f"{min(abs(final_signal), 100):.1f}%"
                        })
                        
                        st.success(f"✅ {symbol} análisis completo")
                    else:
                        st.warning(f"⚠️ {symbol}: Error en cálculo de indicadores")
                else:
                    st.warning(f"⚠️ {symbol}: Sin datos históricos suficientes")
    
    # Mostrar resultados
    if results:
        st.success(f"🎉 Análisis completo para {len(results)} criptomonedas")
        
        # Tabla principal
        df = pd.DataFrame(results)
        
        # Aplicar estilos
        def style_table(df):
            def color_signals(val):
                if '🟢' in val:
                    return 'background-color: #d4edda; color: #155724; font-weight: bold'
                elif '🔴' in val:
                    return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
                else:
                    return 'background-color: #f8f9fa; color: #495057; font-weight: bold'
            
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
            
            def color_contrib(val):
                try:
                    contrib = float(val)
                    if contrib > 5:
                        return 'color: #28a745; font-weight: bold'
                    elif contrib < -5:
                        return 'color: #dc3545; font-weight: bold'
                    else:
                        return 'color: #6c757d'
                except:
                    return ''
            
            styled = df.style.applymap(color_signals, subset=['Señal'])
            styled = styled.applymap(color_score, subset=['Score Final'])
            
            contrib_cols = [col for col in df.columns if 'Contrib' in col]
            for col in contrib_cols:
                styled = styled.applymap(color_contrib, subset=[col])
                
            return styled
        
        st.dataframe(style_table(df), use_container_width=True)
        
        # Explicación de contribuciones
        st.subheader("📈 Explicación de Contribuciones")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Pesos de Indicadores:**
            - 🔴 **RSI (20%)**: Sobrecompra/Sobreventa
            - 🟠 **MACD (25%)**: Momentum y cambios de tendencia
            - 🟡 **Bollinger Bands (15%)**: Posición relativa del precio
            """)
        
        with col2:
            st.markdown("""
            **Pesos de Indicadores:**
            - 🔵 **EMA Trend (20%)**: Dirección de tendencia
            - 🟣 **Volume (10%)**: Confirmación de movimientos
            - ⚫ **Volatility (10%)**: Gestión de riesgo
            """)
        
        # Gráficos detallados
        st.header("📊 Análisis Técnico Detallado")
        
        selected_coin = st.selectbox(
            "Selecciona criptomoneda para análisis:",
            options=list(model.coins.keys()),
            format_func=lambda x: model.coins[x]
        )
        
        # Crear gráfico para la moneda seleccionada
        historical_df = model.fetch_historical_data(selected_coin, 30)
        if historical_df is not None:
            historical_df = model.calculate_all_indicators(historical_df)
            
            if historical_df is not None:
                fig = make_subplots(
                    rows=4, cols=1,
                    subplot_titles=(
                        f'{model.coins[selected_coin]} - Precio y Bollinger Bands',
                        'RSI (Peso: 20%)',
                        'MACD (Peso: 25%)',
                        'Contribuciones por Indicador'
                    ),
                    vertical_spacing=0.08,
                    row_heights=[0.4, 0.2, 0.2, 0.2]
                )
                
                # Precio y Bollinger Bands
                fig.add_trace(go.Scatter(
                    x=historical_df.index, y=historical_df['price'],
                    name='Precio', line=dict(color='blue', width=2)
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=historical_df.index, y=historical_df['bb_upper'],
                    name='BB Superior', line=dict(color='red', dash='dash')
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=historical_df.index, y=historical_df['bb_lower'],
                    name='BB Inferior', line=dict(color='red', dash='dash'),
                    fill='tonexty', fillcolor='rgba(255,0,0,0.1)'
                ), row=1, col=1)
                
                # RSI
                fig.add_trace(go.Scatter(
                    x=historical_df.index, y=historical_df['rsi'],
                    name='RSI', line=dict(color='purple')
                ), row=2, col=1)
                
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # MACD
                fig.add_trace(go.Scatter(
                    x=historical_df.index, y=historical_df['macd'],
                    name='MACD', line=dict(color='blue')
                ), row=3, col=1)
                
                fig.add_trace(go.Scatter(
                    x=historical_df.index, y=historical_df['macd_signal'],
                    name='Signal', line=dict(color='red')
                ), row=3, col=1)
                
                # Contribuciones (ejemplo con datos más recientes)
                recent_data = historical_df.tail(10)
                contributions = []
                dates = []
                
                for idx, row in recent_data.iterrows():
                    final_signal, rsi_val, macd_sig, bb_sig, ema_sig, vol_sig, vol_penalty = model.generate_signals(
                        historical_df.loc[:idx], None
                    )
                    contributions.append(final_signal)
                    dates.append(idx)
                
                fig.add_trace(go.Scatter(
                    x=dates, y=contributions,
                    name='Score Final', line=dict(color='black', width=3)
                ), row=4, col=1)
                
                fig.add_hline(y=25, line_dash="dash", line_color="green", row=4, col=1)
                fig.add_hline(y=-25, line_dash="dash", line_color="red", row=4, col=1)
                
                fig.update_layout(height=1000, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
        
        # Métricas resumen
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            buy_signals = sum(1 for r in results if 'COMPRA' in r['Señal'])
            st.metric("🟢 Señales Compra", buy_signals)
        
        with col2:
            sell_signals = sum(1 for r in results if 'VENTA' in r['Señal'])
            st.metric("🔴 Señales Venta", sell_signals)
            
        with col3:
            neutral_signals = sum(1 for r in results if 'NEUTRO' in r['Señal'])
            st.metric("⚪ Señales Neutras", neutral_signals)
            
        with col4:
            avg_confidence = np.mean([float(r['Confianza'].replace('%', '')) for r in results])
            st.metric("📊 Confianza Promedio", f"{avg_confidence:.1f}%")
    
    else:
        st.error("❌ No se pudieron obtener datos de ninguna criptomoneda.")
        st.info("🔄 Intenta actualizar en unos minutos.")
    
    # Info
    st.info(f"📅 Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **🎯 Modelo Completo - 6 Indicadores Técnicos:**
    
    **Señales de Entrada/Salida:**
    - Score > +25: 🟢 **COMPRA** (Señal alcista fuerte)
    - Score < -25: 🔴 **VENTA** (Señal bajista fuerte)  
    - Score -25 a +25: ⚪ **NEUTRO** (Sin señal clara)
    
    **Metodología de Ponderación:**
    - Cada indicador aporta una señal de -100 a +100
    - Se multiplica por su peso específico
    - Score final = suma ponderada de todas las contribuciones
    
    **⚠️ Disclaimer:** Modelo para fines educativos. No constituye asesoramiento financiero.
    """)

if __name__ == "__main__":
    main()
