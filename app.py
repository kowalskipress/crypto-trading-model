import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone, timedelta
import time
import json

# Configuración de la página
st.set_page_config(
    page_title="Daily Crypto Trading Signals",
    page_icon="📈",
    layout="wide"
)

# Suprimir warnings
import warnings
warnings.filterwarnings('ignore')

class DailyCryptoModel:
    def __init__(self):
        self.coins = {
            'bitcoin': 'BTC',
            'ethereum': 'ETH', 
            'binancecoin': 'BNB',
            'ripple': 'XRP'
        }
        
        # Pesos del modelo
        self.weights = {
            'rsi': 0.20,
            'macd': 0.25,
            'bb': 0.15,
            'ema': 0.20,
            'volume': 0.10,
            'volatility': 0.10
        }
        
        # Timezone Buenos Aires
        self.ba_tz = timezone(timedelta(hours=-3))
        
    def get_update_status(self):
        """Verifica si necesitamos actualizar datos (7:00 AM BA)"""
        now_ba = datetime.now(self.ba_tz)
        today_7am = now_ba.replace(hour=7, minute=0, second=0, microsecond=0)
        
        # Si es antes de las 7 AM, usar datos del día anterior
        if now_ba < today_7am:
            target_update = today_7am - timedelta(days=1)
        else:
            target_update = today_7am
            
        return target_update, now_ba
    
    @st.cache_data(ttl=3600)  # Cache por 1 hora, pero controlado por fecha
    def fetch_daily_data(_self, update_date_str):
        """Obtiene todos los datos una vez por día"""
        
        st.info(f"🔄 Actualizando datos para {update_date_str}...")
        
        all_data = {}
        success_count = 0
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        for i, (coin_id, symbol) in enumerate(_self.coins.items()):
            try:
                st.info(f"📊 Procesando {symbol}... ({i+1}/4)")
                
                # Pausa entre requests
                if i > 0:
                    time.sleep(3)
                
                # 1. Datos actuales
                current_url = "https://api.coingecko.com/api/v3/simple/price"
                current_params = {
                    'ids': coin_id,
                    'vs_currencies': 'usd',
                    'include_24hr_change': 'true',
                    'include_24hr_vol': 'true'
                }
                
                current_response = requests.get(current_url, params=current_params, headers=headers, timeout=20)
                
                if current_response.status_code != 200:
                    st.warning(f"⚠️ Error en datos actuales para {symbol}")
                    continue
                    
                current_data = current_response.json()[coin_id]
                
                # Pausa antes de datos históricos
                time.sleep(2)
                
                # 2. Datos históricos
                hist_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                hist_params = {
                    'vs_currency': 'usd',
                    'days': 30,
                    'interval': 'daily'
                }
                
                hist_response = requests.get(hist_url, params=hist_params, headers=headers, timeout=25)
                
                if hist_response.status_code != 200:
                    st.warning(f"⚠️ Error en datos históricos para {symbol}")
                    continue
                    
                hist_data = hist_response.json()
                
                # Crear DataFrame
                prices_data = hist_data['prices']
                volumes_data = hist_data['total_volumes']
                
                df = pd.DataFrame({
                    'timestamp': [p[0] for p in prices_data],
                    'price': [p[1] for p in prices_data],
                    'volume': [v[1] for v in volumes_data]
                })
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # Calcular todos los indicadores
                df = _self.calculate_all_indicators(df)
                
                if df is not None:
                    # Generar señales
                    signals = _self.generate_complete_signals(df, current_data)
                    
                    all_data[coin_id] = {
                        'symbol': symbol,
                        'current_data': current_data,
                        'historical_df': df,
                        'signals': signals,
                        'timestamp': datetime.now(_self.ba_tz).isoformat()
                    }
                    
                    success_count += 1
                    st.success(f"✅ {symbol} completado exitosamente")
                else:
                    st.warning(f"⚠️ Error calculando indicadores para {symbol}")
                    
            except Exception as e:
                st.error(f"❌ Error procesando {symbol}: {str(e)}")
                continue
        
        if success_count > 0:
            st.success(f"🎉 Datos actualizados exitosamente para {success_count}/4 criptomonedas")
            return all_data
        else:
            st.error("❌ No se pudieron actualizar los datos")
            return None
    
    def calculate_rsi(self, prices, period=14):
        """Calcula RSI"""
        if len(prices) < period + 1:
            return pd.Series([50] * len(prices), index=prices.index)
            
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
            zeros = pd.Series([0] * len(prices), index=prices.index)
            return zeros, zeros
            
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
            df['ema_12'] = df['price'].ewm(span=12).mean()
            df['ema_26'] = df['price'].ewm(span=26).mean()
            
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
    
    def generate_complete_signals(self, df, current_data):
        """Genera señales completas con contribuciones"""
        if df is None or len(df) == 0:
            return None
            
        try:
            latest = df.iloc[-1]
            
            # 1. RSI Signal (20%)
            rsi_val = latest['rsi']
            if rsi_val < 30:
                rsi_signal = 100
            elif rsi_val > 70:
                rsi_signal = -100
            else:
                rsi_signal = (50 - rsi_val) * 2
            
            # 2. MACD Signal (25%)
            macd_val = latest['macd']
            macd_signal_val = latest['macd_signal']
            if macd_val > macd_signal_val:
                macd_signal = min(latest['macd_histogram'] * 1000, 100)
            else:
                macd_signal = max(latest['macd_histogram'] * 1000, -100)
            
            # 3. Bollinger Bands Signal (15%)
            bb_pos = latest['bb_position']
            if pd.isna(bb_pos):
                bb_signal = 0
            elif bb_pos < 0.2:
                bb_signal = 100
            elif bb_pos > 0.8:
                bb_signal = -100
            else:
                bb_signal = (0.5 - bb_pos) * 200
            
            # 4. EMA Trend Signal (20%)
            ema_12 = latest['ema_12']
            ema_26 = latest['ema_26']
            if ema_12 > ema_26:
                ema_signal = min((ema_12 / ema_26 - 1) * 10000, 100)
            else:
                ema_signal = max((ema_12 / ema_26 - 1) * 10000, -100)
            
            # 5. Volume Signal (10%)
            vol_ratio = latest['volume_ratio']
            if pd.isna(vol_ratio):
                volume_signal = 0
            elif vol_ratio > 1.5:
                volume_signal = 50
            elif vol_ratio < 0.5:
                volume_signal = -50
            else:
                volume_signal = 0
            
            # 6. Volatility Signal (10%)
            volatility = latest['volatility']
            if pd.isna(volatility):
                volatility_signal = 0
            else:
                vol_percentile = (df['volatility'].tail(30).rank(pct=True).iloc[-1]) if len(df) >= 30 else 0.5
                if vol_percentile > 0.8:
                    volatility_signal = -30
                elif vol_percentile < 0.2:
                    volatility_signal = 30
                else:
                    volatility_signal = 0
            
            # Calcular contribuciones ponderadas
            rsi_contrib = rsi_signal * self.weights['rsi']
            macd_contrib = macd_signal * self.weights['macd']
            bb_contrib = bb_signal * self.weights['bb']
            ema_contrib = ema_signal * self.weights['ema']
            volume_contrib = volume_signal * self.weights['volume']
            volatility_contrib = volatility_signal * self.weights['volatility']
            
            # Score final
            final_signal = rsi_contrib + macd_contrib + bb_contrib + ema_contrib + volume_contrib + volatility_contrib
            
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
            
            return {
                'final_signal': final_signal,
                'signal_class': signal_class,
                'signal_emoji': signal_emoji,
                'confidence': min(abs(final_signal), 100),
                'rsi_val': rsi_val,
                'rsi_contrib': rsi_contrib,
                'macd_contrib': macd_contrib,
                'bb_contrib': bb_contrib,
                'ema_contrib': ema_contrib,
                'volume_contrib': volume_contrib,
                'volatility_contrib': volatility_contrib
            }
            
        except Exception as e:
            st.error(f"Error generando señales: {str(e)}")
            return None

def main():
    st.title("📊 Daily Crypto Trading Signals")
    st.markdown("**Actualización automática diaria a las 7:00 AM (Buenos Aires)**")
    
    model = DailyCryptoModel()
    
    # Verificar estado de actualización
    target_update, now_ba = model.get_update_status()
    
    # Sidebar con información
    st.sidebar.header("⏰ Estado de Actualización")
    st.sidebar.info(f"🕰️ Hora actual BA: {now_ba.strftime('%H:%M:%S')}")
    st.sidebar.info(f"📅 Última actualización: {target_update.strftime('%d/%m/%Y 07:00')}")
    
    # Próxima actualización
    next_update = target_update + timedelta(days=1)
    time_to_next = next_update - now_ba
    hours_to_next = int(time_to_next.total_seconds() // 3600)
    
    if hours_to_next > 0:
        st.sidebar.success(f"⏳ Próxima actualización en {hours_to_next}h")
    else:
        st.sidebar.warning("🔄 Actualizando datos...")
    
    # Mostrar pesos del modelo
    st.sidebar.subheader("📊 Pesos del Modelo")
    for indicator, weight in model.weights.items():
        st.sidebar.write(f"**{indicator.upper()}**: {weight*100:.0f}%")
    
    # Botón de actualización manual (solo para debugging)
    if st.sidebar.button("🔄 Forzar Actualización", help="Solo para pruebas"):
        st.cache_data.clear()
        st.rerun()
    
    # Obtener datos del día
    update_key = target_update.strftime('%Y%m%d')
    daily_data = model.fetch_daily_data(update_key)
    
    if daily_data and len(daily_data) > 0:
        # Crear tabla resumen
        st.header("📈 Señales de Trading del Día")
        
        results = []
        
        for coin_id, data in daily_data.items():
            symbol = data['symbol']
            current = data['current_data']
            signals = data['signals']
            
            if signals:
                results.append({
                    'Crypto': symbol,
                    'Precio': f"${current['usd']:,.2f}",
                    'Cambio 24h': f"{current.get('usd_24h_change', 0):.2f}%",
                    'Señal': f"{signals['signal_emoji']} {signals['signal_class']}",
                    'Score Final': f"{signals['final_signal']:.1f}",
                    'Confianza': f"{signals['confidence']:.1f}%",
                    'RSI': f"{signals['rsi_val']:.1f}",
                    'RSI Contrib': f"{signals['rsi_contrib']:.1f}",
                    'MACD Contrib': f"{signals['macd_contrib']:.1f}",
                    'BB Contrib': f"{signals['bb_contrib']:.1f}",
                    'EMA Contrib': f"{signals['ema_contrib']:.1f}",
                    'Vol Contrib': f"{signals['volume_contrib']:.1f}",
                    'Volat Contrib': f"{signals['volatility_contrib']:.1f}",
                    'Volumen 24h': f"${current.get('usd_24h_vol', 0):,.0f}"
                })
        
        if results:
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
            
            # Análisis de mercado
            st.subheader("📊 Análisis del Mercado")
            
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
            
            # Gráfico detallado
            st.header("📈 Análisis Técnico Detallado")
            
            selected_coin = st.selectbox(
                "Selecciona criptomoneda:",
                options=list(daily_data.keys()),
                format_func=lambda x: daily_data[x]['symbol']
            )
            
            if selected_coin in daily_data:
                coin_data = daily_data[selected_coin]
                df_hist = coin_data['historical_df']
                symbol = coin_data['symbol']
                
                # Crear gráfico completo
                fig = make_subplots(
                    rows=4, cols=1,
                    subplot_titles=(
                        f'{symbol} - Precio y Bollinger Bands',
                        'RSI (20% del Score)',
                        'MACD (25% del Score)',
                        'Score Final por Día'
                    ),
                    vertical_spacing=0.08,
                    row_heights=[0.4, 0.2, 0.2, 0.2]
                )
                
                # Precio y Bollinger Bands
                fig.add_trace(go.Scatter(
                    x=df_hist.index, y=df_hist['price'],
                    name='Precio', line=dict(color='blue', width=2)
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df_hist.index, y=df_hist['bb_upper'],
                    name='BB Superior', line=dict(color='red', dash='dash')
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df_hist.index, y=df_hist['bb_lower'],
                    name='BB Inferior', line=dict(color='red', dash='dash'),
                    fill='tonexty', fillcolor='rgba(255,0,0,0.1)'
                ), row=1, col=1)
                
                # RSI
                fig.add_trace(go.Scatter(
                    x=df_hist.index, y=df_hist['rsi'],
                    name='RSI', line=dict(color='purple')
                ), row=2, col=1)
                
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # MACD
                fig.add_trace(go.Scatter(
                    x=df_hist.index, y=df_hist['macd'],
                    name='MACD', line=dict(color='blue')
                ), row=3, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df_hist.index, y=df_hist['macd_signal'],
                    name='Signal', line=dict(color='red')
                ), row=3, col=1)
                
                # Score histórico (últimos 15 días)
                recent_data = df_hist.tail(15)
                scores = []
                
                for idx in range(len(recent_data)):
                    temp_df = df_hist.iloc[:-(len(recent_data)-idx-1)] if idx < len(recent_data)-1 else df_hist
                    temp_signals = model.generate_complete_signals(temp_df, coin_data['current_data'])
                    if temp_signals:
                        scores.append(temp_signals['final_signal'])
                    else:
                        scores.append(0)
                
                fig.add_trace(go.Scatter(
                    x=recent_data.index, y=scores,
                    name='Score Final', line=dict(color='black', width=3),
                    mode='lines+markers'
                ), row=4, col=1)
                
                fig.add_hline(y=25, line_dash="dash", line_color="green", row=4, col=1)
                fig.add_hline(y=-25, line_dash="dash", line_color="red", row=4, col=1)
                
                fig.update_layout(height=1000, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Métricas específicas
                signals = coin_data['signals']
                current = coin_data['current_data']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Precio Actual", 
                        f"${current['usd']:,.2f}",
                        delta=f"{current.get('usd_24h_change', 0):.2f}%"
                    )
                
                with col2:
                    st.metric(
                        "Score Final",
                        f"{signals['final_signal']:.1f}",
                        delta=f"Confianza: {signals['confidence']:.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "RSI",
                        f"{signals['rsi_val']:.1f}",
                        delta=f"Contrib: {signals['rsi_contrib']:.1f}"
                    )
                
                with col4:
                    vol_24h = current.get('usd_24h_vol', 0)
                    st.metric(
                        "Volumen 24h",
                        f"${vol_24h/1e9:.1f}B" if vol_24h > 1e9 else f"${vol_24h/1e6:.0f}M",
                        delta=f"Vol Contrib: {signals['volume_contrib']:.1f}"
                    )
        
        else:
            st.warning("⚠️ No se pudieron procesar las señales.")
            
    else:
        st.error("❌ No hay datos disponibles.")
        st.info("🔄 Los datos se actualizan automáticamente todos los días a las 7:00 AM (Buenos Aires)")
        
        # Mostrar datos de ejemplo
        st.header("📋 Ejemplo de Funcionamiento")
        demo_data = [
            {'Crypto': 'BTC', 'Precio': '$65,432', 'Cambio 24h': '2.45%', 'Señal': '🟢 COMPRA', 'Score Final': '32.5', 'Confianza': '85%'},
            {'Crypto': 'ETH', 'Precio': '$3,245', 'Cambio 24h': '-1.23%', 'Señal': '⚪ NEUTRO', 'Score Final': '5.8', 'Confianza': '45%'},
            {'Crypto': 'BNB', 'Precio': '$542', 'Cambio 24h': '0.87%', 'Señal': '⚪ NEUTRO', 'Score Final': '-8.2', 'Confianza': '38%'},
            {'Crypto': 'XRP', 'Precio': '$0.623', 'Cambio 24h': '-3.45%', 'Señal': '🔴 VENTA', 'Score Final': '-28.7', 'Confianza': '78%'}
        ]
        
        demo_df = pd.DataFrame(demo_data)
        st.dataframe(demo_df, use_container_width=True)
    
    # Footer informativo
    st.markdown("---")
    st.markdown(f"""
    **⏰ Sistema de Actualización Automática:**
    
    - **Horario fijo**: Todos los días a las 7:00 AM (Buenos Aires)
    - **Datos frescos**: Precios y volúmenes de las últimas 24 horas
    - **Análisis completo**: 30 días de historia para cada indicador
    - **Sin interrupciones**: Datos pre-calculados, carga instantánea
    
    **🎯 Metodología - 6 Indicadores Ponderados:**
    - RSI (20%) + MACD (25%) + Bollinger Bands (15%) + EMA Trend (20%) + Volumen (10%) + Volatilidad (10%)
    
    **📊 Interpretación de Señales:**
    - Score > +25: 🟢 **COMPRA** | Score < -25: 🔴 **VENTA** | Score -25 a +25: ⚪ **NEUTRO**
    
    **⚠️ Disclaimer:** Para fines educativos únicamente. No constituye asesoramiento financiero.
    """)

if __name__ == "__main__":
    main()
