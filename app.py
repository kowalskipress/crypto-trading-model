import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Configuración
st.set_page_config(
    page_title="Robust Crypto Trading Model",
    page_icon="📈",
    layout="wide"
)

class RobustCryptoModel:
    def __init__(self):
        self.coins = {
            'bitcoin': 'BTC',
            'ethereum': 'ETH',
            'binancecoin': 'BNB',
            'ripple': 'XRP'
        }
        
        # Las 6 métricas clásicas de análisis técnico (adaptadas para tendencias semanales)
        self.weights = {
            'rsi_weekly': 0.20,      # 20% - RSI suavizado para tendencias
            'macd_weekly': 0.25,     # 25% - MACD sobre datos diarios (el más importante)
            'bollinger_weekly': 0.15, # 15% - Posición en Bollinger Bands
            'ema_trend': 0.20,       # 20% - Tendencia EMA (7d vs 14d)
            'volume_trend': 0.10,    # 10% - Confirmación por volumen
            'volatility_adj': 0.10   # 10% - Penalización por alta volatilidad
        }
    
    @st.cache_data(ttl=1800)  # Cache 30 minutos
    def fetch_robust_data(_self, coin_id, days=30):
        """Obtiene datos históricos robustos para análisis técnico completo"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily'
            }
            
            response = requests.get(url, params=params, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                
                # Crear DataFrame robusto
                prices = data['prices']
                volumes = data['total_volumes']
                
                df = pd.DataFrame({
                    'timestamp': [p[0] for p in prices],
                    'price': [p[1] for p in prices],
                    'volume': [v[1] for v in volumes]
                })
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['date'] = df['timestamp'].dt.date
                df.set_index('timestamp', inplace=True)
                
                # Agrupar por día para eliminar ruido intradiario
                daily_df = df.groupby('date').agg({
                    'price': 'last',    # Precio de cierre
                    'volume': 'sum'     # Volumen total del día
                }).reset_index()
                
                daily_df['date'] = pd.to_datetime(daily_df['date'])
                daily_df.set_index('date', inplace=True)
                daily_df.sort_index(inplace=True)
                
                return daily_df
                
            return None
            
        except Exception as e:
            return None
    
    def calculate_rsi_weekly(self, prices, period=14):
        """RSI suavizado para tendencias semanales"""
        if len(prices) < period + 5:
            return 50  # Valor neutro si no hay datos suficientes
            
        # Suavizar precios con media móvil de 3 días
        smoothed_prices = prices.rolling(window=3, center=True).mean().fillna(prices)
        
        deltas = smoothed_prices.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    
    def calculate_macd_weekly(self, prices):
        """MACD adaptado para análisis semanal"""
        if len(prices) < 26:
            return 0
            
        # EMAs para MACD (usando períodos adaptados para datos diarios)
        ema_fast = prices.ewm(span=5).mean()   # ~1 semana
        ema_slow = prices.ewm(span=12).mean()  # ~2.5 semanas
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=4).mean()  # Línea de señal
        histogram = macd_line - signal_line
        
        return histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0
    
    def calculate_bollinger_weekly(self, prices, period=14, std_dev=2):
        """Bollinger Bands para posicionamiento semanal"""
        if len(prices) < period:
            return 0.5  # Posición neutra
            
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        current_price = prices.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        # Posición relativa (0 = banda inferior, 1 = banda superior)
        if pd.isna(current_upper) or pd.isna(current_lower):
            return 0.5
            
        position = (current_price - current_lower) / (current_upper - current_lower)
        return np.clip(position, 0, 1)
    
    def calculate_ema_trend(self, prices):
        """Tendencia EMA: 7 días vs 14 días"""
        if len(prices) < 14:
            return 0
            
        ema_7 = prices.ewm(span=7).mean()
        ema_14 = prices.ewm(span=14).mean()
        
        current_ema_7 = ema_7.iloc[-1]
        current_ema_14 = ema_14.iloc[-1]
        
        if pd.isna(current_ema_7) or pd.isna(current_ema_14):
            return 0
            
        # Diferencia porcentual
        trend_strength = ((current_ema_7 / current_ema_14) - 1) * 100
        return trend_strength
    
    def calculate_volume_trend(self, volumes):
        """Tendencia de volumen: reciente vs histórico"""
        if len(volumes) < 14:
            return 0
            
        recent_volume = volumes.tail(7).mean()  # Última semana
        historic_volume = volumes.head(14).mean()  # Primeras 2 semanas
        
        if pd.isna(recent_volume) or pd.isna(historic_volume) or historic_volume == 0:
            return 0
            
        volume_change = ((recent_volume / historic_volume) - 1) * 100
        return volume_change
    
    def calculate_volatility_adjustment(self, prices):
        """Penalización por alta volatilidad"""
        if len(prices) < 10:
            return 0
            
        returns = prices.pct_change().dropna()
        volatility = returns.rolling(window=7).std().iloc[-1]  # Volatilidad semanal
        
        if pd.isna(volatility):
            return 0
            
        # Normalizar volatilidad (0-100 scale)
        vol_percentile = (volatility * 100) / returns.std() if returns.std() > 0 else 0
        
        # Alta volatilidad = penalización negativa
        if vol_percentile > 80:
            return -30  # Penalización fuerte
        elif vol_percentile > 60:
            return -15  # Penalización moderada
        elif vol_percentile < 20:
            return 15   # Baja volatilidad = oportunidad
        else:
            return 0    # Volatilidad normal
    
    def generate_robust_signals(self, df):
        """Genera señales usando las 6 métricas clásicas"""
        if df is None or len(df) < 20:
            return None
            
        try:
            prices = df['price']
            volumes = df['volume']
            
            # 1. RSI Semanal (20%)
            rsi_val = self.calculate_rsi_weekly(prices)
            if rsi_val < 30:
                rsi_signal = 100  # Fuerte compra
            elif rsi_val > 70:
                rsi_signal = -100  # Fuerte venta
            else:
                rsi_signal = (50 - rsi_val) * 2  # Escala lineal
            
            # 2. MACD Semanal (25%) - El más importante
            macd_hist = self.calculate_macd_weekly(prices)
            macd_signal = np.clip(macd_hist * 1000, -100, 100)  # Escalar apropiadamente
            
            # 3. Bollinger Bands (15%)
            bb_position = self.calculate_bollinger_weekly(prices)
            if bb_position < 0.2:
                bb_signal = 80  # Cerca de banda inferior = compra
            elif bb_position > 0.8:
                bb_signal = -80  # Cerca de banda superior = venta
            else:
                bb_signal = (0.5 - bb_position) * 160  # Escala lineal
            
            # 4. EMA Trend (20%)
            ema_trend_pct = self.calculate_ema_trend(prices)
            ema_signal = np.clip(ema_trend_pct * 15, -100, 100)
            
            # 5. Volume Trend (10%)
            volume_trend_pct = self.calculate_volume_trend(volumes)
            volume_signal = np.clip(volume_trend_pct * 2, -50, 50)
            
            # 6. Volatility Adjustment (10%)
            volatility_signal = self.calculate_volatility_adjustment(prices)
            
            # Calcular contribuciones ponderadas
            contributions = {
                'rsi_weekly': rsi_signal * self.weights['rsi_weekly'],
                'macd_weekly': macd_signal * self.weights['macd_weekly'],
                'bollinger_weekly': bb_signal * self.weights['bollinger_weekly'],
                'ema_trend': ema_signal * self.weights['ema_trend'],
                'volume_trend': volume_signal * self.weights['volume_trend'],
                'volatility_adj': volatility_signal * self.weights['volatility_adj']
            }
            
            # Score final
            final_score = sum(contributions.values())
            
            return {
                'final_score': final_score,
                'contributions': contributions,
                'raw_metrics': {
                    'rsi_value': rsi_val,
                    'macd_histogram': macd_hist,
                    'bb_position': bb_position,
                    'ema_trend_pct': ema_trend_pct,
                    'volume_trend_pct': volume_trend_pct,
                    'volatility_penalty': volatility_signal
                },
                'signals': {
                    'rsi_signal': rsi_signal,
                    'macd_signal': macd_signal,
                    'bb_signal': bb_signal,
                    'ema_signal': ema_signal,
                    'volume_signal': volume_signal,
                    'volatility_signal': volatility_signal
                }
            }
            
        except Exception as e:
            st.error(f"Error en cálculo de señales: {str(e)}")
            return None
    
    def classify_signal(self, score):
        """Clasifica la señal con umbrales conservadores"""
        if score > 35:
            return "🟢 COMPRA", "Señal alcista muy fuerte", "#155724"
        elif score > 20:
            return "🟢 COMPRA", "Señal alcista moderada", "#28a745"
        elif score < -35:
            return "🔴 VENTA", "Señal bajista muy fuerte", "#721c24"
        elif score < -20:
            return "🔴 VENTA", "Señal bajista moderada", "#dc3545"
        else:
            return "⚪ NEUTRO", "Sin tendencia clara definida", "#6c757d"

def main():
    st.title("📊 Robust Crypto Trading Model")
    st.markdown("**6 Métricas Clásicas - Enfoque en Tendencias Semanales**")
    
    model = RobustCryptoModel()
    
    # Sidebar con información del modelo
    st.sidebar.header("⚙️ Modelo de 6 Métricas")
    st.sidebar.markdown("**🎯 Enfoque: Análisis Técnico Clásico**")
    
    # Mostrar pesos
    st.sidebar.subheader("📊 Distribución de Pesos")
    for metric, weight in model.weights.items():
        display_names = {
            'rsi_weekly': 'RSI Semanal',
            'macd_weekly': 'MACD Semanal',
            'bollinger_weekly': 'Bollinger Bands',
            'ema_trend': 'Tendencia EMA',
            'volume_trend': 'Tendencia Volumen',
            'volatility_adj': 'Ajuste Volatilidad'
        }
        st.sidebar.write(f"**{display_names[metric]}**: {weight*100:.0f}%")
    
    # Información metodológica
    st.sidebar.subheader("🔬 Metodología")
    st.sidebar.info("""
    **Datos suavizados diarios**
    - Elimina ruido intradiario
    - Enfoque en tendencias sostenibles
    - Umbrales conservadores
    - 30 días de análisis histórico
    """)
    
    if st.sidebar.button("🔄 Actualizar Análisis"):
        st.cache_data.clear()
        st.rerun()
    
    # Análisis principal
    st.header("📈 Análisis Técnico Completo")
    
    results = []
    progress_bar = st.progress(0)
    status_container = st.container()
    
    for i, (coin_id, symbol) in enumerate(model.coins.items()):
        progress_bar.progress((i + 1) / len(model.coins))
        
        with status_container:
            with st.spinner(f"🔍 Analizando {symbol} con 6 métricas..."):
                # Obtener datos históricos
                df = model.fetch_robust_data(coin_id, days=30)
                
                if df is not None and len(df) >= 20:
                    # Generar análisis completo
                    analysis = model.generate_robust_signals(df)
                    
                    if analysis:
                        score = analysis['final_score']
                        contributions = analysis['contributions']
                        raw_metrics = analysis['raw_metrics']
                        
                        # Clasificar señal
                        signal_class, signal_reason, signal_color = model.classify_signal(score)
                        
                        # Datos de precio
                        current_price = df['price'].iloc[-1]
                        week_ago_price = df['price'].iloc[-7] if len(df) >= 7 else df['price'].iloc[0]
                        weekly_change = ((current_price / week_ago_price) - 1) * 100
                        
                        results.append({
                            'Crypto': symbol,
                            'Precio': f"${current_price:,.2f}",
                            'Cambio 7d': f"{weekly_change:+.2f}%",
                            'Señal': signal_class,
                            'Score Final': f"{score:.1f}",
                            'Confianza': f"{min(abs(score), 100):.0f}%",
                            
                            # Métricas individuales
                            'RSI': f"{raw_metrics['rsi_value']:.1f}",
                            'MACD': f"{raw_metrics['macd_histogram']:.4f}",
                            'BB Pos': f"{raw_metrics['bb_position']*100:.1f}%",
                            'EMA Trend': f"{raw_metrics['ema_trend_pct']:+.2f}%",
                            'Vol Trend': f"{raw_metrics['volume_trend_pct']:+.1f}%",
                            
                            # Contribuciones ponderadas
                            'RSI Contrib': f"{contributions['rsi_weekly']:+.1f}",
                            'MACD Contrib': f"{contributions['macd_weekly']:+.1f}",
                            'BB Contrib': f"{contributions['bollinger_weekly']:+.1f}",
                            'EMA Contrib': f"{contributions['ema_trend']:+.1f}",
                            'Vol Contrib': f"{contributions['volume_trend']:+.1f}",
                            'Volat Contrib': f"{contributions['volatility_adj']:+.1f}",
                            
                            'Razón': signal_reason,
                            'Color': signal_color
                        })
                        
                        st.success(f"✅ {symbol}: {signal_class.split()[1]} (Score: {score:.1f})")
                    else:
                        st.warning(f"⚠️ {symbol}: Error en análisis técnico")
                else:
                    st.warning(f"⚠️ {symbol}: Datos históricos insuficientes")
                
                # Pausa entre requests
                time.sleep(1.5)
    
    progress_bar.empty()
    status_container.empty()
    
    # Mostrar resultados
    if results:
        st.success(f"🎉 Análisis técnico completado para {len(results)} criptomonedas")
        
        # Tabla principal con métricas
        df_results = pd.DataFrame(results)
        
        # Crear tabs para diferentes vistas
        tab1, tab2, tab3 = st.tabs(["📊 Resumen Principal", "🔍 Métricas Detalladas", "📈 Contribuciones"])
        
        with tab1:
            # Vista resumida
            summary_cols = ['Crypto', 'Precio', 'Cambio 7d', 'Señal', 'Score Final', 'Confianza', 'Razón']
            summary_df = df_results[summary_cols]
            
            def style_summary(df):
                def color_signals(val):
                    if '🟢' in val and 'muy fuerte' in val:
                        return 'background-color: #155724; color: white; font-weight: bold'
                    elif '🟢' in val:
                        return 'background-color: #d4edda; color: #155724; font-weight: bold'
                    elif '🔴' in val and 'muy fuerte' in val:
                        return 'background-color: #721c24; color: white; font-weight: bold'
                    elif '🔴' in val:
                        return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
                    else:
                        return 'background-color: #f8f9fa; color: #495057; font-weight: bold'
                
                def color_score(val):
                    try:
                        score = float(val)
                        if abs(score) > 35:
                            return 'font-weight: bold; font-size: 14px'
                        elif abs(score) > 20:
                            return 'font-weight: bold'
                        return ''
                    except:
                        return ''
                
                return df.style.applymap(color_signals, subset=['Señal']).applymap(color_score, subset=['Score Final'])
            
            st.dataframe(style_summary(summary_df), use_container_width=True)
        
        with tab2:
            # Métricas técnicas detalladas
            metrics_cols = ['Crypto', 'RSI', 'MACD', 'BB Pos', 'EMA Trend', 'Vol Trend', 'Score Final']
            metrics_df = df_results[metrics_cols]
            st.dataframe(metrics_df, use_container_width=True)
        
        with tab3:
            # Contribuciones ponderadas
            contrib_cols = ['Crypto', 'RSI Contrib', 'MACD Contrib', 'BB Contrib', 'EMA Contrib', 'Vol Contrib', 'Volat Contrib']
            contrib_df = df_results[contrib_cols]
            
            def style_contributions(df):
                def color_contrib(val):
                    try:
                        contrib = float(val)
                        if contrib > 8:
                            return 'color: #155724; font-weight: bold'
                        elif contrib > 3:
                            return 'color: #28a745; font-weight: bold'
                        elif contrib < -8:
                            return 'color: #721c24; font-weight: bold'
                        elif contrib < -3:
                            return 'color: #dc3545; font-weight: bold'
                        else:
                            return 'color: #6c757d'
                    except:
                        return ''
                
                contrib_columns = [col for col in df.columns if 'Contrib' in col]
                styled = df.style
                for col in contrib_columns:
                    styled = styled.applymap(color_contrib, subset=[col])
                return styled
            
            st.dataframe(style_contributions(contrib_df), use_container_width=True)
        
        # Métricas del mercado
        st.subheader("📊 Estado del Mercado")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            buy_signals = sum(1 for r in results if '🟢' in r['Señal'])
            st.metric("🟢 Señales Alcistas", buy_signals, help="Señales de compra generadas")
        
        with col2:
            sell_signals = sum(1 for r in results if '🔴' in r['Señal'])
            st.metric("🔴 Señales Bajistas", sell_signals, help="Señales de venta generadas")
            
        with col3:
            neutral_signals = sum(1 for r in results if '⚪' in r['Señal'])
            st.metric("⚪ Señales Neutras", neutral_signals, help="Sin tendencia clara")
            
        with col4:
            avg_score = np.mean([float(r['Score Final']) for r in results])
            market_sentiment = "Alcista" if avg_score > 10 else "Bajista" if avg_score < -10 else "Neutral"
            st.metric("📈 Sentimiento", market_sentiment, f"Score: {avg_score:.1f}")
        
        # Análisis individual detallado
        st.subheader("🔍 Análisis Individual Detallado")
        
        for result in results:
            with st.expander(f"{result['Crypto']} - {result['Señal']} | Score: {result['Score Final']} | {result['Razón']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**📊 Métricas Principales:**")
                    st.write(f"• Precio: {result['Precio']}")
                    st.write(f"• Cambio 7d: {result['Cambio 7d']}")
                    st.write(f"• Confianza: {result['Confianza']}")
                    st.write(f"• Score Final: {result['Score Final']}")
                
                with col2:
                    st.write("**🔬 Indicadores Técnicos:**")
                    st.write(f"• RSI: {result['RSI']} ({'Sobrecompra' if float(result['RSI']) > 70 else 'Sobreventa' if float(result['RSI']) < 30 else 'Normal'})")
                    st.write(f"• MACD: {result['MACD']}")
                    st.write(f"• Bollinger: {result['BB Pos']}")
                    st.write(f"• EMA Trend: {result['EMA Trend']}")
                
                with col3:
                    st.write("**⚖️ Contribuciones (Ponderadas):**")
                    st.write(f"• RSI: {result['RSI Contrib']} (20%)")
                    st.write(f"• MACD: {result['MACD Contrib']} (25%)")
                    st.write(f"• Bollinger: {result['BB Contrib']} (15%)")
                    st.write(f"• EMA: {result['EMA Contrib']} (20%)")
                    st.write(f"• Volume: {result['Vol Contrib']} (10%)")
                    st.write(f"• Volatilidad: {result['Volat Contrib']} (10%)")
    
    else:
        st.error("❌ No se pudieron obtener suficientes datos para el análisis")
        st.info("🔄 Intenta actualizar en unos minutos. El modelo requiere al menos 20 días de datos históricos.")
    
    # Footer metodológico
    st.markdown("---")
    st.markdown("""
    **🎯 Modelo Robusto de 6 Métricas - Metodología Completa:**
    
    **📊 Indicadores Clásicos Utilizados:**
    1. **RSI Semanal (20%)**: Detecta sobrecompra/sobreventa con datos suavizados
    2. **MACD Semanal (25%)**: Convergencia/divergencia de medias móviles - *Indicador principal*
    3. **Bollinger Bands (15%)**: Posición relativa del precio respecto a bandas de volatilidad
    4. **Tendencia EMA (20%)**: Cruce de medias exponenciales (7d vs 14d)
    5. **Tendencia Volumen (10%)**: Confirmación mediante análisis de volumen
    6. **Ajuste Volatilidad (10%)**: Penalización por alta volatilidad / premio por baja volatilidad
    
    **🎯 Umbrales de Señal (Conservadores):**
    - Score > +35: 🟢 **COMPRA MUY FUERTE**
    - Score +20 a +35: 🟢 **COMPRA MODERADA**
    - Score -20 a +20: ⚪ **NEUTRO** (Zona de espera)
    - Score -35 a -20: 🔴 **VENTA MODERADA**  
    - Score < -35: 🔴 **VENTA MUY FUERTE**
    
    **✅ Características de Robustez:**
    - Datos agrupados diarios (elimina ruido intradiario)
    - 30 días de análisis histórico mínimo
    - Métricas suavizadas para tendencias sostenibles
    - Ponderación basada en literatura financiera académica
    - Validación cruzada entre múltiples indicadores
    
    **⚠️ Disclaimer:** Modelo para análisis técnico educativo. No constituye asesoramiento financiero profesional.
    """)

if __name__ == "__main__":
    main()
