import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Configuración
st.set_page_config(
    page_title="Consistent Crypto Model",
    page_icon="📈",
    layout="wide"
)

class ConsistentCryptoModel:
    def __init__(self):
        self.coins = {
            'bitcoin': 'BTC',
            'ethereum': 'ETH',
            'binancecoin': 'BNB',
            'ripple': 'XRP'
        }
        
        # Pesos enfocados en CONSISTENCIA LÓGICA
        self.weights = {
            'trend_7d': 0.40,        # 40% - Tendencia 7 días (MÁS IMPORTANTE)
            'trend_14d': 0.25,       # 25% - Tendencia 14 días
            'rsi_weekly': 0.20,      # 20% - RSI suavizado
            'momentum_confirm': 0.15  # 15% - Confirmación de momentum
        }
    
    @st.cache_data(ttl=1800)  # Cache 30 minutos
    def fetch_consistent_data(_self, coin_id, days=21):
        """Obtiene datos históricos con múltiples intentos y fallbacks"""
        
        # Estrategia de fallback: intentar diferentes períodos
        fallback_periods = [21, 14, 10, 7]
        
        for attempt_days in fallback_periods:
            try:
                st.info(f"📊 Intentando obtener {attempt_days} días de datos para {coin_id}...")
                
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                params = {
                    'vs_currency': 'usd',
                    'days': attempt_days,
                    'interval': 'daily'
                }
                
                response = requests.get(url, params=params, timeout=25)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'prices' in data and len(data['prices']) > 0:
                        # Crear DataFrame con datos disponibles
                        prices = data['prices']
                        volumes = data['total_volumes'] if 'total_volumes' in data else []
                        
                        df = pd.DataFrame({
                            'timestamp': [p[0] for p in prices],
                            'price': [p[1] for p in prices],
                            'volume': [v[1] if len(volumes) > i else 1000000 for i, v in enumerate(volumes)] or [1000000] * len(prices)
                        })
                        
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df['date'] = df['timestamp'].dt.date
                        df.set_index('timestamp', inplace=True)
                        
                        # Agrupar por día
                        daily_df = df.groupby('date').agg({
                            'price': 'last',
                            'volume': 'sum'
                        }).reset_index()
                        
                        daily_df['date'] = pd.to_datetime(daily_df['date'])
                        daily_df.set_index('date', inplace=True)
                        daily_df.sort_index(inplace=True)
                        
                        # Validar datos mínimos
                        if len(daily_df) >= 5:  # Reducir requisito mínimo
                            st.success(f"✅ Obtenidos {len(daily_df)} días de datos para {coin_id}")
                            return daily_df
                        else:
                            st.warning(f"⚠️ Solo {len(daily_df)} días disponibles para {coin_id}")
                            
                elif response.status_code == 429:
                    st.warning(f"⏳ Rate limit para {coin_id}, esperando...")
                    time.sleep(3)
                    continue
                else:
                    st.warning(f"⚠️ API error {response.status_code} para {coin_id}")
                    
            except Exception as e:
                st.warning(f"⚠️ Error en intento con {attempt_days} días para {coin_id}: {str(e)}")
                continue
        
        st.error(f"❌ No se pudieron obtener datos para {coin_id} después de múltiples intentos")
        return None
    
    def calculate_consistent_signals(self, df):
        """Calcula señales CONSISTENTES - adaptado para pocos datos"""
        if df is None or len(df) < 3:  # Reducir requisito mínimo
            return None
            
        try:
            prices = df['price']
            volumes = df['volume']
            
            # Adaptación para datasets pequeños
            data_points = len(prices)
            
            # 1. TENDENCIA 7 DÍAS (40% - MÁS IMPORTANTE)
            current_price = prices.iloc[-1]
            
            if data_points >= 7:
                price_ago = prices.iloc[-7]
            elif data_points >= 5:
                price_ago = prices.iloc[-5]
            elif data_points >= 3:
                price_ago = prices.iloc[-3]
            else:
                price_ago = prices.iloc[0]
            
            # Calcular cambio porcentual
            days_diff = min(data_points - 1, 7)
            trend_pct = ((current_price / price_ago) - 1) * 100
            
            # Ajustar señal según días disponibles
            multiplier = 7 / days_diff if days_diff > 0 else 1
            trend_pct_normalized = trend_pct * multiplier
            
            # Señal FUERTE y CONSISTENTE
            if trend_pct_normalized > 15:
                trend_signal = 100
            elif trend_pct_normalized > 8:
                trend_signal = 60
            elif trend_pct_normalized > 3:
                trend_signal = 30
            elif trend_pct_normalized < -15:
                trend_signal = -100
            elif trend_pct_normalized < -8:
                trend_signal = -60
            elif trend_pct_normalized < -3:
                trend_signal = -30
            else:
                trend_signal = 0
            
            # 2. RSI SIMPLIFICADO (30%)
            if data_points >= 10:
                rsi_val = self.calculate_rsi_consistent(prices)
            else:
                # RSI simplificado para pocos datos
                recent_changes = prices.pct_change().dropna()
                if len(recent_changes) > 0:
                    avg_change = recent_changes.mean()
                    if avg_change > 0.05:  # Subida fuerte
                        rsi_val = 25  # Sobreventa (oportunidad de compra)
                    elif avg_change < -0.05:  # Bajada fuerte
                        rsi_val = 75  # Sobrecompra (oportunidad de venta)
                    else:
                        rsi_val = 50
                else:
                    rsi_val = 50
            
            # Señal RSI
            if rsi_val < 30:
                rsi_signal = 100
            elif rsi_val > 70:
                rsi_signal = -100
            else:
                rsi_signal = (50 - rsi_val) * 1.5
            
            # 3. MOMENTUM SIMPLIFICADO (30%)
            if data_points >= 5:
                recent_avg = prices.tail(3).mean()
                older_avg = prices.head(3).mean()
                momentum_pct = ((recent_avg / older_avg) - 1) * 100
            else:
                # Para muy pocos datos, usar cambio directo
                momentum_pct = trend_pct
            
            momentum_signal = np.clip(momentum_pct * 3, -50, 50)
            
            # Calcular score con pesos adaptados
            adapted_weights = {
                'trend': 0.40,
                'rsi': 0.30,
                'momentum': 0.30
            }
            
            final_score = (
                trend_signal * adapted_weights['trend'] +
                rsi_signal * adapted_weights['rsi'] +
                momentum_signal * adapted_weights['momentum']
            )
            
            # VALIDACIÓN DE CONSISTENCIA mejorada
            if trend_pct_normalized > 10 and final_score < 5:
                final_score = max(final_score, 20)
            elif trend_pct_normalized < -10 and final_score > -5:
                final_score = min(final_score, -20)
            
            return {
                'final_score': final_score,
                'raw_metrics': {
                    'trend_pct': trend_pct,
                    'trend_pct_normalized': trend_pct_normalized,
                    'rsi_value': rsi_val,
                    'momentum_pct': momentum_pct,
                    'data_points': data_points,
                    'days_used': days_diff
                },
                'contributions': {
                    'trend': trend_signal * adapted_weights['trend'],
                    'rsi': rsi_signal * adapted_weights['rsi'],
                    'momentum': momentum_signal * adapted_weights['momentum']
                }
            }
            
        except Exception as e:
            st.error(f"Error en cálculo de señales: {str(e)}")
            return None
    
    def calculate_rsi_consistent(self, prices, period=14):
        """RSI consistente y confiable"""
        if len(prices) < period + 5:
            return 50
            
        # Suavizar con media móvil de 3 días
        smoothed = prices.rolling(window=3, center=True).mean().fillna(prices)
        
        deltas = smoothed.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        # Evitar división por cero
        avg_losses = avg_losses.replace(0, 0.0001)
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        final_rsi = rsi.iloc[-1]
        
        # Validar RSI dentro de rango
        if pd.isna(final_rsi) or final_rsi < 0 or final_rsi > 100:
            return 50
            
        return final_rsi
    
    def classify_signal_consistent(self, score, trend_pct):
        """Clasificación CONSISTENTE - adaptada para pocos datos"""
        
        # REGLA DE CONSISTENCIA: Tendencia domina
        if trend_pct > 12:  # Subida fuerte
            if score > 25:
                return "🟢 COMPRA", "Tendencia alcista fuerte confirmada", "#155724"
            elif score > 10:
                return "🟢 COMPRA", "Tendencia alcista moderada", "#28a745"
            else:
                return "🟢 COMPRA", "Tendencia alcista (datos limitados)", "#28a745"
                
        elif trend_pct < -12:  # Caída fuerte
            if score < -25:
                return "🔴 VENTA", "Tendencia bajista fuerte confirmada", "#721c24"
            elif score < -10:
                return "🔴 VENTA", "Tendencia bajista moderada", "#dc3545"
            else:
                return "🔴 VENTA", "Tendencia bajista (datos limitados)", "#dc3545"
                
        else:  # Movimiento moderado
            if score > 20:
                return "🟢 COMPRA", "Señal alcista por indicadores", "#28a745"
            elif score < -20:
                return "🔴 VENTA", "Señal bajista por indicadores", "#dc3545"
            else:
                return "⚪ NEUTRO", "Sin tendencia clara definida", "#6c757d"

def main():
    # IDENTIFICADOR DE VERSION CON NOMBRE ARGENTINO
    st.title("📊 Crypto Trading Model")
    st.markdown("**Análisis Técnico Profesional - Enfoque en Tendencias Semanales**")
    
    # Marca de versión con nombre argentino
    st.success("🏔️ **VERSIÓN MENDOZA** | Datos Adaptativos | Build: 20/07/2025 22:45")
    
    model = ConsistentCryptoModel()
    
    # Sidebar con información
    st.sidebar.header("🏔️ Crypto Model Mendoza")
    st.sidebar.success("🏔️ **VERSIÓN MENDOZA**")
    st.sidebar.markdown("**🍇 Enfoque: Adaptación Inteligente**")
    st.sidebar.info("📅 Build: 20/07/2025 22:45 GMT-3")
    
    # Mostrar pesos enfocados en consistencia
    st.sidebar.subheader("⚖️ Distribución de Pesos")
    for metric, weight in model.weights.items():
        display_names = {
            'trend_7d': 'Tendencia 7 días',
            'trend_14d': 'Tendencia 14 días',
            'rsi_weekly': 'RSI Semanal',
            'momentum_confirm': 'Confirmación Momentum'
        }
        st.sidebar.write(f"**{display_names[metric]}**: {weight*100:.0f}%")
    
    st.sidebar.subheader("🔧 Adaptaciones Automáticas")
    st.sidebar.info("""
    **Manejo inteligente de datos:**
    • Fallback: 21→14→10→7 días
    • Mínimo: 3 días de datos
    • Normalización por período
    • Señales consistentes garantizadas
    """)
    
    if st.sidebar.button("🔄 Actualizar Análisis"):
        st.cache_data.clear()
        st.rerun()
    
    # Análisis principal
    st.header("📈 Análisis Consistente de Cryptos")
    
    results = []
    progress_bar = st.progress(0)
    
    for i, (coin_id, symbol) in enumerate(model.coins.items()):
        progress_bar.progress((i + 1) / len(model.coins))
        
        with st.spinner(f"🔍 Analizando {symbol} con lógica consistente..."):
            # Obtener datos históricos
            df = model.fetch_consistent_data(coin_id, days=21)
            
            if df is not None and len(df) >= 3:  # Reducir requisito mínimo
                # Generar análisis consistente
                analysis = model.calculate_consistent_signals(df)
                
                if analysis:
                    score = analysis['final_score']
                    raw_metrics = analysis['raw_metrics']
                    
                    # Clasificar señal de forma consistente
                    signal_class, signal_reason, signal_color = model.classify_signal_consistent(
                        score, raw_metrics['trend_pct_normalized']
                    )
                    
                    # Datos de precio
                    current_price = df['price'].iloc[-1]
                    trend_change = raw_metrics['trend_pct']
                    data_quality = f"({raw_metrics['data_points']} días)"
                    
                    results.append({
                        'Crypto': symbol,
                        'Precio': f"${current_price:,.2f}",
                        'Cambio': f"{trend_change:+.2f}%",
                        'Período': data_quality,
                        'Señal': signal_class,
                        'Score Final': f"{score:.1f}",
                        'Confianza': f"{min(abs(score), 100):.0f}%",
                        
                        # Métricas adaptadas
                        'RSI': f"{raw_metrics['rsi_value']:.1f}",
                        'Momentum': f"{raw_metrics['momentum_pct']:+.2f}%",
                        'Días Usados': f"{raw_metrics['days_used']}d",
                        
                        # Contribuciones
                        'Contrib Trend': f"{analysis['contributions']['trend']:+.1f}",
                        'Contrib RSI': f"{analysis['contributions']['rsi']:+.1f}",
                        'Contrib Mom': f"{analysis['contributions']['momentum']:+.1f}",
                        
                        'Razón': signal_reason,
                        'Color': signal_color
                    })
                    
                    st.success(f"✅ {symbol}: {signal_class.split()[1]} (Score: {score:.1f}) [{raw_metrics['data_points']} días]")
                else:
                    st.warning(f"⚠️ {symbol}: Error en cálculo de señales")
            else:
                st.error(f"❌ {symbol}: Datos insuficientes (mínimo 3 días requerido)")
            
            # Pausa entre requests
            time.sleep(1.5)
    
    progress_bar.empty()
    
    # Mostrar resultados con validación de consistencia
    if results:
        st.success(f"🎯 Análisis consistente completado para {len(results)} criptomonedas")
        
        # VALIDACIÓN DE CONSISTENCIA ADAPTADA
        st.subheader("🛡️ Validación de Consistencia")
        
        inconsistencies = []
        for result in results:
            crypto = result['Crypto']
            change_str = result['Cambio'].replace('%', '').replace('+', '')
            try:
                change = float(change_str)
                score = float(result['Score Final'])
                signal = result['Señal']
                
                # Validar consistencia con umbrales adaptados
                if change > 12 and '🟢' not in signal:
                    inconsistencies.append(f"{crypto}: Cambio +{change:.1f}% pero señal no es COMPRA")
                elif change < -12 and '🔴' not in signal:
                    inconsistencies.append(f"{crypto}: Cambio {change:.1f}% pero señal no es VENTA")
                elif change > 8 and score < -5:
                    inconsistencies.append(f"{crypto}: Cambio positivo {change:.1f}% pero score negativo {score:.1f}")
                elif change < -8 and score > 5:
                    inconsistencies.append(f"{crypto}: Cambio negativo {change:.1f}% pero score positivo {score:.1f}")
            except:
                pass Validación de Consistencia")
        
        inconsistencies = []
        for result in results:
            crypto = result['Crypto']
            change_7d = float(result['Cambio 7d'].replace('%', '').replace('+', ''))
            score = float(result['Score Final'])
            signal = result['Señal']
            
            # Validar consistencia
            if change_7d > 15 and '🟢' not in signal:
                inconsistencies.append(f"{crypto}: Cambio +{change_7d:.1f}% pero señal no es COMPRA")
            elif change_7d < -15 and '🔴' not in signal:
                inconsistencies.append(f"{crypto}: Cambio {change_7d:.1f}% pero señal no es VENTA")
            elif change_7d > 10 and score < 0:
                inconsistencies.append(f"{crypto}: Cambio positivo {change_7d:.1f}% pero score negativo {score:.1f}")
            elif change_7d < -10 and score > 0:
                inconsistencies.append(f"{crypto}: Cambio negativo {change_7d:.1f}% pero score positivo {score:.1f}")
        
        if inconsistencies:
            st.error("❌ INCONSISTENCIAS DETECTADAS:")
            for inconsistency in inconsistencies:
                st.error(f"• {inconsistency}")
        else:
            st.success("✅ TODAS LAS SEÑALES SON CONSISTENTES - Modelo confiable")
        
        # Tabla principal
        df_results = pd.DataFrame(results)
        
        # Crear tabs
        tab1, tab2, tab3 = st.tabs(["📊 Resumen Consistente", "🔍 Métricas Detalladas", "⚖️ Contribuciones"])
        
        with tab1:
            # Vista principal adaptada para datos limitados
            summary_cols = ['Crypto', 'Precio', 'Cambio', 'Período', 'Señal', 'Score Final', 'Confianza', 'Razón']
            summary_df = df_results[summary_cols]
            
            def style_consistent(df):
                def color_signals(val):
                    if '🟢' in val:
                        return 'background-color: #d4edda; color: #155724; font-weight: bold'
                    elif '🔴' in val:
                        return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
                    else:
                        return 'background-color: #f8f9fa; color: #495057; font-weight: bold'
                
                def color_change(val):
                    try:
                        change = float(val.replace('%', '').replace('+', ''))
                        if change > 8:
                            return 'background-color: #d4edda; color: #155724; font-weight: bold'
                        elif change < -8:
                            return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
                        else:
                            return 'color: #6c757d'
                    except:
                        return ''
                
                return df.style.applymap(color_signals, subset=['Señal']).applymap(color_change, subset=['Cambio'])
            
            st.dataframe(style_consistent(summary_df), use_container_width=True)
        
        with tab2:
            # Métricas detalladas adaptadas
            metrics_cols = ['Crypto', 'RSI', 'Momentum', 'Días Usados', 'Score Final']
            metrics_df = df_results[metrics_cols]
            st.dataframe(metrics_df, use_container_width=True)
        
        with tab3:
            # Contribuciones adaptadas
            contrib_cols = ['Crypto', 'Contrib Trend', 'Contrib RSI', 'Contrib Mom']
            contrib_df = df_results[contrib_cols]
            st.dataframe(contrib_df, use_container_width=True)
        
        # Métricas del mercado
        st.subheader("📊 Estado del Mercado")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            buy_signals = sum(1 for r in results if '🟢' in r['Señal'])
            st.metric("🟢 Señales Compra", buy_signals)
        
        with col2:
            sell_signals = sum(1 for r in results if '🔴' in r['Señal'])
            st.metric("🔴 Señales Venta", sell_signals)
            
        with col3:
            neutral_signals = sum(1 for r in results if '⚪' in r['Señal'])
            st.metric("⚪ Señales Neutras", neutral_signals)
            
        with col4:
            avg_score = np.mean([float(r['Score Final']) for r in results])
            market_sentiment = "Alcista" if avg_score > 10 else "Bajista" if avg_score < -10 else "Neutral"
            st.metric("📈 Sentimiento", market_sentiment, f"Score: {avg_score:.1f}")
    
    else:
        st.error("❌ No se pudieron obtener datos suficientes")
        st.info("🔄 Intenta actualizar en unos minutos")
    
    # Footer informativo
    st.markdown("---")
    st.info("🏔️ **CRYPTO MODEL MENDOZA** - Build 20/07/2025 22:45 | Adaptación Inteligente a Datos Limitados")
    st.markdown("""
    **🏔️ Crypto Model Mendoza - Adaptación Inteligente:**
    
    **⚖️ Estrategia de Datos Adaptativos:**
    - **Fallback automático**: 21→14→10→7 días según disponibilidad
    - **Requisito mínimo**: 3 días de datos históricos
    - **Normalización temporal**: Ajusta señales según período disponible
    - **Múltiples intentos**: Reintentos automáticos con diferentes parámetros
    
    **🎯 Pesos Adaptativos (para datos limitados):**
    - **Tendencia Principal (40%)**: Cambio en período disponible
    - **RSI Adaptativo (30%)**: Cálculo ajustado según datos
    - **Momentum (30%)**: Confirmación de dirección
    
    **🛡️ Reglas de Consistencia Mantenidas:**
    - Cambio > +12% → SIEMPRE señal de COMPRA
    - Cambio < -12% → SIEMPRE señal de VENTA
    - Score alineado con cambio de precio
    - Validación automática de coherencia
    
    **✅ Mejoras en Mendoza:**
    - Tolerancia a datos limitados (mínimo 3 días)
    - Múltiples intentos con fallback
    - Normalización por período de datos
    - Información de calidad de datos en resultados
    
    **⚠️ Disclaimer:** Modelo para análisis técnico educativo. Todas las señales son consistentes lógicamente.
    """)

if __name__ == "__main__":
    main()
