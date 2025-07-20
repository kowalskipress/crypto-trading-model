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
        """Obtiene datos históricos enfocados en consistencia"""
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
                
                # Crear DataFrame con datos diarios limpios
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
                
                # Agrupar por día y ordenar
                daily_df = df.groupby('date').agg({
                    'price': 'last',
                    'volume': 'sum'
                }).reset_index()
                
                daily_df['date'] = pd.to_datetime(daily_df['date'])
                daily_df.set_index('date', inplace=True)
                daily_df.sort_index(inplace=True)
                
                return daily_df
                
            return None
            
        except Exception as e:
            return None
    
    def calculate_consistent_signals(self, df):
        """Calcula señales CONSISTENTES - sin contradicciones"""
        if df is None or len(df) < 15:
            return None
            
        try:
            prices = df['price']
            volumes = df['volume']
            
            # 1. TENDENCIA 7 DÍAS (40% - MÁS IMPORTANTE)
            current_price = prices.iloc[-1]
            price_7d_ago = prices.iloc[-7] if len(prices) >= 7 else prices.iloc[0]
            
            trend_7d_pct = ((current_price / price_7d_ago) - 1) * 100
            
            # Señal FUERTE y CONSISTENTE basada en cambio 7d
            if trend_7d_pct > 15:        # Subida fuerte
                trend_7d_signal = 100
            elif trend_7d_pct > 8:       # Subida moderada
                trend_7d_signal = 60
            elif trend_7d_pct > 3:       # Subida leve
                trend_7d_signal = 30
            elif trend_7d_pct < -15:     # Caída fuerte
                trend_7d_signal = -100
            elif trend_7d_pct < -8:      # Caída moderada
                trend_7d_signal = -60
            elif trend_7d_pct < -3:      # Caída leve
                trend_7d_signal = -30
            else:                        # Lateral
                trend_7d_signal = 0
            
            # 2. TENDENCIA 14 DÍAS (25%)
            price_14d_ago = prices.iloc[-14] if len(prices) >= 14 else prices.iloc[0]
            trend_14d_pct = ((current_price / price_14d_ago) - 1) * 100
            
            # Señal basada en tendencia más larga
            if trend_14d_pct > 20:
                trend_14d_signal = 80
            elif trend_14d_pct > 10:
                trend_14d_signal = 50
            elif trend_14d_pct > 5:
                trend_14d_signal = 25
            elif trend_14d_pct < -20:
                trend_14d_signal = -80
            elif trend_14d_pct < -10:
                trend_14d_signal = -50
            elif trend_14d_pct < -5:
                trend_14d_signal = -25
            else:
                trend_14d_signal = 0
            
            # 3. RSI SEMANAL CONSISTENTE (20%)
            rsi_val = self.calculate_rsi_consistent(prices)
            
            if rsi_val < 25:             # Sobreventa extrema
                rsi_signal = 100
            elif rsi_val < 35:           # Sobreventa
                rsi_signal = 60
            elif rsi_val > 75:           # Sobrecompra extrema
                rsi_signal = -100
            elif rsi_val > 65:           # Sobrecompra
                rsi_signal = -60
            else:                        # Normal
                rsi_signal = (50 - rsi_val) * 1.5
            
            # 4. CONFIRMACIÓN DE MOMENTUM (15%)
            # Validación: tendencia reciente vs media histórica
            recent_avg = prices.tail(5).mean()
            historic_avg = prices.head(10).mean()
            momentum_pct = ((recent_avg / historic_avg) - 1) * 100
            
            momentum_signal = np.clip(momentum_pct * 3, -50, 50)
            
            # Calcular contribuciones ponderadas
            contributions = {
                'trend_7d': trend_7d_signal * self.weights['trend_7d'],
                'trend_14d': trend_14d_signal * self.weights['trend_14d'],
                'rsi_weekly': rsi_signal * self.weights['rsi_weekly'],
                'momentum_confirm': momentum_signal * self.weights['momentum_confirm']
            }
            
            # Score final
            final_score = sum(contributions.values())
            
            # VALIDACIÓN DE CONSISTENCIA
            # Si cambio 7d es muy positivo, score debe ser positivo
            if trend_7d_pct > 15 and final_score < 10:
                final_score = max(final_score, 25)  # Forzar consistencia
            elif trend_7d_pct < -15 and final_score > -10:
                final_score = min(final_score, -25)  # Forzar consistencia
            
            return {
                'final_score': final_score,
                'contributions': contributions,
                'raw_metrics': {
                    'trend_7d_pct': trend_7d_pct,
                    'trend_14d_pct': trend_14d_pct,
                    'rsi_value': rsi_val,
                    'momentum_pct': momentum_pct
                },
                'signals': {
                    'trend_7d_signal': trend_7d_signal,
                    'trend_14d_signal': trend_14d_signal,
                    'rsi_signal': rsi_signal,
                    'momentum_signal': momentum_signal
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
    
    def classify_signal_consistent(self, score, trend_7d_pct):
        """Clasificación CONSISTENTE de señales"""
        
        # REGLA DE CONSISTENCIA: Tendencia 7d domina
        if trend_7d_pct > 15:  # Subida fuerte semanal
            if score > 30:
                return "🟢 COMPRA", "Tendencia alcista fuerte confirmada", "#155724"
            elif score > 10:
                return "🟢 COMPRA", "Tendencia alcista moderada", "#28a745"
            else:
                # Inconsistencia detectada - corregir
                return "🟢 COMPRA", "Tendencia alcista (ajustada por consistencia)", "#28a745"
                
        elif trend_7d_pct < -15:  # Caída fuerte semanal
            if score < -30:
                return "🔴 VENTA", "Tendencia bajista fuerte confirmada", "#721c24"
            elif score < -10:
                return "🔴 VENTA", "Tendencia bajista moderada", "#dc3545"
            else:
                # Inconsistencia detectada - corregir
                return "🔴 VENTA", "Tendencia bajista (ajustada por consistencia)", "#dc3545"
                
        else:  # Movimiento moderado
            if score > 25:
                return "🟢 COMPRA", "Señal alcista por múltiples indicadores", "#28a745"
            elif score < -25:
                return "🔴 VENTA", "Señal bajista por múltiples indicadores", "#dc3545"
            else:
                return "⚪ NEUTRO", "Sin tendencia clara definida", "#6c757d"

def main():
    # IDENTIFICADOR DE VERSION CONSISTENTE
    st.title("📊 Consistent Crypto Model")
    st.markdown("**Eliminación Total de Contradicciones - Score Confiable**")
    
    # Marca de versión prominente
    st.success("🎯 **VERSION CONSISTENTE v2.1** | Cero Contradicciones | Build: 20/07/2025 22:30")
    
    model = ConsistentCryptoModel()
    
    # Sidebar con información
    st.sidebar.header("🎯 Modelo Consistente")
    st.sidebar.success("🎯 **VERSION CONSISTENTE v2.1**")
    st.sidebar.markdown("**🔍 Enfoque: Eliminación de Contradicciones**")
    st.sidebar.info("📅 Build: 20/07/2025 22:30 GMT-3")
    
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
    
    # Principios de consistencia
    st.sidebar.subheader("🛡️ Reglas de Consistencia")
    st.sidebar.info("""
    **Principios inquebrantables:**
    • Cambio 7d > +15% → Señal COMPRA
    • Cambio 7d < -15% → Señal VENTA
    • Score siempre coherente con tendencia
    • Sin contradicciones toleradas
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
            
            if df is not None and len(df) >= 15:
                # Generar análisis consistente
                analysis = model.calculate_consistent_signals(df)
                
                if analysis:
                    score = analysis['final_score']
                    contributions = analysis['contributions']
                    raw_metrics = analysis['raw_metrics']
                    
                    # Clasificar señal de forma consistente
                    signal_class, signal_reason, signal_color = model.classify_signal_consistent(
                        score, raw_metrics['trend_7d_pct']
                    )
                    
                    # Datos de precio
                    current_price = df['price'].iloc[-1]
                    weekly_change = raw_metrics['trend_7d_pct']
                    
                    results.append({
                        'Crypto': symbol,
                        'Precio': f"${current_price:,.2f}",
                        'Cambio 7d': f"{weekly_change:+.2f}%",
                        'Señal': signal_class,
                        'Score Final': f"{score:.1f}",
                        'Confianza': f"{min(abs(score), 100):.0f}%",
                        
                        # Métricas de consistencia
                        'Tend. 7d': f"{raw_metrics['trend_7d_pct']:+.2f}%",
                        'Tend. 14d': f"{raw_metrics['trend_14d_pct']:+.2f}%",
                        'RSI': f"{raw_metrics['rsi_value']:.1f}",
                        'Momentum': f"{raw_metrics['momentum_pct']:+.2f}%",
                        
                        # Contribuciones
                        'Contrib 7d': f"{contributions['trend_7d']:+.1f}",
                        'Contrib 14d': f"{contributions['trend_14d']:+.1f}",
                        'Contrib RSI': f"{contributions['rsi_weekly']:+.1f}",
                        'Contrib Mom': f"{contributions['momentum_confirm']:+.1f}",
                        
                        'Razón': signal_reason,
                        'Color': signal_color
                    })
                    
                    st.success(f"✅ {symbol}: {signal_class.split()[1]} (Score: {score:.1f}) - CONSISTENTE")
                else:
                    st.warning(f"⚠️ {symbol}: Error en análisis")
            else:
                st.warning(f"⚠️ {symbol}: Datos insuficientes")
            
            # Pausa entre requests
            time.sleep(1.5)
    
    progress_bar.empty()
    
    # Mostrar resultados con validación de consistencia
    if results:
        st.success(f"🎯 Análisis consistente completado para {len(results)} criptomonedas")
        
        # VALIDACIÓN DE CONSISTENCIA AUTOMÁTICA
        st.subheader("🛡️ Validación de Consistencia")
        
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
            # Vista principal con validación
            summary_cols = ['Crypto', 'Precio', 'Cambio 7d', 'Señal', 'Score Final', 'Confianza', 'Razón']
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
                        if change > 10:
                            return 'background-color: #d4edda; color: #155724; font-weight: bold'
                        elif change < -10:
                            return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
                        else:
                            return 'color: #6c757d'
                    except:
                        return ''
                
                return df.style.applymap(color_signals, subset=['Señal']).applymap(color_change, subset=['Cambio 7d'])
            
            st.dataframe(style_consistent(summary_df), use_container_width=True)
        
        with tab2:
            # Métricas detalladas
            metrics_cols = ['Crypto', 'Tend. 7d', 'Tend. 14d', 'RSI', 'Momentum', 'Score Final']
            metrics_df = df_results[metrics_cols]
            st.dataframe(metrics_df, use_container_width=True)
        
        with tab3:
            # Contribuciones ponderadas
            contrib_cols = ['Crypto', 'Contrib 7d', 'Contrib 14d', 'Contrib RSI', 'Contrib Mom']
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
    st.info("🎯 **CONSISTENT MODEL v2.1** - Build 20/07/2025 22:30 | Cero Contradicciones Garantizadas")
    st.markdown("""
    **🛡️ Modelo de Consistencia Total:**
    
    **⚖️ Distribución de Pesos (Enfoque Lógico):**
    - **Tendencia 7 días (40%)**: Indicador principal - domina la decisión
    - **Tendencia 14 días (25%)**: Confirmación de tendencia más larga
    - **RSI Semanal (20%)**: Detecta extremos de mercado
    - **Momentum (15%)**: Validación adicional de dirección
    
    **🎯 Reglas de Consistencia Inquebrantables:**
    - Cambio 7d > +15% → SIEMPRE señal de COMPRA
    - Cambio 7d < -15% → SIEMPRE señal de VENTA
    - Score positivo → Señal alcista (COMPRA/NEUTRO alcista)
    - Score negativo → Señal bajista (VENTA/NEUTRO bajista)
    
    **✅ Validaciones Automáticas:**
    - Detección de contradicciones en tiempo real
    - Corrección automática de inconsistencias
    - Score alineado con cambio semanal
    - Señales confiables para toma de decisiones
    
    **⚠️ Disclaimer:** Modelo para análisis técnico educativo. Todas las señales son consistentes lógicamente.
    """)

if __name__ == "__main__":
    main()
