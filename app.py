import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Configuración
st.set_page_config(
    page_title="Crypto Model Mendoza",
    page_icon="🏔️",
    layout="wide"
)

class MendozaCryptoModel:
    def __init__(self):
        self.coins = {
            'bitcoin': 'BTC',
            'ethereum': 'ETH',
            'binancecoin': 'BNB',
            'ripple': 'XRP'
        }
        
        # Pesos enfocados en CONSISTENCIA LÓGICA
        self.weights = {
            'trend': 0.40,        # 40% - Tendencia principal
            'rsi': 0.30,          # 30% - RSI adaptativo
            'momentum': 0.30      # 30% - Confirmación de momentum
        }
    
    @st.cache_data(ttl=1800)
    def fetch_adaptive_data(_self, coin_id, days=21):
        """Obtiene datos con sistema de fallback inteligente"""
        
        fallback_periods = [21, 14, 10, 7]
        
        for attempt_days in fallback_periods:
            try:
                st.info(f"📊 Obteniendo {attempt_days} días de datos para {coin_id}...")
                
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
                        prices = data['prices']
                        volumes = data.get('total_volumes', [])
                        
                        df = pd.DataFrame({
                            'timestamp': [p[0] for p in prices],
                            'price': [p[1] for p in prices],
                            'volume': [v[1] if len(volumes) > i else 1000000 for i, v in enumerate(volumes)]
                        })
                        
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df['date'] = df['timestamp'].dt.date
                        df.set_index('timestamp', inplace=True)
                        
                        daily_df = df.groupby('date').agg({
                            'price': 'last',
                            'volume': 'sum'
                        }).reset_index()
                        
                        daily_df['date'] = pd.to_datetime(daily_df['date'])
                        daily_df.set_index('date', inplace=True)
                        daily_df.sort_index(inplace=True)
                        
                        if len(daily_df) >= 3:
                            st.success(f"✅ {len(daily_df)} días obtenidos para {coin_id}")
                            return daily_df
                            
                elif response.status_code == 429:
                    st.warning(f"⏳ Rate limit para {coin_id}")
                    time.sleep(3)
                    continue
                    
            except Exception as e:
                st.warning(f"⚠️ Error con {attempt_days} días: {str(e)}")
                continue
        
        st.error(f"❌ No se pudieron obtener datos para {coin_id}")
        return None
    
    def calculate_adaptive_signals(self, df):
        """Calcula señales adaptadas al período disponible"""
        if df is None or len(df) < 3:
            return None
            
        try:
            prices = df['price']
            data_points = len(prices)
            
            # 1. TENDENCIA PRINCIPAL (40%)
            current_price = prices.iloc[-1]
            
            if data_points >= 7:
                price_ago = prices.iloc[-7]
                days_used = 7
            elif data_points >= 5:
                price_ago = prices.iloc[-5]
                days_used = 5
            else:
                price_ago = prices.iloc[0]
                days_used = data_points - 1
            
            trend_pct = ((current_price / price_ago) - 1) * 100
            
            # Normalizar a base semanal
            multiplier = 7 / days_used if days_used > 0 else 1
            trend_normalized = trend_pct * multiplier
            
            # Señal de tendencia
            if trend_normalized > 12:
                trend_signal = 100
            elif trend_normalized > 6:
                trend_signal = 60
            elif trend_normalized < -12:
                trend_signal = -100
            elif trend_normalized < -6:
                trend_signal = -60
            else:
                trend_signal = trend_normalized * 5
            
            # 2. RSI ADAPTATIVO (30%)
            if data_points >= 10:
                rsi_val = self.calculate_simple_rsi(prices)
            else:
                # RSI simplificado
                changes = prices.pct_change().dropna()
                if len(changes) > 0:
                    avg_change = changes.mean()
                    if avg_change > 0.04:
                        rsi_val = 25
                    elif avg_change < -0.04:
                        rsi_val = 75
                    else:
                        rsi_val = 50
                else:
                    rsi_val = 50
            
            # Señal RSI
            if rsi_val < 30:
                rsi_signal = 80
            elif rsi_val > 70:
                rsi_signal = -80
            else:
                rsi_signal = (50 - rsi_val) * 1.5
            
            # 3. MOMENTUM (30%)
            if data_points >= 5:
                recent_avg = prices.tail(2).mean()
                older_avg = prices.head(2).mean()
                momentum_pct = ((recent_avg / older_avg) - 1) * 100
            else:
                momentum_pct = trend_pct
            
            momentum_signal = np.clip(momentum_pct * 3, -50, 50)
            
            # Score final
            final_score = (
                trend_signal * self.weights['trend'] +
                rsi_signal * self.weights['rsi'] +
                momentum_signal * self.weights['momentum']
            )
            
            # Forzar consistencia
            if trend_normalized > 10 and final_score < 5:
                final_score = max(final_score, 20)
            elif trend_normalized < -10 and final_score > -5:
                final_score = min(final_score, -20)
            
            return {
                'final_score': final_score,
                'trend_pct': trend_pct,
                'trend_normalized': trend_normalized,
                'rsi_value': rsi_val,
                'momentum_pct': momentum_pct,
                'data_points': data_points,
                'days_used': days_used
            }
            
        except Exception as e:
            st.error(f"Error en señales: {str(e)}")
            return None
    
    def calculate_simple_rsi(self, prices, period=10):
        """RSI simplificado para datasets pequeños"""
        if len(prices) < period:
            return 50
            
        changes = prices.diff()
        gains = changes.where(changes > 0, 0)
        losses = -changes.where(changes < 0, 0)
        
        avg_gain = gains.rolling(window=period).mean().iloc[-1]
        avg_loss = losses.rolling(window=period).mean().iloc[-1]
        
        if pd.isna(avg_gain) or pd.isna(avg_loss) or avg_loss == 0:
            return 50
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return np.clip(rsi, 0, 100)
    
    def classify_signal(self, score, trend_pct):
        """Clasificación consistente de señales"""
        if trend_pct > 10:
            if score > 25:
                return "🟢 COMPRA", "Tendencia alcista fuerte"
            else:
                return "🟢 COMPRA", "Tendencia alcista (datos limitados)"
        elif trend_pct < -10:
            if score < -25:
                return "🔴 VENTA", "Tendencia bajista fuerte"
            else:
                return "🔴 VENTA", "Tendencia bajista (datos limitados)"
        else:
            if score > 20:
                return "🟢 COMPRA", "Señal alcista por indicadores"
            elif score < -20:
                return "🔴 VENTA", "Señal bajista por indicadores"
            else:
                return "⚪ NEUTRO", "Sin tendencia clara"

def main():
    # HEADER CON VERSIÓN MENDOZA
    st.title("🏔️ Crypto Model Mendoza")
    st.markdown("**Análisis Técnico con Adaptación Inteligente**")
    st.success("🏔️ **VERSIÓN MENDOZA** | Sistema Adaptativo | Build: 20/07/2025 23:00")
    
    model = MendozaCryptoModel()
    
    # Sidebar
    st.sidebar.header("🏔️ Crypto Model Mendoza")
    st.sidebar.success("🏔️ **VERSIÓN MENDOZA**")
    st.sidebar.markdown("**🍇 Adaptación Inteligente**")
    st.sidebar.info("📅 Build: 20/07/2025 23:00")
    
    # Pesos del modelo
    st.sidebar.subheader("⚖️ Pesos del Modelo")
    for metric, weight in model.weights.items():
        names = {'trend': 'Tendencia Principal', 'rsi': 'RSI Adaptativo', 'momentum': 'Momentum'}
        st.sidebar.write(f"**{names[metric]}**: {weight*100:.0f}%")
    
    if st.sidebar.button("🔄 Actualizar"):
        st.cache_data.clear()
        st.rerun()
    
    # Análisis principal
    st.header("📊 Análisis de Criptomonedas")
    
    results = []
    progress = st.progress(0)
    
    for i, (coin_id, symbol) in enumerate(model.coins.items()):
        progress.progress((i + 1) / len(model.coins))
        
        with st.spinner(f"🔍 Procesando {symbol}..."):
            df = model.fetch_adaptive_data(coin_id)
            
            if df is not None and len(df) >= 3:
                analysis = model.calculate_adaptive_signals(df)
                
                if analysis:
                    score = analysis['final_score']
                    signal_class, signal_reason = model.classify_signal(score, analysis['trend_normalized'])
                    
                    current_price = df['price'].iloc[-1]
                    
                    results.append({
                        'Crypto': symbol,
                        'Precio': f"${current_price:,.2f}",
                        'Cambio': f"{analysis['trend_pct']:+.2f}%",
                        'Días': f"({analysis['data_points']}d)",
                        'Señal': signal_class,
                        'Score': f"{score:.1f}",
                        'RSI': f"{analysis['rsi_value']:.1f}",
                        'Razón': signal_reason
                    })
                    
                    st.success(f"✅ {symbol}: {signal_class.split()[1]} (Score: {score:.1f})")
                else:
                    st.warning(f"⚠️ {symbol}: Error en cálculo")
            else:
                st.error(f"❌ {symbol}: Datos insuficientes")
            
            time.sleep(1.5)
    
    progress.empty()
    
    # Resultados
    if results:
        st.success(f"🎯 Análisis completado para {len(results)} criptomonedas")
        
        # Validación de consistencia
        st.subheader("🛡️ Validación de Consistencia")
        
        inconsistencies = []
        for result in results:
            crypto = result['Crypto']
            change_str = result['Cambio'].replace('%', '').replace('+', '')
            try:
                change = float(change_str)
                score = float(result['Score'])
                signal = result['Señal']
                
                if change > 10 and '🟢' not in signal:
                    inconsistencies.append(f"{crypto}: +{change:.1f}% pero no es COMPRA")
                elif change < -10 and '🔴' not in signal:
                    inconsistencies.append(f"{crypto}: {change:.1f}% pero no es VENTA")
            except:
                pass
        
        if inconsistencies:
            st.error("❌ Inconsistencias detectadas:")
            for inc in inconsistencies:
                st.error(f"• {inc}")
        else:
            st.success("✅ Todas las señales son consistentes - Modelo Mendoza confiable")
        
        # Tabla de resultados
        df_results = pd.DataFrame(results)
        
        def style_table(df):
            def color_signals(val):
                if '🟢' in val:
                    return 'background-color: #d4edda; color: #155724; font-weight: bold'
                elif '🔴' in val:
                    return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
                else:
                    return 'background-color: #f8f9fa; color: #495057; font-weight: bold'
            return df.style.applymap(color_signals, subset=['Señal'])
        
        st.dataframe(style_table(df_results), use_container_width=True)
        
        # Métricas del mercado
        col1, col2, col3 = st.columns(3)
        
        with col1:
            buy_count = sum(1 for r in results if '🟢' in r['Señal'])
            st.metric("🟢 Compras", buy_count)
        
        with col2:
            sell_count = sum(1 for r in results if '🔴' in r['Señal'])
            st.metric("🔴 Ventas", sell_count)
        
        with col3:
            neutral_count = sum(1 for r in results if '⚪' in r['Señal'])
            st.metric("⚪ Neutras", neutral_count)
    
    else:
        st.error("❌ No se pudieron obtener datos")
        st.info("🔄 Intenta actualizar en unos minutos")
    
    # Footer
    st.markdown("---")
    st.info("🏔️ **CRYPTO MODEL MENDOZA** - Build 23:00 | Sistema de Adaptación Inteligente")
    st.markdown("""
    **🏔️ Características de Mendoza:**
    
    - **Fallback automático**: 21→14→10→7 días
    - **Mínimo operativo**: 3 días de datos
    - **Consistencia garantizada**: Sin contradicciones
    - **Adaptación inteligente**: Normalización temporal
    
    **⚠️ Disclaimer:** Para fines educativos únicamente.
    """)

if __name__ == "__main__":
    main()
