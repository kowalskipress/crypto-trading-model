import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Configuración
st.set_page_config(
    page_title="Weekly Trend Crypto Model",
    page_icon="📈",
    layout="wide"
)

class WeeklyTrendModel:
    def __init__(self):
        self.coins = {
            'bitcoin': 'BTC',
            'ethereum': 'ETH',
            'binancecoin': 'BNB',
            'ripple': 'XRP'
        }
        
        # Pesos enfocados en tendencias de mediano plazo
        self.weights = {
            'trend_7d': 0.30,      # 30% - Tendencia 7 días
            'trend_14d': 0.25,     # 25% - Tendencia 14 días  
            'rsi_weekly': 0.20,    # 20% - RSI suavizado
            'ma_cross': 0.15,      # 15% - Cruce de medias móviles
            'volume_trend': 0.10   # 10% - Tendencia de volumen
        }
    
    @st.cache_data(ttl=1800)  # Cache 30 minutos
    def fetch_historical_data(_self, coin_id, days=21):
        """Obtiene datos históricos para análisis de tendencias"""
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
                
                # Crear DataFrame con datos diarios
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
                    'price': 'last',    # Precio de cierre del día
                    'volume': 'sum'     # Volumen total del día
                }).reset_index()
                
                daily_df['date'] = pd.to_datetime(daily_df['date'])
                daily_df.set_index('date', inplace=True)
                daily_df.sort_index(inplace=True)
                
                return daily_df
                
            return None
            
        except Exception as e:
            return None
    
    def calculate_trend_signals(self, df):
        """Calcula señales basadas en tendencias de mediano plazo"""
        if df is None or len(df) < 15:
            return None
            
        try:
            # 1. TENDENCIA 7 DÍAS (30% peso)
            # Comparar últimos 7 días vs 7 días anteriores
            recent_7d = df['price'].tail(7).mean()
            previous_7d = df['price'].iloc[-14:-7].mean()
            trend_7d_pct = ((recent_7d / previous_7d) - 1) * 100
            
            # 2. TENDENCIA 14 DÍAS (25% peso)  
            # Comparar últimos 14 días vs promedio total
            recent_14d = df['price'].tail(14).mean()
            total_avg = df['price'].mean()
            trend_14d_pct = ((recent_14d / total_avg) - 1) * 100
            
            # 3. RSI SEMANAL SUAVIZADO (20% peso)
            # RSI calculado sobre promedios de 3 días para suavizar
            df['price_3d'] = df['price'].rolling(window=3, center=True).mean()
            price_changes = df['price_3d'].diff()
            gains = price_changes.where(price_changes > 0, 0)
            losses = -price_changes.where(price_changes < 0, 0)
            
            avg_gain = gains.rolling(window=14).mean()
            avg_loss = losses.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi_weekly = 100 - (100 / (1 + rs))
            current_rsi = rsi_weekly.iloc[-1] if not pd.isna(rsi_weekly.iloc[-1]) else 50
            
            # 4. CRUCE DE MEDIAS MÓVILES (15% peso)
            # MA de 5 días vs MA de 10 días (suavizadas)
            ma_5 = df['price'].rolling(window=5).mean()
            ma_10 = df['price'].rolling(window=10).mean()
            
            current_ma5 = ma_5.iloc[-1]
            current_ma10 = ma_10.iloc[-1]
            ma_cross_signal = ((current_ma5 / current_ma10) - 1) * 100
            
            # 5. TENDENCIA DE VOLUMEN (10% peso)
            # Volumen reciente vs volumen histórico
            recent_volume = df['volume'].tail(7).mean()
            historic_volume = df['volume'].head(14).mean()
            volume_trend = ((recent_volume / historic_volume) - 1) * 100
            
            # Convertir a señales normalizadas (-100 a +100)
            signals = {
                'trend_7d': np.clip(trend_7d_pct * 4, -100, 100),  # Amplificar señal
                'trend_14d': np.clip(trend_14d_pct * 5, -100, 100),
                'rsi_weekly': self._normalize_rsi(current_rsi),
                'ma_cross': np.clip(ma_cross_signal * 20, -100, 100),
                'volume_trend': np.clip(volume_trend * 2, -50, 50)  # Menor impacto
            }
            
            # Calcular score ponderado final
            final_score = sum(signals[key] * self.weights[key] for key in signals.keys())
            
            return {
                'final_score': final_score,
                'signals': signals,
                'metrics': {
                    'trend_7d_pct': trend_7d_pct,
                    'trend_14d_pct': trend_14d_pct,
                    'rsi_weekly': current_rsi,
                    'ma_cross_pct': ma_cross_signal,
                    'volume_trend_pct': volume_trend,
                    'recent_7d_avg': recent_7d,
                    'previous_7d_avg': previous_7d
                }
            }
            
        except Exception as e:
            st.error(f"Error calculando tendencias: {str(e)}")
            return None
    
    def _normalize_rsi(self, rsi):
        """Normaliza RSI a escala -100 a +100"""
        if rsi < 30:
            return 100  # Sobreventa = Compra
        elif rsi > 70:
            return -100  # Sobrecompra = Venta
        else:
            return (50 - rsi) * 2  # Escala lineal
    
    def classify_signal(self, score):
        """Clasifica la señal final"""
        if score > 30:
            return "🟢 COMPRA", "Tendencia alcista fuerte"
        elif score > 15:
            return "🟢 COMPRA", "Tendencia alcista moderada"
        elif score < -30:
            return "🔴 VENTA", "Tendencia bajista fuerte"
        elif score < -15:
            return "🔴 VENTA", "Tendencia bajista moderada"
        else:
            return "⚪ NEUTRO", "Sin tendencia clara"

def main():
    st.title("📊 Weekly Trend Crypto Model")
    st.markdown("**Análisis de tendencias de mediano plazo - Ignora ruido intradiario**")
    
    model = WeeklyTrendModel()
    
    # Sidebar con configuración
    st.sidebar.header("⚙️ Configuración del Modelo")
    st.sidebar.markdown("**🎯 Enfoque: Tendencias Semanales**")
    
    # Mostrar pesos
    st.sidebar.subheader("📊 Pesos de Indicadores")
    for indicator, weight in model.weights.items():
        display_name = {
            'trend_7d': 'Tendencia 7 días',
            'trend_14d': 'Tendencia 14 días',
            'rsi_weekly': 'RSI Semanal',
            'ma_cross': 'Cruce Medias Móviles',
            'volume_trend': 'Tendencia Volumen'
        }
        st.sidebar.write(f"**{display_name[indicator]}**: {weight*100:.0f}%")
    
    if st.sidebar.button("🔄 Actualizar Datos"):
        st.cache_data.clear()
        st.rerun()
    
    # Obtener y analizar datos
    st.header("📈 Análisis de Tendencias Semanales")
    
    results = []
    progress_bar = st.progress(0)
    
    for i, (coin_id, symbol) in enumerate(model.coins.items()):
        progress_bar.progress((i + 1) / len(model.coins))
        
        with st.spinner(f"Analizando tendencias de {symbol}..."):
            # Obtener datos históricos (21 días para análisis robusto)
            df = model.fetch_historical_data(coin_id, days=21)
            
            if df is not None and len(df) >= 15:
                # Calcular señales de tendencia
                analysis = model.calculate_trend_signals(df)
                
                if analysis:
                    score = analysis['final_score']
                    signals = analysis['signals']
                    metrics = analysis['metrics']
                    
                    # Clasificar señal
                    signal_class, signal_reason = model.classify_signal(score)
                    
                    # Precio actual (último día)
                    current_price = df['price'].iloc[-1]
                    
                    # Cambio semanal (7 días)
                    week_ago_price = df['price'].iloc[-7] if len(df) >= 7 else df['price'].iloc[0]
                    weekly_change = ((current_price / week_ago_price) - 1) * 100
                    
                    results.append({
                        'Crypto': symbol,
                        'Precio': f"${current_price:,.2f}",
                        'Cambio 7d': f"{weekly_change:+.2f}%",
                        'Señal': signal_class,
                        'Score Final': f"{score:.1f}",
                        'Confianza': f"{min(abs(score), 100):.0f}%",
                        'Tend. 7d': f"{metrics['trend_7d_pct']:+.2f}%",
                        'Tend. 14d': f"{metrics['trend_14d_pct']:+.2f}%",
                        'RSI Sem': f"{metrics['rsi_weekly']:.1f}",
                        'MA Cross': f"{metrics['ma_cross_pct']:+.2f}%",
                        'Vol Trend': f"{metrics['volume_trend_pct']:+.1f}%",
                        'Razón': signal_reason
                    })
                    
                    st.success(f"✅ {symbol}: {signal_class.split()[1]} (Score: {score:.1f})")
                else:
                    st.warning(f"⚠️ {symbol}: Error en análisis")
            else:
                st.warning(f"⚠️ {symbol}: Datos insuficientes")
            
            # Pausa entre requests
            time.sleep(1)
    
    progress_bar.empty()
    
    # Mostrar resultados
    if results:
        st.success(f"🎉 Análisis completado para {len(results)} criptomonedas")
        
        # Tabla principal
        df_results = pd.DataFrame(results)
        
        # Aplicar estilos
        def style_table(df):
            def color_signals(val):
                if '🟢' in val and 'fuerte' in val:
                    return 'background-color: #155724; color: white; font-weight: bold'
                elif '🟢' in val:
                    return 'background-color: #d4edda; color: #155724; font-weight: bold'
                elif '🔴' in val and 'fuerte' in val:
                    return 'background-color: #721c24; color: white; font-weight: bold'
                elif '🔴' in val:
                    return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
                else:
                    return 'background-color: #f8f9fa; color: #495057; font-weight: bold'
            
            def color_score(val):
                try:
                    score = float(val)
                    if score > 30:
                        return 'background-color: #155724; color: white; font-weight: bold'
                    elif score > 15:
                        return 'background-color: #d4edda; color: #155724; font-weight: bold'
                    elif score < -30:
                        return 'background-color: #721c24; color: white; font-weight: bold'
                    elif score < -15:
                        return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
                    else:
                        return 'color: #495057'
                except:
                    return ''
            
            def color_percentage(val):
                try:
                    pct = float(val.replace('%', '').replace('+', ''))
                    if pct > 5:
                        return 'color: #28a745; font-weight: bold'
                    elif pct > 2:
                        return 'color: #28a745'
                    elif pct < -5:
                        return 'color: #dc3545; font-weight: bold'
                    elif pct < -2:
                        return 'color: #dc3545'
                    else:
                        return 'color: #6c757d'
                except:
                    return ''
            
            styled = df.style.applymap(color_signals, subset=['Señal'])
            styled = styled.applymap(color_score, subset=['Score Final'])
            
            pct_cols = ['Cambio 7d', 'Tend. 7d', 'Tend. 14d', 'MA Cross', 'Vol Trend']
            for col in pct_cols:
                if col in df.columns:
                    styled = styled.applymap(color_percentage, subset=[col])
                    
            return styled
        
        st.dataframe(style_table(df_results), use_container_width=True)
        
        # Análisis de mercado
        st.subheader("📊 Resumen del Mercado")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            buy_signals = sum(1 for r in results if '🟢' in r['Señal'])
            st.metric("🟢 Señales Alcistas", buy_signals)
        
        with col2:
            sell_signals = sum(1 for r in results if '🔴' in r['Señal'])
            st.metric("🔴 Señales Bajistas", sell_signals)
            
        with col3:
            neutral_signals = sum(1 for r in results if '⚪' in r['Señal'])
            st.metric("⚪ Señales Neutras", neutral_signals)
            
        with col4:
            avg_score = np.mean([float(r['Score Final']) for r in results])
            st.metric("📊 Score Promedio", f"{avg_score:.1f}")
        
        # Interpretación de señales
        st.subheader("🎯 Interpretación de Señales")
        
        for result in results:
            with st.expander(f"{result['Crypto']} - {result['Señal']} (Score: {result['Score Final']})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Precio actual**: {result['Precio']}")
                    st.write(f"**Cambio 7 días**: {result['Cambio 7d']}")
                    st.write(f"**Razón**: {result['Razón']}")
                    st.write(f"**Confianza**: {result['Confianza']}")
                
                with col2:
                    st.write(f"**Tendencia 7d**: {result['Tend. 7d']}")
                    st.write(f"**Tendencia 14d**: {result['Tend. 14d']}")
                    st.write(f"**RSI Semanal**: {result['RSI Sem']}")
                    st.write(f"**Cruce MA**: {result['MA Cross']}")
    
    else:
        st.error("❌ No se pudieron obtener datos suficientes")
        st.info("🔄 Intenta actualizar en unos minutos")
    
    # Footer informativo
    st.markdown("---")
    st.markdown("""
    **🎯 Metodología - Enfoque en Tendencias de Mediano Plazo:**
    
    **📊 Indicadores Utilizados:**
    - **Tendencia 7 días (30%)**: Compara últimos 7 días vs 7 días anteriores
    - **Tendencia 14 días (25%)**: Compara últimos 14 días vs promedio histórico
    - **RSI Semanal (20%)**: RSI suavizado para eliminar ruido diario
    - **Cruce MA (15%)**: Media móvil 5 días vs 10 días
    - **Tendencia Volumen (10%)**: Volumen reciente vs histórico
    
    **🎯 Umbrales de Señal:**
    - Score > +30: 🟢 **COMPRA FUERTE**
    - Score +15 a +30: 🟢 **COMPRA MODERADA**
    - Score -15 a +15: ⚪ **NEUTRO**
    - Score -30 a -15: 🔴 **VENTA MODERADA**  
    - Score < -30: 🔴 **VENTA FUERTE**
    
    **✅ Beneficios del Modelo:**
    - Ignora volatilidad intradiaria
    - Se enfoca en tendencias sostenibles
    - Reduce señales falsas por ruido de mercado
    - Ideal para decisiones de inversión de mediano plazo
    
    **⚠️ Disclaimer:** Para fines educativos. No constituye asesoramiento financiero.
    """)

if __name__ == "__main__":
    main()
