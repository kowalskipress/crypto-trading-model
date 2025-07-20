import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time

# Configuración
st.set_page_config(
    page_title="Crypto Trading MVP",
    page_icon="📈",
    layout="wide"
)

def get_crypto_data():
    """Obtiene datos básicos de 4 criptomonedas - SIMPLE"""
    
    # API más simple y confiable
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        'ids': 'bitcoin,ethereum,binancecoin,ripple',
        'vs_currencies': 'usd',
        'include_24hr_change': 'true'
    }
    
    try:
        # Request simple sin headers complejos
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
            
    except:
        return None

def generate_simple_signal(price_change_24h):
    """Señal súper simple basada solo en cambio 24h"""
    
    if price_change_24h > 5:
        return "🟢 COMPRA", "Subida fuerte (+5%)"
    elif price_change_24h < -5:
        return "🔴 VENTA", "Caída fuerte (-5%)"
    elif price_change_24h > 2:
        return "🟢 COMPRA", "Subida moderada (+2%)"
    elif price_change_24h < -2:
        return "🔴 VENTA", "Caída moderada (-2%)"
    else:
        return "⚪ NEUTRO", "Movimiento lateral"

def main():
    st.title("📈 Crypto Trading MVP")
    st.markdown("**Modelo simple y confiable - Solo datos básicos**")
    
    # Botón actualizar
    if st.button("🔄 Actualizar Datos"):
        st.cache_data.clear()
    
    # Obtener datos
    with st.spinner("Obteniendo datos..."):
        data = get_crypto_data()
    
    if data:
        st.success("✅ Datos obtenidos correctamente")
        
        # Crear tabla
        results = []
        
        coins = {
            'bitcoin': 'BTC',
            'ethereum': 'ETH',
            'binancecoin': 'BNB', 
            'ripple': 'XRP'
        }
        
        for coin_id, symbol in coins.items():
            if coin_id in data:
                coin_data = data[coin_id]
                price = coin_data['usd']
                change_24h = coin_data.get('usd_24h_change', 0)
                
                # Generar señal simple
                signal, reason = generate_simple_signal(change_24h)
                
                results.append({
                    'Crypto': symbol,
                    'Precio': f"${price:,.2f}",
                    'Cambio 24h': f"{change_24h:+.2f}%",
                    'Señal': signal,
                    'Razón': reason
                })
        
        # Mostrar tabla
        if results:
            df = pd.DataFrame(results)
            
            # Colores simples
            def color_row(row):
                if '🟢' in row['Señal']:
                    return ['background-color: #d4edda'] * len(row)
                elif '🔴' in row['Señal']:
                    return ['background-color: #f8d7da'] * len(row)
                else:
                    return ['background-color: #f8f9fa'] * len(row)
            
            styled_df = df.style.apply(color_row, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            
            # Estadísticas simples
            col1, col2, col3 = st.columns(3)
            
            with col1:
                buy_count = sum(1 for r in results if '🟢' in r['Señal'])
                st.metric("Señales Compra", buy_count)
            
            with col2:
                sell_count = sum(1 for r in results if '🔴' in r['Señal'])
                st.metric("Señales Venta", sell_count)
            
            with col3:
                neutral_count = sum(1 for r in results if '⚪' in r['Señal'])
                st.metric("Señales Neutras", neutral_count)
        
    else:
        st.error("❌ No se pudieron obtener datos")
        st.info("💡 Mostrando datos de ejemplo:")
        
        # Datos de fallback
        demo_data = [
            {'Crypto': 'BTC', 'Precio': '$65,432', 'Cambio 24h': '+2.34%', 'Señal': '🟢 COMPRA', 'Razón': 'Subida moderada'},
            {'Crypto': 'ETH', 'Precio': '$3,245', 'Cambio 24h': '-1.23%', 'Señal': '⚪ NEUTRO', 'Razón': 'Movimiento lateral'},
            {'Crypto': 'BNB', 'Precio': '$542', 'Cambio 24h': '+0.87%', 'Señal': '⚪ NEUTRO', 'Razón': 'Movimiento lateral'},
            {'Crypto': 'XRP', 'Precio': '$0.623', 'Cambio 24h': '-6.45%', 'Señal': '🔴 VENTA', 'Razón': 'Caída fuerte'}
        ]
        
        demo_df = pd.DataFrame(demo_data)
        st.dataframe(demo_df, use_container_width=True)
    
    # Info
    st.info(f"📅 Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Metodología simple
    st.markdown("---")
    st.markdown("""
    **🎯 Metodología MVP - Súper Simple:**
    
    **Señales basadas en cambio 24h:**
    - 🟢 **COMPRA**: Subida > +2%
    - 🔴 **VENTA**: Caída > -2%  
    - ⚪ **NEUTRO**: Movimiento entre -2% y +2%
    
    **Umbrales fuertes:**
    - 🟢 **COMPRA FUERTE**: Subida > +5%
    - 🔴 **VENTA FUERTE**: Caída > -5%
    
    ✅ **MVP Funcional** - Sin dependencias complejas
    """)

if __name__ == "__main__":
    main()
