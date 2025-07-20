"""
Configuraciones principales del modelo de trading
"""

# Criptomonedas a analizar
CRYPTO_SYMBOLS = [
    'bitcoin',
    'ethereum', 
    'bnb',
    'xrp'
]

# Configuración del modelo
MODEL_CONFIG = {
    # Timeframe y frecuencia
    'timeframe': '6H',
    'analysis_frequency': 'Cada 6 horas',
    'historical_days': 90,
    'min_periods': 30,
    
    # Ponderaciones de métricas (deben sumar 1.0)
    'weights': {
        # Tier 1 - Alto Impacto (35%)
        'rsi': 0.15,        # RSI (14) - Mejor predictor de reversiones
        'macd': 0.20,       # MACD - Excelente para tendencias crypto
        
        # Tier 2 - Impacto Medio (40%)
        'bb_position': 0.15,  # Bollinger Bands Position - Volatilidad normalizada
        'vwap': 0.12,          # VWAP - Soporte/resistencia institucional
        'stoch_k': 0.13,       # Stochastic %K - Momentum confirmación
        
        # Tier 3 - Impacto Bajo (25%)
        'obv': 0.10,           # On Balance Volume - Flujo de dinero
        'atr': 0.08,           # ATR - Volatilidad
        'roc': 0.07            # Rate of Change - Momentum puro
    },
    
    # Umbrales para señales
    'thresholds': {
        'buy': 65,      # Índice >= 65 = COMPRA
        'sell': 35,     # Índice <= 35 = VENTA  
        'neutral_min': 35,  # 35 < Índice < 65 = NEUTRO
        'neutral_max': 65
    },
    
    # Parámetros de indicadores técnicos
    'indicator_params': {
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'bb_std': 2,
        'stoch_period': 14,
        'atr_period': 14,
        'roc_period': 14,
        'obv_comparison_period': 14
    }
}

# Configuración de la interfaz
UI_CONFIG = {
    'page_title': 'Crypto Trading Model',
    'page_icon': '📊',
    'layout': 'wide',
    'refresh_interval': 21600,  # 6 horas en segundos
    
    # Colores para señales
    'colors': {
        'buy': '#28a745',
        'sell': '#dc3545', 
        'neutral': '#ffc107',
        'background': '#f0f2f6'
    },
    
    # Emojis para señales
    'emojis': {
        'buy': '🟢',
        'sell': '🔴',
        'neutral': '🟡'
    }
}

# Configuración de APIs
API_CONFIG = {
    'coingecko': {
        'base_url': 'https://api.coingecko.com/api/v3',
        'rate_limit': 10,  # requests per minute for free tier
        'timeout': 30
    },
    
    'backup_apis': [
        {
            'name': 'cryptocompare',
            'base_url': 'https://min-api.cryptocompare.com/data',
            'rate_limit': 100  # requests per second for free tier
        }
    ]
}

# Validación de configuraciones
def validate_config():
    """
    Validar que las configuraciones sean correctas
    """
    # Verificar que los pesos sumen 1.0
    total_weight = sum(MODEL_CONFIG['weights'].values())
    if abs(total_weight - 1.0) > 0.001:
        raise ValueError(f"Los pesos deben sumar 1.0, actual: {total_weight}")
    
    # Verificar umbrales lógicos
    thresholds = MODEL_CONFIG['thresholds']
    if thresholds['sell'] >= thresholds['buy']:
        raise ValueError("El umbral de venta debe ser menor que el de compra")
    
    if thresholds['neutral_min'] != thresholds['sell'] or thresholds['neutral_max'] != thresholds['buy']:
        raise ValueError("Los umbrales neutrales deben coincidir con compra/venta")
    
    # Verificar que hay al menos una criptomoneda
    if not CRYPTO_SYMBOLS:
        raise ValueError("Debe haber al menos una criptomoneda configurada")
    
    return True

# Configuraciones específicas para Streamlit Cloud
STREAMLIT_CONFIG = {
    'secrets_required': [
        'COINGECKO_API_KEY'  # Opcional pero recomendado
    ],
    
    'cache_ttl': 21600,  # 6 horas en segundos
    'max_retries': 3,
    'retry_delay': 5  # segundos
}

# Información del modelo para documentación
MODEL_INFO = {
    'name': 'Crypto Trading Model',
    'version': '1.0.0',
    'description': 'Modelo de trading algorítmico para criptomonedas basado en múltiples indicadores técnicos',
    'author': 'Trading Model Team',
    'created_date': '2025-07-19',
    'last_updated': '2025-07-19',
    
    'features': [
        'Análisis cada 6 horas',
        '8 indicadores técnicos combinados',
        'Ponderación basada en estadística financiera',
        'Señales de compra/venta/neutro',
        'Backtesting integrado',
        'Interface web interactiva'
    ],
    
    'supported_cryptos': CRYPTO_SYMBOLS,
    
    'methodology': {
        'data_source': 'CoinGecko API',
        'timeframe': '6 horas',
        'historical_data': '90 días',
        'signal_generation': 'Índice compuesto ponderado',
        'normalization': 'Escala 0-100 con función sigmoid'
    }
}

# Constantes para cálculos
CONSTANTS = {
    'DAYS_PER_YEAR': 365,
    'HOURS_PER_DAY': 24,
    'MINUTES_PER_HOUR': 60,
    'SECONDS_PER_MINUTE': 60,
    
    # Para cálculos financieros
    'TRADING_DAYS_PER_YEAR': 365,  # Crypto opera 24/7
    'RISK_FREE_RATE': 0.02,  # 2% anual como aproximación
    
    # Límites para normalización
    'MAX_RSI': 100,
    'MIN_RSI': 0,
    'MAX_STOCH': 100,
    'MIN_STOCH': 0,
    'SIGMOID_STEEPNESS': 1.0
}

# Mensajes de error y advertencias
MESSAGES = {
    'errors': {
        'no_data': 'No se pudieron obtener datos para esta criptomoneda',
        'api_limit': 'Límite de API alcanzado. Intente más tarde',
        'connection_error': 'Error de conexión. Verifique su internet',
        'invalid_data': 'Los datos recibidos no son válidos',
        'calculation_error': 'Error en los cálculos. Datos insuficientes'
    },
    
    'warnings': {
        'limited_data': 'Datos limitados disponibles. Los resultados pueden ser menos precisos',
        'high_volatility': 'Alta volatilidad detectada. Use con precaución',
        'low_volume': 'Volumen bajo detectado. Señales menos confiables'
    },
    
    'info': {
        'data_updated': 'Datos actualizados exitosamente',
        'signal_generated': 'Nueva señal generada',
        'backtest_complete': 'Backtesting completado'
    }
}

# Configuración de logging (para debugging)
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_enabled': False,  # No usar archivos en Streamlit Cloud
    'console_enabled': True
}

# Validar configuración al importar
if __name__ == "__main__":
    try:
        validate_config()
        print("✅ Configuración validada correctamente")
    except ValueError as e:
        print(f"❌ Error en configuración: {e}")
else:
    # Validar silenciosamente cuando se importa
    try:
        validate_config()
    except ValueError as e:
        import streamlit as st
        st.error(f"Error en configuración: {e}")
