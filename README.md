# 🚀 Crypto Trading Model

Modelo algorítmico de trading para criptomonedas que combina múltiples indicadores técnicos para generar señales de compra, venta o posición neutra cada 6 horas.

## 📊 Características Principales

- **Análisis automatizado** cada 6 horas (00:00, 06:00, 12:00, 18:00 UTC)
- **8 indicadores técnicos** combinados con ponderación estadística
- **4 criptomonedas** soportadas: Bitcoin, Ethereum, BNB, XRP
- **Interfaz web interactiva** con Streamlit
- **Backtesting integrado** para validación histórica
- **Datos en tiempo real** desde CoinGecko API

## 🎯 Metodología

### Indicadores Técnicos Utilizados

**Tier 1 - Alto Impacto (35%)**
- **RSI (14)**: 15% - Identificación de reversiones
- **MACD**: 20% - Análisis de tendencias

**Tier 2 - Impacto Medio (40%)**
- **Bollinger Bands**: 15% - Volatilidad normalizada
- **VWAP**: 12% - Soporte/resistencia institucional  
- **Stochastic %K**: 13% - Confirmación de momentum

**Tier 3 - Impacto Bajo (25%)**
- **OBV**: 10% - Flujo de dinero
- **ATR**: 8% - Volatilidad
- **ROC**: 7% - Momentum puro

### Generación de Señales

El modelo genera un **índice compuesto (0-100)** que se traduce en:

- 🟢 **COMPRA**: Índice ≥ 65
- 🟡 **NEUTRO**: 35 < Índice < 65  
- 🔴 **VENTA**: Índice ≤ 35

## 🚀 Deployment en Streamlit Cloud

### 1. Preparación del Repositorio

```bash
# Clonar el repositorio
git clone https://github.com/kowalskipress/crypto-trading-model.git
cd crypto-trading-model

# Crear estructura de carpetas
mkdir -p data models utils config
```

### 2. Subir Archivos al Repositorio

Copia todos los archivos proporcionados a sus respectivas carpetas:

```
crypto-trading-model/
├── app.py                 # Aplicación principal
├── requirements.txt       # Dependencias
├── README.md             # Esta documentación
├── data/
│   └── crypto_data.py    # Manejo de datos
├── models/
│   └── trading_model.py  # Lógica del modelo
├── utils/
│   └── indicators.py     # Indicadores técnicos
└── config/
    └── settings.py       # Configuraciones
```

### 3. Configurar Streamlit Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Conecta tu cuenta de GitHub
3. Selecciona el repositorio `crypto-trading-model`
4. Configura:
   - **Main file path**: `app.py`
   - **Python version**: 3.9+

### 4. Configurar Secretos (Opcional)

En Streamlit Cloud, ve a "Settings" > "Secrets" y agrega:

```toml
# .streamlit/secrets.toml
COINGECKO_API_KEY = "tu_api_key_aquí"
```

> **Nota**: La API key es opcional. El modelo funciona con la API pública de CoinGecko.

### 5. Deploy

1. Click "Deploy!"
2. Espera a que se complete la instalación
3. Tu aplicación estará disponible en `https://[tu-app].streamlit.app`

## 🔧 Uso Local (Opcional)

Si quieres probar localmente antes del deploy:

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
streamlit run app.py
```

## 📈 Cómo Usar la Aplicación

### Dashboard Principal
- **Vista general** de todas las señales
- **Gráfico de índices** por criptomoneda
- **Métricas resumidas** (compras/ventas/neutros)

### Análisis Detallado
- **Selección individual** de criptomonedas
- **Desglose de métricas** con ponderaciones
- **Información de confianza** de cada señal

### Backtesting
- **Rendimiento histórico** del modelo
- **Estadísticas de acierto**
- **Métricas de riesgo**

## ⚙️ Configuración Avanzada

### Modificar Ponderaciones

Edita `config/settings.py`:

```python
'weights': {
    'rsi': 0.15,        # Ajustar peso RSI
    'macd': 0.20,       # Ajustar peso MACD
    # ... otros indicadores
}
```

### Cambiar Umbrales

```python
'thresholds': {
    'buy': 65,      # Cambiar umbral de compra
    'sell': 35,     # Cambiar umbral de venta
}
```

### Agregar Criptomonedas

```python
CRYPTO_SYMBOLS = [
    'bitcoin',
    'ethereum', 
    'bnb',
    'xrp',
    'cardano',      # Agregar nueva crypto
    'solana'        # Agregar nueva crypto
]
```

## 📊 API de Datos

### CoinGecko API (Principal)
- **URL**: `https://api.coingecko.com/api/v3`
- **Rate Limit**: 10-30 requests/min (gratuita)
- **Datos**: OHLCV, volumen, precios actuales

### Endpoints Utilizados
- `/coins/{id}/ohlc` - Datos OHLC históricos
- `/coins/{id}/market_chart` - Datos de volumen
- `/simple/price` - Precios actuales

## 🛠️ Mantenimiento

### Actualización de Datos
- Los datos se actualizan automáticamente cada 6 horas
- Cache de 6 horas para optimizar rendimiento
- Botón manual de actualización disponible

### Monitoreo
- Logs de errores en la aplicación
- Validación automática de datos
- Fallbacks en caso de errores de API

## 📋 Troubleshooting

### Errores Comunes

**"No se pudieron cargar los datos"**
- Verificar conexión a internet
- API de CoinGecko puede estar temporalmente down
- Rate limit alcanzado (esperar 1 hora)

**"Error en cálculos"**
- Datos insuficientes (< 30 períodos)
- Valores nulos en los datos
- Verificar configuración de indicadores

**"App no carga en Streamlit Cloud"**
- Verificar que todos los archivos están en el repositorio
- Revisar requirements.txt
- Verificar logs en Streamlit Cloud

### Logs y Debugging

Para ver logs detallados, temporalmente cambia en `config/settings.py`:

```python
LOGGING_CONFIG = {
    'level': 'DEBUG',  # Cambiar de INFO a DEBUG
    # ...
}
```

## 🔒 Seguridad

- No hardcodear API keys en el código
- Usar Streamlit Secrets para configuración sensible
- La aplicación es read-only (no ejecuta trades reales)
- Datos públicos únicamente

## 📄 Licencia

Este proyecto es de código abierto. Úsenlo responsablemente y a su propio riesgo.

## ⚠️ Disclaimer

**Este modelo es solo para fines educativos y de investigación. No constituye consejo financiero. Las inversiones en criptomonedas conllevan riesgos significativos. Siempre haga su propia investigación antes de tomar decisiones de inversión.**

## 🤝 Contribuciones

Las contribuciones son bienvenidas:

1. Fork el repositorio
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📞 Soporte

Para preguntas o problemas:
- Abre un Issue en GitHub
- Revisa la documentación en el código
- Consulta los logs de la aplicación

---

**¡Happy Trading! 🚀📈**
