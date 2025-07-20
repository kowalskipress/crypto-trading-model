import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import streamlit as st
from typing import Optional, Dict, List

class CryptoDataFetcher:
    """
    Clase para obtener datos de criptomonedas desde CoinGecko API
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.api_key = api_key
        self.session = requests.Session()
        
        # Headers
        headers = {
            'User-Agent': 'Crypto-Trading-Model/1.0',
            'Accept': 'application/json'
        }
        
        if self.api_key:
            headers['x-cg-demo-api-key'] = self.api_key
            
        self.session.headers.update(headers)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 3.0  # Segundos entre requests (rate limit)
    
    def _make_request(self, url: str, params: Dict = None) -> Optional[Dict]:
        """
        Realizar request con rate limiting y manejo de errores
        """
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limit exceeded
                st.warning("Rate limit excedido. Esperando...")
                time.sleep(60)
                return self._make_request(url, params)  # Retry
            else:
                st.error(f"Error API: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            st.error(f"Error de conexión: {str(e)}")
            return None
    
    def get_coin_id(self, symbol: str) -> Optional[str]:
        """
        Obtener el ID de CoinGecko para un símbolo
        """
        # Mapeo directo para las cryptos principales
        symbol_to_id = {
            'bitcoin': 'bitcoin',
            'ethereum': 'ethereum', 
            'bnb': 'binancecoin',
            'xrp': 'ripple'
        }
        
        return symbol_to_id.get(symbol.lower())
    
    def get_historical_data(self, symbol: str, days: int = 90) -> Optional[pd.DataFrame]:
        """
        Obtener datos históricos de una criptomoneda
        
        Args:
            symbol: Símbolo de la crypto (bitcoin, ethereum, etc.)
            days: Número de días de historia
            
        Returns:
            DataFrame con columnas: timestamp, open, high, low, close, volume
        """
        coin_id = self.get_coin_id(symbol)
        if not coin_id:
            st.error(f"No se encontró ID para {symbol}")
            return None
        
        # Parámetros para obtener datos OHLC
        url = f"{self.base_url}/coins/{coin_id}/ohlc"
        params = {
            'vs_currency': 'usd',
            'days': str(days)
        }
        
        data = self._make_request(url, params)
        
        if not data:
            return None
        
        try:
            # Convertir a DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            
            # Convertir timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convertir a float
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col].astype(float)
            
            # Obtener datos de volumen por separado (CoinGecko OHLC no incluye volumen)
            volume_data = self.get_volume_data(coin_id, days)
            if volume_data is not None:
                df = df.join(volume_data, how='left')
                df['volume'] = df['volume'].fillna(0)
            else:
                df['volume'] = 0
            
            # Resample a 6 horas si es necesario
            df = self._resample_to_6h(df)
            
            return df.dropna()
            
        except Exception as e:
            st.error(f"Error procesando datos para {symbol}: {str(e)}")
            return None
    
    def get_volume_data(self, coin_id: str, days: int) -> Optional[pd.DataFrame]:
        """
        Obtener datos de volumen por separado
        """
        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': str(days),
            'interval': 'daily' if days > 1 else 'hourly'
        }
        
        data = self._make_request(url, params)
        
        if not data or 'total_volumes' not in data:
            return None
        
        try:
            volumes = data['total_volumes']
            df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['volume'] = df['volume'].astype(float)
            
            return df
            
        except Exception as e:
            st.warning(f"Error obteniendo volumen: {str(e)}")
            return None
    
    def _resample_to_6h(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resamplear datos a intervalos de 6 horas
        """
        try:
            # Resample a 6 horas
            df_6h = df.resample('6H').agg({
                'open': 'first',
                'high': 'max', 
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            return df_6h.dropna()
            
        except Exception as e:
            st.warning(f"Error en resample: {str(e)}")
            return df
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Obtener precios actuales de múltiples cryptos
        """
        coin_ids = []
        for symbol in symbols:
            coin_id = self.get_coin_id(symbol)
            if coin_id:
                coin_ids.append(coin_id)
        
        if not coin_ids:
            return {}
        
        url = f"{self.base_url}/simple/price"
        params = {
            'ids': ','.join(coin_ids),
            'vs_currencies': 'usd',
            'include_24hr_change': 'true'
        }
        
        data = self._make_request(url, params)
        
        if not data:
            return {}
        
        # Mapear de vuelta a símbolos
        id_to_symbol = {
            'bitcoin': 'bitcoin',
            'ethereum': 'ethereum',
            'binancecoin': 'bnb', 
            'ripple': 'xrp'
        }
        
        prices = {}
        for coin_id, price_data in data.items():
            symbol = id_to_symbol.get(coin_id)
            if symbol:
                prices[symbol] = {
                    'price': price_data['usd'],
                    'change_24h': price_data.get('usd_24h_change', 0)
                }
        
        return prices
    
    def test_connection(self) -> bool:
        """
        Probar conexión con la API
        """
        url = f"{self.base_url}/ping"
        data = self._make_request(url)
        return data is not None

# Función de utilidad para validar datos
def validate_dataframe(df: pd.DataFrame, symbol: str) -> bool:
    """
    Validar que el DataFrame tiene los datos necesarios
    """
    if df is None or df.empty:
        return False
    
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.warning(f"Columnas faltantes para {symbol}: {missing_columns}")
        return False
    
    # Verificar que hay suficientes datos
    if len(df) < 30:
        st.warning(f"Datos insuficientes para {symbol}: {len(df)} períodos")
        return False
    
    # Verificar valores nulos
    if df[required_columns].isnull().any().any():
        st.warning(f"Valores nulos detectados en {symbol}")
        return False
    
    return True
