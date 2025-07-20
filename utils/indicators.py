import pandas as pd
import numpy as np
from typing import Tuple, Optional
import streamlit as st

class TechnicalIndicators:
    """
    Clase con implementaciones de indicadores técnicos para trading
    """
    
    def __init__(self):
        pass
    
    def rsi(self, prices: pd.Series, period: int = 14) -> float:
        """
        Relative Strength Index (RSI)
        
        Args:
            prices: Serie de precios
            period: Período para el cálculo (default: 14)
            
        Returns:
            Valor RSI (0-100)
        """
        try:
            delta = prices.diff()
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            avg_gain = gains.rolling(window=period).mean()
            avg_loss = losses.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
            
        except Exception as e:
            st.warning(f"Error calculando RSI: {str(e)}")
            return 50.0
    
    def macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """
        Moving Average Convergence Divergence (MACD)
        
        Args:
            prices: Serie de precios
            fast: Período EMA rápida (default: 12)
            slow: Período EMA lenta (default: 26)
            signal: Período línea de señal (default: 9)
            
        Returns:
            Tuple con (macd_line, signal_line, histogram)
        """
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            return (
                float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
                float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0,
                float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0
            )
            
        except Exception as e:
            st.warning(f"Error calculando MACD: {str(e)}")
            return 0.0, 0.0, 0.0
    
    def bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[float, float, float]:
        """
        Bollinger Bands
        
        Args:
            prices: Serie de precios
            period: Período para la media móvil (default: 20)
            std_dev: Número de desviaciones estándar (default: 2)
            
        Returns:
            Tuple con (upper_band, middle_band, lower_band)
        """
        try:
            middle_band = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)
            
            return (
                float(upper_band.iloc[-1]) if not pd.isna(upper_band.iloc[-1]) else 0.0,
                float(middle_band.iloc[-1]) if not pd.isna(middle_band.iloc[-1]) else 0.0,
                float(lower_band.iloc[-1]) if not pd.isna(lower_band.iloc[-1]) else 0.0
            )
            
        except Exception as e:
            st.warning(f"Error calculando Bollinger Bands: {str(e)}")
            current_price = float(prices.iloc[-1]) if len(prices) > 0 else 0.0
            return current_price, current_price, current_price
    
    def bollinger_position(self, prices: pd.Series, upper_band: float, lower_band: float) -> float:
        """
        Posición del precio dentro de las Bollinger Bands (0-100)
        
        Args:
            prices: Serie de precios
            upper_band: Banda superior
            lower_band: Banda inferior
            
        Returns:
            Posición normalizada (0-100)
        """
        try:
            current_price = float(prices.iloc[-1])
            
            if upper_band == lower_band:
                return 50.0
            
            position = (current_price - lower_band) / (upper_band - lower_band) * 100
            return np.clip(position, 0, 100)
            
        except Exception as e:
            st.warning(f"Error calculando posición Bollinger: {str(e)}")
            return 50.0
    
    def vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        Volume Weighted Average Price (VWAP)
        
        Args:
            df: DataFrame con columnas 'high', 'low', 'close', 'volume'
            
        Returns:
            Serie VWAP
        """
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            
            return vwap.fillna(df['close'])
            
        except Exception as e:
            st.warning(f"Error calculando VWAP: {str(e)}")
            return df['close'] if 'close' in df.columns else pd.Series([0])
    
    def vwap_signal(self, prices: pd.Series, vwap: pd.Series) -> float:
        """
        Señal basada en VWAP (precio vs VWAP)
        
        Args:
            prices: Serie de precios
            vwap: Serie VWAP
            
        Returns:
            Señal normalizada (-1 a 1)
        """
        try:
            current_price = float(prices.iloc[-1])
            current_vwap = float(vwap.iloc[-1])
            
            if current_vwap == 0:
                return 0.0
            
            # Diferencia porcentual
            diff = (current_price - current_vwap) / current_vwap
            
            # Normalizar usando tanh para mantener en rango -1 a 1
            return float(np.tanh(diff * 5))  # Factor 5 para sensibilidad
            
        except Exception as e:
            st.warning(f"Error calculando señal VWAP: {str(e)}")
            return 0.0
    
    def stochastic_k(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Stochastic %K
        
        Args:
            df: DataFrame con columnas 'high', 'low', 'close'
            period: Período para el cálculo (default: 14)
            
        Returns:
            Valor %K (0-100)
        """
        try:
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            
            stoch_k = ((df['close'] - low_min) / (high_max - low_min)) * 100
            
            return float(stoch_k.iloc[-1]) if not pd.isna(stoch_k.iloc[-1]) else 50.0
            
        except Exception as e:
            st.warning(f"Error calculando Stochastic %K: {str(e)}")
            return 50.0
    
    def obv(self, df: pd.DataFrame) -> pd.Series:
        """
        On Balance Volume (OBV)
        
        Args:
            df: DataFrame con columnas 'close', 'volume'
            
        Returns:
            Serie OBV
        """
        try:
            obv = pd.Series(index=df.index, dtype=float)
            obv.iloc[0] = df['volume'].iloc[0]
            
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            return obv
            
        except Exception as e:
            st.warning(f"Error calculando OBV: {str(e)}")
            return pd.Series([0] * len(df), index=df.index)
    
    def obv_signal(self, obv: pd.Series, period: int = 14) -> float:
        """
        Señal basada en OBV (momentum del volumen)
        
        Args:
            obv: Serie OBV
            period: Período para la comparación
            
        Returns:
            Señal normalizada (-1 a 1)
        """
        try:
            if len(obv) < period:
                return 0.0
            
            current_obv = obv.iloc[-1]
            past_obv = obv.iloc[-period]
            
            if past_obv == 0:
                return 0.0
            
            # Cambio porcentual en OBV
            obv_change = (current_obv - past_obv) / abs(past_obv)
            
            # Normalizar usando tanh
            return float(np.tanh(obv_change))
            
        except Exception as e:
            st.warning(f"Error calculando señal OBV: {str(e)}")
            return 0.0
    
    def atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average True Range (ATR)
        
        Args:
            df: DataFrame con columnas 'high', 'low', 'close'
            period: Período para el cálculo (default: 14)
            
        Returns:
            Serie ATR
        """
        try:
            high_low = df['high'] - df['low']
            high_close_prev = abs(df['high'] - df['close'].shift(1))
            low_close_prev = abs(df['low'] - df['close'].shift(1))
            
            true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return atr.fillna(0)
            
        except Exception as e:
            st.warning(f"Error calculando ATR: {str(e)}")
            return pd.Series([0] * len(df), index=df.index)
    
    def atr_signal(self, atr: pd.Series, period: int = 14) -> float:
        """
        Señal basada en ATR (volatilidad)
        
        Args:
            atr: Serie ATR
            period: Período para la comparación
            
        Returns:
            Señal normalizada (0-1, donde 1 = alta volatilidad)
        """
        try:
            if len(atr) < period:
                return 0.5
            
            current_atr = atr.iloc[-1]
            avg_atr = atr.tail(period).mean()
            
            if avg_atr == 0:
                return 0.5
            
            # Ratio de volatilidad actual vs promedio
            volatility_ratio = current_atr / avg_atr
            
            # Normalizar a 0-1 usando función logística
            signal = 1 / (1 + np.exp(-2 * (volatility_ratio - 1)))
            
            return float(signal)
            
        except Exception as e:
            st.warning(f"Error calculando señal ATR: {str(e)}")
            return 0.5
    
    def roc(self, prices: pd.Series, period: int = 14) -> float:
        """
        Rate of Change (ROC)
        
        Args:
            prices: Serie de precios
            period: Período para el cálculo (default: 14)
            
        Returns:
            ROC en porcentaje
        """
        try:
            if len(prices) < period + 1:
                return 0.0
            
            current_price = prices.iloc[-1]
            past_price = prices.iloc[-period-1]
            
            if past_price == 0:
                return 0.0
            
            roc = ((current_price - past_price) / past_price) * 100
            
            return float(roc)
            
        except Exception as e:
            st.warning(f"Error calculando ROC: {str(e)}")
            return 0.0
    
    def sma(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average (SMA)
        
        Args:
            prices: Serie de precios
            period: Período para la media
            
        Returns:
            Serie SMA
        """
        try:
            return prices.rolling(window=period).mean()
        except Exception as e:
            st.warning(f"Error calculando SMA: {str(e)}")
            return prices
    
    def ema(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Exponential Moving Average (EMA)
        
        Args:
            prices: Serie de precios
            period: Período para la media
            
        Returns:
            Serie EMA
        """
        try:
            return prices.ewm(span=period).mean()
        except Exception as e:
            st.warning(f"Error calculando EMA: {str(e)}")
            return prices
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validar que el DataFrame tiene las columnas necesarias
        
        Args:
            df: DataFrame a validar
            
        Returns:
            True si es válido, False si no
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if df is None or df.empty:
            return False
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.warning(f"Columnas faltantes: {missing_columns}")
            return False
        
        # Verificar datos nulos
        if df[required_columns].isnull().any().any():
            st.warning("Se encontraron valores nulos en los datos")
            return False
        
        return True
