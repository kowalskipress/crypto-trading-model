import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import streamlit as st
from utils.indicators import TechnicalIndicators
from config.settings import MODEL_CONFIG

class TradingModel:
    """
    Modelo principal de trading que combina múltiples indicadores técnicos
    para generar señales de compra, venta o neutro.
    """
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.weights = MODEL_CONFIG['weights']
        self.thresholds = MODEL_CONFIG['thresholds']
        
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calcular todos los indicadores técnicos necesarios
        
        Args:
            df: DataFrame con OHLCV data
            
        Returns:
            Dict con valores de todos los indicadores
        """
        try:
            indicators = {}
            
            # RSI (14)
            indicators['rsi'] = self.indicators.rsi(df['close'])
            
            # MACD
            macd_line, macd_signal, _ = self.indicators.macd(df['close'])
            indicators['macd_line'] = macd_line
            indicators['macd_signal'] = macd_line - macd_signal  # Diferencia
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.indicators.bollinger_bands(df['close'])
            indicators['bb_position'] = self.indicators.bollinger_position(df['close'], bb_upper, bb_lower)
            
            # VWAP
            vwap = self.indicators.vwap(df)
            indicators['vwap_signal'] = self.indicators.vwap_signal(df['close'], vwap)
            
            # Stochastic %K
            indicators['stoch_k'] = self.indicators.stochastic_k(df)
            
            # On Balance Volume (OBV)
            obv = self.indicators.obv(df)
            indicators['obv_signal'] = self.indicators.obv_signal(obv)
            
            # Average True Range (ATR)
            atr = self.indicators.atr(df)
            indicators['atr_signal'] = self.indicators.atr_signal(atr)
            
            # Rate of Change (ROC)
            indicators['roc'] = self.indicators.roc(df['close'])
            
            return indicators
            
        except Exception as e:
            st.error(f"Error calculando indicadores: {str(e)}")
            return {}
    
    def normalize_indicators(self, indicators: Dict[str, float]) -> Dict[str, float]:
        """
        Normalizar todos los indicadores a escala 0-100
        
        Args:
            indicators: Dict con valores de indicadores
            
        Returns:
            Dict con indicadores normalizados
        """
        normalized = {}
        
        try:
            # RSI ya está en escala 0-100
            normalized['rsi'] = np.clip(indicators.get('rsi', 50), 0, 100)
            
            # MACD: convertir a escala 0-100 usando función sigmoid
            macd_signal = indicators.get('macd_signal', 0)
            normalized['macd'] = self._sigmoid_normalize(macd_signal) * 100
            
            # Bollinger Bands position ya está normalizada (0-100)
            normalized['bb_position'] = np.clip(indicators.get('bb_position', 50), 0, 100)
            
            # VWAP signal: normalizar usando sigmoid
            vwap_signal = indicators.get('vwap_signal', 0)
            normalized['vwap'] = self._sigmoid_normalize(vwap_signal) * 100
            
            # Stochastic %K ya está en escala 0-100
            normalized['stoch_k'] = np.clip(indicators.get('stoch_k', 50), 0, 100)
            
            # OBV signal: normalizar usando sigmoid
            obv_signal = indicators.get('obv_signal', 0)
            normalized['obv'] = self._sigmoid_normalize(obv_signal) * 100
            
            # ATR signal: normalizar usando sigmoid
            atr_signal = indicators.get('atr_signal', 0)
            normalized['atr'] = self._sigmoid_normalize(atr_signal) * 100
            
            # ROC: normalizar a escala 0-100
            roc = indicators.get('roc', 0)
            normalized['roc'] = np.clip((roc + 50), 0, 100)  # Asumir ROC en rango ±50
            
            return normalized
            
        except Exception as e:
            st.error(f"Error normalizando indicadores: {str(e)}")
            return {key: 50 for key in self.weights.keys()}  # Valores neutros por defecto
    
    def _sigmoid_normalize(self, value: float, steepness: float = 1.0) -> float:
        """
        Normalizar usando función sigmoid (0-1)
        """
        return 1 / (1 + np.exp(-steepness * value))
    
    def calculate_composite_index(self, normalized_indicators: Dict[str, float]) -> float:
        """
        Calcular el índice compuesto ponderado
        
        Args:
            normalized_indicators: Indicadores normalizados (0-100)
            
        Returns:
            Índice compuesto (0-100)
        """
        try:
            total_score = 0
            total_weight = 0
            
            for indicator, weight in self.weights.items():
                if indicator in normalized_indicators:
                    score = normalized_indicators[indicator]
                    total_score += score * weight
                    total_weight += weight
            
            if total_weight == 0:
                return 50  # Neutral si no hay datos
            
            composite_index = total_score / total_weight
            return np.clip(composite_index, 0, 100)
            
        except Exception as e:
            st.error(f"Error calculando índice compuesto: {str(e)}")
            return 50
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Generar señal de trading basada en los datos
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            Dict con señal, índice, confianza y métricas
        """
        if df is None or len(df) < MODEL_CONFIG['min_periods']:
            return None
        
        try:
            # Usar solo el último período para la señal
            latest_df = df.tail(MODEL_CONFIG['min_periods'])
            
            # Calcular indicadores
            raw_indicators = self.calculate_all_indicators(latest_df)
            
            if not raw_indicators:
                return None
            
            # Obtener valores más recientes (último período)
            current_indicators = {}
            for key, value in raw_indicators.items():
                if isinstance(value, (pd.Series, np.ndarray)):
                    current_indicators[key] = float(value.iloc[-1]) if hasattr(value, 'iloc') else float(value[-1])
                else:
                    current_indicators[key] = float(value)
            
            # Normalizar indicadores
            normalized = self.normalize_indicators(current_indicators)
            
            # Calcular índice compuesto
            composite_index = self.calculate_composite_index(normalized)
            
            # Determinar señal
            if composite_index >= self.thresholds['buy']:
                signal = 'COMPRA'
                confidence = min(95, 50 + (composite_index - self.thresholds['buy']) * 1.5)
            elif composite_index <= self.thresholds['sell']:
                signal = 'VENTA'
                confidence = min(95, 50 + (self.thresholds['sell'] - composite_index) * 1.5)
            else:
                signal = 'NEUTRO'
                # Confianza basada en qué tan cerca está del centro
                distance_from_center = abs(composite_index - 50)
                confidence = max(20, 50 - distance_from_center)
            
            # Preparar resultado
            result = {
                'signal': signal,
                'index': composite_index,
                'confidence': confidence,
                'metrics': {
                    'rsi': current_indicators.get('rsi', 50),
                    'macd_signal': current_indicators.get('macd_signal', 0),
                    'bb_position': current_indicators.get('bb_position', 50),
                    'vwap_signal': current_indicators.get('vwap_signal', 0),
                    'stoch_k': current_indicators.get('stoch_k', 50),
                    'obv_signal': current_indicators.get('obv_signal', 0),
                    'atr_signal': current_indicators.get('atr_signal', 0),
                    'roc': current_indicators.get('roc', 0)
                },
                'normalized_metrics': normalized,
                'timestamp': df.index[-1],
                'price': float(df['close'].iloc[-1])
            }
            
            return result
            
        except Exception as e:
            st.error(f"Error generando señal: {str(e)}")
            return None
    
    def backtest_strategy(self, df: pd.DataFrame, initial_capital: float = 10000) -> Dict:
        """
        Realizar backtesting de la estrategia
        
        Args:
            df: DataFrame con datos históricos
            initial_capital: Capital inicial para el backtest
            
        Returns:
            Dict con resultados del backtest
        """
        if df is None or len(df) < MODEL_CONFIG['min_periods'] * 2:
            return {}
        
        try:
            results = {
                'signals': [],
                'returns': [],
                'equity_curve': [],
                'trades': 0,
                'winning_trades': 0,
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
            
            capital = initial_capital
            position = 0  # 0 = no position, 1 = long, -1 = short
            entry_price = 0
            
            # Simular trading período por período
            min_periods = MODEL_CONFIG['min_periods']
            
            for i in range(min_periods, len(df)):
                # Obtener datos hasta el período actual
                current_data = df.iloc[:i+1]
                
                # Generar señal
                signal_data = self.generate_signal(current_data)
                
                if not signal_data:
                    continue
                
                signal = signal_data['signal']
                current_price = signal_data['price']
                
                results['signals'].append({
                    'timestamp': signal_data['timestamp'],
                    'signal': signal,
                    'price': current_price,
                    'index': signal_data['index']
                })
                
                # Ejecutar trading logic
                if signal == 'COMPRA' and position <= 0:
                    # Cerrar posición short si existe
                    if position == -1:
                        profit = (entry_price - current_price) / entry_price
                        capital *= (1 + profit)
                        results['trades'] += 1
                        if profit > 0:
                            results['winning_trades'] += 1
                    
                    # Abrir posición long
                    position = 1
                    entry_price = current_price
                
                elif signal == 'VENTA' and position >= 0:
                    # Cerrar posición long si existe
                    if position == 1:
                        profit = (current_price - entry_price) / entry_price
                        capital *= (1 + profit)
                        results['trades'] += 1
                        if profit > 0:
                            results['winning_trades'] += 1
                    
                    # Abrir posición short
                    position = -1
                    entry_price = current_price
                
                results['equity_curve'].append(capital)
            
            # Cerrar posición final si existe
            if position != 0:
                final_price = df['close'].iloc[-1]
                if position == 1:
                    profit = (final_price - entry_price) / entry_price
                else:
                    profit = (entry_price - final_price) / entry_price
                capital *= (1 + profit)
                results['trades'] += 1
                if profit > 0:
                    results['winning_trades'] += 1
            
            # Calcular métricas finales
            results['total_return'] = (capital - initial_capital) / initial_capital * 100
            results['win_rate'] = (results['winning_trades'] / results['trades'] * 100) if results['trades'] > 0 else 0
            
            # Calcular Sharpe ratio y max drawdown
            if len(results['equity_curve']) > 1:
                equity_series = pd.Series(results['equity_curve'])
                returns_series = equity_series.pct_change().dropna()
                
                if len(returns_series) > 0 and returns_series.std() != 0:
                    results['sharpe_ratio'] = returns_series.mean() / returns_series.std() * np.sqrt(252)  # Anualizado
                
                # Max drawdown
                peak = equity_series.cummax()
                drawdown = (equity_series - peak) / peak * 100
                results['max_drawdown'] = drawdown.min()
            
            return results
            
        except Exception as e:
            st.error(f"Error en backtesting: {str(e)}")
            return {}
    
    def get_model_info(self) -> Dict:
        """
        Obtener información del modelo
        """
        return {
            'name': 'Crypto Trading Model v1.0',
            'indicators_count': len(self.weights),
            'weights': self.weights,
            'thresholds': self.thresholds,
            'timeframe': MODEL_CONFIG['timeframe'],
            'min_periods': MODEL_CONFIG['min_periods']
        }
