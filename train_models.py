#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
机器学习模型训练脚本

该脚本用于训练交易策略所需的机器学习模型，包括LSTM、随机森林等。

作者: 高级Python工程师
日期: 2024-05-22
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from user_data.strategies.utils.logging_utils import get_logger
from user_data.strategies.utils.config_manager import get_config_manager
from user_data.strategies.modules.signal_generator.ml_signals import MLSignalGenerator, ModelType
from user_data.strategies.utils.data_preprocessor import DataPreprocessor

# 获取日志记录器
logger = get_logger("train_models")

class ModelTrainer:
    """
    模型训练器

    负责训练各种机器学习模型用于交易信号生成
    """

    def __init__(self, data_dir: str = "user_data/data"):
        """
        初始化模型训练器

        参数:
            data_dir: 历史数据目录
        """
        self.data_dir = data_dir
        self.data_preprocessor = DataPreprocessor(data_dir)
        self.ml_generator = MLSignalGenerator()

        logger.info("模型训练器初始化完成")

    def prepare_training_data(self, symbol: str, timeframe: str,
                            start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        准备训练数据

        参数:
            symbol: 交易对符号
            timeframe: 时间周期
            start_date: 开始日期
            end_date: 结束日期

        返回:
            处理后的训练数据
        """
        logger.info(f"准备训练数据: {symbol} {timeframe} {start_date} - {end_date}")

        # 加载原始数据
        raw_data = self.data_preprocessor.load_ohlcv_data(
            symbol, timeframe, start_date, end_date
        )

        if raw_data.empty:
            logger.error(f"未找到数据: {symbol} {timeframe}")
            return pd.DataFrame()

        logger.info(f"加载原始数据: {len(raw_data)} 条记录")

        # 数据预处理
        # 1. 清洗数据
        cleaned_data = self.data_preprocessor.clean_data(raw_data)

        # 2. 添加技术指标
        processed_data = self.data_preprocessor.add_technical_indicators(cleaned_data)

        # 生成目标标签
        processed_data = self._generate_labels(processed_data)

        # 清理数据 - 移除包含NaN的行（特别是future_return为NaN的行）
        processed_data = processed_data.dropna()

        # 确保有足够的数据
        if len(processed_data) < 50:
            logger.warning(f"清理后数据不足: {len(processed_data)} 条记录")
            return pd.DataFrame()

        logger.info(f"处理后数据: {len(processed_data)} 条记录")
        return processed_data

    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标特征

        参数:
            data: 原始OHLCV数据

        返回:
            包含技术指标的数据
        """
        df = data.copy()

        # 价格相关特征
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']

        # 移动平均线
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # 布林带
        sma20 = df['close'].rolling(window=20).mean()
        std20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma20 + (std20 * 2)
        df['bb_lower'] = sma20 - (std20 * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma20
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # 成交量指标
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # 波动率
        df['volatility'] = df['close'].rolling(window=20).std()

        return df

    def _generate_labels(self, data: pd.DataFrame,
                        future_periods: int = 5,
                        threshold: float = 0.005) -> pd.DataFrame:
        """
        生成目标标签

        参数:
            data: 包含特征的数据
            future_periods: 未来周期数
            threshold: 价格变化阈值

        返回:
            包含标签的数据
        """
        df = data.copy()

        # 计算未来收益率
        df['future_return'] = df['close'].shift(-future_periods) / df['close'] - 1

        # 生成二分类标签 (1: 买入, 0: 卖出/持有)
        df['target'] = (df['future_return'] > threshold).astype(int)

        # 生成多分类标签
        conditions = [
            df['future_return'] > threshold,
            df['future_return'] < -threshold
        ]
        choices = [1, -1]  # 1: 买入, -1: 卖出, 0: 持有
        df['target_multiclass'] = np.select(conditions, choices, default=0)

        return df

    def train_lstm_model(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        训练LSTM模型

        参数:
            data: 训练数据
            symbol: 交易对符号

        返回:
            训练结果
        """
        logger.info(f"开始训练LSTM模型: {symbol}")

        # 选择特征列（使用DataPreprocessor生成的列名）
        feature_columns = [
            'price_change', 'rsi14', 'macd', 'macdsignal', 'macdhist',
            'sma20', 'ema20', 'bb_width', 'bb_position', 'volatility',
            'atr', 'cci', 'adx', 'obv', 'mfi'
        ]

        # 确保所有特征列都存在
        available_features = [col for col in feature_columns if col in data.columns]

        if len(available_features) < 5:
            logger.error(f"可用特征太少: {len(available_features)}")
            return {'success': False, 'message': '可用特征太少'}

        model_name = f"lstm_{symbol.replace('/', '_')}_5m"

        # 训练模型
        result = self.ml_generator.train_model(
            data=data,
            model_type=ModelType.LSTM,
            target_column='target',
            feature_columns=available_features,
            model_name=model_name,
            lookback=10,
            units=50,
            dropout=0.2,
            epochs=50,
            batch_size=32,
            n_features=len(available_features)
        )

        return result

    def train_random_forest_model(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        训练随机森林模型

        参数:
            data: 训练数据
            symbol: 交易对符号

        返回:
            训练结果
        """
        logger.info(f"开始训练随机森林模型: {symbol}")

        # 选择特征列（使用DataPreprocessor生成的列名）
        feature_columns = [
            'price_change', 'rsi14', 'macd', 'macdsignal', 'macdhist',
            'sma20', 'ema20', 'bb_width', 'bb_position', 'volatility',
            'atr', 'cci', 'adx', 'obv', 'mfi'
        ]

        # 确保所有特征列都存在
        available_features = [col for col in feature_columns if col in data.columns]

        model_name = f"rf_{symbol.replace('/', '_')}_5m"

        # 训练模型
        result = self.ml_generator.train_model(
            data=data,
            model_type=ModelType.RANDOM_FOREST,
            target_column='target',
            feature_columns=available_features,
            model_name=model_name,
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        return result

    def train_all_models(self, symbols: List[str], timeframe: str = "5m",
                        start_date: datetime = None, end_date: datetime = None) -> Dict[str, Dict[str, Any]]:
        """
        训练所有模型

        参数:
            symbols: 交易对符号列表
            timeframe: 时间周期
            start_date: 开始日期
            end_date: 结束日期

        返回:
            所有模型的训练结果
        """
        results = {}

        # 设置训练数据时间范围
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)  # 默认使用最近30天的数据

        for symbol in symbols:
            logger.info(f"开始训练 {symbol} 的模型")
            results[symbol] = {}

            # 准备训练数据
            training_data = self.prepare_training_data(symbol, timeframe, start_date, end_date)

            if training_data.empty:
                logger.warning(f"跳过 {symbol}，无训练数据")
                continue

            # 训练LSTM模型
            lstm_result = self.train_lstm_model(training_data, symbol)
            results[symbol]['lstm'] = lstm_result

            # 训练随机森林模型
            rf_result = self.train_random_forest_model(training_data, symbol)
            results[symbol]['random_forest'] = rf_result

        return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='机器学习模型训练工具')

    parser.add_argument('--symbols', type=str, nargs='+', default=['BTC/USDT'],
                       help='交易对符号列表，例如 BTC/USDT ETH/USDT')

    parser.add_argument('--timeframe', type=str, default='5m',
                       help='时间周期，例如 5m 1h')

    parser.add_argument('--start-date', type=str, default=None,
                       help='开始日期，格式为 YYYY-MM-DD，例如 2024-01-01')

    parser.add_argument('--end-date', type=str, default=None,
                       help='结束日期，格式为 YYYY-MM-DD，例如 2024-05-22')

    parser.add_argument('--data-dir', type=str, default='user_data/data',
                       help='历史数据目录')

    args = parser.parse_args()

    # 解析日期
    start_date = None
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"无效的开始日期格式: {args.start_date}，请使用 YYYY-MM-DD 格式")
            return

    end_date = None
    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"无效的结束日期格式: {args.end_date}，请使用 YYYY-MM-DD 格式")
            return

    # 初始化训练器
    trainer = ModelTrainer(args.data_dir)

    # 训练所有模型
    results = trainer.train_all_models(args.symbols, args.timeframe, start_date, end_date)

    # 输出结果
    logger.info("模型训练完成！")
    for symbol, symbol_results in results.items():
        logger.info(f"\n{symbol} 训练结果:")
        for model_type, result in symbol_results.items():
            if result['success']:
                metrics = result['metrics']
                logger.info(f"  {model_type}: 准确率={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")
            else:
                logger.error(f"  {model_type}: 训练失败 - {result['message']}")

if __name__ == '__main__':
    main()
