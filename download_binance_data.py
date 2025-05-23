#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
币安数据下载工具

该脚本用于从币安API下载历史K线数据，支持多个交易对和时间周期。

作者: 高级Python工程师
日期: 2024-05-21
"""

import os
import sys
import time
import argparse
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any
import ccxt
import json

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from user_data.strategies.utils.logging_utils import get_logger
from user_data.strategies.utils.config_manager import get_config_manager

# 获取日志记录器
logger = get_logger("download_binance_data")

class BinanceDataDownloader:
    """
    币安数据下载器
    
    从币安API下载历史K线数据，支持多个交易对和时间周期
    """
    
    def __init__(self, api_key: str = "", api_secret: str = "", 
                data_dir: str = "user_data/data/historical/binance"):
        """
        初始化数据下载器
        
        参数:
            api_key: 币安API密钥
            api_secret: 币安API密钥
            data_dir: 数据保存目录
        """
        self.data_dir = data_dir
        
        # 确保数据目录存在
        os.makedirs(data_dir, exist_ok=True)
        
        # 初始化币安API客户端
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # 使用永续合约
                'adjustForTimeDifference': True,
                'recvWindow': 10000
            }
        })
        
        # 支持的时间周期
        self.timeframes = {
            '1m': {'seconds': 60, 'limit': 1000},
            '3m': {'seconds': 180, 'limit': 1000},
            '5m': {'seconds': 300, 'limit': 1000},
            '15m': {'seconds': 900, 'limit': 1000},
            '30m': {'seconds': 1800, 'limit': 1000},
            '1h': {'seconds': 3600, 'limit': 1000},
            '2h': {'seconds': 7200, 'limit': 1000},
            '4h': {'seconds': 14400, 'limit': 1000},
            '6h': {'seconds': 21600, 'limit': 1000},
            '8h': {'seconds': 28800, 'limit': 1000},
            '12h': {'seconds': 43200, 'limit': 1000},
            '1d': {'seconds': 86400, 'limit': 1000},
            '3d': {'seconds': 259200, 'limit': 1000},
            '1w': {'seconds': 604800, 'limit': 1000},
            '1M': {'seconds': 2592000, 'limit': 1000}
        }
        
    def download_data(self, symbol: str, timeframe: str, 
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None,
                     save: bool = True) -> pd.DataFrame:
        """
        下载指定交易对和时间周期的历史数据
        
        参数:
            symbol: 交易对符号
            timeframe: 时间周期
            start_date: 开始日期
            end_date: 结束日期
            save: 是否保存到文件
            
        返回:
            历史数据
        """
        if timeframe not in self.timeframes:
            logger.error(f"不支持的时间周期: {timeframe}")
            return pd.DataFrame()
            
        # 设置默认日期范围
        if end_date is None:
            end_date = datetime.now()
            
        if start_date is None:
            # 默认下载最近30天的数据
            start_date = end_date - timedelta(days=30)
            
        logger.info(f"下载 {symbol} {timeframe} 数据，时间范围: {start_date} - {end_date}")
        
        # 转换为毫秒时间戳
        start_timestamp = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        
        # 计算需要下载的数据批次
        timeframe_seconds = self.timeframes[timeframe]['seconds']
        limit = self.timeframes[timeframe]['limit']
        
        # 每批次数据的时间范围
        batch_timerange = timeframe_seconds * limit * 1000  # 毫秒
        
        # 初始化结果列表
        all_candles = []
        
        # 分批下载数据
        current_start = start_timestamp
        
        while current_start < end_timestamp:
            current_end = min(current_start + batch_timerange, end_timestamp)
            
            try:
                # 下载一批数据
                candles = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_start,
                    limit=limit
                )
                
                if not candles:
                    logger.warning(f"未获取到数据: {symbol} {timeframe} {datetime.fromtimestamp(current_start/1000)}")
                    # 移动到下一批
                    current_start = current_end
                    continue
                    
                # 添加到结果列表
                all_candles.extend(candles)
                
                # 获取最后一条数据的时间戳作为下一批的起始时间
                last_timestamp = candles[-1][0]
                current_start = last_timestamp + timeframe_seconds * 1000
                
                # 打印进度
                progress = (current_start - start_timestamp) / (end_timestamp - start_timestamp) * 100
                logger.info(f"下载进度: {progress:.2f}% ({len(all_candles)} 条数据)")
                
                # 限制API请求频率
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"下载数据失败: {e}")
                # 限制API请求频率，出错后等待更长时间
                time.sleep(2)
                # 移动到下一批
                current_start = current_end
                
        # 转换为DataFrame
        if all_candles:
            df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 转换时间戳为datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 去重并排序
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            # 保存到文件
            if save:
                self._save_data(df, symbol, timeframe)
                
            return df
        else:
            logger.warning(f"未获取到任何数据: {symbol} {timeframe}")
            return pd.DataFrame()
    
    def download_multiple_data(self, symbols: List[str], timeframes: List[str],
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        下载多个交易对和时间周期的历史数据
        
        参数:
            symbols: 交易对符号列表
            timeframes: 时间周期列表
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            数据字典 {symbol: {timeframe: dataframe}}
        """
        results = {}
        
        for symbol in symbols:
            results[symbol] = {}
            
            for timeframe in timeframes:
                logger.info(f"下载 {symbol} {timeframe} 数据")
                
                try:
                    df = self.download_data(symbol, timeframe, start_date, end_date)
                    results[symbol][timeframe] = df
                    
                except Exception as e:
                    logger.error(f"下载 {symbol} {timeframe} 数据失败: {e}")
                    results[symbol][timeframe] = pd.DataFrame()
                    
                # 限制API请求频率
                time.sleep(1)
                
        return results
    
    def _save_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """
        保存数据到文件
        
        参数:
            df: 数据
            symbol: 交易对符号
            timeframe: 时间周期
            
        返回:
            是否保存成功
        """
        if df.empty:
            logger.warning(f"数据为空，无法保存: {symbol} {timeframe}")
            return False
            
        try:
            # 构建保存路径
            symbol_dir = os.path.join(self.data_dir, symbol.replace('/', '_'))
            timeframe_dir = os.path.join(symbol_dir, timeframe)
            
            # 确保目录存在
            os.makedirs(timeframe_dir, exist_ok=True)
            
            # 按月分割数据
            df['month'] = df['timestamp'].dt.strftime('%Y%m')
            
            for month, month_df in df.groupby('month'):
                # 生成文件名
                filename = f"{month}_ohlcv.csv"
                file_path = os.path.join(timeframe_dir, filename)
                
                # 删除辅助列
                month_df = month_df.drop(columns=['month'])
                
                # 保存到CSV文件
                month_df.to_csv(file_path, index=False)
                
                logger.info(f"已保存数据: {file_path} ({len(month_df)} 条记录)")
                
            return True
            
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
            return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='币安数据下载工具')
    
    parser.add_argument('--symbols', type=str, nargs='+', default=['BTC/USDT'],
                       help='交易对符号列表，例如 BTC/USDT ETH/USDT')
                       
    parser.add_argument('--timeframes', type=str, nargs='+', 
                       default=['1m', '5m', '15m', '1h', '4h'],
                       help='时间周期列表，例如 1m 5m 15m 1h 4h')
                       
    parser.add_argument('--start-date', type=str, default=None,
                       help='开始日期，格式为 YYYY-MM-DD')
                       
    parser.add_argument('--end-date', type=str, default=None,
                       help='结束日期，格式为 YYYY-MM-DD')
                       
    parser.add_argument('--config', type=str, default='user_data/config/config.json',
                       help='配置文件路径')
                       
    args = parser.parse_args()
    
    # 加载配置
    config_manager = get_config_manager()
    config = config_manager.load_config(args.config)
    
    # 获取API密钥
    api_key = config.get('exchange.key', '')
    api_secret = config.get('exchange.secret', '')
    
    # 解析日期
    start_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        
    end_date = None
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        
    # 初始化下载器
    downloader = BinanceDataDownloader(api_key, api_secret)
    
    # 下载数据
    downloader.download_multiple_data(args.symbols, args.timeframes, start_date, end_date)
    
    logger.info("数据下载完成")

if __name__ == '__main__':
    main()
