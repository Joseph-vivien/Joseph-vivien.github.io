#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
币安智能杠杆交易系统主程序入口

该脚本是系统的主入口，负责初始化各个组件并启动交易系统。

作者: 高级Python工程师
日期: 2024-05-21
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any
import time
from datetime import datetime
import pandas as pd

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.analyzers as btanalyzers

from user_data.strategies.utils.logging_utils import setup_logging, get_logger
from user_data.strategies.utils.config_manager import get_config_manager
from user_data.strategies.MasterStrategy import MasterStrategy

# 设置日志
setup_logging()
logger = get_logger("main")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='币安智能杠杆交易系统')

    parser.add_argument('--config', type=str, default='user_data/config/config.json',
                       help='配置文件路径')

    parser.add_argument('--mode', type=str, choices=['live', 'dry_run', 'backtest'],
                       default='dry_run', help='运行模式: live(实盘), dry_run(模拟), backtest(回测)')

    parser.add_argument('--strategy', type=str, default='MasterStrategy',
                       help='策略名称')

    parser.add_argument('--log-level', type=str,
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default='INFO', help='日志级别')

    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    logger.info(f"加载配置文件: {config_path}")

    config_manager = get_config_manager()
    config = config_manager.load_config(config_path)

    return config

def initialize_system(config: Dict[str, Any], mode: str) -> bt.Cerebro:
    """初始化系统"""
    logger.info(f"初始化系统，运行模式: {mode}")

    # 创建Cerebro引擎
    cerebro = bt.Cerebro()

    # 设置初始资金
    initial_cash = config.get('dry_run_wallet', 10000)
    cerebro.broker.setcash(initial_cash)

    # 设置佣金
    commission = config.get('commission', 0.001)  # 0.1%
    cerebro.broker.setcommission(commission=commission)

    # 添加策略（稍后会设置total_bars参数）
    cerebro.addstrategy(MasterStrategy)

    # 添加分析器
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(btanalyzers.Returns, _name='returns')

    # 根据模式加载数据
    symbols = config.get('symbols', ['BTC/USDT'])
    timeframe = config.get('timeframe', '1h')

    if mode == 'backtest':
        # 加载回测数据
        data_dir = config.get('data_dir', 'user_data/data/historical/binance')

        # 获取时间范围设置
        backtest_config = config.get('backtest', {})
        timerange = backtest_config.get('timerange', None)

        start_date = None
        end_date = None

        if timerange:
            try:
                # 解析时间范围 "20240101-20240102"
                if '-' in timerange:
                    start_str, end_str = timerange.split('-')
                    start_date = pd.to_datetime(start_str, format='%Y%m%d')
                    end_date = pd.to_datetime(end_str, format='%Y%m%d')
                    logger.info(f"回测时间范围: {start_date.date()} 到 {end_date.date()}")
                else:
                    logger.warning(f"无效的时间范围格式: {timerange}")
            except Exception as e:
                logger.error(f"解析时间范围失败: {e}")

        for symbol in symbols:
            symbol_path = os.path.join(data_dir, symbol.replace('/', '_'), timeframe)

            if not os.path.exists(symbol_path):
                logger.warning(f"数据目录不存在: {symbol_path}")
                continue

            # 获取所有CSV文件
            csv_files = [f for f in os.listdir(symbol_path) if f.endswith('.csv')]

            if not csv_files:
                logger.warning(f"未找到数据文件: {symbol_path}")
                continue

            # 合并所有CSV文件
            dfs = []
            for csv_file in csv_files:
                file_path = os.path.join(symbol_path, csv_file)
                try:
                    df = pd.read_csv(file_path)
                    df['datetime'] = pd.to_datetime(df['timestamp'])
                    dfs.append(df)
                except Exception as e:
                    logger.error(f"加载数据文件失败: {file_path}, {e}")

            if not dfs:
                logger.warning(f"未加载到任何数据: {symbol}")
                continue

            # 合并数据
            df = pd.concat(dfs)
            df = df.sort_values('datetime').reset_index(drop=True)

            # 应用时间范围过滤
            if start_date is not None or end_date is not None:
                original_len = len(df)
                if start_date is not None:
                    df = df[df['datetime'] >= start_date]
                if end_date is not None:
                    df = df[df['datetime'] <= end_date]
                df = df.reset_index(drop=True)
                logger.info(f"时间过滤: {original_len} -> {len(df)} 条记录")

            # 添加技术指标计算
            from user_data.strategies.utils.data_preprocessor import DataPreprocessor
            preprocessor = DataPreprocessor()

            # 重命名列以匹配DataPreprocessor期望的格式
            df_for_indicators = df.rename(columns={'datetime': 'timestamp'})

            # 计算技术指标
            df_with_indicators = preprocessor.add_technical_indicators(df_for_indicators)

            # 将技术指标添加回原始DataFrame
            for col in df_with_indicators.columns:
                if col not in df.columns and col != 'timestamp':
                    df[col] = df_with_indicators[col].values

            # 创建数据源
            data = btfeeds.PandasData(
                dataname=df,
                datetime='datetime',
                open='open',
                high='high',
                low='low',
                close='close',
                volume='volume',
                openinterest=-1,
                name=symbol
            )

            # 添加数据
            cerebro.adddata(data)
            logger.info(f"已加载 {symbol} 数据: {len(df)} 条记录")

    elif mode in ['live', 'dry_run']:
        # 使用实时数据
        try:
            # 导入CCXT
            import ccxt

            # 获取交易所配置
            exchange_config = config.get('exchange', {})
            exchange_name = exchange_config.get('name', 'binance')
            api_key = exchange_config.get('key', '')
            api_secret = exchange_config.get('secret', '')

            # 创建交易所对象
            if exchange_name == 'binance':
                if mode == 'live' and api_key and api_secret:
                    exchange = ccxt.binance({
                        'apiKey': api_key,
                        'secret': api_secret,
                        'enableRateLimit': True,
                        'options': {
                            'defaultType': 'future'
                        }
                    })
                    logger.info("已连接到币安交易所（实盘模式）")
                else:
                    exchange = ccxt.binance({
                        'enableRateLimit': True,
                        'options': {
                            'defaultType': 'future'
                        }
                    })
                    logger.info("已连接到币安交易所（模拟模式）")
            else:
                logger.error(f"不支持的交易所: {exchange_name}")
                return cerebro

            # 创建CCXT数据源
            from user_data.strategies.data_feeds.ccxt_data import CCXTStore

            # 创建CCXT Store
            store = CCXTStore(
                exchange_name=exchange_name,
                config={
                    'apiKey': api_key,
                    'secret': api_secret,
                    'sandbox': mode != 'live',
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'future'
                    }
                }
            )

            # 添加实时数据
            for symbol in symbols:
                data = store.getdata(
                    symbol=symbol,
                    timeframe=timeframe,
                    historical=True,
                    live=(mode == 'live'),
                    limit=2000
                )
                cerebro.adddata(data)
                logger.info(f"已添加实时数据: {symbol}")

            # 注意：我们的自定义CCXT数据源不包含broker功能
            # 如果需要实盘交易，需要单独实现broker

        except ImportError:
            logger.error("无法导入CCXT模块，请安装: pip install ccxt backtrader-ccxt")
        except Exception as e:
            logger.error(f"设置实时数据失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    return cerebro

def run_trading_system(cerebro: bt.Cerebro, mode: str) -> None:
    """运行交易系统"""
    logger.info(f"启动交易系统，模式: {mode}")

    try:
        # 记录初始资金
        initial_cash = cerebro.broker.getvalue()
        logger.info(f"初始资金: {initial_cash:.2f}")

        # 运行回测
        results = cerebro.run()

        if not results:
            logger.error("没有策略运行结果，可能是因为没有添加策略或数据")
            return

        strategy = results[0]

        # 记录最终资金
        final_cash = cerebro.broker.getvalue()
        logger.info(f"最终资金: {final_cash:.2f}")
        logger.info(f"收益率: {(final_cash - initial_cash) / initial_cash * 100:.2f}%")

        # 输出分析结果
        sharpe_ratio = strategy.analyzers.sharpe.get_analysis()['sharperatio']
        if sharpe_ratio is not None:
            logger.info(f"夏普比率: {sharpe_ratio:.2f}")

        drawdown = strategy.analyzers.drawdown.get_analysis()
        logger.info(f"最大回撤: {drawdown.max.drawdown:.2f}%")

        trade_analysis = strategy.analyzers.trades.get_analysis()

        # 安全地获取交易统计
        total_trades = 0
        winning_trades = 0
        losing_trades = 0

        try:
            if hasattr(trade_analysis, 'total') and hasattr(trade_analysis.total, 'closed'):
                total_trades = trade_analysis.total.closed
            if hasattr(trade_analysis, 'won') and hasattr(trade_analysis.won, 'total'):
                winning_trades = trade_analysis.won.total
            if hasattr(trade_analysis, 'lost') and hasattr(trade_analysis.lost, 'total'):
                losing_trades = trade_analysis.lost.total
        except (AttributeError, KeyError):
            logger.warning("无法获取详细交易统计")

        if total_trades > 0:
            win_rate = winning_trades / total_trades * 100
            logger.info(f"总交易次数: {total_trades}")
            logger.info(f"盈利交易: {winning_trades}")
            logger.info(f"亏损交易: {losing_trades}")
            logger.info(f"胜率: {win_rate:.2f}%")
        else:
            logger.info("本次运行没有执行任何交易")

        # 如果是回测模式，绘制图表
        if mode == 'backtest':
            cerebro.plot(style='candle', barup='green', bardown='red',
                        volup='green', voldown='red', grid=True,
                        plotdist=0.1, subplot=True)

    except Exception as e:
        logger.error(f"系统错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # 加载配置
    config_path = args.config
    if args.mode == 'backtest':
        config_path = 'user_data/config/config_backtest.json'
    elif args.mode == 'dry_run':
        config_path = 'user_data/config/config_dry_run.json'

    config = load_config(config_path)

    # 初始化系统
    cerebro = initialize_system(config, args.mode)

    # 运行交易系统
    run_trading_system(cerebro, args.mode)

if __name__ == '__main__':
    main()
