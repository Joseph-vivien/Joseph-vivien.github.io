#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
币安智能杠杆交易系统回测脚本

该脚本用于执行策略回测，支持多个交易对和时间周期。

作者: 高级Python工程师
日期: 2024-05-21
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from user_data.strategies.utils.logging_utils import setup_logging, get_logger
from user_data.strategies.utils.config_manager import get_config_manager
from user_data.strategies.MasterStrategy import MasterStrategy
from user_data.strategies.utils.performance_metrics import calculate_performance_metrics

# 设置日志
setup_logging()
logger = get_logger("backtest")

class BacktestEngine:
    """
    回测引擎
    
    执行策略回测，计算性能指标，生成回测报告
    """
    
    def __init__(self, config_path: str):
        """
        初始化回测引擎
        
        参数:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        
        # 加载配置
        self.config_manager = get_config_manager()
        self.config = self.config_manager.load_config(config_path)
        
        # 设置回测参数
        self.backtest_params = self.config.get('backtest', {})
        
        # 回测结果目录
        self.results_dir = self.backtest_params.get('results_dir', 'user_data/backtest_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 回测报告目录
        self.reports_dir = os.path.join(self.results_dir, 'reports')
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # 回测图表目录
        self.plots_dir = os.path.join(self.results_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # 回测交易记录目录
        self.trades_dir = os.path.join(self.results_dir, 'trades')
        os.makedirs(self.trades_dir, exist_ok=True)
        
    def load_data(self, symbols: List[str], timeframes: List[str], 
                 start_date: datetime, end_date: datetime) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        加载回测数据
        
        参数:
            symbols: 交易对符号列表
            timeframes: 时间周期列表
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            数据字典 {symbol: {timeframe: dataframe}}
        """
        logger.info(f"加载回测数据: {symbols}, {timeframes}, {start_date} - {end_date}")
        
        data_dir = self.config.get('data_dir', 'user_data/data/historical/binance')
        
        data = {}
        
        for symbol in symbols:
            data[symbol] = {}
            symbol_dir = os.path.join(data_dir, symbol.replace('/', '_'))
            
            for timeframe in timeframes:
                timeframe_dir = os.path.join(symbol_dir, timeframe)
                
                if not os.path.exists(timeframe_dir):
                    logger.warning(f"数据目录不存在: {timeframe_dir}")
                    continue
                    
                # 获取所有月度数据文件
                files = [f for f in os.listdir(timeframe_dir) if f.endswith('_ohlcv.csv')]
                
                if not files:
                    logger.warning(f"未找到数据文件: {timeframe_dir}")
                    continue
                    
                # 加载并合并数据
                dfs = []
                
                for file in files:
                    file_path = os.path.join(timeframe_dir, file)
                    try:
                        df = pd.read_csv(file_path)
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        dfs.append(df)
                    except Exception as e:
                        logger.error(f"加载数据文件失败: {file_path}, {e}")
                        
                if not dfs:
                    logger.warning(f"未加载到任何数据: {symbol} {timeframe}")
                    continue
                    
                # 合并数据
                df = pd.concat(dfs)
                
                # 去重并排序
                df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                
                # 筛选时间范围
                df = df[(df['timestamp'] >= pd.Timestamp(start_date)) & 
                        (df['timestamp'] <= pd.Timestamp(end_date))]
                
                if df.empty:
                    logger.warning(f"筛选后数据为空: {symbol} {timeframe}")
                    continue
                    
                data[symbol][timeframe] = df
                logger.info(f"已加载 {symbol} {timeframe} 数据: {len(df)} 条记录")
                
        return data
        
    def run_backtest(self, strategy_name: str, symbols: List[str], timeframes: List[str],
                    start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        执行回测
        
        参数:
            strategy_name: 策略名称
            symbols: 交易对符号列表
            timeframes: 时间周期列表
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            回测结果
        """
        logger.info(f"开始回测: {strategy_name}, {symbols}, {timeframes}, {start_date} - {end_date}")
        
        # 加载数据
        data = self.load_data(symbols, timeframes, start_date, end_date)
        
        if not data:
            logger.error("未加载到任何数据，无法执行回测")
            return {}
            
        # 初始化策略
        strategy_config = self.config.copy()
        strategy_config['symbols'] = symbols
        strategy_config['timeframes'] = timeframes
        strategy_config['start_date'] = start_date
        strategy_config['end_date'] = end_date
        strategy_config['mode'] = 'backtest'
        
        strategy = MasterStrategy(strategy_config)
        
        # 执行回测
        backtest_results = strategy.run_backtest(data)
        
        # 保存回测结果
        self.save_backtest_results(strategy_name, backtest_results, start_date, end_date)
        
        return backtest_results
        
    def save_backtest_results(self, strategy_name: str, results: Dict[str, Any],
                             start_date: datetime, end_date: datetime) -> None:
        """
        保存回测结果
        
        参数:
            strategy_name: 策略名称
            results: 回测结果
            start_date: 开始日期
            end_date: 结束日期
        """
        # 生成结果文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_name = f"{strategy_name}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{timestamp}"
        
        # 保存交易记录
        trades_df = pd.DataFrame(results.get('trades', []))
        if not trades_df.empty:
            trades_path = os.path.join(self.trades_dir, f"{result_name}_trades.csv")
            trades_df.to_csv(trades_path, index=False)
            logger.info(f"已保存交易记录: {trades_path}")
            
        # 保存性能指标
        metrics = results.get('metrics', {})
        metrics_path = os.path.join(self.reports_dir, f"{result_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"已保存性能指标: {metrics_path}")
            
        # 生成回测报告
        report_path = os.path.join(self.reports_dir, f"{result_name}_report.html")
        self.generate_html_report(results, report_path)
        logger.info(f"已生成回测报告: {report_path}")
            
        # 生成回测图表
        if not trades_df.empty:
            plot_path = os.path.join(self.plots_dir, f"{result_name}_equity_curve.png")
            self.plot_equity_curve(results, plot_path)
            logger.info(f"已生成权益曲线图: {plot_path}")
            
    def generate_html_report(self, results: Dict[str, Any], report_path: str) -> None:
        """
        生成HTML回测报告
        
        参数:
            results: 回测结果
            report_path: 报告保存路径
        """
        # 生成HTML报告的代码
        # 这里简化处理，实际应用中可以使用模板引擎生成更复杂的报告
        metrics = results.get('metrics', {})
        trades = results.get('trades', [])
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>回测报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>回测报告</h1>
            <h2>性能指标</h2>
            <table>
                <tr><th>指标</th><th>值</th></tr>
        """
        
        for key, value in metrics.items():
            if isinstance(value, float):
                value_str = f"{value:.4f}"
                if key in ['total_profit', 'profit_factor', 'sharpe_ratio', 'sortino_ratio']:
                    cls = 'positive' if value > 0 else 'negative'
                    html += f'<tr><td>{key}</td><td class="{cls}">{value_str}</td></tr>\n'
                else:
                    html += f'<tr><td>{key}</td><td>{value_str}</td></tr>\n'
            else:
                html += f'<tr><td>{key}</td><td>{value}</td></tr>\n'
                
        html += """
            </table>
            <h2>交易记录</h2>
            <table>
                <tr>
                    <th>交易ID</th>
                    <th>交易对</th>
                    <th>方向</th>
                    <th>开仓时间</th>
                    <th>开仓价格</th>
                    <th>平仓时间</th>
                    <th>平仓价格</th>
                    <th>数量</th>
                    <th>盈亏</th>
                    <th>盈亏%</th>
                </tr>
        """
        
        for trade in trades[:100]:  # 限制显示前100条交易记录
            profit = trade.get('profit', 0)
            profit_pct = trade.get('profit_pct', 0)
            cls = 'positive' if profit >= 0 else 'negative'
            
            html += f"""
                <tr>
                    <td>{trade.get('id', '')}</td>
                    <td>{trade.get('symbol', '')}</td>
                    <td>{trade.get('side', '')}</td>
                    <td>{trade.get('open_time', '')}</td>
                    <td>{trade.get('open_price', '')}</td>
                    <td>{trade.get('close_time', '')}</td>
                    <td>{trade.get('close_price', '')}</td>
                    <td>{trade.get('amount', '')}</td>
                    <td class="{cls}">{profit:.4f}</td>
                    <td class="{cls}">{profit_pct:.2f}%</td>
                </tr>
            """
            
        html += """
            </table>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html)
            
    def plot_equity_curve(self, results: Dict[str, Any], plot_path: str) -> None:
        """
        绘制权益曲线图
        
        参数:
            results: 回测结果
            plot_path: 图表保存路径
        """
        trades = results.get('trades', [])
        if not trades:
            logger.warning("没有交易记录，无法绘制权益曲线")
            return
            
        # 转换为DataFrame
        trades_df = pd.DataFrame(trades)
        
        # 确保时间列是datetime类型
        trades_df['open_time'] = pd.to_datetime(trades_df['open_time'])
        trades_df['close_time'] = pd.to_datetime(trades_df['close_time'])
        
        # 按平仓时间排序
        trades_df = trades_df.sort_values('close_time')
        
        # 计算累计收益
        trades_df['cumulative_profit'] = trades_df['profit'].cumsum()
        
        # 绘制权益曲线
        plt.figure(figsize=(12, 6))
        plt.plot(trades_df['close_time'], trades_df['cumulative_profit'])
        plt.title('权益曲线')
        plt.xlabel('时间')
        plt.ylabel('累计收益')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='币安智能杠杆交易系统回测工具')
    
    parser.add_argument('--config', type=str, default='user_data/config/config_backtest.json',
                       help='回测配置文件路径')
                       
    parser.add_argument('--strategy', type=str, default='MasterStrategy',
                       help='策略名称')
                       
    parser.add_argument('--symbols', type=str, nargs='+', default=['BTC/USDT'],
                       help='交易对符号列表，例如 BTC/USDT ETH/USDT')
                       
    parser.add_argument('--timeframes', type=str, nargs='+', 
                       default=['1h'],
                       help='时间周期列表，例如 1m 5m 15m 1h 4h')
                       
    parser.add_argument('--start-date', type=str, required=True,
                       help='开始日期，格式为 YYYY-MM-DD')
                       
    parser.add_argument('--end-date', type=str, required=True,
                       help='结束日期，格式为 YYYY-MM-DD')
                       
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 解析日期
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # 初始化回测引擎
    engine = BacktestEngine(args.config)
    
    # 执行回测
    results = engine.run_backtest(
        args.strategy,
        args.symbols,
        args.timeframes,
        start_date,
        end_date
    )
    
    # 打印回测结果摘要
    metrics = results.get('metrics', {})
    print("\n回测结果摘要:")
    print(f"总收益: {metrics.get('total_profit', 0):.2f}")
    print(f"胜率: {metrics.get('win_rate', 0):.2f}%")
    print(f"盈亏比: {metrics.get('profit_factor', 0):.2f}")
    print(f"最大回撤: {metrics.get('max_drawdown', 0):.2f}%")
    print(f"夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"交易次数: {metrics.get('total_trades', 0)}")
    
    logger.info("回测完成")

if __name__ == '__main__':
    main()
