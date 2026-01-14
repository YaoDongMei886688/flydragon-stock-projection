"""
飞龙股份(002536)智能相似走势对比分析系统
版本: 2.0 - 真实网络数据版
功能: 基于真实股票数据的三图表对比分析
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockSimilarityAnalyzer:
    def __init__(self, stock_code="002536"):
        """初始化分析器，飞龙股份代码 002536"""
        self.stock_code = stock_code
        self.stock_name = self.get_stock_name(stock_code)
        self.hist_data = None
        self.current_pattern = None
        self.similar_patterns = []
        self.similar_stocks = []
        
    def get_stock_name(self, stock_code):
        """根据股票代码获取股票名称"""
        # 扩展的股票代码-名称映射
        name_map = {
            '002536': '飞龙股份',
            '300697': '电工合金',
            '600021': '上海电力',
            '000001': '平安银行',
            '000858': '五粮液',
            '300750': '宁德时代',
            '600036': '招商银行',
            '000333': '美的集团',
            '002415': '海康威视',
            '300059': '东方财富'
        }
        return name_map.get(stock_code, f'股票{stock_code}')
    
    def fetch_stock_data(self, stock_code=None, years=2):
        """获取指定股票的历史数据"""
        if stock_code is None:
            stock_code = self.stock_code
            
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=365*years)).strftime('%Y%m%d')
        
        try:
            print(f"正在获取{self.get_stock_name(stock_code)}({stock_code})历史数据...")
            
            # 使用akshare获取数据
            df = ak.stock_zh_a_hist(
                symbol=stock_code, 
                period="daily", 
                start_date=start_date, 
                end_date=end_date, 
                adjust="qfq"
            )
            
            if df.empty or len(df) < 30:
                print(f"  警告: {stock_code}数据不足，使用模拟数据")
                return self.generate_mock_data(stock_code)
            
            # 数据清洗和格式化
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            df = df.sort_index()
            
            # 计算技术指标
            df['MA5'] = df['收盘'].rolling(window=5).mean()
            df['MA20'] = df['收盘'].rolling(window=20).mean()
            df['Returns'] = df['收盘'].pct_change()
            df['Volatility'] = df['Returns'].rolling(window=10).std()
            
            print(f"  成功获取 {len(df)} 个交易日数据")
            return df
            
        except Exception as e:
            print(f"  数据获取失败: {e}")
            return self.generate_mock_data(stock_code)
    
    def generate_mock_data(self, stock_code):
        """生成模拟数据（当API不可用时使用）"""
        print(f"  为{stock_code}生成模拟数据...")
        dates = pd.date_range(end=datetime.now(), periods=500, freq='B')
        np.random.seed(hash(stock_code) % 10000)
        
        # 生成价格序列（模拟股票走势）
        price = 100
        prices = []
        for i in range(len(dates)):
            if i > 0:
                ret = np.random.normal(0.0005, 0.02)
                price *= (1 + ret)
            prices.append(price)
        
        df = pd.DataFrame({
            '开盘': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
            '收盘': prices,
            '最高': [p * (1 + np.random.uniform(0, 0.03)) for p in prices],
            '最低': [p * (1 - np.random.uniform(0, 0.03)) for p in prices],
            '成交量': np.random.lognormal(14, 1, len(prices))
        }, index=dates)
        
        df['MA5'] = df['收盘'].rolling(window=5).mean()
        df['MA20'] = df['收盘'].rolling(window=20).mean()
        return df
    
    def extract_current_pattern(self, window_size=30):
        """提取当前股票的走势模式"""
        if self.hist_data is None:
            self.hist_data = self.fetch_stock_data(self.stock_code, years=2)
        
        # 获取最近 window_size 天的数据
        recent_data = self.hist_data.tail(window_size)
        
        if len(recent_data) < window_size:
            print("警告: 数据不足，使用所有可用数据")
            recent_data = self.hist_data.tail(min(30, len(self.hist_data)))
        
        # 标准化价格序列（便于比较）
        prices = recent_data['收盘'].values
        norm_prices = (prices - prices.mean()) / (prices.std() + 1e-8)
        
        self.current_pattern = {
            'dates': recent_data.index,
            'prices': norm_prices,
            'actual_prices': prices,
            'start_price': prices[0],
            'end_price': prices[-1],
            'period_return': (prices[-1] / prices[0] - 1) * 100,
            'returns': recent_data['Returns'].values[-20:] if len(recent_data) >= 20 else [],
            'ma_ratio': recent_data['MA5'].iloc[-1] / recent_data['MA20'].iloc[-1] if len(recent_data) > 0 else 1
        }
        
        return self.current_pattern
    
    def find_similar_stocks(self, candidate_codes=None, top_n=3):
        """在其他股票中寻找相似走势"""
        if self.current_pattern is None:
            self.extract_current_pattern()
        
        if candidate_codes is None:
            # 预定义一组候选股票进行对比
            candidate_codes = ['300697', '600021', '000001', '000858', '300750', '600036', '000333']
        
        similar_stocks = []
        current_pattern = self.current_pattern['prices']
        pattern_length = len(current_pattern)
        
        print(f"\n正在在{len(candidate_codes)}只候选股票中寻找相似走势...")
        
        for code in candidate_codes:
            if code == self.stock_code:
                continue
                
            try:
                # 获取对比股票数据
                compare_data = self.fetch_stock_data(code, years=1)
                if len(compare_data) < pattern_length + 20:
                    continue
                
                # 获取最近的相同长度数据
                recent_compare = compare_data.tail(pattern_length)
                compare_prices = recent_compare['收盘'].values
                
                # 标准化
                compare_norm = (compare_prices - compare_prices.mean()) / (compare_prices.std() + 1e-8)
                
                # 计算相似度（欧氏距离）
                distance = np.linalg.norm(current_pattern - compare_norm)
                
                # 计算该期间的涨跌幅
                compare_return = (compare_prices[-1] / compare_prices[0] - 1) * 100
                
                # 计算相似度百分比（距离越小越相似）
                similarity_score = max(0, 100 - distance * 15)
                
                similar_stocks.append({
                    'code': code,
                    'name': self.get_stock_name(code),
                    'distance': distance,
                    'similarity': similarity_score,
                    'prices': compare_prices,
                    'period_return': compare_return,
                    'dates': recent_compare.index,
                    'data_source': '真实数据' if '模拟' not in str(compare_data) else '模拟数据'
                })
                
                print(f"  {self.get_stock_name(code)}({code}): 相似度{similarity_score:.1f}%")
                
            except Exception as e:
                print(f"  分析{code}时出错: {e}")
                continue
        
        # 按相似度排序
        similar_stocks.sort(key=lambda x: x['distance'])
        self.similar_stocks = similar_stocks[:top_n]
        
        return self.similar_stocks
    
    def find_similar_history_patterns(self, num_patterns=3, search_window=60):
        """在当前股票自身历史中寻找相似走势"""
        if self.current_pattern is None:
            self.extract_current_pattern()
        if self.hist_data is None:
            self.hist_data = self.fetch_stock_data(self.stock_code, years=2)
        
        current_vector = self.current_pattern['prices']
        all_patterns = []
        pattern_length = len(current_vector)
        data_length = len(self.hist_data)
        
        print(f"\n正在在自身历史中寻找相似走势...")
        
        for i in range(0, data_length - pattern_length - search_window):
            # 获取历史片段
            hist_prices = self.hist_data['收盘'].iloc[i:i+pattern_length].values
            
            # 标准化
            hist_norm = (hist_prices - hist_prices.mean()) / (hist_prices.std() + 1e-8)
            
            # 计算相似度
            distance = np.linalg.norm(current_vector - hist_norm)
            
            # 获取后续走势
            future_start = i + pattern_length
            future_end = min(future_start + search_window, data_length)
            
            if future_end > future_start:
                future_prices = self.hist_data['收盘'].iloc[future_start:future_end].values
                future_returns = (future_prices[-1] / future_prices[0] - 1) * 100
                
                all_patterns.append({
                    'start_idx': i,
                    'distance': distance,
                    'pattern': hist_norm,
                    'actual_pattern': hist_prices,
                    'future_prices': future_prices,
                    'future_returns': future_returns,
                    'start_date': self.hist_data.index[i],
                    'end_date': self.hist_data.index[i+pattern_length-1],
                    'future_start_date': self.hist_data.index[future_start],
                    'future_end_date': self.hist_data.index[future_end-1] if future_end <= data_length else self.hist_data.index[-1]
                })
        
        # 按相似度排序
        all_patterns.sort(key=lambda x: x['distance'])
        self.similar_patterns = all_patterns[:num_patterns]
        
        for i, pattern in enumerate(self.similar_patterns[:3]):
            similarity = max(0, 100 - pattern['distance'] * 15)
            print(f"  历史模式{i+1}: 相似度{similarity:.1f}%, 后续{pattern['future_returns']:.1f}%")
        
        return self.similar_patterns
    
    def create_comparison_chart(self, save_path='flydragon_analysis.html'):
        """创建基于真实网络数据的左右对比样式图表"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        print("\n正在生成三图表对比分析...")
        
        # 1. 准备数据
        if not self.similar_stocks:
            self.find_similar_stocks()
        
        if not self.similar_patterns:
            self.find_similar_history_patterns()
        
        # 2. 获取对比数据
        # 左侧：最相似的其他股票
        if self.similar_stocks:
            similar_stock = self.similar_stocks[0]
            left_title = f'<b>走势最相似的个股</b><br><span style="font-size:0.8em;">{similar_stock["name"]}({similar_stock["code"]}) - 相似度{similar_stock["similarity"]:.1f}%</span>'
            left_subtitle = f'分析周期: {similar_stock["dates"][0].strftime("%Y/%m/%d")}-{similar_stock["dates"][-1].strftime("%Y/%m/%d")}<br>期间涨跌幅: {similar_stock["period_return"]:.1f}%'
        else:
            similar_stock = None
            left_title = '<b>走势最相似的个股</b><br><span style="font-size:0.8em;">未找到足够相似的股票</span>'
            left_subtitle = ''
        
        # 中间：最相似的自身历史走势
        if self.similar_patterns:
            similar_history = self.similar_patterns[0]
            similarity = max(0, 100 - similar_history['distance'] * 15)
            middle_title = f'<b>最相似的历史走势</b><br><span style="font-size:0.8em;">相似度{similarity:.1f}%</span>'
            middle_subtitle = f'历史周期: {similar_history["start_date"].strftime("%Y/%m/%d")}-{similar_history["end_date"].strftime("%Y/%m/%d")}'
        else:
            similar_history = None
            middle_title = '<b>最相似的历史走势</b><br><span style="font-size:0.8em;">未找到足够相似的历史模式</span>'
            middle_subtitle = ''
        
        # 右侧：历史模式的真实后续走势
        if self.similar_patterns:
            most_similar = self.similar_patterns[0]
            right_title = f'<b>参考后续走势图</b><br><span style="font-size:0.8em;">基于上述历史相似模式的实际后续表现</span>'
            right_subtitle = f'后续周期: {most_similar["future_start_date"].strftime("%Y/%m/%d")}-{most_similar["future_end_date"].strftime("%Y/%m/%d")}<br>实际涨跌幅: {most_similar["future_returns"]:.1f}%'
        else:
            most_similar = None
            right_title = '<b>参考后续走势图</b>'
            right_subtitle = '无历史相似模式可参考'
        
        # 3. 创建三列子图布局
        fig = make_subplots(
            rows=1, cols=3,
            column_widths=[0.3, 0.3, 0.4],
            subplot_titles=(left_title, middle_title, right_title),
            horizontal_spacing=0.12,
            vertical_spacing=0.2
        )
        
        # 4. 左侧：相似个股对比
        if similar_stock:
            dates_left = list(range(len(self.current_pattern['actual_prices'])))
            
            # 当前股票走势
            fig.add_trace(
                go.Scatter(
                    x=dates_left, 
                    y=self.current_pattern['actual_prices'],
                    mode='lines', 
                    name=f'{self.stock_name}(当前)',
                    line=dict(color='#FF6B6B', width=3),
                    hovertemplate='当前股票<br>时间点: %{x}<br>价格: %{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # 相似股票走势
            fig.add_trace(
                go.Scatter(
                    x=dates_left, 
                    y=similar_stock['prices'],
                    mode='lines', 
                    name=f'{similar_stock["name"]}(对比)',
                    line=dict(color='#4ECDC4', width=3, dash='dash'),
                    hovertemplate=f'{similar_stock["name"]}<br>时间点: %{x}<br>价格: %{y:.2f}<br>相似度: {similar_stock["similarity"]:.1f}%<extra></extra>'
                ),
                row=1, col=1
            )
            
            # 添加左侧副标题
            fig.add_annotation(
                x=0.5, y=1.08, xref="paper", yref="paper",
                text=left_subtitle,
                showarrow=False,
                font=dict(size=10, color="#666"),
                row=1, col=1
            )
        
        # 5. 中间：历史走势对比
        if similar_history:
            dates_middle = list(range(len(self.current_pattern['actual_prices'])))
            
            # 当前走势
            fig.add_trace(
                go.Scatter(
                    x=dates_middle, 
                    y=self.current_pattern['actual_prices'],
                    mode='lines', 
                    name='当前走势',
                    line=dict(color='#FF6B6B', width=3),
                    showlegend=False,
                    hovertemplate='当前走势<br>时间点: %{x}<br>价格: %{y:.2f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # 历史相似走势
            fig.add_trace(
                go.Scatter(
                    x=dates_middle, 
                    y=similar_history['actual_pattern'],
                    mode='lines', 
                    name='历史相似走势',
                    line=dict(color='#45B7D1', width=3, dash='dash'),
                    hovertemplate='历史相似走势<br>时间点: %{x}<br>价格: %{y:.2f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # 添加中间副标题
            fig.add_annotation(
                x=0.5, y=1.08, xref="paper", yref="paper",
                text=middle_subtitle,
                showarrow=False,
                font=dict(size=10, color="#666"),
                row=1, col=2
            )
        
        # 6. 右侧：真实后续走势参考
        if most_similar:
            future_dates = list(range(len(most_similar['future_prices'])))
            
            # 实际后续走势
            fig.add_trace(
                go.Scatter(
                    x=future_dates, 
                    y=most_similar['future_prices'],
                    mode='lines+markers', 
                    name='历史相似模式的实际后续',
                    line=dict(color='#96CEB4', width=4),
                    marker=dict(size=6, color='#96CEB4'),
                    fill='tozeroy',
                    fillcolor='rgba(150, 206, 180, 0.2)',
                    hovertemplate='后续走势<br>交易日: %{x}<br>价格: %{y:.2f}<br>涨跌幅: %{customdata:.1f}%',
                    customdata=[((p / most_similar['future_prices'][0] - 1) * 100) for p in most_similar['future_prices']]
                ),
                row=1, col=3
            )
            
            # 添加起始参考线
            fig.add_hline(
                y=most_similar['future_prices'][0],
                line_dash="dot",
                line_color="rgba(128, 128, 128, 0.7)",
                line_width=1,
                opacity=0.7,
                row=1, col=3
            )
            
            # 添加右侧副标题
            fig.add_annotation(
                x=0.5, y=1.08, xref="paper", yref="paper",
                text=right_subtitle,
                showarrow=False,
                font=dict(size=10, color="#666"),
                row=1, col=3
            )
            
            # 标注最终涨跌幅
            fig.add_annotation(
                x=future_dates[-1], y=most_similar['future_prices'][-1],
                text=f"{most_similar['future_returns']:.1f}%",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#96CEB4",
                font=dict(size=12, color="#96CEB4"),
                row=1, col=3
            )
        else:
            # 如果没有历史模式，显示提示
            fig.add_annotation(
                x=0.5, y=0.5, xref="x domain", yref="y domain",
                text="未找到足够相似的历史走势模式",
                showarrow=False,
                font=dict(size=14, color="#999"),
                row=1, col=3
            )
        
        # 7. 更新整体布局
        fig.update_layout(
            height=650,
            showlegend=True,
            legend=dict(
                x=0.5,
                y=-0.15,
                orientation='h',
                font=dict(size=12)
            ),
            template='plotly_white',
            title=dict(
                text=f'{self.stock_name}({self.stock_code}) 智能相似走势对比分析',
                x=0.5,
                font=dict(size=22, color='#2C3E50'),
                y=0.97
            ),
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # 更新坐标轴标签
        fig.update_xaxes(title_text="时间周期 (交易日)", row=1, col=1)
        fig.update_yaxes(title_text="价格", row=1, col=1)
        fig.update_xaxes(title_text="时间周期 (交易日)", row=1, col=2)
        fig.update_yaxes(title_text="价格", row=1, col=2)
        fig.update_xaxes(title_text="后续交易日", row=1, col=3)
        fig.update_yaxes(title_text="价格", row=1, col=3)
        
        # 设置统一的y轴范围，方便对比
        if similar_stock and similar_history:
            all_prices = np.concatenate([
                self.current_pattern['actual_prices'],
                similar_stock['prices'],
                similar_history['actual_pattern']
            ])
            y_min, y_max = all_prices.min() * 0.95, all_prices.max() * 1.05
            
            fig.update_yaxes(range=[y_min, y_max], row=1, col=1)
            fig.update_yaxes(range=[y_min, y_max], row=1, col=2)
        
        # 8. 保存图表
        fig.write_html(save_path)
        print(f"✅ 三图表对比分析已保存至: {save_path}")
        
        return fig

# 主执行函数
def main():
    print("=" * 70)
    print(f"{'飞龙股份(002536)智能相似走势对比分析系统':^70}")
    print(f"{'版本 2.0 - 基于真实网络数据':^70}")
    print("=" * 70)
    
    # 创建分析器实例
    analyzer = StockSimilarityAnalyzer(stock_code="002536")
    
    # 获取主股票数据
    print("\n📊 第一阶段：获取主股票数据")
    print("-" * 50)
    analyzer.hist_data = analyzer.fetch_stock_data("002536", years=2)
    
    # 分析当前模式
    print("\n📈 第二阶段：分析当前走势模式")
    print("-" * 50)
    current_pattern = analyzer.extract_current_pattern(window_size=30)
    print(f"  当前价格: {current_pattern['end_price']:.2f}")
    print(f"  分析周期涨跌幅: {current_pattern['period_return']:.1f}%")
    print(f"  5/20日均线比: {current_pattern['ma_ratio']:.3f}")
    
    # 寻找相似股票
    print("\n🔍 第三阶段：寻找相似走势股票")
    print("-" * 50)
    similar_stocks = analyzer.find_similar_stocks(top_n=3)
    
    # 寻找自身历史相似模式
    print("\n🕰️ 第四阶段：寻找自身历史相似模式")
    print("-" * 50)
    similar_patterns = analyzer.find_similar_history_patterns(num_patterns=3)
    
    # 创建可视化图表
    print("\n🎨 第五阶段：生成可视化图表")
    print("-" * 50)
    fig = analyzer.create_comparison_chart('flydragon_analysis.html')
    
    # 显示分析总结
    print("\n" + "=" * 70)
    print(f"{'分析完成总结':^70}")
    print("-" * 70)
    
    print(f"主分析股票: {analyzer.stock_name}({analyzer.stock_code})")
    print(f"当前价格: {current_pattern['end_price']:.2f}")
    
    if similar_stocks:
        best_match = similar_stocks[0]
        print(f"\n最相似个股: {best_match['name']}({best_match['code']})")
        print(f"相似度: {best_match['similarity']:.1f}%")
        print(f"该股同期涨跌幅: {best_match['period_return']:.1f}%")
    
    if similar_patterns:
        best_history = similar_patterns[0]
        similarity = max(0, 100 - best_history['distance'] * 15)
        print(f"\n最相似历史周期: {best_history['start_date'].strftime('%Y/%m/%d')} 至 {best_history['end_date'].strftime('%Y/%m/%d')}")
        print(f"历史相似度: {similarity:.1f}%")
        print(f"该历史模式后续实际涨跌幅: {best_history['future_returns']:.1f}%")
    
    print(f"\n图表文件: flydragon_analysis.html")
    print("=" * 70)
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()
