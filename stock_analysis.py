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
        self.hist_data = None
        self.current_pattern = None
        self.similar_patterns = []
        
    def fetch_stock_data(self, years=2):
        """获取股票历史数据"""
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=365*years)).strftime('%Y%m%d')
        
        try:
            print(f"正在获取飞龙股份({self.stock_code})历史数据...")
            # 使用akshare获取数据
            df = ak.stock_zh_a_hist(
                symbol=self.stock_code, 
                period="daily", 
                start_date=start_date, 
                end_date=end_date, 
                adjust="qfq"
            )
            
            if df.empty:
                raise ValueError("获取的数据为空")
                
            # 数据清洗和格式化
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            df = df.sort_index()
            
            # 计算技术指标
            df['MA5'] = df['收盘'].rolling(window=5).mean()
            df['MA20'] = df['收盘'].rolling(window=20).mean()
            df['Returns'] = df['收盘'].pct_change()
            df['Volatility'] = df['Returns'].rolling(window=10).std()
            
            self.hist_data = df
            print(f"数据获取成功！共 {len(df)} 个交易日数据")
            return df
            
        except Exception as e:
            print(f"数据获取失败: {e}")
            # 生成模拟数据作为后备
            return self.generate_mock_data()
    
    def generate_mock_data(self):
        """生成模拟数据（当API不可用时使用）"""
        print("使用模拟数据生成演示图表...")
        dates = pd.date_range(end=datetime.now(), periods=500, freq='B')
        np.random.seed(42)
        
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
        self.hist_data = df
        return df
    
    def extract_current_pattern(self, window_size=30):
        """提取当前走势模式"""
        if self.hist_data is None:
            self.fetch_stock_data()
        
        # 获取最近 window_size 天的数据
        recent_data = self.hist_data.tail(window_size)
        
        # 标准化价格序列（便于比较）
        prices = recent_data['收盘'].values
        norm_prices = (prices - prices.mean()) / prices.std()
        
        self.current_pattern = {
            'dates': recent_data.index,
            'prices': norm_prices,
            'actual_prices': prices,
            'returns': recent_data['Returns'].values[-20:],
            'ma_ratio': recent_data['MA5'].iloc[-1] / recent_data['MA20'].iloc[-1]
        }
        
        return self.current_pattern
    
    def find_similar_patterns(self, num_patterns=3, search_window=60):
        """在历史中寻找相似走势模式"""
        if self.current_pattern is None:
            self.extract_current_pattern()
        
        current_vector = self.current_pattern['prices']
        all_patterns = []
        
        # 在历史数据中滑动窗口搜索
        data_length = len(self.hist_data)
        pattern_length = len(current_vector)
        
        for i in range(0, data_length - pattern_length - search_window):
            hist_prices = self.hist_data['收盘'].iloc[i:i+pattern_length].values
            hist_norm = (hist_prices - hist_prices.mean()) / hist_prices.std()
            
            # 计算相似度（欧氏距离）
            distance = np.linalg.norm(current_vector - hist_norm)
            
            # 获取后续走势
            future_start = i + pattern_length
            future_end = future_start + search_window
            if future_end <= data_length:
                future_prices = self.hist_data['收盘'].iloc[future_start:future_end].values
                future_returns = (future_prices[-1] / future_prices[0] - 1) * 100
                
                all_patterns.append({
                    'start_idx': i,
                    'distance': distance,
                    'pattern': hist_norm,
                    'actual_pattern': hist_prices,
                    'future_prices': future_prices,
                    'future_returns': future_returns,
                    'dates': self.hist_data.index[i:i+pattern_length]
                })
        
        # 按相似度排序，取最相似的几个
        all_patterns.sort(key=lambda x: x['distance'])
        self.similar_patterns = all_patterns[:num_patterns]
        
        return self.similar_patterns
    
    def generate_projection(self, projection_days=15):
        """基于相似模式生成走势推演"""
        if not self.similar_patterns:
            self.find_similar_patterns()
        
        projections = []
        current_price = self.hist_data['收盘'].iloc[-1]
        
        for i, pattern in enumerate(self.similar_patterns):
            # 获取相似模式的历史后续表现
            future_returns = pattern['future_returns'] / 100  # 转换为小数
            
            # 生成推演序列
            projected_prices = [current_price]
            for day in range(1, projection_days + 1):
                # 基于历史相似模式的回报率进行调整
                daily_return = future_returns / projection_days
                noise = np.random.normal(0, 0.005)  # 添加随机噪声
                new_price = projected_prices[-1] * (1 + daily_return + noise)
                projected_prices.append(new_price)
            
            projections.append({
                'pattern_id': i,
                'projected_prices': projected_prices[1:],  # 排除当前价格
                'avg_return': future_returns,
                'confidence': 1 / (1 + pattern['distance'])  # 相似度越高，置信度越高
            })
        
        return projections
    
        def create_comparison_chart(self, save_path='flydragon_analysis.html'):
        """创建直观的左右对比样式图表"""
        ... # [前面的代码保持不变，直到第5部分]

        # --- 5. 右侧：基于最相似历史模式的真实后续走势 ---
        if self.similar_patterns:
            # 获取相似度最高的历史模式
            most_similar = self.similar_patterns[0]
            
            # 使用该模式历史上真实的后续价格数据
            real_future_prices = most_similar['future_prices']
            real_future_dates = list(range(len(real_future_prices)))
            
            # 计算这段历史后续的实际涨跌幅
            actual_return = most_similar['future_returns']
            
            fig.add_trace(
                go.Scatter(
                    x=real_future_dates,
                    y=real_future_prices,
                    mode='lines+markers',
                    name=f'历史相似模式后续走势 (实际涨跌幅: {actual_return:.1f}%)',
                    line=dict(color='#FF6B6B', width=4),
                    fill='tozeroy',
                    fillcolor='rgba(255, 107, 107, 0.1)'
                ),
                row=1, col=3
            )
            
            # 添加一条水平起始线作为参考
            fig.add_hline(
                y=real_future_prices[0],
                line_dash="dot",
                line_color="gray",
                opacity=0.7,
                row=1, col=3
            )
            
            # 更新右侧子图标题，加入实际收益率信息
            fig.layout.annotations[2].update(
                text=f'<b>参考后续走势</b><br><span style="font-size:0.7em;">基于相似度{1/(1+most_similar["distance"]):.1%}的历史模式 | 其后续实际涨跌幅: {actual_return:.1f}%</span>'
            )
        else:
            # 如果没有找到相似模式，显示提示信息
            fig.add_annotation(
                x=0.5, y=0.5, xref="x domain", yref="y domain",
                text="未找到足够相似的历史走势模式",
                showarrow=False,
                row=1, col=3
            )
        
        # --- [后面的布局和保存代码保持不变] ---
        
        # 5. 右侧：后续走势参考（示例）
        future_dates = list(range(20))
        future_prices = [100 * (1 + 0.02*i) for i in future_dates]
        fig.add_trace(
            go.Scatter(x=future_dates, y=future_prices,
                       mode='lines+markers', name='参考后续走势',
                       line=dict(color='purple', width=4),
                       fill='tozeroy', fillcolor='rgba(128,0,128,0.1)'),
            row=1, col=3
        )
        
        # 6. 更新布局
        fig.update_layout(
            height=600,
            showlegend=True,
            legend=dict(x=0.5, y=-0.1, orientation='h'),
            template='plotly_white',
            title=dict(
                text=f'飞龙股份(002536)相似走势对比分析',
                x=0.5,
                font=dict(size=24, color='darkblue')
            )
        )
        
        # 更新坐标轴标签
        fig.update_xaxes(title_text="时间周期", row=1, col=1)
        fig.update_yaxes(title_text="价格", row=1, col=1)
        fig.update_xaxes(title_text="时间周期", row=1, col=2)
        fig.update_yaxes(title_text="价格", row=1, col=2)
        fig.update_xaxes(title_text="后续交易日", row=1, col=3)
        fig.update_yaxes(title_text="价格", row=1, col=3)
        
        # 7. 保存图表
        fig.write_html(save_path)
        print(f"直观对比图表已保存至: {save_path}")
        return fig

# 主执行函数
def main():
    print("=" * 60)
    print("飞龙股份(002536)趋势分析与相似模式对比")
    print("=" * 60)
    
    # 创建分析器实例
    analyzer = StockSimilarityAnalyzer(stock_code="002536")
    
    # 获取数据
    df = analyzer.fetch_stock_data(years=2)
    
    # 分析当前模式
    print("\n1. 分析当前走势模式...")
    current_pattern = analyzer.extract_current_pattern(window_size=30)
    
    # 寻找相似历史模式
    print("2. 在历史数据中寻找相似走势...")
    similar_patterns = analyzer.find_similar_patterns(num_patterns=5)
    
    print(f"   找到 {len(similar_patterns)} 个相似历史模式")
    for i, pattern in enumerate(similar_patterns[:3]):
        print(f"   模式{i+1}: 距离={pattern['distance']:.3f}, "
              f"后续收益={pattern['future_returns']:.2f}%")
    
    # 生成推演
    print("3. 生成未来走势推演...")
    projections = analyzer.generate_projection()
    
    # 创建可视化图表
    print("4. 生成可视化图表...")
    fig = analyzer.create_comparison_chart('flydragon_analysis.html')
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print(f"当前价格: {df['收盘'].iloc[-1]:.2f}")
    print(f"5日均线: {df['MA5'].iloc[-1]:.2f}")
    print(f"20日均线: {df['MA20'].iloc[-1]:.2f}")
    print(f"图表文件: flydragon_analysis.html")
    print("=" * 60)
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()
