#!/usr/bin/env python3
"""
自动更新图表脚本 - 每天运行一次更新数据
"""

import subprocess
import sys
import os
from datetime import datetime

def update_chart():
    """更新股票图表"""
    print(f"[{datetime.now()}] 开始更新飞龙股份分析图表...")
    
    try:
        # 运行分析脚本
        result = subprocess.run(
            [sys.executable, "stock_analysis.py"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("图表更新成功！")
            
            # 如果使用GitHub Pages，可以在这里添加git commit和push
            # 但需要配置GitHub Actions的密钥
            return True
        else:
            print(f"更新失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"执行错误: {e}")
        return False

if __name__ == "__main__":
    success = update_chart()
    sys.exit(0 if success else 1)
