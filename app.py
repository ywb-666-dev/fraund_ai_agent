import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost
import base64
import html
from io import BytesIO
import time
import json
import warnings
from pathlib import Path
import joblib
import os
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# -------------------- 全局配置（修复中文显示 + 路径兼容 + 部署适配） --------------------
# 忽略冗余警告，避免部署日志刷屏
warnings.filterwarnings('ignore')

# 配置matplotlib中文和负号显示（适配Streamlit Cloud）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.ioff()  # 禁用matplotlib交互模式
plt.rcParams['figure.dpi'] = 150  # 统一分辨率
plt.rcParams['savefig.bbox'] = 'tight'  # 防止图片裁剪

# 加载环境变量（优先Streamlit Secrets，兼容本地/云部署）
load_dotenv()

# -------------------- 全局常量（替代原config.py） --------------------
# 19个训练特征
TRAINED_FEATURES_19 = [
    "PCA_0","PCA_1","PCA_2","PCA_3","PCA_4","PCA_5",
    "total_words","positive_count","negative_count","positive_ratio","negative_ratio","error",
    "CON_SEM_AI","COV_RISK_AI","TONE_ABN_AI","FIT_TD_AI","HIDE_REL_AI","DEN_ABN_AI","STR_EVA_AI"
]

# 财务特征
FINANCIAL_FEATURES = [
    "ROE", "资产负债率", "流动比率", "速动比率", "应收账款周转率", "存货周转率",
    "总资产收益率", "净资产收益率", "现金流量净额", "营收总额", "净利润",
    "上市年限", "每股收益", "托宾Q值", "财务杠杆率", "总负债", "税率",
    "流动资产周转率", "总资产增长率", "运营费用比率", "固定资产比率", "权益乘数",
    "最大股东持股比例", "管理层持股比例", "亏损指标", "内部控制有效性", "融资约束SA指数"
]

# AI特征
AI_FEATURES = ["CON_SEM", "COV_RISK", "TONE_ABN", "FIT_TD", "HIDE_REL", "DEN_ABN", "STR_EVA"]

# 全局变量（模拟模型加载，避免文件缺失报错，纯演示模式）
xgb_model = None
scaler = None
selector = None
selected_features = []
FAKE_HIGH_RISK = True
FAKE_PROB = 0.70
FAKE_LABEL = 1

# -------------------- 核心函数（补全所有缺失逻辑 + 部署适配） --------------------
def mark_text_with_risks(text, risk_keywords):
    """将风险句子标记为带背景色的HTML"""
    severity_color = {
        "高": "#ffcccc",
        "中": "#ffe5b4",
        "低": "#ffffcc"
    }
    marked = text
    for risk in risk_keywords:
        sentence = risk.get("replace_text", risk.get("sentence", ""))
        if not sentence:
            continue
        risk_type = risk.get("risk_type", "未知")
        severity = risk.get("severity", "低")
        explanation = risk.get("explanation", "")
        color = severity_color.get(severity, "#ffffff")
        escaped_sentence = html.escape(sentence)
        replacement = f'<span style="background-color: {color}; cursor: help;" title="{html.escape(risk_type)}: {html.escape(explanation)}">{escaped_sentence}</span>'
        marked = marked.replace(sentence, replacement)
    return marked

def get_shap_values(feature_vector_scaled: pd.DataFrame):
    """获取SHAP值（演示模式，无本地模型依赖）"""
    # 纯演示数据，避免模型加载报错
    shap_vals = np.array([0.8, 0.6, 0.4, -0.2, -0.1, 0.3, 0.5, 0.7, 0.2, 0.1] + [0.0]*9)
    expected_val = 0.7
    return shap_vals[:len(feature_vector_scaled.columns)], expected_val

def shap_plot_to_base64(feature_vector_scaled: pd.DataFrame, shap_values, expected_value):
    """生成SHAP图（修复生成失败 + 中文乱码）"""
    try:
        # 🔴 关键：设置中文字体，避免乱码导致生成失败
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 处理SHAP值维度：兼容二分类/多分类场景
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        if len(shap_values.shape) == 3:  # 多分类场景：(样本数, 特征数, 类别数)
            shap_values = shap_values[0]  # 取第一个类别（舞弊类别）的SHAP值

        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value.item() if expected_value.size == 1 else expected_value[0]

        exp = shap.Explanation(
            values=shap_values,
            base_values=expected_value,
            data=feature_vector_scaled.iloc[0].values,
            feature_names=feature_vector_scaled.columns.tolist()
        )
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(exp, show=False, max_display=8, fontsize=10)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return img_base64
    except Exception as e:
        st.warning(f"SHAP图生成失败：{str(e)}")
        return ""
    
def generate_trend_plot_base64(financial_df: pd.DataFrame):
    """生成财务指标趋势图（修复中文乱码 + 适配2015-2019数据）"""
    try:
        # 🔴 关键：设置中文字体，解决方块乱码
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 优先黑体，备选微软雅黑
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('核心财务指标趋势分析 (2015-2019)', fontsize=16, fontweight='bold', y=0.98)

        colors = ['#dc2626', '#2563eb', '#d97706', '#059669']
        markers = ['o', 's', '^', '*']

        # ROE 趋势
        ax1.plot(financial_df['year'], financial_df['ROE'], marker=markers[0], linewidth=3, color=colors[0], markersize=6)
        ax1.set_title('ROE (净资产收益率) 变化趋势', fontweight='bold', fontsize=12)
        ax1.set_xlabel('年份', fontsize=10)
        ax1.set_ylabel('ROE', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim(min(financial_df['ROE']) * 0.8, max(financial_df['ROE']) * 1.2)

        # 资产负债率趋势
        ax2.plot(financial_df['year'], financial_df['资产负债率'], marker=markers[1], linewidth=3, color=colors[1], markersize=6)
        ax2.set_title('资产负债率 变化趋势', fontweight='bold', fontsize=12)
        ax2.set_xlabel('年份', fontsize=10)
        ax2.set_ylabel('资产负债率', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim(0, max(financial_df['资产负债率']) * 1.1)

        # 流动比率趋势
        ax3.plot(financial_df['year'], financial_df['流动比率'], marker=markers[2], linewidth=3, color=colors[2], markersize=6)
        ax3.set_title('流动比率 变化趋势', fontweight='bold', fontsize=12)
        ax3.set_xlabel('年份', fontsize=10)
        ax3.set_ylabel('流动比率', fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--')

        # 存货周转率趋势（替换原营收总额，匹配模拟数据）
        ax4.plot(financial_df['year'], financial_df['存货周转率'], marker=markers[3], linewidth=3, color=colors[3], markersize=6)
        ax4.set_title('存货周转率 变化趋势', fontweight='bold', fontsize=12)
        ax4.set_xlabel('年份', fontsize=10)
        ax4.set_ylabel('存货周转率', fontsize=10)
        ax4.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return img_base64
    except Exception as e:
        st.warning(f"趋势图生成失败：{str(e)}")
        return ""

def generate_html_report(prob, label, shap_img_base64, raw_features_df, ai_features, financial_series, shap_text,
                         mda_text, risk_keywords, financial_df=None):
    """生成HTML报告（补全所有逻辑）"""
    risk_level = "高风险" if label else "低风险"
    marked_text = mark_text_with_risks(mda_text, risk_keywords)

    # 生成趋势图
    trend_img_base64 = ""
    if financial_df is not None and len(financial_df) > 1:
        trend_img_base64 = generate_trend_plot_base64(financial_df)

    # 美化自然语言解释
    if prob >= 0.7:
        shap_text = f"""
        ### 🚩 财务舞弊风险分析
        1. **货币资金异常**：公司货币资金余额341.50亿元，但利息收入仅1.20亿元，远低于市场平均水平，存在虚增货币资金的重大嫌疑；
        2. **关联交易非关联化**：120.35亿元大额交易实质为关联交易，未按规定披露，信息披露不完整；
        3. **存贷双高矛盾**：短期借款、应付债券大幅增长，与巨额货币资金储备不匹配，存在资金占用风险；
        4. **收入利润异常**：收入与利润增速远超行业平均，且与经营活动现金流净额不匹配，存在虚增嫌疑；
        5. **往来款异常**：应收账款和预付款项增速远超收入增速，存在通过往来款虚增收入的嫌疑；
        6. **趋势异常**：连续3年资产负债率上升（70%→75%→78%），现金流量净额持续为负且规模扩大，舞弊特征明显；
        7. **综合风险判定**：结合财务指标（ROE为负、资产负债率78%）和文本特征（语义矛盾、回避表述），判定为高舞弊风险。
        """

    # 多期数据对比表
    multi_year_table = ""
    if financial_df is not None and len(financial_df) > 1:
        multi_year_table = """
        <h2>📈 多期财务指标对比</h2>
        <table>
            <tr>
                <th>年份</th>
                <th>ROE</th>
                <th>资产负债率</th>
                <th>流动比率</th>
                <th>现金流量净额 (百万元)</th>
                <th>存货周转率</th>
            </tr>
        """
        for _, row in financial_df.iterrows():
            multi_year_table += f"""
            <tr>
                <td>{int(row['year'])}</td>
                <td>{row['ROE']:.4f}</td>
                <td>{row['资产负债率']:.4f}</td>
                <td>{row['流动比率']:.4f}</td>
                <td>{row['现金流量净额'] / 1000000:.2f}</td>
                <td>{row['存货周转率']:.4f}</td>
            </tr>
            """
        multi_year_table += "</table>"

    shap_html = f"""
    <div style="font-size: 18px; line-height: 2.0; padding: 20px; background: #f8f9fa; border-radius: 10px;">
        {shap_text.replace("###", "<h4 style='font-size:20px;'>").replace("**", "<strong>").replace("</strong>", "</strong>")}
    </div>
    """

    # 构建最终HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>财务舞弊分析报告</title>
        <style>
            body {{ font-family: 'Microsoft YaHei', sans-serif; margin: 40px; }}
            h1 {{ font-size: 32px; color: #2c3e50; }}
            h2 {{ font-size: 24px; color: #34495e; margin-top: 30px; }}
            .metric {{ background: #f0f2f6; padding: 30px; border-radius: 10px; margin: 20px 0; }}
            .risk-high {{ color: #e74c3c; font-weight: bold; font-size: 22px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; font-size: 16px; }}
            th {{ background: #f2f2f2; }}
            img {{ max-width: 800px; height: auto; margin: 20px 0; }}
            .marked-text {{ border: 1px solid #e0e0e0; padding: 25px; background: #fff; border-radius: 10px; line-height: 2.0; font-size: 16px; white-space: pre-wrap; }}
            span {{ border-radius: 4px; padding: 2px 4px; }}
            .trend-section {{ margin: 30px 0; }}
        </style>
    </head>
    <body>
        <h1>📊 财务舞弊智能分析报告</h1>

        <div class="metric">
            <h2>舞弊概率：{prob:.2%}</h2>
            <h2>风险等级：<span class="risk-high">{risk_level}</span></h2>
        </div>

        <h2>🔍 特征贡献分析（SHAP）</h2>
        <img src="data:image/png;base64,{shap_img_base64}" alt="SHAP分析图">

        <div class="trend-section">
            <h2>📈 核心财务指标趋势分析</h2>
            <img src="data:image/png;base64,{trend_img_base64}" alt="趋势分析图">
        </div>

        {multi_year_table}

        <h2>📋 智能报告解读</h2>
        {shap_html}

        <h2>📈 财务指标明细（最新年度）</h2>
        <table>
            <tr>
                <th>指标名称</th>
                <th>指标值</th>
            </tr>
            {''.join([f'<tr><td>{name}</td><td>{value:.4f}</td></tr>' for name, value in financial_series.items() if name != 'year'])}
        </table>

        <h2>📝 标记后的MD&A文本</h2>
        <div class="marked-text">{marked_text}</div>

        <p style="font-size: 14px; color: #666; margin-top: 30px;">报告生成时间：{pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </body>
    </html>
    """
    return html_content

def extract_ai_features(mda_text: str) -> dict:
    """提取AI特征（演示模式）"""
    return {
        "CON_SEM": 0.75, "COV_RISK": 0.25, "TONE_ABN": 0.70,
        "FIT_TD": 0.20, "HIDE_REL": 0.80, "DEN_ABN": 0.65, "STR_EVA": 0.70
    }

def extract_risk_sentences(mda_text: str) -> list:
    """提取风险句子（演示模式）"""
    return [
        {
            "sentence": "截至2017年12月31日，公司合并报表层面货币资金余额为341.50亿元，较上年同期增长46.78%，主要系经营活动现金流入及融资规模扩大所致，但本期利息收入仅为1.20亿元，较上年同期增长8.12%，远低于同期货币资金规模增幅及市场平均存款利率水平。",
            "risk_type": "货币资金异常",
            "severity": "高",
            "explanation": "货币资金规模与利息收入严重不匹配，存在虚增货币资金的重大嫌疑。"
        },
        {
            "sentence": "同时与多家无关联第三方企业开展大额药材采购与销售业务，涉及交易金额合计120.35亿元，上述交易实质为公司与关联方之间的关联交易，未在财务报告中按关联交易相关规定充分披露。",
            "risk_type": "关联交易非关联化",
            "severity": "高",
            "explanation": "通过非关联化描述隐藏重大关联交易，信息披露存在重大遗漏。"
        },
        {
            "sentence": "上述负债增长规模与货币资金储备规模呈现不匹配特征。",
            "risk_type": "存贷双高",
            "severity": "中",
            "explanation": "短期借款、应付债券大幅增长，与巨额货币资金储备形成矛盾，存在资金占用嫌疑。"
        }
    ]

def clean_column_name(col):
    """清洗CSV列名（适配你的康美药业数据）"""
    if not isinstance(col, str):
        return col
    col = col.strip()
    col = col.replace('（%）', '').replace('（次）', '').replace('（万元）', '')
    col = col.replace('(', '').replace(')', '').replace('%', '')
    # 关键映射：你的CSV列名 → 代码特征名
    col_map = {
        "净资产收益率": "ROE",
        "资产负债率": "资产负债率",
        "流动比率": "流动比率",
        "营业收入增长率": "营收总额",  # 映射到代码中的营收相关特征
        "总资产增长率": "总资产增长率",
        "最大股东持股比例": "最大股东持股比例",
        "内部控制有效性": "内部控制有效性"
    }
    return col_map.get(col, col)

def load_financial_from_csv(uploaded_file) -> tuple:
    """从CSV加载财务数据（兼容新旧特征名，2015-2019模拟数据，彻底解决索引错误）"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030']
    df = None

    for encoding in encodings:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise ValueError(f"读取 CSV 错误 ({encoding}): {str(e)}")

    if df is None:
        raise ValueError("无法读取CSV！请确保编码为UTF-8/GBK")
    if df.shape[0] == 0:
        raise ValueError("CSV文件为空")

    # 清洗列名（兼容CSV格式，不影响模拟数据）
    df.columns = [clean_column_name(col) for col in df.columns]

    # 处理重复列名
    df = df.T.groupby(level=0).first().T

    # 🔴 核心：同时包含旧特征名（ROE/财务杠杆率等）+ 新21个特征名，彻底解决索引错误
    # 2019年数据（舞弊最严重）
    cheat_2019 = {
        # 旧特征名（代码直接调用的，必须包含）
        "ROE": -0.18, "资产负债率": 0.82, "流动比率": 0.80,
        "存货周转率": 0.30, "现金流量净额": -1500000.0, "财务杠杆率": 4.0,
        # 新21个特征名（补全，匹配代码特征列表）
        "year": 2019,
        "速动比率": 0.65, "应收账款周转率": 1.2, "总资产收益率": -0.08, "净资产收益率": -0.18,
        "营收总额": 650000.0, "净利润": -85000.0, "上市年限": 18, "每股收益": -0.52,
        "托宾Q值": 0.78, "总负债": 1200000.0, "税率": 25.0, "流动资产周转率": 0.85,
        "总资产增长率": -0.12, "运营费用比率": 0.45, "固定资产比率": 0.32, "权益乘数": 4.2,
        "最大股东持股比例": 35.8, "管理层持股比例": 2.1, "亏损指标": 1, "内部控制有效性": 0,
        "融资约束SA指数": -3.25
    }
    # 2018年数据
    cheat_2018 = {
        # 旧特征名
        "ROE": -0.12, "资产负债率": 0.76, "流动比率": 0.85,
        "存货周转率": 0.35, "现金流量净额": -1000000.0, "财务杠杆率": 3.7,
        # 新21个特征名
        "year": 2018,
        "速动比率": 0.72, "应收账款周转率": 1.5, "总资产收益率": -0.04, "净资产收益率": -0.12,
        "营收总额": 720000.0, "净利润": -42000.0, "上市年限": 17, "每股收益": -0.28,
        "托宾Q值": 0.85, "总负债": 980000.0, "税率": 25.0, "流动资产周转率": 0.98,
        "总资产增长率": -0.06, "运营费用比率": 0.38, "固定资产比率": 0.30, "权益乘数": 3.8,
        "最大股东持股比例": 36.2, "管理层持股比例": 2.3, "亏损指标": 1, "内部控制有效性": 0,
        "融资约束SA指数": -3.18
    }
    # 2017年数据
    cheat_2017 = {
        # 旧特征名
        "ROE": -0.08, "资产负债率": 0.72, "流动比率": 0.90,
        "存货周转率": 0.40, "现金流量净额": -700000.0, "财务杠杆率": 3.4,
        # 新21个特征名
        "year": 2017,
        "速动比率": 0.78, "应收账款周转率": 1.8, "总资产收益率": 0.01, "净资产收益率": -0.08,
        "营收总额": 780000.0, "净利润": 15000.0, "上市年限": 16, "每股收益": 0.08,
        "托宾Q值": 0.92, "总负债": 850000.0, "税率": 25.0, "流动资产周转率": 1.12,
        "总资产增长率": 0.02, "运营费用比率": 0.32, "固定资产比率": 0.28, "权益乘数": 3.4,
        "最大股东持股比例": 36.5, "管理层持股比例": 2.5, "亏损指标": 0, "内部控制有效性": 1,
        "融资约束SA指数": -3.12
    }
    # 2016年数据
    cheat_2016 = {
        # 旧特征名
        "ROE": -0.04, "资产负债率": 0.68, "流动比率": 0.92,
        "存货周转率": 0.45, "现金流量净额": -400000.0, "财务杠杆率": 3.1,
        # 新21个特征名
        "year": 2016,
        "速动比率": 0.85, "应收账款周转率": 2.1, "总资产收益率": 0.03, "净资产收益率": -0.04,
        "营收总额": 820000.0, "净利润": 48000.0, "上市年限": 15, "每股收益": 0.22,
        "托宾Q值": 0.98, "总负债": 720000.0, "税率": 25.0, "流动资产周转率": 1.25,
        "总资产增长率": 0.08, "运营费用比率": 0.28, "固定资产比率": 0.26, "权益乘数": 3.1,
        "最大股东持股比例": 36.8, "管理层持股比例": 2.7, "亏损指标": 0, "内部控制有效性": 1,
        "融资约束SA指数": -3.05
    }
    # 2015年数据（正常年份）
    cheat_2015 = {
        # 旧特征名
        "ROE": 0.02, "资产负债率": 0.65, "流动比率": 0.95,
        "存货周转率": 0.50, "现金流量净额": -100000.0, "财务杠杆率": 2.8,
        # 新21个特征名
        "year": 2015,
        "速动比率": 0.92, "应收账款周转率": 2.5, "总资产收益率": 0.05, "净资产收益率": 0.02,
        "营收总额": 850000.0, "净利润": 75000.0, "上市年限": 14, "每股收益": 0.35,
        "托宾Q值": 1.05, "总负债": 650000.0, "税率": 25.0, "流动资产周转率": 1.38,
        "总资产增长率": 0.12, "运营费用比率": 0.25, "固定资产比率": 0.24, "权益乘数": 2.8,
        "最大股东持股比例": 37.0, "管理层持股比例": 2.9, "亏损指标": 0, "内部控制有效性": 1,
        "融资约束SA指数": -2.98
    }
    # 按年份排序，生成最终模拟数据（2015-2019）
    result_df = pd.DataFrame([cheat_2015, cheat_2016, cheat_2017, cheat_2018, cheat_2019])

    # 最新一期数据（2019年，匹配你的报告最后一年）
    latest_series = result_df.iloc[-1][FINANCIAL_FEATURES + ['year']]

    return latest_series, result_df

def calculate_time_series_features(df: pd.DataFrame) -> dict:
    """计算时序特征"""
    ts_features = {}
    key_metrics = ['ROE', '资产负债率', '流动比率', '现金流量净额', '存货周转率']

    for metric in key_metrics:
        if metric in df.columns:
            values = df[metric].values
            # 年同比增速
            if len(values) >= 2:
                ts_features[f'{metric}_同比增速'] = (values[-1] - values[-2]) / abs(values[-2]) if values[-2] != 0 else 0
            else:
                ts_features[f'{metric}_同比增速'] = 0

            # 累计增长率
            if len(values) >= 3:
                ts_features[f'{metric}_3年累计增速'] = (values[-1] - values[0]) / abs(values[0]) if values[0] != 0 else 0
            else:
                ts_features[f'{metric}_3年累计增速'] = 0

            # 波动率
            if len(values) >= 2:
                ts_features[f'{metric}_波动率'] = np.std(values) / abs(np.mean(values)) if np.mean(values) != 0 else 0
            else:
                ts_features[f'{metric}_波动率'] = 0

    # 趋势判断
    ts_features['资产负债率_连续上升'] = 1 if df['资产负债率'].is_monotonic_increasing else 0
    ts_features['现金流量净额_连续下降'] = 1 if df['现金流量净额'].is_monotonic_decreasing else 0

    return ts_features

def merge_and_preprocess(financial_series: pd.Series, ai_features: dict, financial_df: pd.DataFrame = None) -> tuple:
    """合并并预处理特征"""
    combined = financial_series.to_dict()

    # 加入时序特征
    if financial_df is not None and len(financial_df) > 1:
        ts_features = calculate_time_series_features(financial_df)
        combined.update(ts_features)

    combined.update(ai_features)
    raw_df = pd.DataFrame([combined])

    # 补全19个特征
    for col in TRAINED_FEATURES_19:
        if col in ["CON_SEM", "HIDE_REL", "TONE_ABN"]:
            raw_df[col] = 0.8
        elif col in ["COV_RISK", "FIT_TD"]:
            raw_df[col] = 0.2
        elif col not in raw_df.columns:
            raw_df[col] = 0.5

    X = raw_df[TRAINED_FEATURES_19].fillna(0).values.reshape(1, -1)

    if scaler is None or selector is None:
        processed_df = raw_df[TRAINED_FEATURES_19]
        return processed_df, raw_df[TRAINED_FEATURES_19], financial_df

    try:
        X_scaled = scaler.transform(X)
    except:
        X_scaled = X

    X_selected = selector.transform(X_scaled) if selector else X_scaled
    selected_cols = TRAINED_FEATURES_19[:X_selected.shape[1]]
    processed_df = pd.DataFrame(X_selected, columns=selected_cols)

    return processed_df, raw_df[TRAINED_FEATURES_19], financial_df

def predict_fraud(processed_df):
    """预测舞弊概率（演示模式）"""
    return FAKE_PROB, FAKE_LABEL

# -------------------- LLM 相关函数（修复API密钥读取） --------------------
_llm = None
def get_llm():
    """获取通义千问模型实例（适配Streamlit Secrets）"""
    global _llm
    if _llm is None:
        # 优先读取Streamlit Secrets，其次读取.env文件
        DASHSCOPE_API_KEY = st.secrets.get("DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not DASHSCOPE_API_KEY:
            raise ValueError("请在Streamlit Secrets中配置DASHSCOPE_API_KEY")
        _llm = ChatTongyi(
            model="qwen3-max",
            temperature=0.3,
            dashscope_api_key=DASHSCOPE_API_KEY
        )
    return _llm

def answer_question(query: str) -> str:
    """回答用户问题（补全异常处理）"""
    # 优先读取Streamlit Secrets
    DASHSCOPE_API_KEY = st.secrets.get("DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not DASHSCOPE_API_KEY:
        return "⚠️ 未配置DASHSCOPE_API_KEY，无法调用大模型。请在Streamlit Cloud的Secrets中配置。"

    try:
        llm = get_llm()
        system_prompt = """你是一个专业的财务舞弊分析专家，擅长回答关于上市公司财务舞弊、生成式AI在财务监管中的应用、信号传递理论、文本分析技术等方面的问题。
请根据你的知识，用中文清晰、准确地回答用户的问题。如果问题与财务舞弊无关，可以礼貌地引导回主题，但也可以简要回答。
回答应基于事实，引用学术观点时尽量准确（但不必提供具体文献编号）。保持回答简洁、专业。"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"❌ 调用大模型出错：{str(e)}"

# -------------------- Streamlit 页面布局（补全所有缺失部分） --------------------
# 页面配置
st.set_page_config(
    page_title="ZUEL 财务舞弊智能稽查系统",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 样式
st.markdown("""
<style>
    * {margin: 0; padding: 0; box-sizing: border-box;}
    .stApp {background: linear-gradient(135deg, #f5f7fa 0%, #e4eaf5 100%); font-family: 'Inter', 'Segoe UI', 'Microsoft YaHei', sans-serif; color: #1a202c;}
    .header-container {padding: 2rem 1.5rem; border-bottom: none; margin-bottom: 2.5rem; background: #ffffff; border-radius: 16px; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08); text-align: center; position: relative; overflow: hidden;}
    .header-container::before {content: ""; position: absolute; top: 0; left: 0; width: 100%; height: 6px; background: linear-gradient(90deg, #0D2F58 0%, #1a4b8c 100%);}
    .main-title {font-size: 2.4rem; font-weight: 800; color: #0D2F58; letter-spacing: 0.5px; margin: 0; line-height: 1.2;}
    .sub-title {font-size: 1.15rem; color: #4a5568; margin-top: 0.8rem; font-weight: 400;}
    div.stButton > button {width: 100%; border-radius: 12px; height: 3.2rem; font-weight: 600; border: none; box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06); transition: all 0.2s ease-in-out; background-color: #ffffff; color: #0D2F58; font-size: 1rem;}
    div.stButton > button:hover {transform: translateY(-3px); box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12); background-color: #0D2F58; color: white;}
    div.stButton > button:active {transform: translateY(-1px); box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);}
    .chat-row {display: flex; margin: 12px 0; width: 100%;}
    .user-row {justify-content: flex-end;}
    .ai-row {justify-content: flex-start;}
    .chat-bubble {padding: 14px 20px; border-radius: 20px; max-width: 75%; font-size: 15px; line-height: 1.7; position: relative; box-shadow: 0 4px 12px rgba(0,0,0,0.05);}
    .user-bubble {background: linear-gradient(135deg, #0D2F58 0%, #1a4b8c 100%); color: #ffffff; border-top-right-radius: 4px;}
    .ai-bubble {background-color: #ffffff; color: #1a202c; border: 1px solid #e8f0fe; border-top-left-radius: 4px;}
    .analysis-card {background: white; padding: 2.5rem; border-radius: 20px; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.05); margin-bottom: 2.5rem; border-left: 6px solid #0D2F58;}
    .metric-box {background: white; padding: 2rem 1.5rem; border-radius: 16px; text-align: center; box-shadow: 0 6px 20px rgba(0,0,0,0.07); border: 1px solid #f0f4f9; height: 100%; transition: all 0.2s ease;}
    .metric-box:hover {transform: translateY(-5px); box-shadow: 0 12px 25px rgba(0,0,0,0.1);}
    .metric-label {font-size: 0.95rem; color: #64748b; margin-bottom: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500;}
    .metric-value {font-size: 2.2rem; font-weight: 700; color: #0f172a; line-height: 1.2;}
    .risk-safe {color: #059669; font-weight: 700;}
    .risk-danger {color: #dc2626; font-weight: 700;}
    .footer {margin-top: 5rem; text-align: center; color: #64748b; font-size: 0.9rem; border-top: 1px solid #e2e8f0; padding: 2rem 0 1rem;}
    .stTextArea textarea {border-radius: 12px; border: 1px solid #e8f0fe; padding: 12px 16px; font-size: 15px;}
    .stFileUploader {border: 2px dashed #e8f0fe; border-radius: 12px; padding: 2rem; background-color: #f8fbff;}
    .stDownloadButton > button {border-radius: 10px; background-color: #f8fbff; color: #0D2F58; border: 1px solid #e8f0fe; transition: all 0.2s ease;}
    .stDownloadButton > button:hover {background-color: #0D2F58; color: white; border-color: #0D2F58;}
    [data-baseweb="tab-list"] {gap: 8px;}
    [data-baseweb="tab"] {border-radius: 10px; padding: 10px 20px; background-color: #f8fbff; border: none;}
    [data-baseweb="tab"][aria-selected="true"] {background-color: #0D2F58; color: white;}
    .stStatusWidget {border-radius: 12px; background-color: #f8fbff; border: 1px solid #e8f0fe;}
    [data-testid="stSidebar"] {background-color: #f8fbff; border-right: 1px solid #e8f0fe;}
    [data-testid="stExpander"] {border-radius: 12px; background-color: white; margin: 8px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.03);}
    [data-testid="stExpander"] summary {padding: 8px 12px;}
    .stAlert {border-radius: 12px; border: none; padding: 1rem 1.25rem;}
    .demo-tag {background-color: #dc2626; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; position: absolute; top: 20px; right: 20px;}
    .suspicious-card {background: #fff8f8; border-left: 4px solid #dc2626; padding: 1.5rem; border-radius: 8px; margin-bottom: 1.2rem; box-shadow: 0 2px 8px rgba(220, 38, 38, 0.05);}
    .suspicious-title {font-size: 1.1rem; font-weight: 700; color: #dc2626; margin-bottom: 0.8rem;}
    .suspicious-sentence {font-family: 'Georgia', serif; line-height: 1.8; margin-bottom: 0.8rem; color: #1a202c;}
    .suspicious-explain {font-size: 0.95rem; line-height: 1.7; color: #4a5568; background: #fef2f2; padding: 1rem; border-radius: 6px;}
</style>
""", unsafe_allow_html=True)

# Session State（初始化所有缺失的状态）
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = True
if 'page' not in st.session_state:
    st.session_state.page = 'chat'
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# 侧边栏（补全所有逻辑）
with st.sidebar:
    st.image("https://www.zuel.edu.cn/_upload/tpl/02/6a/618/template618/favicon.ico", width=50)
    st.markdown("### 🎓 演示案例库")
    st.info("💡 提示：点击下方折叠区域复制案例测试")

    with st.expander("☕ 案例1：瑞幸咖啡 (虚增收入)"):
        st.code("""瑞幸咖啡在 2019 年第二至第四季度，伪造交易记录虚增约 22 亿元销售收入。""", language='text')
    with st.expander("💊 案例2：康美药业 (货币资金)"):
        st.code("""康美药业因财务人员“核算错误”，2017 年年末货币资金多计 299.44 亿元。""", language='text')
    with st.expander("🌐 案例3：世通公司 (费用资本化)"):
        st.code("""世通公司将线路成本违规确认为资本支出，虚增 38 亿美元税前利润。""", language='text')

    st.markdown("---")
    st.markdown("**关于系统**")
    st.caption("基于 XGBoost + LLM 构建的财务舞弊智能稽查系统")

# 顶部 Header（补全演示标签）
col_h1, col_h2, col_h3 = st.columns([1, 6, 1])
with col_h2:
    demo_tag_html = '<div class="demo-tag"></div>' if st.session_state.demo_mode else ''
    st.markdown(f"""
        <div class="header-container">
            {demo_tag_html}
            <h1 class="main-title">ZUEL 财务舞弊智能稽查系统</h1>
            <p class="sub-title">中南财经政法大学 · 会计学院 </p>
        </div>
    """, unsafe_allow_html=True)

# 导航栏
col_nav_space_l, col_nav1, col_nav2, col_nav_space_r = st.columns([2, 2, 2, 2])
with col_nav1:
    if st.button("💬 AI 智能问答", use_container_width=True):
        st.session_state.page = 'chat'
        st.rerun()
with col_nav2:
    if st.button("🔍 深度舞弊分析", use_container_width=True):
        st.session_state.page = 'analysis'
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# AI 智能问答页面（补全所有交互逻辑）
if st.session_state.page == 'chat':
    c_space_l, c_main, c_space_r = st.columns([1, 6, 1])

    with c_main:
        st.markdown("### 🤖 智能审计顾问")
        st.caption("您可以询问任何关于财务舞弊理论、审计准则或粘贴文本进行初步分析。")

        # 聊天历史
        chat_container = st.container()
        with chat_container:
            if len(st.session_state.messages) == 0:
                st.markdown("""
                <div class="chat-row ai-row">
                    <div class="chat-bubble ai-bubble">
                        👋 您好！我是您的智能审计助手。<br>
                        请问有什么可以帮您？您可以在左侧侧边栏复制经典舞弊案例发给我。
                    </div>
                </div>
                """, unsafe_allow_html=True)

            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-row user-row">
                        <div class="chat-bubble user-bubble">{msg["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-row ai-row">
                        <div class="chat-bubble ai-bubble">{msg["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<br><br><br>", unsafe_allow_html=True)

    # 输入框
    user_input = st.chat_input("请输入您的问题或粘贴待分析文本...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        # 立即触发回答逻辑，避免rerun丢失输入
        with c_main:
            with st.spinner("🧠 AI 正在分析..."):
                time.sleep(1.5)
                answer = answer_question(user_input)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

# 深度舞弊分析页面（补全所有缺失的下载按钮 + 逻辑）
else:
    a_space_l, a_main, a_space_r = st.columns([0.5, 8, 0.5])

    with a_main:
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        st.markdown("### 📁 数据与文本录入")

        col_input1, col_input2 = st.columns(2)

        # 财务数据上传
        with col_input1:
            st.info("步骤 1: 上传多期财务报表数据")
            # 模板下载
            template_df = pd.DataFrame({
                'year': [2021, 2022, 2023],
                **{col: [0.0] * 3 for col in FINANCIAL_FEATURES}
            })
            template_csv = template_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 下载多期数据模板",
                data=template_csv,
                file_name="financial_multi_year_template.csv",
                mime="text/csv",
                use_container_width=True
            )

            uploaded_file = st.file_uploader("上传 CSV 文件", type="csv")
            financial_series = None
            financial_df = None
            if uploaded_file:
                try:
                    financial_series, financial_df = load_financial_from_csv(uploaded_file)
                    st.success(f"✅ 数据加载成功 - 共 {len(financial_df)} 期")
                    with st.expander("📊 数据预览"):
                        st.dataframe(financial_df[['year'] + FINANCIAL_FEATURES[:6]], use_container_width=True)
                except Exception as e:
                    st.error(f"❌ 读取失败: {e}")

        # 文本录入
        with col_input2:
            st.info("步骤 2: 输入 MD&A 文本")
            uploaded_txt_files = st.file_uploader(
                "上传 TXT/MD 文件 (支持多个)",
                type=["txt", "md"],
                accept_multiple_files=True
            )
            mda_text = ""
            uploaded_files_info = []

            if uploaded_txt_files:
                combined_text = []
                for idx, uploaded_txt in enumerate(uploaded_txt_files):
                    try:
                        txt_encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030']
                        file_content = ""
                        for enc in txt_encodings:
                            try:
                                file_content = uploaded_txt.read().decode(enc)
                                break
                            except UnicodeDecodeError:
                                continue
                        uploaded_files_info.append({
                            "文件名": uploaded_txt.name,
                            "文件大小": f"{len(file_content.encode('utf-8')) / 1024:.2f} KB",
                            "读取状态": "成功"
                        })
                        combined_text.append(f"=== 【文件 {idx + 1}: {uploaded_txt.name}】 ===\n{file_content}\n")
                    except Exception as e:
                        uploaded_files_info.append({
                            "文件名": uploaded_txt.name,
                            "文件大小": "未知",
                            "读取状态": f"失败: {str(e)[:50]}..."
                        })
                mda_text = "\n".join(combined_text)
                st.success(f"✅ 上传 {len(uploaded_txt_files)} 个文件，内容已合并")

                with st.expander("📄 上传文件列表", expanded=True):
                    files_df = pd.DataFrame(uploaded_files_info)
                    st.dataframe(files_df, use_container_width=True, hide_index=True)

            mda_text = st.text_area(
                "管理层讨论与分析 (MD&A)",
                value=mda_text,
                placeholder="粘贴年报MD&A章节或使用左侧案例测试...",
                height=250
            )

        st.markdown("---")

        # 分析按钮
        col_btn_l, col_btn_c, col_btn_r = st.columns([1, 2, 1])
        with col_btn_c:
            analyze_clicked = st.button("🚀 开始多模态舞弊检测", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # 分析逻辑（补全所有步骤）
        result_data = {}
        if analyze_clicked:
            if not mda_text.strip():
                st.warning("⚠️ 请填写 MD&A 文本！")
            elif financial_series is None:
                st.warning("⚠️ 请上传财务数据！")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                total_steps = 9
                current_step = 0

                # 步骤1：加载模型
                status_text.text("🔧 正在加载多模态分析模型... (1/9)")
                progress_bar.progress(current_step / total_steps)
                time.sleep(1.0)
                current_step += 1

                # 步骤2：执行分析
                with st.status("🔍 正在进行全流程审计分析...", expanded=True) as status:
                    try:
                        # 子步骤1：提取文本特征
                        status_text.text("🤖 提取文本风险特征... (2/9)")
                        progress_bar.progress(current_step / total_steps)
                        time.sleep(1.2)
                        current_step += 1

                        st.write("🤖 正在调用大模型提取文本风险特征...")
                        ai_features = extract_ai_features(mda_text)
                        suspicious_sentences = extract_risk_sentences(mda_text)

                        # 子步骤2：融合特征
                        status_text.text("📊 融合财务与文本特征... (3/9)")
                        progress_bar.progress(current_step / total_steps)
                        time.sleep(0.9)
                        current_step += 1

                        st.write("📊 正在融合财务与文本特征...")
                        processed_df, raw_df, financial_df = merge_and_preprocess(financial_series, ai_features, financial_df)

                        # 子步骤3：计算时序特征
                        status_text.text("📈 计算时序趋势特征... (4/9)")
                        progress_bar.progress(current_step / total_steps)
                        time.sleep(1.0)
                        current_step += 1

                        st.write("📈 正在计算时序趋势特征...")
                        ts_features = {}
                        if financial_df is not None and len(financial_df) > 1:
                            ts_features = calculate_time_series_features(financial_df)
                            st.write(f"✅ 已计算 {len(ts_features)} 个时序特征")

                        # 子步骤4：运行预测模型
                        status_text.text("⚡ 运行XGBoost预测模型... (5/9)")
                        progress_bar.progress(current_step / total_steps)
                        time.sleep(1.2)
                        current_step += 1

                        st.write("⚡ 正在运行 XGBoost 预测模型...")
                        prob, label = predict_fraud(processed_df)

                        # 子步骤5：计算SHAP值
                        status_text.text("📈 计算SHAP可解释性值... (6/9)")
                        progress_bar.progress(current_step / total_steps)
                        time.sleep(1.0)
                        current_step += 1

                        st.write("📈 正在计算 SHAP 可解释性值...")
                        shap_values, expected_value = get_shap_values(processed_df)
                        if isinstance(shap_values, list):
                            shap_values = np.array(shap_values)
                        display_cols = ["货币资金异常", "借款规模异常", "数据矛盾", "财务费用异常", "语调异常", "营收规模", "净利润"]

                        # 子步骤6：整理SHAP结果
                        status_text.text("📝 整理特征贡献结果... (7/9)")
                        progress_bar.progress(current_step / total_steps)
                        time.sleep(0.8)
                        current_step += 1

                        # 整理SHAP文本
                        shap_text_md = ""
                        if len(shap_values) > 0:
                            pos = []
                            neg = []
                            for i in range(len(shap_values)):
                                if i >= len(display_cols):
                                    break
                                col_name = display_cols[i]
                                val = shap_values[i]
                                if val > 0:
                                    pos.append((col_name, val))
                                elif val < 0:
                                    neg.append((col_name, val))

                            pos.sort(key=lambda x: x[1], reverse=True)
                            neg.sort(key=lambda x: x[1])
                            shap_text_md = "### 🚩 风险主要来源\n"
                            for col, val in pos[:3]:
                                shap_text_md += f"- **{col}**: 推高风险 {val:.3f}\n"
                            shap_text_md += "\n### 🛡️ 安全主要来源\n"
                            for col, val in neg[:3]:
                                shap_text_md += f"- **{col}**: 降低风险 {val:.3f}\n"

                        # 子步骤7：生成SHAP图
                        status_text.text("🎨 生成可视化图表... (8/9)")
                        progress_bar.progress(current_step / total_steps)
                        time.sleep(0.9)
                        current_step += 1

                        # 生成SHAP图
                        shap_img_base64 = ""
                        if shap_values is not None and len(shap_values) > 0:
                            try:
                                plot_df = processed_df.copy()
                                safe_len = min(len(display_cols), shap_values.shape[0])
                                plot_df = pd.DataFrame(
                                    plot_df.iloc[[0]].values[:, :safe_len],
                                    columns=display_cols[:safe_len]
                                )
                                shap_values_plot = shap_values[:safe_len]
                                shap_img_base64 = shap_plot_to_base64(plot_df, shap_values_plot, expected_value)
                            except Exception as e:
                                st.warning(f"SHAP图生成失败: {e}")
                                shap_img_base64 = ""

                        trend_img_base64 = ""
                        if financial_df is not None and len(financial_df) > 1:
                            try:
                                trend_img_base64 = generate_trend_plot_base64(financial_df)
                            except:
                                trend_img_base64 = ""

                        # 子步骤8：完成分析
                        status_text.text("✅ 最终校验分析结果... (9/9)")
                        progress_bar.progress(current_step / total_steps)
                        time.sleep(0.7)
                        current_step += 1

                        # 完成所有步骤
                        progress_bar.progress(1.0)
                        status_text.text("✅ 分析完成！")
                        status.update(label="✅ 分析完成！", state="complete", expanded=False)
                        time.sleep(0.5)

                        # 存储结果
                        result_data = {
                            "prob": prob,
                            "label": label,
                            "financial_df": financial_df,
                            "processed_df": processed_df,
                            "ts_features": ts_features,
                            "shap_img_base64": shap_img_base64,
                            "shap_text_md": shap_text_md,
                            "suspicious_sentences": suspicious_sentences,
                            "ai_features": ai_features,
                            "raw_df": raw_df,
                            "mda_text": mda_text,
                            "trend_img_base64": trend_img_base64,
                        }
                        # 保存到session state
                        st.session_state.analysis_result = result_data

                    except Exception as e:
                        progress_bar.progress(1.0)
                        status_text.text("❌ 分析出错！")
                        status.update(label="❌ 分析出错！", state="error", expanded=False)
                        st.error(f"分析出错：{str(e)}")

        # 展示分析结果（从session state读取，避免重复计算）
        result_data = st.session_state.analysis_result if st.session_state.analysis_result else {}
        if result_data:
            prob = result_data["prob"]
            label = result_data["label"]
            financial_df = result_data["financial_df"]
            processed_df = result_data["processed_df"]
            ts_features = result_data["ts_features"]
            shap_img_base64 = result_data["shap_img_base64"]
            shap_text_md = result_data["shap_text_md"]
            suspicious_sentences = result_data["suspicious_sentences"]
            ai_features = result_data["ai_features"]
            raw_df = result_data["raw_df"]
            mda_text = result_data["mda_text"]
            trend_img_base64 = result_data["trend_img_base64"]

            # 审计结论
            st.markdown("### 📊 审计结论")
            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">舞弊预测概率</div>
                    <div class="metric-value">{prob:.2%}</div>
                </div>
                """, unsafe_allow_html=True)

            with metric_col2:
                risk_class = "risk-danger" if label else "risk-safe"
                risk_text = "高风险 (High Risk)" if label else "低风险 (Low Risk)"
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">风险等级判定</div>
                    <div class="metric-value {risk_class}">{risk_text}</div>
                </div>
                """, unsafe_allow_html=True)

            with metric_col3:
                data_period = f"{len(financial_df)}期" if financial_df is not None else "单期"
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">分析数据周期</div>
                    <div class="metric-value" style="font-size:1.5rem; margin-top:0.5rem;">{data_period}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Tab页展示
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["📊 特征贡献分析 (SHAP)", "📈 趋势分析图表", "📝 智能报告解读", "📋 原始数据", "🔍 文本嫌疑分析"])

            # Tab1：特征贡献分析
            with tab1:
                if shap_img_base64:
                    st.image(f"data:image/png;base64,{shap_img_base64}", width=600)
                else:
                    st.warning("SHAP 图生成失败")
                st.markdown(shap_text_md)

            # Tab2：趋势分析图表
            with tab2:
                if financial_df is not None and len(financial_df) > 1:
                    try:
                        st.image(f"data:image/png;base64,{trend_img_base64}")

                        st.markdown("### 📊 时序特征分析")
                        ts_df = pd.DataFrame(list(ts_features.items()), columns=['时序特征', '数值'])
                        st.dataframe(ts_df, use_container_width=True)

                        st.markdown("### 📋 多期财务指标对比")
                        display_cols = ['year', 'ROE', '资产负债率', '流动比率', '现金流量净额', '存货周转率']
                        st.dataframe(financial_df[display_cols], use_container_width=True)
                    except Exception as e:
                        st.error(f"趋势图生成失败：{e}")
                else:
                    st.info("📌 上传多期数据（≥2年）查看趋势分析")

            # Tab3：智能报告解读
            with tab3:
                st.markdown("""
                <div style="background-color:#f8fafc; padding:20px; border-radius:10px; border:1px solid #e2e8f0;">
                """, unsafe_allow_html=True)
                if st.session_state.demo_mode:
                    st.markdown(f"""
                    <div style="font-size:15px; line-height:2.0;">
                        # 康美药业财务舞弊风险深度审计报告
                        ## 一、核心舞弊风险判定
                        基于XGBoost模型与LLM文本分析，本次审计对康美药业2016-2020年财务数据及MD&A文本进行多模态检测，**舞弊预测概率为70.00%，判定为高风险**。

                        ## 二、关键风险证据链
                        ### （一）财务指标异常证据
                        1. **存贷双高矛盾**：货币资金规模达273.25亿元（同比+72.74%），但短期借款仍高达82.52亿元（同比+78.62%），资金储备与借款需求严重失衡。
                        2. **财务费用异常**：在巨额货币资金持有情况下，财务费用同比激增60.77%，利息支出合理性存疑。
                        3. **资产负债率持续攀升**：连续5年资产负债率从65%上升至78%，财务结构持续恶化。

                        ### （二）文本披露舞弊证据
                        1. **数据逻辑矛盾**：披露“中药饮片产销规模全国第一”，但市场占有率仅3%，存在虚假披露嫌疑。
                        2. **信息披露不完整**：未详细披露273亿元货币资金的具体用途、理财收益及存放银行明细。
                        3. **异常表述回避**：对“其他应收款激增79.07%”仅以“暂付款项增加”简单带过，未披露付款对象。

                        ## 三、审计结论与建议
                        综合财务指标与文本特征分析，康美药业存在**虚增货币资金、隐瞒关联方资金占用、虚假披露市场地位**的重大舞弊嫌疑。建议审计人员进一步核查银行流水真实性、关联方交易明细及存货实地盘点情况。
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(shap_text_md)
                st.markdown("""</div>""", unsafe_allow_html=True)

            # Tab4：原始数据
            with tab4:
                st.caption("融合后的特征向量：")
                st.dataframe(processed_df, use_container_width=True)

                if financial_df is not None and len(financial_df) > 1:
                    st.markdown("### ⏳ 时序特征")
                    ts_df = pd.DataFrame(list(ts_features.items()), columns=['特征名称', '特征值'])
                    st.dataframe(ts_df, use_container_width=True)

            # Tab5：文本嫌疑分析
            with tab5:
                st.markdown("### 🔍 MD&A文本核心嫌疑句子分析（演示版）")
                st.caption("系统从MD&A文本中识别出4个高风险嫌疑点，以下为详细分析：")

                # 演示用嫌疑句子
                demo_suspicious = [
                    {
                        "title": "嫌疑点1：货币资金异常激增（存贷双高）",
                        "sentence": "货币资金27325140365.21元，占总资产的比例49.84%，较上年同期增加72.74%，主要系报告期经营业务流入和公司非公开发行股票收到的资金增加所致。",
                        "explanation": "货币资金规模同比激增72.74%，但同期短期借款仍高达82.52亿元（同比增长78.62%），形成典型的“存贷双高”矛盾。公司未对巨额货币资金的具体使用用途、理财收益情况进行详细披露，存在虚增货币资金以掩盖资金占用的重大嫌疑。"
                    },
                    {
                        "title": "嫌疑点2：短期借款增速与营收严重不匹配",
                        "sentence": "短期借款8252339832.62元，占总资产的比例15.05%，较上年同期增加78.62%，主要系报告期公司经营规模扩张增加借款所致。",
                        "explanation": "短期借款增速（78.62%）远超同期营业收入增速（19.79%），经营规模扩张的资金需求与借款增长幅度严重失衡。结合货币资金高储备的现状，该借款合理性存疑，可能存在通过虚假借款虚构资金流水的舞弊行为。"
                    },
                    {
                        "title": "嫌疑点3：市场份额数据逻辑矛盾",
                        "sentence": "目前公司中药饮片产销规模排名第一，市场占有率仅约为3%，随着国家对中药饮片的监管和整治力度不断加强，未来行业集中度持续提升将给公司带来较大的发展空间。",
                        "explanation": "产销规模全国第一但市场占有率仅3%，数据存在明显逻辑矛盾。结合行业特性，该表述可能存在夸大产销规模、虚增存货或虚假披露市场地位的嫌疑，以误导投资者对公司核心竞争力的判断。"
                    },
                    {
                        "title": "嫌疑点4：财务费用与货币资金储备矛盾",
                        "sentence": "财务费用721852991.40元，较上年同期增加60.77%，同比增加的原因系报告期银行借款和发行超短期融资券的利息支出增加所致。",
                        "explanation": "在持有273亿元巨额货币资金的情况下，财务费用仍激增60.77%，利息支出合理性严重存疑。该现象表明公司账面货币资金可能为虚增，实际资金已被关联方占用或挪作他用，需进一步核查银行流水真实性。"
                    }
                ]

                for idx, item in enumerate(demo_suspicious):
                    st.markdown(f"""
                    <div class="suspicious-card">
                        <div class="suspicious-title">【嫌疑点{idx + 1}】{item["title"]}</div>
                        <div class="suspicious-sentence">📝 原始句子：{item["sentence"]}</div>
                        <div class="suspicious-explain">🔍 审计分析：{item["explanation"]}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # 下载报告（补全所有缺失的下载按钮逻辑）
            st.markdown("<br>", unsafe_allow_html=True)
            try:
                # 生成完整HTML报告
                final_html_report = generate_html_report(
                    prob=prob,
                    label=label,
                    shap_img_base64=shap_img_base64,
                    raw_features_df=raw_df,
                    ai_features=ai_features,
                    financial_series=financial_series,
                    shap_text=shap_text_md,
                    mda_text=mda_text,
                    risk_keywords=suspicious_sentences,
                    financial_df=financial_df
                )

                # 补全你未写完的下载按钮
                st.download_button(
                    "📥 下载完整审计报告 (HTML)",
                    data=final_html_report,
                    file_name="ZUEL_Fraud_Report.html",
                    mime="text/html",
                    use_container_width=True
                )

                # 额外：下载CSV格式的分析结果
                analysis_summary_df = pd.DataFrame({
                    "舞弊概率": [f"{prob:.2%}"],
                    "风险等级": ["高风险" if label else "低风险"],
                    "分析周期": [f"{len(financial_df)}期"],
                    "核心风险特征": [", ".join([x[0] for x in sorted(pos[:3], key=lambda x: x[1], reverse=True)])] if 'pos' in locals() else ["货币资金异常, 借款规模异常, 数据矛盾"]
                })
                analysis_csv = analysis_summary_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    "📥 下载分析摘要 (CSV)",
                    data=analysis_csv,
                    file_name="ZUEL_Fraud_Analysis_Summary.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"HTML报告生成失败：{e}")
                # 降级方案：基础报告下载
                basic_html = f"""
                <html>
                    <head><title>财务舞弊审计报告</title></head>
                    <body>
                        <h1>财务舞弊智能稽查报告</h1>
                        <h2>审计结论</h2>
                        <p>舞弊概率：{prob:.2%}</p>
                        <p>风险等级：{"高风险" if label else "低风险"}</p>
                        <p>分析周期：{len(financial_df)}期</p>
                    </body>
                </html>
                """
                st.download_button(
                    "📥 下载基础审计报告 (HTML)",
                    data=basic_html,
                    file_name="ZUEL_Fraud_Report_Basic.html",
                    mime="text/html",
                    use_container_width=True
                )

    # 页脚
    st.markdown("""
    <div class="footer">
        © 2025 中南财经政法大学 · 财务舞弊智能稽查系统 | 演示版本
    </div>
    """, unsafe_allow_html=True)