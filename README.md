# 多因子选股 · 机器学习分类模型（BA 工作坊）

基于 A 股多因子面板数据，构建**二分类机器学习模型**，预测"下期表现相对更好的股票"，并把全流程结果整理成一份可交互的 HTML 分析报告。

> 配色遵循 A 股惯例：**红色 = 涨 / 绿色 = 跌**。

---

## 1. 项目背景

在量化选股中，常用"多因子"框架：用估值、成长、规模等维度的因子刻画一只股票，再训练模型把股票分成"好 / 差"两类。
本项目以教学为目的，演示如何把机器学习分类模型落地到股票数据上，并强调**金融场景特有的工程纪律**（防止未来信息泄露、样本外验证、分层回测等）。

## 2. 数据

- 文件：`model_data_stock.csv`
- 规模：**20,772 行 × 20 列** = 4,281 只股票 × 5 个季度报告期（2021Q2 ~ 2022Q2）
- 字段构成：
  - `Date`、`Code`：报告期与股票代码
  - `Y`：二分类标签（TRUE = 下期表现好，FALSE = 其余）
  - 17 个数值特征：
    - 估值 / 规模因子（9）：`EV_EBITDA`、`PB`、`PCF_cash`、`PCF_oper`、`PE`、`PE_ex`、`PS`、`DIV`、`MV`
    - 成长因子（8）：`G_profit`、`G_equity`、`G_totalprofit`、`G_eps`、`G_assets`、`G_cashflow`、`G_opprofit`、`G_revenue`
- 标签分布：TRUE 8,389 条（约 40%）/ FALSE 12,383 条 —— 轻度不平衡，故评估以 **AUC / F1** 为主而非准确率。
- 数据质量：无缺失、全数值；主要风险是**极端值**（如接近零利润的公司 PE 可达 ±数十万）。

## 3. 方法与工程纪律

1. **时间序列切分（防未来泄露）**：前 3 期训练 / 第 4 期验证 / 第 5 期（2022Q2）作为**样本外测试**，绝不随机切分。
2. **横截面预处理**：按报告期分别做去极值（分位截尾）+ 标准化；所有拟合参数**仅用训练期**估计，避免泄露。
3. **模型集合**：逻辑回归、决策树、随机森林、KNN、SVM、朴素贝叶斯、**XGBoost**（已启用；本例未用 GBDT / LightGBM）。
4. **评估**：混淆矩阵、ROC、AUC、P-R、F1；并在样本外做**分层回测**（按预测概率分 5 组看多空区分度）。

## 4. 主要结果（样本外 2022Q2）

| 指标 | 数值 |
|------|------|
| 最优模型 | 随机森林 |
| 随机森林 AUC | **0.585** |
| XGBoost AUC | **0.566** |
| 分层多空区分度 | **13.0%** |
| 最强因子 | `EV_EBITDA`、`G_profit`、`PE` |

> 注：AUC 仅 0.585 属"弱信号"，符合 A 股因子选股的现实难度；重点在于演示完整、合规的建模流程，而非追求高收益。

## 5. 文件说明

| 文件 | 说明 |
|------|------|
| `stock_ml_classification_demo.ipynb` | 主 notebook：EDA → 切分 → 预处理 → 7 模型训练 → 单模型详解 → 跨模型对比 → 结论 |
| `model_data_stock.csv` | 原始数据 |
| `requirements.txt` | Python 依赖 |
| `SPEC_股票ML分类模型notebook规划.md` | notebook 的设计规划文档（只规划未执行阶段的产物） |
| `多因子选股ML分类_交互报告.html` | 可交互分析报告（Plotly，CDN 加载） |

## 6. 运行方式

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Apple Silicon 注意：XGBoost 需要 arm64 版 libomp
#   brew install libomp   （或用已有的 arm64 dylib 软链到 python 库路径）

jupyter notebook stock_ml_classification_demo.ipynb
# 或端到端执行：
jupyter nbconvert --to notebook --execute stock_ml_classification_demo.ipynb
```

环境：Python 3.9+，scikit-learn / seaborn / xgboost / nbconvert / jupyter。

## 7. 交互报告

- GitHub 文件页：<https://github.com/piaolij/BA-workshop/blob/main/多因子选股ML分类_交互报告.html>
- 在线渲染：<https://raw.githubusercontent.com/piaolij/BA-workshop/main/多因子选股ML分类_交互报告.html>

报告含：样本分布、按 Y 均值对比、与 Y 相关性 Top 因子、因子相关性矩阵、TOP6 因子箱线图（Y=0 / Y=1）、各模型混淆矩阵 + ROC + AUC + 分析、跨模型对比、总结分析。

## 8. 注意事项与结论要点

- `Y` 的业务定义以课程要求为准（本例假设 TRUE = 下期表现好）。
- 未做行业 / 市值中性化（基础教学版）。
- **PE 与 PE_ex 的 Pearson 相关仅 0.007**（受极端离群值失真），但 **Spearman 秩相关达 0.72**，本质刻画同一估值维度；本数据集无任何因子对 |r| > 0.7。
- 量化选股关键教训：时间序列切分防泄露、横截面标准化、用 AUC 而非准确率、样本外分层回测验证区分度。
