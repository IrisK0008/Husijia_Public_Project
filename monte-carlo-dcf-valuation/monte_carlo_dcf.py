import yfinance as yf
import pandas as pd
import numpy as np
import random
import sys

pd.options.display.float_format = '{:,.2f}'.format

DEFAULT_MARKET_INDEX_SYMBOL = "^GSPC"

def fetch_single_ticker(ticker_symbol):
    """
    创建一个 yfinance.Ticker 对象。
    
    逻辑:
    仅实例化对象，不立即下载数据。
    
    参数:
    - ticker_symbol: 股票代码 (字符串, 例如 'AAPL')
    
    返回:
    - ticker: yfinance Ticker 对象 (如果失败则返回 None)
    
    """
    ticker = yf.Ticker(ticker_symbol)
    return ticker

def fetch_tickers(ticker_symbol, market_index_symbol=DEFAULT_MARKET_INDEX_SYMBOL):
    """
    同时创建目标股票和市场指数的 yfinance 对象。
    
    逻辑:
    用于后续计算 Beta 或市场对比时，同时准备好两个数据源。
    
    参数:
    - ticker_symbol: 目标股票代码 (字符串)
    - market_index_symbol: 市场指数代码 (字符串, 默认标普500 '^GSPC')
    
    返回:
    - ticker: 目标股票对象
    - market: 市场指数对象
    
    """
    ticker = yf.Ticker(ticker_symbol)
    market = yf.Ticker(market_index_symbol)
    return ticker, market

def fetch_raw_financials(ticker):
    """
    从 Ticker 对象中提取原始的三大财务报表。
    
    逻辑:
    尝试访问 income_stmt, cashflow, balance_sheet 属性。
    如果不包含数据或访问失败，返回 None。
    
    参数:
    - ticker: 初始化的 yfinance.Ticker 对象
    
    返回:
    - raw_income_stmt: 利润表 (DataFrame)
    - raw_cashflow: 现金流量表 (DataFrame)
    - raw_balance: 资产负债表 (DataFrame)
    
    财务数据搬运工
    """
    try:
        raw_income_stmt = ticker.income_stmt
        raw_cashflow = ticker.cashflow
        raw_balance = ticker.balance_sheet
        return raw_income_stmt, raw_cashflow, raw_balance
    except Exception as error:
        print(f" [警告] 无法获取原始报表: {error}")
        return None, None, None 

def compute_market_statistics(stock_or_market_ticker, period="5y", interval="1mo"):
    """
    抓取市场/指数的历史数据，并计算关键统计量 (Mu, Sigma)。
    
    逻辑:
    1. 获取历史价格。
    2. 计算收益率。
    3. 年化处理: 
       - 收益率 (Mu) = 均值 * 时间因子
       - 波动率 (Sigma) = 标准差 * sqrt(时间因子)

    随机漫步理论，Variance 方差是可以按照时间线性相加的，sigma‘标准差/波动率’是方差的平方根。

    参数:
    - stock_or_market_ticker: yfinance 对象或字符串代码 (如 "^GSPC")
    - period: 历史长度 (如 "5y", "max")
    - interval: 采样频率 (如 "1mo", "1d")

    返回字典:
      - requested_period: 请求的 period 字符串（便于追踪）
      - requested_interval: 请求的 interval 字符串（便于追踪）
      - price_history: 抓取到的原始历史 DataFrame（若成功）
      - returns_series: 基于 Close 收盘价 的收益率序列（若成功）
      - annual_mu: 年化平均收益（基于 returns_series）
      - annual_sigma: 年化波动率（基于 returns_series）

    说明:
      - 日度年化因子使用 252，周度 52，月度 12；对非常短的 intraday interval（如 '1m','5m'）默认按日化处理。
      - 若抓取失败会尽量返回none。

    """
    # 1. 预先初始化所有 Key，确保“兜底”承诺生效
    stats = {
        'requested_period': period,
        'requested_interval': interval,
        'price_history': None,    
        'returns_series': None,   
        'annual_mu': None,        
        'annual_sigma': None      
    }
    stats['requested_period'] = period
    stats['requested_interval'] = interval

    # 2. 输入对象标准化
    try:
        if isinstance(stock_or_market_ticker, str):
            market_ticker = yf.Ticker(stock_or_market_ticker)
        else:
            market_ticker = stock_or_market_ticker
    except Exception as error:
        print(f"   [错误] cannot create ticker (无法创建市场指数对象)")
        print(f"   具体原因: {error}")

    # 3. 设定时间因子 (Time Factor)
    interval_to_annual = {
        '1d': 252, '1D': 252,
        '1wk': 52, '1w': 52,
        '1mo': 12,
        # 对于分钟/小时级别，默认按日化处理
        '1m': 252, '5m': 252, '15m': 252, '30m': 252, '60m': 252, '1h': 252
    }

    # 防止静默失败”（Silent Failure）
    if interval not in interval_to_annual:
        raise ValueError(f" [错误] 未知的统计间隔 '{interval}'，无法确定年化因子，请在代码中手动添加。")
    
    periods_per_year = interval_to_annual[interval]

    try:
        # 4. 抓取历史数据
        hist = market_ticker.history(period=period, interval=interval)

        # 5. 数据有效性验证
        if 'Close' not in hist.columns or hist['Close'].dropna().empty:
            raise ValueError("No Close price data returned for given period/interval.")

       # 6. 计算收益率序列（基于 Close）
        returns_series = hist['Close'].pct_change().dropna()

        # 7. 计算年化指标（随机漫步理论）
        mu_annual = returns_series.mean() * periods_per_year
        sigma_annual = returns_series.std() * np.sqrt(periods_per_year)

        # 8. 填充数据
        stats['price_history'] = hist
        stats['returns_series'] = returns_series
        stats['annual_mu'] = mu_annual
        stats['annual_sigma'] = sigma_annual

    except Exception as error:
        print(f"   [错误] 抓取市场数据失败")
        print(f"   具体原因: {error}")

    return stats

def clean_financial_df(data_frame, fill_method='ffill', axis=1, warn_threshold=0.1) -> pd.DataFrame:
    """
    清洗财务数据表格的通用函数
    
    逻辑:
    1. 质量体检: 检查缺失率是否过高。
    2. 时间梳理: 强制按时间正序排列，防止倒序导致未来函数，无论 axis 是 0 还是 1，都强制从小到大排 (ascending=True) 这样左边/上边 永远是过去，右边/下边 永远是未来
    3. 缝补漏洞: 根据策略填充缺失值 (默认向前填充)。
    4. 缺失率报警

    参数:
      - fill_method: 
         'ffill' (【默认推荐】: 用前一年的填，最安全，无未来函数)
         'zero'  (填0，适合某些特定科目)
         'interpolate' (警告: 包含未来数据，仅用于画图或写总结报告，不可用于回测。)

        - axis: 
            0 (竖着填): 适合股价历史 (Time series, Date is Index)
                            [ 此时时间在左边！]
                            ↓
                    +------------+-------+
                行0   | 2020-01-01 |  100  |
                    +------------+-------+
                行1   | 2020-01-02 |  105  |  <-- 想要理顺时间，
                    +------------+-------+      必须竖着(axis=0)排。
                行2   | 2020-01-03 |  102  |
                    +------------+-------+

            1 (横着填): 适合财务报表 (Financials, Date is Columns)

                        [ 此时时间在头顶上！]
                                    ↓
                +------------+------------+------------+
                |    2020    |    2021    |    2022    |
                +------------+------------+------------+
            营收  |    100     |    200     |    300     |
                +------------+------------+------------+
                
                ^ 想要理顺时间，必须横着(axis=1)排。

    返回:
    - cleaned_df: 清洗后的 DataFrame
    
    """
    # 1. 防御性检查
    if data_frame is None or data_frame.empty:
        return data_frame

    # 2. 质量监控
    missing_ratio = data_frame.isnull().sum().sum() / data_frame.size
    if missing_ratio > warn_threshold:
        print(f" 警告: 缺失率 {missing_ratio:.2%} 超过警戒线，请检查数据源质量。")

    cleaned_df = data_frame.copy()

    # 3. 时间流向校正
    cleaned_df = cleaned_df.sort_index(axis=axis, ascending=True)

    # 4. 执行清洗策略
    if fill_method == 'ffill':
        # 对应功能点 3: 严格遵守'时间序列'逻辑
        cleaned_df = cleaned_df.ffill(axis=axis)
       # 兜底动作：修补开局 (如果第一天就是空的，前面没有历史，只能填0)
        cleaned_df = cleaned_df.fillna(0)

    elif fill_method == 'zero':
        cleaned_df = cleaned_df.fillna(0)

    elif fill_method == 'linear_fill':
        cleaned_df = cleaned_df.interpolate(method='linear', axis=axis, limit_direction='both')
        cleaned_df = cleaned_df.fillna(0)

    return cleaned_df

def get_fcf_and_details(income_stmt, balance_sheet, cashflow, show_interactive=False):
    """
    从三大报表中挖掘并计算自由现金流 (FCF) 及其组成部分。
    
    逻辑:
    1. 挖掘数据: 
       - 逐个提取 EBIT, Depr, CapEx, WC，包含备用列名搜索。
       
    2. 税率熔断机制 (Risk Control): 
       - 现象: 原始数据的有效税率可能出现负数 (退税) 或极高值 (罚款/数据错误)。
       - 修正: 强制 Clip 在 0% ~ 35% 之间。
       - 对 DCF 估值的影响:
         (a) 若原税率 < 0% (如 -15%): 修正为 0% -> (1-t) 变小 -> FCF 降低。
             [目的: 挤出“退税泡沫”，防止高估未来现金流]
         (b) 若原税率 > 50% (如 80%): 修正为 35% -> (1-t) 变大 -> FCF 升高。
             [目的: 填补“异常深坑”，防止因单年异常低估长期价值]

    3. 组装 FCF: 
       - 公式: EBIT*(1-t) + Depr + CapEx + WC
       - 注意: Yahoo 数据的 CapEx 和 WC 通常为负数，故使用加法。
       - 原公式: CF = NOPAT + Depreciation - CapEx - ChangeInWC

    4. 日志格式化:
       - 打印警告时使用 str(date)[:10] 切片，仅显示 "YYYY-MM-DD"，保持日志整洁。
    
    参数:
    - income_stmt: 清洗后的利润表
    - balance_sheet: 清洗后的资产负债表
    - cashflow: 清洗后的现金流量表
    
    返回 (字典):
    - income_stmt, cashflow, balance_sheet: 原始表
    - ebit_series: 息税前利润序列
    - tax_rate_series: 有效税率序列 (已修正)
    - depr_series: 折旧摊销序列
    - capex_series: 资本支出序列
    - change_in_wc_series: 营运资本变动序列
    - fcf_series: 最终计算出的 FCF 序列
    - valid_fcf: 过滤掉 0 值并按时间正序排列的 FCF (用于估值)
    
    """
    years = income_stmt.columns if income_stmt is not None else []

    # 1. 挖掘 EBIT (息税前利润)
    try:
        ebit_series = income_stmt.loc['EBIT'].fillna(0)
    except Exception:
        try:
            ebit_series = income_stmt.loc['Operating Income'].fillna(0)
        except Exception:
            ebit_series = pd.Series(0, index=years)

    # 2. 计算有效税率 (Tax Rate)
    # 逻辑: Tax Provision / Pretax Income = Tax Rate
    try:
        tax_provision_series = income_stmt.loc['Tax Provision'].fillna(0)
    except Exception:
        try:
            tax_provision_series = income_stmt.loc['Tax provision'].fillna(0)
        except Exception:
            tax_provision_series = pd.Series(0, index=years)

    try:
        # 避免分母为 0
        pretax_income_series = income_stmt.loc['Pretax Income'].replace(0, 1)
    except Exception:
        pretax_income_series = pd.Series(1, index=years)

    tax_rate_series = tax_provision_series / pretax_income_series
    
    # 3. 税率熔断机制 (异常修正)
    # 逻辑: 现实中企业的长期有效税率极少为负或超过 50%。
    abnormal_tax_rate = (tax_rate_series < 0) | (tax_rate_series > 0.50)
    
    if abnormal_tax_rate.any():
        # 记录异常年份
        bad_years = tax_rate_series[abnormal_tax_rate].index.tolist()
        bad_values = tax_rate_series[abnormal_tax_rate].values

        # 格式化打印
        bad_years_str = [str(bad_year)[:10] for bad_year in bad_years] 
        
        print(f"   [警告] 检测到异常税率。年份: {bad_years_str}")
        print(f"   -> 原始异常值: {[f'{bad_value:.2%}' for bad_value in bad_values]}")

        # 执行修正 (Clip)
        tax_rate_series = tax_rate_series.clip(lower=0.0, upper=0.35)
        print(f"  [警告] 已修正为 0% ~ 35% 范围，以免模型崩溃。")
    
    # 4. 挖掘折旧与摊销 (Depreciation)
    try:
        depr_series = cashflow.loc['Depreciation Amortization Depletion'].fillna(0)
    except Exception:
        try:
            depr_series = cashflow.loc['Depreciation amortisation depletion'].fillna(0)
        except Exception:
            depr_series = pd.Series(0, index=years)

    # 5. 挖掘资本支出 (CapEx)
    try:
        capex_series = cashflow.loc['Capital Expenditures'].fillna(0)
    except Exception:
        try:
            # 尝试不同的别名
            capex_series = cashflow.loc['Capital Expenditure'].fillna(0)
        except Exception:
            try:
                capex_series = cashflow.loc['Purchase Of PPE'].fillna(0)
            except Exception:
                capex_series = pd.Series(0, index=years)

    # 6. 挖掘营运资本变动 (Change in WC)
    try:
        change_in_wc_series = cashflow.loc['Change In Working Capital'].fillna(0)
    except Exception:
        change_in_wc_series = pd.Series(0, index=years)

    # 7. 组装 FCF (核心公式)
    # =======================================================
    # 公式: FCF = NOPAT + Depreciation - CapEx - ChangeInWC
    # 
    # [关键警告]: 
    # Yahoo Finance 数据中，CapEx (资本支出) 和 WC 变动通常已经是【负数】(流出)。
    # 所以这里必须用【加号 +】，否则就变成“负负得正”了。
    # =======================================================
    fcf_series = ebit_series * (1 - tax_rate_series) + depr_series + capex_series + change_in_wc_series

    # 8. 清洗与打包
    # 过滤有效 FCF（非 0），并按时间正序
    valid_fcf = fcf_series[fcf_series != 0].sort_index(ascending=True)

    result = {
        'income_stmt': income_stmt,
        'cashflow': cashflow,
        'balance_sheet': balance_sheet,
        'ebit_series': ebit_series,
        'tax_rate_series': tax_rate_series,
        'depr_series': depr_series,
        'capex_series': capex_series,
        'change_in_wc_series': change_in_wc_series,
        'fcf_series': fcf_series,
        'valid_fcf': valid_fcf
    }

    if show_interactive:
        print("已提取并计算财务指标，包含 FCF 等。")
    return result

def compute_risk_metrics(target_ticker, market_ticker, period="5y", interval="1mo"):
    """
    计算资产的 Beta 系数及风险正交分解指标 (CAPM模型)。
    
    数理逻辑:
    根据单因子模型 (Single Factor Model): r_i = alpha + beta * r_m + epsilon
    总方差可分解为 (Variance Decomposition):
        Var(r_i) = beta^2 * Var(r_m) + Var(epsilon)
        总风险²  = 系统性风险²    + 特质性风险²
    
    参数:
    - target_ticker: 目标资产 yfinance 对象
    - market_ticker: 市场基准 yfinance 对象
    - period: 回测周期
    - interval: 数据频率
    
    返回 (字典):
    - beta: 贝塔系数 (系统性风险敞口)
    - correlation: 皮尔逊相关系数
    - r_squared: 拟合优度 (解释力度)
    - annual_sigma: 年化总波动率 (Total Volatility)
    - annual_systematic_sigma: 年化系统性波动率 (Systematic Volatility)
    - annual_idiosyncratic_sigma: 年化特质波动率 (Idiosyncratic Volatility)
    - idiosyncratic_ratio: 特质方差占比 (Variance Ratio)
    
    Risk Decomposition Module
    """
    
    result = {
        "beta": None,
        "correlation": None,
        "r_squared": None,
        "annual_sigma": None,                # 年化总波动率 (Total Risk)
        "annual_systematic_sigma": None,     # 年化系统性风险 (Systematic Risk) 
        "annual_idiosyncratic_sigma": None,  # 年化特质风险 (Idiosyncratic Risk) 
        "idiosyncratic_ratio": None,         # 特质风险占比
        "data_points": 0,
        "warning": None
    }

    try:
        # 1. 获取基础统计量
        # 注: 此处获取的 annual_sigma 为标准差 (Standard Deviation)
        stock_stats = compute_market_statistics(target_ticker, period, interval)
        market_stats = compute_market_statistics(market_ticker, period, interval)

        if 'returns_series' not in stock_stats or stock_stats['returns_series'] is None:
            raise ValueError("目标股票数据获取失败")
        if 'returns_series' not in market_stats or market_stats['returns_series'] is None:
            raise ValueError("市场指数数据获取失败")

        stock_ret = stock_stats['returns_series']
        market_ret = market_stats['returns_series']
        
        # 赋值总波动率 (Sigma)
        result["annual_sigma"] = stock_stats['annual_sigma']
        market_annual_sigma = market_stats['annual_sigma']

        # 2. 数据对齐 (Data Alignment)
        df = pd.concat([stock_ret, market_ret], axis=1, join="inner")
        df.columns = ["stock_ret", "market_ret"]
        
        if len(df) < 2:
            raise ValueError("有效重叠数据点太少，无法计算")
            
        result["data_points"] = len(df)

        # 3. 计算 Beta (系统性敞口)
        # Formula: Beta = Cov(r_i, r_m) / Var(r_m)
        covariance = df["stock_ret"].cov(df["market_ret"])
        market_var = df["market_ret"].var()
        
        if market_var > 0:
            result["beta"] = covariance / market_var
        
        # 4. 计算相关性与拟合优度
        # Correlation (rho)
        result["correlation"] = df["stock_ret"].corr(df["market_ret"])

        # R-Squared (R^2): 衡量模型解释力度
        # 在单变量线性回归中，R^2 = Correlation^2
        if result["correlation"] is not None:
            result["r_squared"] = result["correlation"] ** 2
        
        # 5. 风险正交分解 (Risk Orthogonal Decomposition)
        # =======================================================
        # 理论基础: Total Variance = Systematic Variance + Idiosyncratic Variance
        # 注意: 分解是基于“方差”(Variance) 进行的，而非“标准差”(Sigma)
        # =======================================================
        if result["beta"] is not None:
            # (A) 系统性波动率 (Systematic Sigma)
            # Formula: Sigma_sys = |Beta| * Sigma_mkt
            result["annual_systematic_sigma"] = abs(result["beta"]) * market_annual_sigma
            
           # (B) 特质性波动率 (Idiosyncratic Sigma)
            # Formula: Sigma_idio = sqrt( Sigma_total^2 - Sigma_sys^2 )
            # 也可以使用: Sigma_idio = Sigma_total * sqrt(1 - R^2)
            var_total = result["annual_sigma"] ** 2
            var_sys = result["annual_systematic_sigma"] ** 2
            
            # 使用 max(0, ...) 防止浮点数精度误差导致的负值
            result["annual_idiosyncratic_sigma"] = np.sqrt(max(0, var_total - var_sys))
            
            # (C) 特质风险占比 (Idiosyncratic Variance Ratio)
            # 该指标衡量有多少方差是无法被市场解释的
            # 特质风险占比 = 特质风险^2 / 总风险^2 = 1 - R²
            if var_total > 0:
                result["idiosyncratic_ratio"] = (result["annual_idiosyncratic_sigma"] ** 2) / var_total

    except Exception as error:
        result["warning"] = str(error)

    return result

def get_historical_target_debt_ratio(ticker, balance_sheet_df, years=5):
    """
    基于历史数据估算公司的目标资本结构 (Target Capital Structure)。
    
    数理逻辑:
    WACC 模型中的权重 (w_d, w_e) 通常应反映公司长期的目标资本结构。
    若假设公司维持当前的杠杆水平，可通过回溯历史计算 D/(D+E) 的均值作为预测依据。
    
    计算步骤:
    1. 提取历史总债务 (Book Value of Debt)。
    2. 回溯财报发布日的股价，计算权益市场价值 (Market Value of Equity)。
    3. 计算每一期的债务权重: w_d = D / (D + E)。
    4. 取算术平均值作为 Target Debt Ratio。
    
    参数:
    - ticker: yfinance Ticker 对象
    - years: 回溯年份数量 (int)
    
    返回:
    - float: 历史平均债务占比 (Target Debt Weight)
    - None: 数据不足或获取失败
    
    """   
    try:
        # 1. 获取资产负债表 (Data Retrieval)
        raw_balance = balance_sheet_df
        
        if raw_balance.empty:
            print(" [警告] 无法获取财报，建议使用当前比率。")
            return None
        
        # 2. 锁定债务数据 (Debt Extraction)
        # 尝试匹配不同的会计科目名称
        debt_names = ['Total Debt', 'Total debt', 'Total Liab']
        debt_row = None
        for name in debt_names:
            if name in raw_balance.index:
                debt_row = raw_balance.loc[name]
                break
        if debt_row is None:
            raise ValueError(" [警告] 在 Balance Sheet 中没找到债务数据")
        debt_row = debt_row.sort_index(ascending=False).head(years)
            
        # 3. 获取流通股数 (Proxy for Market Cap)
        # 简化假设 (Simplified Assumption): 
        # 使用"当前"流通股数乘以"历史"股价来估算历史市值。
        # 虽然忽略了历史回购/增发的影响，但在缺乏历史 share count 数据时是通用的工程近似解。
        shares_outstanding = ticker.info.get('sharesOutstanding')
        if not shares_outstanding:
            raise ValueError("[警告] 无法获取流通股数")
        
        # 4. 回溯历史资本结构 (Backtesting Loop)
        ratios = {} 
       # 遍历每一个财报日期 (Reporting Date)
        for date, debt_value in debt_row.items():
            
            # (A) 时间窗口对齐 (Time Alignment)
            # 取前后 5 天是为了防止财报日是非交易日 (周末/节假日) 导致取不到股价
            start_date = date - pd.Timedelta(days=5)
            end_date = date + pd.Timedelta(days=5)
            
            start_end_hist = ticker.history(start=start_date, end=end_date)

            if start_end_hist.empty:
                continue 
                
            # (B) 计算权益市场价值 (Market Value of Equity, E)
            # 使用窗口期均价平滑波动
            avg_price = start_end_hist['Close'].mean()
            market_cap = avg_price * shares_outstanding
            
            # (C) 计算债务权重 (Debt Weight, w_d)
            # Formula: w_d = D / (D + E)
            if pd.isna(debt_value) or debt_value == 0:
                continue
            current_ratio = debt_value / (debt_value + market_cap)
            
            # 记录: {2023: 0.25, 2022: 0.30}
            ratios[date.year] = current_ratio
        
        if not ratios:
            return None
            
        # 5. 统计聚合 (Statistical Aggregation)
        # 取算术平均值代表长期目标杠杆率
        avg_ratio = sum(ratios.values()) / len(ratios)
        
        print(f"   历史平均债务占比 (D/(D+E)): {avg_ratio:.2%}")
        return avg_ratio

    except Exception as error:
        print(f" [警告] 计算历史债务占比失败 ({error})，建议回退使用当前 Current Ratio")
        return None

def adjust_beta(raw_beta, current_debt, current_equity, tax_rate, guess_target_debt_ratio=None):
    """
    基于哈马达公式 (Hamada Equation) 对 Beta 进行去杠杆与再杠杆调节。
    
    数理逻辑:
    观察到的 Beta (Raw Beta) 包含了 "业务风险" 和 "财务风险" (当前杠杆)。
    为了预测不同资本结构下的 WACC，必须执行如下变换:
    
    1. Unlever (去杠杆): 剔除当前财务杠杆，提取纯粹的资产 Beta (Asset Beta)。
       Formula: Beta_Asset = Beta_Levered / [ 1 + (1 - Tax) * (D/E)_current ]
       
    2. Relever (再杠杆): 将资产 Beta 映射到目标资本结构，得到目标权益 Beta。
       Formula: Beta_Target = Beta_Asset * [ 1 + (1 - Tax) * (D/E)_target ]
       
    参数:
    - raw_beta: 历史回归得到的 Beta (Levered Equity Beta)
    - current_debt: 当前总债务 (Market/Book Value)
    - current_equity: 当前权益价值 (Market Cap)
    - tax_rate: 有效税率
    - guess_target_debt_ratio: 目标债务占比 (Total Debt / Total Capital)。
      用于寻找最佳资本结构 (Optimal Capital Structure)。若为 None，则假设维持现状。
    
    返回:
    - final_beta: 调整后的权益 Beta (用于 CAPM 模型计算 Ke)

    """
    
    # 1. 计算当前杠杆比率 (Current Leverage Ratio)
    # D/E Ratio = Debt / Equity
    current_de_ratio = current_debt / current_equity
    
    # 2. 去杠杆 (Unlevering Process)
    # 目的: 获取 Asset Beta (即衡量纯粹业务风险的 Beta)
    unlevered_beta = raw_beta / (1 + (1 - tax_rate) * current_de_ratio)
    
    # 策略分支: 如果没有设定目标结构，则默认维持当前结构
    if guess_target_debt_ratio is None:
        return raw_beta

    # 3. 再杠杆 (Relevering Process)
    # 目的: 计算目标资本结构下的 Equity Beta
    # -------------------------------------------------------
    # 数学转换: 输入的是 Debt_Ratio (w_d = D / (D+E))
    # 公式需要的是 D/E Ratio
    # 推导: D/E = w_d / (1 - w_d)
    target_equity_ratio = 1 - guess_target_debt_ratio
    target_de_ratio = guess_target_debt_ratio / target_equity_ratio
    
    relevered_beta = unlevered_beta * (1 + (1 - tax_rate) * target_de_ratio)
    
    return relevered_beta

def get_risk_free_rate(duration='10y', default_rf_rate=0.04):
    """
    获取无风险利率 (Risk-Free Rate)，即资金的时间价值基准。
    
    逻辑:
    1. 概念定义: 代表当下的机会成本 (Opportunity Cost)。
    2. 标的选择: 根据期限映射到对应的国债收益率指数 (5年 ^FVX, 10年 ^TNX, 30年 ^TYX)。
    3. 数据清洗: 取最近 5 个交易日的收盘价平均值。
       (目的: 平滑单日市场噪音，防止因某一天数据异常导致模型跳变)
    4. 单位换算: Yahoo Finance 返回的是百分数 (如 4.25)，需除以 100 转换为小数 (0.0425)。

    参数:
    - duration: 期限选择 '5y', '10y', '30y' (默认 '10y' 为行业标准)
    - default_rf_rate: 兜底值 (默认 4%)，防止网络故障导致模型卡死。

    返回:
    - current_rf_rate: 无风险利率 (小数, e.g. 0.0425)

    """
    # 1. 期限映射 (Ticker Mapping)
    tickers_choice = {
        '5y':  '^FVX',  # 5年期国债
        '10y': '^TNX',  # 10年期国债
        '30y': '^TYX'   # 30年期国债
    }
    if duration not in tickers_choice:
        raise ValueError(f"  [警告] 不支持期限 '{duration}'。只能选: {list(tickers_choice.keys())}")

    ticker_symbol = tickers_choice[duration]

    try:
        # 2. 获取数据 (Data Retrieval)
        ticker = fetch_single_ticker(ticker_symbol)
        # 只取最近 1 个月的数据就足够计算 5 日均线
        hist = ticker.history(period="1mo")
        if hist.empty:
            raise ValueError(" [警告] 无风险利率数据为空")
        
        # 3. 平滑处理与单位换算 (Smoothing & Conversion)
        # 逻辑: 取最后 5 天的平均值 -> 除以 100
        # 注意: 国债 Indices (如 ^TNX) 的报价 4.25 代表 4.25%，不是 4.25 美元
        current_rf_rate = hist['Close'].tail(5).mean() / 100
        print(f"  无风险利率 获取成功 ({duration}): {current_rf_rate:.2%} (5日均值)")        
        return current_rf_rate

    except Exception as error:
        # 4. 熔断兜底 (Fallback)
        print(f" [警告] 获取无风险利率失败 ({error})，启用默认值: {default_rf_rate:.2%}")
        return default_rf_rate
    
def get_real_historical_erp(current_rf_rate, years=20, default_erp=0.05):
    """
    计算基于历史数据的真实市场风险溢价 (Historical ERP)。
    
    定义:
    ERP (Equity Risk Premium) = Rm (市场预期回报) - Rf (无风险利率)
    它代表了投资者为了承担股票市场的波动风险，要求比买国债多拿多少收益。
    
    逻辑:
    1. 抓取基准: 获取标普500 (^GSPC) 过去 N 年的历史数据。
    2. 计算 Rm: 使用 CAGR (复合年化增长率) 计算市场的长期几何平均回报。
       (相比算术平均，几何平均更能反映长期持有的真实复利效果)
    3. 动态调整: 用历史 Rm 减去 *当前* 的 Rf。
    4. 熔断机制: 若计算结果为负 (即 倒挂)，强制修正为 2%，防止 CAPM 失效。
    
    参数:
    - current_rf_rate: 当前的无风险利率 (由 get_risk_free_rate 获取)
    - years: 回溯周期 (默认 20 年，涵盖完整的康波/朱格拉周期)
    - default_erp: 兜底默认值 (通常取 5%~5.5%)
    
    返回:
    - real_erp: 市场风险溢价 (小数)

    """    
    try:
        # 1. 抓取市场基准 (Market Benchmark)
        market_ticker = fetch_single_ticker("^GSPC")
        hist = market_ticker.history(period=f"{years}y")
        
        if hist.empty:
            raise ValueError("[警告] sp500数据为空")
            
        # 2. 计算年化复合增长率 (CAGR) 
        # 逻辑:CAGR = ((期末价格 / 期初价格) ^ (1 / 年数)) - 1
        start_price = hist['Close'].iloc[0]   
        end_price = hist['Close'].iloc[-1]    
        
       # 计算几何平均收益率
        rm_cagr = ((end_price / start_price) ** (1 / years)) - 1
        
        # 3. 核心计算 (The Spread)
        # ERP = 股市回报 - 国债回报
        real_erp = rm_cagr - current_rf_rate
        
        # 4. 熔断机制 (Sanity Check)
        # -------------------------------------------------------
        # 异常场景: 当国债收益率飙升 (如加息周期) 或股市长期低迷时，
        # real_erp 可能会变成负数。这意味着“买股票不如买国债”。
        # 但在估值模型中，负的 ERP 会导致 Ke < Rf 甚至 Ke < 0，逻辑上不成立。
        # -------------------------------------------------------
        if real_erp < 0:
            print(f" [警告] 计算出的 ERP 为负值({real_erp:.2%})。(说明近期国债收益率高于股市历史回报)，修正为 2.00%")
            real_erp = 0.02
            return real_erp 

        print(f"  历史市场风险溢价 ERP = {rm_cagr:.2%} (市场) - {current_rf_rate:.2%} (国债) = {real_erp:.2%}")
        return real_erp

    except Exception as error:
        print(f" [警告] 计算历史 ERP 失败 ({error})，启用默认值: {default_erp:.2%}")
        return default_erp

def guess_equity_risk_premium(current_rf_rate, manual_input=None, years=20):
    """
    智能 ERP 路由与决策函数。
    
    逻辑:
    实现 "人机结合" (Quantamental) 的估值策略：
    1. 优先模式 (Analyst Override): 如果分析师提供了主观判断 (manual_input)，具有最高优先级。
       (适用于：需要直接使用分析师的最新数据，或对未来市场有特殊预判时)
       
    2. 自动模式 (Quantitative Fallback): 如果没有手动输入，则回退到历史数据模型。
       (适用于：批量处理股票，或希望完全依赖客观历史数据时)
    
    参数:
    - current_rf_rate: 当前无风险利率 (用于自动计算模式)
    - manual_input: 手动指定的 ERP (小数，如 0.055)。若为 None 则启用自动模式。
    - years: 自动模式下的回测周期 (默认 20 年)
    
    返回:
    - final_erp: 最终决定的 ERP 值

    """
    # 1. 分析师覆盖模式 (Analyst Override)
    # 逻辑: 人的判断 > 历史数据。如果输入了数值，直接返回。
    if manual_input is not None:
        return manual_input
    
    # 2. 量化模型模式 (Quantitative Fallback)
    # 逻辑: 调用历史测算模块，基于 SP500 和国债的利差自动计算
    return get_real_historical_erp(current_rf_rate, years=years, default_erp=0.05)

def get_capm_ke(relevered_beta, risk_free_rate, guess_equity_risk_premium):
    """
    基于 CAPM 模型计算股权成本 (Cost of Equity, Ke)。
    
    数理逻辑:
    Ke 代表股东为了承担持有该股票的风险，所要求的最低回报率。
    公式: Ke = Rf + Beta * (Rm - Rf)
             = Rf + Beta * ERP
    
    逻辑流程:
    1. 核心计算: 将风险倍数 (Beta) 与市场溢价 (ERP) 结合，叠加无风险基准。
    2. 市场水位评估: 反推隐含的市场预期回报 (Rm)，用于辅助判断宏观环境。
    3. 逻辑熔断 (Sanity Check):
       - 现象: 对于低 Beta 股票，计算出的 Ke 可能低于国债收益率。
       - 修正: 强制要求 Ke >= Rf + 2.00%。
       - 原理: 理性投资者不会在承担更高风险的情况下，接受比国债还低的回报 (风险收益不对等)。
    
    参数:
    - relevered_beta: 权益贝塔 (已包含财务杠杆的最终风险系数)
    - risk_free_rate: 无风险利率 (Rf,通常指 10 年期国债收益率 )
    - guess_equity_risk_premium: 市场风险溢价 (ERP, 市场温度)
    
    返回:
    - ke: 股权成本 (小数)
    
    """

    # 1. 基础检查
    if relevered_beta is None:
        print(" [错误] Beta 为空，无法计算 Ke。")
        return None
        
    try:
        # 2. 核心计算
        ke = risk_free_rate + (relevered_beta * guess_equity_risk_premium)
        
        # 辅助指标: 反推市场预期回报 (Implicit Market Return)
        expect_rm = risk_free_rate + guess_equity_risk_premium

        print(f"  市场风险溢价 (ERP): {guess_equity_risk_premium:.2%} (Rm - Rf)")
        print(f"  -> 隐含的市场预期回报 (Rm): {expect_rm:.2%} (大盘平均分)")

        # 3. 逻辑修正 (防御性风控)
        # -------------------------------------------------------
        # 场景 A: 修正地板价 (Floor Correction)
        # 即使 Beta 很低 (如 0.5)，买股票也不应低于存银行。
        # -------------------------------------------------------
        if ke < risk_free_rate:
            print(f"  [修正] 计算出的 Ke ({ke:.2%}) 低于无风险利率，逻辑不合理。")
            print(f"  -> 已强制修正为: Rf + 2.00% (设定最低风险门槛)")
            ke = risk_free_rate + 0.02

        # 场景 B: 高风险预警 (Ceiling Warning)
        if ke > 0.30:
            print(f"  [警告] 计算出的 Ke 高达 {ke:.2%}！这会导致估值极低，请检查数据。")
            
        print(f"  最终股权成本 (Ke): {ke:.2%}")
        return ke

    except Exception as error:
        print(f" [警告] 计算 Ke 发生未知错误: {error}")
        return None

def get_cost_of_debt_kd(income_stmt, balance_sheet, risk_free_rate):
    """
    计算税前债务成本 (Pre-tax Cost of Debt, Kd)。
    
    数理逻辑:
    Kd 代表公司借钱的平均利率。最粗略的算法是：年利息支出 / 总债务。
    
    逻辑流程:
    1. 锁定窗口: 只看最近一年的财报数据。
    2. 模糊挖掘: 
       - 分子 (利息): 尝试匹配 'Interest Expense', 'Interest expense' 等字段。
       - 分母 (债务): 优先找 'Total Debt'，找不到则尝试用 '短期债务 + 长期债务' 合成。
    3. 逻辑熔断 (Sanity Check):
       - 现象: 计算出的 Kd 可能低于无风险利率 (Rf)。
       - 修正: 强制要求 Kd >= Rf。
       - 原理: 公司信用再好也不可能比印钞票的政府更好 (Default Spread >= 0)。
    
    参数:
    - income_stmt: 清洗后的利润表
    - balance_sheet: 清洗后的资产负债表
    - risk_free_rate: 无风险利率 (Rf)
    
    返回:
    - final_kd: 修正后的债务成本 (小数)
    - interest_expense: 利息支出金额
    - total_debt: 总债务金额
    
    """
    try:
        # 1. 锁定最新财报窗口 (Time Window)
        # 取 DataFrame 的最后一列 (通常是最近的一年/一季)
        latest_income_data = income_stmt.iloc[:,-1] 
        latest_balance_data = balance_sheet.iloc[:,-1] 

        # 2. 挖掘利息支出 (分子, Numerator)
        interest_expense = None 
        # 不同会计准则下的科目别名
        possible_interest_names = ['Interest Expense', 'Interest expense', 'Interest Expense Non Operating']
        
        for name in possible_interest_names:
            if name in latest_income_data.index:
                interest_expense = latest_income_data[name]
                break 
        
        if interest_expense is None:
            print(f" [警告] 利润表中找不到利息支出科目。")
            print(f" 尝试过的列名: {possible_interest_names}")
            print(f" 实际可用列名: {latest_income_data.index.tolist()}")
            return None, None, None

        # 3. 挖掘债务总额 (分母, Denominator)
        # -------------------------------------------------------
        total_debt = None 
        possible_debt_names = ['Total Debt', 'Total debt', 'Long Term Debt And Capital Lease Obligation']
        
        # 策略 A: 直接查找总债务科目
        for name in possible_debt_names:
            if name in latest_balance_data.index:
                total_debt = latest_balance_data[name]
                break
        
        # 策略 B: 如果没找到总数，尝试“拼凑法” (短期 + 长期)
        if total_debt is None:
            current_debt_names = ['Current Debt', 'Current debt', 'Short Term Debt', 'Short Term Borrowings']
            long_term_debt_names = ['Long Term Debt', 'Long term debt', 'Non-Current Debt', 'Long Term Debt And Capital Lease Obligation']
            
            current_debt = None
            long_term_debt = None
            
            # 找短期债务
            for name in current_debt_names:
                if name in latest_balance_data.index:
                    current_debt = latest_balance_data[name]
                    break
            
            # 找长期债务
            for name in long_term_debt_names:
                if name in latest_balance_data.index:
                    long_term_debt = latest_balance_data[name]
                    break
            
            # 只有当两者都找到时，才能合成总债务
            if current_debt is not None and long_term_debt is not None:
                total_debt = current_debt + long_term_debt
            else:
                print(f" [警告] 找不到总债务，且无法通过短期+长期债务计算。")
                print(f" - 短期债务查找结果: {'找到' if current_debt is not None else '未找到'} (尝试列表: {current_debt_names})")
                print(f" - 长期债务查找结果: {'找到' if long_term_debt is not None else '未找到'} (尝试列表: {long_term_debt_names})")
                return None, None, None

        # 4. 计算原始 Kd (Raw Calculation)
        # 取绝对值防止数据源中的负号干扰
        final_interest = abs(interest_expense)
        final_debt = abs(total_debt)

        # 边界情况: 无负债公司
        if final_debt == 0:
            print(" [警告] 公司债务为 0 (无杠杆)，Kd 设为 0")
            return 0.0, 0.0, 0.0

        raw_kd = final_interest / final_debt
        final_kd = raw_kd
        
        # 5. 逻辑熔断 (Synthetic Rating Floor)
        # -------------------------------------------------------
        # 逻辑: 任何公司的融资成本都不应低于无风险利率 (Rf)。
        # 如果算出 Kd < Rf，说明要么数据有误，要么是一次性因素导致利息偏低。
        # -------------------------------------------------------
        if raw_kd < risk_free_rate:
            print(f" [警告] 计算出的 Kd ({raw_kd:.2%}) 低于无风险利率 ({risk_free_rate:.2%})，已修正。")
            final_kd = max(raw_kd, risk_free_rate) 

        return final_kd, final_interest, final_debt

    except Exception as error:
        print(f" [警告] 计算 Kd 过程发生异常: {error}")
        return None, None, None 

def get_current_equity_value(ticker):
    """
    获取公司当前的股权价值 (Market Capitalization / Equity Value)。
    
    数理逻辑:
    在 WACC 模型中，E (Equity) 指的是权益的市场价值，而非账面价值。
    
    获取策略 (双重熔断机制):
    1. 高速通道 (Primary): 尝试从交易所元数据 (fast_info) 直接读取市值。
    2. 兜底通道 (Fallback): 若读取失败，通过 "最新收盘价 * 流通股数" 手动合成。
    
    参数:
    - ticker: yfinance.Ticker 对象
    
    返回:
    - equity_value: 市值 (浮点数, 单位: 元)
    """
    equity_value = None

    # 1. 高速通道: 直接读取元数据 (fast_info)
    try:
        equity_value = ticker.fast_info['market_cap']
        if equity_value is not None and equity_value > 0:
            print(f" 通过 fast_info 获取市值: {equity_value/1e9:.2f}B")
            return equity_value
    except Exception:
        pass  # 接口调用失败，静默进入下一层兜底逻辑

    # 2. 兜底通道: 手动合成 (最新股价 * 流通股数)
    # Price * Shares Outstanding
    try:
        # 逻辑:以此规避周末、节假日或临时停牌导致的单日数据缺失，取最近有效收盘价
        hist = ticker.history(period="5d")
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            shares = ticker.info.get('sharesOutstanding')
            if shares:
                equity_value = current_price * shares
                print(f"  手动计算市值: {equity_value/1e9:.2f}Billion, (价格 {current_price:.2f} * 股数)")
                return equity_value
    except Exception as error:
        print(f"  [警告] 无法计算市值: {error}")
        
    if equity_value is None:
        print("  [警告] 无法获取 Equity Value。")

    return None

def compute_wacc(ke, kd, tax_rate, equity_value, debt_value):
    """
    计算加权平均资本成本 (WACC / Weighted Average Cost of Capital)。
    
    数理逻辑:
    WACC 是企业融资的整体机会成本，也是 DCF 模型中将未来现金流折现回现在的核心折现率。
    公式: WACC = (E/V * Ke) + (D/V * Kd * (1 - Tax))
    其中 V = E + D (总资本)
    
    参数:
    - ke: 股权成本 (Cost of Equity, 股东回报率)
    - kd: 税前债务成本 (Pre-tax Cost of Debt, 债权人的利率)
    - tax_rate: 有效税率 (用于计算税盾)
    - equity_value: 权益市场价值 (Market Value of Equity / Market Cap)
    - debt_value: 债务总额 (Total Debt, 通常用账面价值代替)
    
    返回:
    - wacc: 最终折现率 (小数)
    """  
    # 1. 输入完整性校验 (Input Validation)
    if ke is None or kd is None:
        print(" [警告] 缺参数: Ke 或 Kd 为空， 无法计算 WACC。")
        return None
        
    # 2. 计算总资本 (Total Invested Capital, V)
    total_value = equity_value + debt_value
    
    if total_value == 0:
        print(" [警告] 总资本为0 (Equity + Debt = 0)，无法计算权重。")
        return None
        
    # 3. 计算资本结构权重 (Capital Structure Weights)
    # Weight of Equity (We) vs Weight of Debt (Wd)
    weight_equity = equity_value / total_value
    weight_debt = debt_value / total_value
    
    print(f"  市值 (E): {equity_value/1e9:.2f}B (占比 {weight_equity:.2%})")
    print(f"  债务 (D): {debt_value/1e9:.2f}B (占比 {weight_debt:.2%})")
    
    ## 4. 核心计算 (Core Component Calculation)
    # -------------------------------------------------------
    # A. 股权贡献部分
    equity_part = weight_equity * ke
    
    # B. 债务贡献部分 (含税盾效应 Tax Shield Effect)
    # 核心原理: 利息支出在税前扣除，降低了应税收入，相当于政府补贴了部分利息成本。
    # 实际债务成本 = Kd * (1 - Tax Rate)
    after_tax_kd = kd * (1 - tax_rate)
    debt_part = weight_debt * after_tax_kd
    
    print(f"  股权成本 (Ke): {ke:.2%}")
    print(f"  债务成本 (Kd): {kd:.2%} -> 税后: {after_tax_kd:.2%}")
    
    # 5. 加权汇总 (Aggregation)
    wacc = equity_part + debt_part
    
    # 6. 合理性检验 (Sanity Check & Bounds)
    # -------------------------------------------------------
    # WACC 作为企业的融资成本，具有一定的合理区间。
    # -------------------------------------------------------
    if wacc <= 0.03:
        # 异常低: 接近无风险利率，通常意味着数据错误或模型假设过于激进。
        print(f"  [警告] WACC ({wacc:.2%}) 低得不正常 (小于 3%)。请检查数据是否输入错误。")
    elif wacc >= 0.20:
        # 异常高: 通常出现在困境反转股 (Distressed) 或极早期高风险初创企业。
        print(f"  [警告] WACC ({wacc:.2%}) 非常高 (大于 20%)。这通常是初创公司或快倒闭的公司。")

    print(f"  WACC: {equity_part:.2%} (股权贡献) + {debt_part:.2%} (债务贡献) = {wacc:.2%}")
    
    return wacc

def run_dcf_valuation(current_fcf, wacc, growth_rate_5y, terminal_growth_rate, shares_outstanding, net_debt):
    """  
    核心估值函数：执行两阶段 DCF 模型计算每股内在价值。
    
    数学模型 (Two-Stage DCF Model):
    1. 显性预测期 (Explicit Forecast Period, T1-T5):
       假设公司以较高的增长率 (growth_rate_5y) 高速发展。
       Formula: FCF[t] = FCF[0] * (1 + g)^t
       
    2. 永续增长期 (Terminal Value Period, T6 -> ∞):
       假设公司进入稳定成熟期，以永续增长率 (g_terminal) 增长。
       使用戈登增长模型 (Gordon Growth Model) 将未来无穷的现金流折算为 T5 时刻的价值。
       Formula: TV_T5 = [ FCF_T5 * (1 + g_terminal) ] / ( WACC - g_terminal )
       
    3. 价值汇总 (Sum of Present Values):
       Enterprise Value = PV(Explicit FCFs) + PV(Terminal Value)


    该函数执行 Discounted Cash Flow (DCF) 模型的完整流程，包含以下五个核心数学步骤：
    核心公式清单 (Mathematical Formulas):    
    1. 显性期预测 (Explicit Period Forecasting):
       计算未来 5 年的自由现金流 (FCF)，假设每年以固定比率增长。
       >> FCF[t] = Current_FCF * (1 + growth_rate_5y)^t
    
    2. 终值计算 (Terminal Value - Gordon Growth Model):
       计算第 5 年之后直到永远的价值。注意：此价值归属于第 5 年年末。
       >> TV = ( FCF[5] * (1 + terminal_growth_rate) ) / ( WACC - terminal_growth_rate )
    
    3. 现金流折现 (Discounting):
       将未来的钱折算回今天 (Year 0)。
       >> PV_FCF = Σ ( FCF[t] / (1 + WACC)^t )      <-- 前5年的现值
       >> PV_TV  = TV / (1 + WACC)^5                <-- 终值的现值
    
    4. 企业价值 (Total Enterprise Value):
       >> TEV = PV_FCF + PV_TV
       
    5. 股权价值与股价 (Equity Value & Intrinsic Value):
       >> Equity_Value = TEV - Net_Debt
       >> Intrinsic_Value = Equity_Value / Shares_Outstanding
    
    参数:
    - current_fcf: 最近一年的自由现金流 (FCF_0)
    - wacc: 加权平均资本成本 (Discount Rate)
    - growth_rate: 显性期(前5年)的预测增长率 (g)
    - terminal_growth_rate: 永续增长率 (g_terminal, 通常 2%~3%)
    - shares_outstanding: 总股本
    - net_debt: 净债务 (Total Debt - Cash)
    
    返回:
    - intrinsic_value: 每股内在价值 (Fair Value Per Share)

    """

    # Phase 1: 显性期预测 (Explicit Period Forecasting)
    # ==========================================
    future_fcfs_5y = {}
    for year in range(1, 6): # i 也就是 1, 2, 3, 4, 5
        fcf = current_fcf * ((1 + growth_rate_5y) ** year)
        future_fcfs_5y[year] = fcf  

    # ==========================================
    # Phase 2: 终值计算 (Terminal Value Calculation) - 戈登增长模型
    # ==========================================
    fcf_year5th = future_fcfs_5y[5]

    # [数学约束检查] Convergence Condition
    # 根据几何级数求和公式，只有当 r (WACC) > g (Terminal Growth) 时，级数才收敛。
    # 否则分母为负或零，意味着价值无穷大，模型崩塌。
    if wacc <= terminal_growth_rate:
        print(f" [错误] WACC ({wacc:.2%}) 必须大于 永续增长率 ({terminal_growth_rate:.2%})，否则模型无效。")
        return None 
    
    # 戈登增长模型 (Gordon Growth Model)
    # 计算的是 T5 时刻的价值 (即 T6 及以后所有现金流在 T5 的现值)
    # TV_T5 = FCF_T6 / (WACC - g) = FCF_T5 * (1+g) / (WACC - g)
    terminal_value = fcf_year5th * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)
    
    # ==========================================
    # Phase 3: 折现回溯 (Discounting to Present)
    # ==========================================

    explicit_period_pv = 0
    # A. 显性期现金流折现
    for year in range(1, 6):
        discount_factor = (1 + wacc) ** year
        explicit_period_pv += future_fcfs_5y[year] / discount_factor

    # B. 终值折现
    # 注意: terminal_value 是 T5 时刻的钱，需要折现 5 年回到 T0
    # 洞察: 对于成熟企业，TV_PV 通常占总企业价值的 60%-80%。
    terminal_value_pv = terminal_value / ((1 + wacc) ** 5)

    # ==========================================
    # Phase 4: 股权价值计算 (Equity Valuation)
    # ==========================================

    # 1. 企业价值 (TEV / Total Enterprise Value)
    total_enterprise_value = explicit_period_pv + terminal_value_pv

    # 2. 股权价值 (Equity Value)
    # TEV 归属于所有出资人 (股东+债权人)，减去净债务后才是归属于股东的价值
    equity_value = total_enterprise_value - net_debt
    
    # 3. 每股内在价值 (Intrinsic Value Per Share)
    if shares_outstanding <= 0:
        print(" [警告] shares_outstanding 流通股数无效 (<=0)")
        return 0.0
    
    # Intrinsic Value (内在价值/公允价值)
    intrinsic_value = equity_value / shares_outstanding

    return intrinsic_value

def run_monte_carlo_simulation(current_fcf, base_wacc, growth_rate_5y, 
                               shares, net_debt, num_simulations=1000, terminal_growth_rate=0.025):
    """
    蒙特卡洛模拟器 (Monte Carlo Simulator) - 随机游走估值。
    
    数理逻辑:
    将传统的 DCF "点估计" (Point Estimate) 转化为 "概率分布" (Probability Distribution)。
    通过引入随机变量，模拟成千上万种可能的未来平行宇宙 (Scenarios)，从而评估估值的置信区间。
    
    核心步骤:
    1. 定义随机过程 (Stochastic Process):
       假设输入的关键参数 (WACC, Growth) 服从正态分布 (Normal Distribution)。
       >> WACC_random ~ N(μ=base_wacc, σ=1.0%)
       >> Growth_random ~ N(μ=base_growth, σ=2.0%)
       
    2. 迭代推演 (Simulation Loop):
       生成每一次独立的市场环境，计算对应的内含价值。
       
    3. 数据清洗 (Sanity Filtering):
       自动剔除因数学收敛失败 (如 WACC < g) 导致的无效样本。
    
    参数:
    - current_fcf: 当前自由现金流
    - base_wacc: 基准 WACC (期望值)
    - growth_rate_5y: 基准显性期增长率 (期望值)
    - shares: 总股本
    - net_debt: 净债务
    - num_simulations: 模拟次数 (默认 1000 次，根据大数定律，次数越多分布越接近真实)
    - terminal_growth_rate: 永续增长率 (固定参数)
    
    返回:
    - simulation_results: 包含所有有效预测股价的列表 (List of Floats)

    """
    simulation_results = []
    
    print(f"\n 开始执行 {num_simulations} 次蒙特卡洛模拟...")
    
    for i in range(num_simulations):
        # 1. 生成随机参数 (Stochastic Parameters)
        # -------------------------------------------------------
        # 使用 normalvariate(mu, sigma) 生成正态分布随机数
        
        # A. 随机 WACC (标准差 σ = 1.0%)
        # 逻辑: WACC 主要受宏观利率 (Rf) 和市场溢价 (ERP) 影响，相对系统化，波动较小。
        random_wacc = random.normalvariate(base_wacc, 0.01)

        # B. 随机 Growth (标准差 σ = 2.0%)
        # 逻辑: 公司业绩受特质风险 (Idiosyncratic Risk) 影响极大。
        # 无论是产品失败、管理层动荡 ("CEO发疯") 还是竞争加剧，都会导致增长率剧烈震荡。
        # 因此，这里的波动率设定通常高于 WACC。
        random_growth = random.normalvariate(growth_rate_5y, 0.02)
        
        # 2. 执行单次估值 (Single Scenario Valuation)
        price = run_dcf_valuation(
            current_fcf=current_fcf,
            wacc=random_wacc,
            growth_rate_5y=random_growth,
            terminal_growth_rate=terminal_growth_rate,
            shares_outstanding=shares,
            net_debt=net_debt
        )
        
        # 3. 结果清洗与收集 (Data Collection)
        # -------------------------------------------------------
        # 只要 run_dcf_valuation 返回的不是 None (即模型收敛，未发生 WACC < g 的崩溃)，
        # 就将其纳入统计样本。
        if price is not None:
            simulation_results.append(price)
        
    return simulation_results

def adjust_wacc_floor(current_wacc, floor=0.075):
    """
    WACC 安全阀与熔断机制 (Valuation Anchor)。
    
    数理逻辑:
    DCF 模型对分母 (WACC) 极度敏感。
    当 WACC 接近永续增长率 (g) 时，分母趋近于 0，导致估值趋近于无穷大 (Singularity)。
    
    业务背景:
    对于像 MDLZ、KO 这样 Beta 极低 (<0.6) 的大盘蓝筹股，纯 CAPM 算出来的 WACC 可能只有 5%~6%。
    但这在现实中是不合理的——投资者通常要求至少 7%~8% 的长期回报率 (Hurdle Rate) 才会投资股票。
    
    逻辑:
    如果 Calculated WACC < Floor (默认 7.5%)，则强制使用 Floor 值。
    
    参数:
    - current_wacc: 模型计算出的原始 WACC
    - floor: 最低资本成本下限 (默认 0.075 即 7.5%)
    
    返回:
    - adjusted_wacc: 修正后的 WACC

    """
    # 检查是否触发生存底线
    if current_wacc < floor:
        print(f" [修正] WACC ({current_wacc:.2%}) 甚至低于国债，强制修正为 {floor:.2%}")
        return floor
    
    return current_wacc

def calculate_historical_cagrs(income_df, fcf_series, years=5):
    """
    财务三维增长性校验 (Growth & Quality Check)
    
    数理逻辑:
    计算营收 (Top-line)、净利润 (Bottom-line) 与 自由现金流 (FCF) 的历史复合增长率 (CAGR)。
    旨在通过对比三者的增速差异 (Divergence)，识别潜在的财务粉饰或经营风险。
    
    核心逻辑 (Financial Forensics):
    1. 趋势一致性 (Trend Consistency): 长期来看，FCF 增速应与 Net Income 增速趋同。
    2. 异常预警 (Red Flags):
    - 场景 A (FCF >> Net Income): 可能源于折旧摊销过大(假摔)或资本开支剧减(吃老本)。
    - 场景 B (FCF << Net Income): 可能源于激进的收入确认(应收账款堆积)或存货积压。
    
    参数:
    - income_df: 包含 'Revenue', 'Net_Income' 的清洗后 DataFrame
    - fcf_series: 之前计算好的 valid_fcf 序列
    - years: 回测窗口 (默认 5 年)
    
    返回:
    - growth_metrics: 包含各指标 CAGR 的字典 (用于后续修正预测期的 g)
    
    """
   
    print(f"\n 正在计算过去 {years} 年的【三大核心增长率】参考...")
    results = {}
    
    try:
        # 1. 锁定时间窗口 (Time Window)
        # income_df 应该是已经按时间正序排列好的
        recent_income = income_df.iloc[:, -years:]
        
        # ==========================================
        # A. 营收 CAGR (Top-line Growth)
        # ==========================================
        rev_names = ['Total Revenue', 'Total revenue', 'Operating Revenue']

        revenue = None
        for name in rev_names:
            if name in recent_income.index:
                revenue = recent_income.loc[name]
                break
        
        if revenue is not None and len(revenue) > 1:
            start_value = revenue.iloc[0]
            end_value = revenue.iloc[-1]    

            if start_value > 0 and end_value > 0:
                # CAGR 公式: (End / Start) ^ (1 / n) - 1
                # 间隔年数 = 数据点数 - 1
                interval_year = len(revenue) - 1
                revnue_cagr = (end_value / start_value) ** (1 / interval_year) - 1
                results['Revenue'] = revnue_cagr
                print(f"  🔹 营收 Total Revenue  CAGR: {revnue_cagr:.2%}")


        # ==========================================
        # B. 净利润 CAGR (Bottom-line Growth)
        # ==========================================
        net_income_names = ['Net Income', 'Net income', 'Net Income Common Stockholders']

        net_income = None
        for name in net_income_names:
            if name in recent_income.index:
                net_income = recent_income.loc[name]
                break
                
        if net_income is not None and len(net_income) > 1:
            start_value = net_income.iloc[0]
            end_value = net_income.iloc[-1]

            if start_value > 0 and end_value > 0: # 只有正数才能算 CAGR
                interval_year = len(net_income) - 1
                net_income_cagr = (end_value / start_value) ** (1 / interval_year) - 1
                results['Net Income'] = net_income_cagr
                print(f"  🔹 净利润 Net Income  CAGR: {net_income_cagr:.2%}")


        # ==========================================
        # C. 自由现金流 CAGR (Cash Flow Growth)
        # ==========================================
        if fcf_series is not None and len(fcf_series) > 1:
            recent_fcf = fcf_series.tail(years)

            start_fcf = recent_fcf.iloc[0]
            end_fcf = recent_fcf.iloc[-1]
            
            if start_fcf > 0 and end_fcf > 0:
                interval_year = len(recent_fcf) - 1
                if interval_year > 0:
                    fcf_cagr = (end_fcf / start_fcf) ** (1 / interval_year) - 1
                    results['FCF'] = fcf_cagr
                    print(f"  🔹 自由现金流 FCF  CAGR: {fcf_cagr:.2%}")
            else:
                print("  🔹[错误] FCF 起点或终点为负，检查是否处于高强度的 CAPEX (资本开支) 扩张期或经营恶化，跳过 FCF CAGR 计算")

        # ==========================================
        # D. 智能财务洞察 (Smart Insight / Red Flag)
        # ==========================================
        # 逻辑：检测 FCF 与 Net Income 的背离程度 (Divergence)
        if 'Net Income' in results and 'FCF' in results:
            net_income_cagr = results['Net Income']
            fcf_cagr = results['FCF']
            
            # 阈值判定: 当 FCF 增速比净利润高出 3 个百分点以上
            if fcf_cagr > (net_income_cagr + 0.03): 
                print("\n" + "="*60)
                print(f"  [警告] FCF 增速 ({fcf_cagr:.1%}) 远超 净利润 ({net_income_cagr:.1%})")
                print("  ----------------------------------------------------------")
                print("  这种“倒挂”通常暗示财务质量极高，但也可能是出问题，原因如下：")
                print("  1. 折旧'假摔' (Non-cash): 老设备折旧压低了利润，但FCF加回了这笔钱。")
                print("  2. 资本收缩 (Low CapEx): 公司可能在“吃老本”，削减了投资，导致 FCF 暴涨。")
                print("  3. 运营压榨: 催账更狠了，或者拖欠供应商货款更久了。")
                print("  建议: 这种高增速很难长期维持，预测未来 Growth 时请保守取值。")
                print("="*60 + "\n")

        return results

    except Exception as error:
        print(f" [警告] 计算CAGR参考指标失败: {error}")
        return {}

if __name__ == "__main__":
    """
    ================================================================================
    主程序入口 (Main Entry Point)
    
    流程概览:
    1. INIT   : 设置股票代码，抓取原始数据。
    2. CLEAN  : 清洗报表，计算自由现金流 (FCF)。
    3. GROWTH : 智能设定增长率 (对比历史营收 CAGR)。
    4. WACC   : 计算加权平均资本成本 (Beta, ERP, Cost of Debt)。
    5. SIM    : 执行 1000 次蒙特卡洛模拟。
    6. REPORT : 输出估值区间与投资建议。
    ================================================================================
    """

    """
    ================================================================================
    STEP 1: 初始化与数据抓取 (Initialization & Data Fetching)
    
    逻辑:
    - 设定参数: 目标股票 (TICKER) 与 市场基准 (MARKET, 通常为标普500)。
    - 数据源: 连接 Yahoo Finance API (yfinance) 获取对象。
    - 抓取内容: 原始利润表 (Income), 现金流量表 (Cash Flow), 资产负债表 (Balance)。
    - 熔断机制: 如果核心财务数据为空 (None/Empty)，直接终止程序防止报错。
    ================================================================================
    """
    print("\n [步骤 1/6] 初始化与数据抓取 (Initialization & Fetching)...")
    
    TICKER = "MDLZ"  
    MARKET = "^GSPC" 
    print(f"正在启动 {TICKER} 的 DCF 蒙特卡洛模型 ...")
    
    # 获取 YF 对象
    ticker, market = fetch_tickers(TICKER, MARKET)
    raw_income, raw_cash, raw_balance = fetch_raw_financials(ticker)
    
    if raw_income is None or raw_income.empty:
        print(" [警告] 无法获取财务数据，程序终止。")
        exit()

    """
    ================================================================================
    STEP 2: 数据清洗与核心指标计算 (Data Cleaning & FCF Calculation)
    
    逻辑:
    - 填充缺失值 (NaN -> 0)。
    - 统一时间轴方向。
    - 计算核心指标: EBIT, Tax Rate, Depreciation, Capex, Change in WC.
    - 最终得到: Valid FCF (Free Cash Flow) 序列。
    ================================================================================
    """
    print("\n [步骤 2/6] 数据清洗与核心指标计算 (Data Cleaning & FCF Calculation)...")

    clean_income = clean_financial_df(raw_income, fill_method='ffill', axis=1)
    clean_balance = clean_financial_df(raw_balance, fill_method='ffill', axis=1)
    clean_cashflow = clean_financial_df(raw_cash, fill_method='ffill', axis=1)

    financials = get_fcf_and_details(clean_income, clean_balance, clean_cashflow)
    
    if financials['valid_fcf'].empty:
        print(" [警告] 无法计算 FCF (数据为空)，程序终止")
        sys.exit()

    """
    ================================================================================
    STEP 3: 增长率智能设定 (Smart Growth Assumption)
    
    核心风控逻辑:
    1. 计算历史营收 (Revenue) 和 现金流 (FCF) 的复合增长率 (CAGR)。
    2. 设定一个预期的手动增长率 (manual_growth_input)。
    3. 【安全阀】: 如果手动设定的增长率 > 历史营收 CAGR，程序将发出警告，仅提示。
       建议将增长率下调至历史营收水平。这防止了对一家慢速增长公司给予过高预期。
    ================================================================================
    """
    print("\n [步骤 3/6] 增长率智能设定 (Smart Growth Assumption)...")

    # 1. 计算历史数据作为锚点
    growth_metrics = calculate_historical_cagrs(
        financials['income_stmt'], 
        financials['valid_fcf'], 
        years=5
    )
    revune_cagr = growth_metrics.get('Revenue') 
    fcf_cagr = growth_metrics.get('FCF')
    
    # 2. 设定你的心理预期 ⭐(此处手动修改)
    # MDLZ保守预设为0.05，TRI保守预设0.065
    manual_growth_input = 0.05
    
    # 3. 自动风控校验
    if manual_growth_input > revune_cagr + 0.02:
        print(f"[警告] 设定的增长率 ({manual_growth_input:.1%}) 远高于历史营收增速 Total Revnue cagr({revune_cagr:.1%}), 检查")
      
    base_growth = manual_growth_input 
    print(f" 🔹 模型使用的未来 5 年 增长率: {base_growth:.2%}")

    # 获取最近一年 FCF 用于起跑
    current_fcf_value = financials['valid_fcf'].iloc[-1]
    print(f" 🔹 最近一年 FCF (起跑线): ${current_fcf_value/1e9:.2f} B")

    """
    ================================================================================
    STEP 4: 风险折现率模型 (Risk & WACC Model)
    
    核心逻辑:
    1. Beta: 获取 5年 Beta，并尝试与 2年 Beta 进行长短期风险对比。
    2. Cost of Equity (Ke): 使用 CAPM 模型，并进行去杠杆/加杠杆 (Unlever/Relever) 调整。
    3. Cost of Debt (Kd): 基于利息支出和总债务计算。
    4. WACC: 加权平均。
    5. 【安全阀】: 设定 WACC 地板价 (7.5%)，防止低利率环境下的估值泡沫。
    ================================================================================
    """
    print("\n [步骤 4/6] 计算风险指标 (Beta & WACC)...")
    
    risk_metrics = compute_risk_metrics(ticker, market)
    
    # 1. Beta 验证
    raw_beta = risk_metrics.get('beta')
    if 'beta_2y' in risk_metrics:
        print(f" 🔹 Beta 验证: 官方 5y ({raw_beta:.2f}) vs 短期 2y ({risk_metrics['beta_2y']:.2f})")
    else:
        print(f"   Beta (Yahoo 5y): {raw_beta:.2f}")

    # 2. ERP 与 Rf
    rf_rate = get_risk_free_rate(duration='10y', default_rf_rate=0.04)
    historical_erp = get_real_historical_erp(current_rf_rate=rf_rate, years=20)
    
    FINAL_ERP = historical_erp  # ⭐可以在此处手动覆盖

    market_premium = guess_equity_risk_premium(rf_rate, manual_input=FINAL_ERP)
    print(f" 🔹 Rf: {rf_rate:.2%} | ERP: {market_premium:.2%}")
    
    # 3. 债务与市值数据
    kd, interest_exp, total_debt = get_cost_of_debt_kd(
        financials['income_stmt'], 
        financials['balance_sheet'], 
        rf_rate
    )
    equity_value = get_current_equity_value(ticker)
    
    # 4. 有效税率
    recent_tax = financials['tax_rate_series'].tail(3)
    tax_rate = recent_tax.mean() if not recent_tax.empty else 0.21

    if equity_value is None or total_debt is None:
        print("[警告] 无法获取市值或债务数据，无法计算 WACC。")
        exit()

    # 5. 计算最终 Ke 和 WACC
    target_debt_ratio = get_historical_target_debt_ratio(ticker, financials['balance_sheet'], years=5)

    # 这里的 adjust_beta 传入第 5 个参数
    adjusted_beta = adjust_beta(raw_beta, total_debt, equity_value, tax_rate, target_debt_ratio)  # ⭐ target_debt_ratio 可手动修改, 进行导数优化部分，最佳资本结构（Optimal Capital Structure）
    ke = get_capm_ke(adjusted_beta, rf_rate, market_premium)
    wacc_raw = compute_wacc(ke, kd, tax_rate, equity_value, total_debt)
    
    # 强制 WACC 地板价
    wacc_final = adjust_wacc_floor(wacc_raw, floor=0.075)

    """
    ================================================================================
    STEP 5: 蒙特卡洛模拟 (Monte Carlo Simulation)
    
    过程:
    - 准备输入: 股份数 (Shares), 净债务 (Net Debt = Debt - Cash)。
    - 运行模拟: 基于正态分布随机波动 Growth 和 WACC，运行 1000 次 DCF。
    - 生成结果: 得到 1000 个可能的股价预测值。
    ================================================================================
    """
    print("\n [步骤 5/6] 计算风险指标 (Beta & WACC)...")

    shares = ticker.info.get('sharesOutstanding')
    
    # 计算净债务
    last_1y_balance = financials['balance_sheet'].iloc[:, -1]
    cash = 0
    for name in ['Cash And Cash Equivalents', 'Cash Financial']:
        if name in last_1y_balance:
            cash = last_1y_balance[name]
            break
    net_debt = total_debt - cash
    
    # 启动模拟
    print(f" 🔹 确定 Growth={base_growth:.1%}, WACC={wacc_final:.1%}")
    
    prices = run_monte_carlo_simulation(
        current_fcf=current_fcf_value,
        base_wacc=wacc_final,
        growth_rate_5y=base_growth, 
        shares=shares,
        net_debt=net_debt,
        num_simulations=1000
    )

    """
    ================================================================================
    STEP 6: 最终估值报告 (Final Valuation Report)
    
    输出:
    - 悲观估值 (10th Percentile): 只有 10% 的概率会低于这个价。
    - 平均估值 (Mean): 统计学期望值。
    - 乐观估值 (90th Percentile): 只有 10% 的概率会高于这个价。
    - 投资建议: 比较当前股价与估值区间，给出 Deep Value / Undervalued / Overvalued 评级。
    ================================================================================
    """
    avg_price = np.mean(prices)
    min_price = np.percentile(prices, 10)  # 悲观
    max_price = np.percentile(prices, 90)  # 乐观
    
    try:
        current_price = ticker.fast_info['last_price']
    except:
        current_price = ticker.history(period='1d')['Close'].iloc[-1]

    print("\n" + "="*40)
    print(f" {TICKER} 最终估值报告")
    print("="*40)
    print(f"当前股价:   ${current_price:.2f}")
    print("-" * 20)
    print(f" 悲观估值: ${min_price:.2f} (安全边际)")
    print(f" 平均估值: ${avg_price:.2f} (中枢)")
    print(f" 乐观估值: ${max_price:.2f} (天花板)")
    print("="*40)
    
    # 结论判定
    diff = (avg_price - current_price) / current_price
    if current_price < min_price:
        print(" [推荐] 股价低于悲观估值，安全边际高。 (Deep Value)")
    elif current_price < avg_price:
        print(f" [低估] 价格合理，还有 {diff:.1%} 的潜在空间。")
    elif current_price > max_price:
        print(" [高估] 股价高于乐观估值。")
    else:
        print(f"[合理区间] 股价在合理范围内 (相对于中枢偏差 {diff:.1%})。")

        

