import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation will drop timezone information")

import os
import re

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# 设置全局字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

__eps__ = 1e-10

class crypto_currency:
    '''
    crypto_currency class: 用于存储单个加密货币的数据，并计算相关指标，例如每日收益率
    '''
    def __init__(self, code, name, file_path=None):
        self.code = code
        self.name = name
        self.df = pd.read_csv(f"{file_path}/code{code}_{name}.csv")
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        self.df = self.df.sort_values(by='timestamp', ascending=True)

        self.begin_date = self.df['timestamp'].iloc[0]
        self.end_date = self.df['timestamp'].iloc[-1]

    def get_daily_return(self, periods=1):
        # 每日收益率计算
        self.df['daily_return'] = self.df['close'].pct_change(periods=periods)

    def get_weekly_return(self):
        df = self.df.copy()
        df['week'] = df['timestamp'].dt.to_period('W').dt.start_time

        weekly_return = (
            df.groupby('week')['daily_return'].transform(
            lambda x: (1 + x).prod() - 1)
            .reset_index(name='weekly_return')  # 复利计算周收益
        )

        df['weekly_return'] = weekly_return['weekly_return']
        self.df = df

class predictor:
    '''
    predictor class: 用于生成因子，方便后续扩展其他因子
    输入：crypto_currency类实例
    输出：带有有关因子的crypto_currency类实例
    '''
    def __init__(self, crypto_currency):
        self.crypto_currency = crypto_currency
        
    def get_predictor_MCAP(self, predictor_name='MCAP'):
        # MCAP因子计算: 上周的对数市值
        df = self.crypto_currency.df.copy()
        df['week'] = df['timestamp'].dt.to_period('W').dt.start_time
        
        # 计算每周最后一天的市值，并滞后一周
        weekly_last_mcap = (
            df.groupby('week')['marketCap'].last()  
            .shift(1)                               
            .reset_index(name='last_mcap')
        )
        
        df = df.merge(weekly_last_mcap, on='week', how='left')
        
        # 计算MCAP因子
        df[predictor_name] = np.log(df['last_mcap']+__eps__)

        df.drop(columns=['last_mcap'], inplace=True)
        self.crypto_currency.df = df

    def get_predictor_MAXDPRC(self, predictor_name='MAXDPRC'):
        # MAXDPRC因子计算: 上周的对数最高价
        df = self.crypto_currency.df.copy()
        df['week'] = df['timestamp'].dt.to_period('W').dt.start_time
        
        # 计算每周最高价，并滞后一周
        weekly_last_maxdprc = (
            df.groupby('week')['high'].max()  
            .shift(1)                               
            .reset_index(name='last_maxdprc')
        )
        df = df.merge(weekly_last_maxdprc, on='week', how='left')
        
        # 计算MCAP因子
        df[predictor_name] = np.log(df['last_maxdprc']+__eps__)

        df.drop(columns=['last_maxdprc'], inplace=True)
        self.crypto_currency.df = df

    def get_predictor_mom_r_1_0(self, predictor_name='mom_r_1_0'):
        # mom_r_1_0因子计算: 过去1周的收益率
        df = self.crypto_currency.df.copy()

        df['week'] = df['timestamp'].dt.to_period('W').dt.start_time

        weekly_return = (
            df.groupby('week')['daily_return'].apply(
                lambda x: (1 + x).prod() - 1 if len(x) == 7 else None
            ).reset_index(name=predictor_name)
        )

        weekly_return[predictor_name] = weekly_return[predictor_name].shift(1)

        df = df.merge(weekly_return, on='week', how='left')

        self.crypto_currency.df = df

    def get_predictor_mom_r_2_0(self, predictor_name='mom_r_2_0'):
        # mom_r_2_0因子计算: 过去两周的收益率
        df = self.crypto_currency.df.copy()

        df['two_weeks'] = df['timestamp'].dt.to_period('W').astype('datetime64[ns]')
        df['two_weeks'] = df.groupby(pd.Grouper(key='two_weeks', freq='2W'))['two_weeks'].transform('first')

        two_weeks_return = (
            df.groupby('two_weeks')['daily_return'].apply(
                lambda x: (1 + x).prod() - 1 if len(x) == 14 else None
            ).reset_index(name=predictor_name)
        )

        # 滞后两周
        two_weeks_return[predictor_name] = two_weeks_return[predictor_name].shift(1)

        df = df.merge(two_weeks_return, on='two_weeks', how='left')

        self.crypto_currency.df = df

    def get_predictor_PRCVOL(self, predictor_name='PRCVOL'):
        # PRCVOL因子计算: 上周的对数平均价格乘以成交量
        df = self.crypto_currency.df.copy()
        df['week'] = df['timestamp'].dt.to_period('W').dt.start_time

        df['prcvol_daily'] = df['close'] * df['volume']

        weekly_prcvol = (
            df.groupby('week')['prcvol_daily'].mean()
            .shift(1).reset_index(name='last_prcvol_weekly')
        )

        df = df.merge(weekly_prcvol, on='week', how='left')

        df[predictor_name] = np.log(df['last_prcvol_weekly']+__eps__)

        self.crypto_currency.df = df


def day2week(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # 定义各列的聚合规则
    agg_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min', 
        'close': 'last',
        'volume': 'sum',
        'daily_return': 'last',
        'MCAP': 'last',
        'MAXDPRC': 'last',
        'mom_r_1_0': 'last',
        'mom_r_2_0': 'last',
        'PRCVOL': 'last',
        'weekly_return': 'last'
    }

    week_df = df.resample('7d', offset='-3d').agg(agg_rules)

    week_df = week_df.dropna(how='all')

    week_df = week_df.reset_index()
    week_df['timestamp'] = week_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return week_df


class zero_investment_strategy:
    '''
    输入：带有有关因子的crypto_currency类实例存储的list
    输出：零投资多头策略的周收益率
    具体操作：生成两个矩阵，一个存储投资组合权重，行为各加密货币，列为时间；另一个存储每周收益率，行各加密货币，列为时间。
    每个时间点的加密货币的收益，即为两矩阵的哈达玛积
    其中，做多因子值最高的10个加密货币，做空因子值最低的10个加密货币，权重分别为1/20(多)和-1/20(空)
    '''
    def __init__(self, crypto_currency_list):
        for c in crypto_currency_list:
            ## 转为周频数据
            c.df = day2week(df=c.df)
        
        self.cc_class_list = crypto_currency_list
        self.time_list = self.cc_class_list[0].df['timestamp'].unique()
        self.cc_list = [cc.name for cc in self.cc_class_list]

    def get_factor_matrix(self, factor='MCAP'):
        # 生成因子矩阵
        factor_matrix = [[float(c.df[c.df['timestamp']==t][factor].values) for c in self.cc_class_list] for t in self.time_list]
        factor_df = pd.DataFrame(factor_matrix, columns=self.cc_list, index=self.time_list).T
        # print(factor_df.iloc[:, :5])
        return factor_df
        
    def get_weight_matrix(self, factor='MCAP'):
        # 生成投资组合权重矩阵: 由因子矩阵生成
        df = self.get_factor_matrix(factor=factor)
        df.reset_index(inplace=True)

        # 遍历每一列
        for col in df.columns:
            col_series = pd.to_numeric(df[col], errors='coerce')
            
            if col_series.isna().all():
                print(f"列 {col} 全为 NaN，跳过处理")
                continue

            top_10_indices = col_series.nlargest(10).index
            bottom_10_indices = col_series.nsmallest(10).index

            if factor == 'MCAP' or factor == 'MAXDPRC' or factor == 'PRCVOL':
                ## 五个因子中MCAP, MAXDPRC, PRCVOL因子需要做多因子小的，做空因子大的，所以需要调换
                temp_top_10_indices = top_10_indices
                top_10_indices = bottom_10_indices
                bottom_10_indices = temp_top_10_indices

            df.loc[:, col] = 0.0

            # 做多最大的10个
            df.loc[top_10_indices, col] = 0.05

            # 做空最小的10个
            df.loc[bottom_10_indices, col] = -0.05

        df.set_index('index', inplace=True)
        self.weight_df = df     

    def get_return_matrix(self):
        # 生成收益率矩阵
        return_matrix = [[float(c.df[c.df['timestamp']==t]['weekly_return'].values) for c in self.cc_class_list] for t in self.time_list]
        return_df = pd.DataFrame(return_matrix, columns=self.cc_list, index=self.time_list).T
        # print(return_df)
        self.return_df = return_df

    def zero_investment_stragety(self, factor='MCAP'):
        # 单因子零投资多空策略，计算其收益率
        self.get_weight_matrix(factor=factor)
        self.get_return_matrix()

        # 哈达玛积计算收益率
        self.strategy_return_df = self.weight_df * self.return_df
        self.strategy_time_return = self.strategy_return_df.sum(axis=0)


def get_eval_result(df, factor='factor'):
    df['cumulative_return'] = (1 + df['weekly_return']).cumprod()

    # 计算年化收益率
    annualized_return = (df['cumulative_return'].iloc[-1]) ** (52 / len(df)) - 1

    # 计算夏普比率（假设无风险利率为0）
    sharpe_ratio = np.sqrt(52) * (df['weekly_return'].mean() / df['weekly_return'].std())

    # 计算最大回撤
    df['cumulative_max'] = df['cumulative_return'].cummax()
    df['drawdown'] = df['cumulative_return'] / df['cumulative_max'] - 1
    max_drawdown = df['drawdown'].min()

    # 计算卡玛比率
    calmar_ratio = annualized_return / abs(max_drawdown+__eps__)

    print(f"年化收益率: {annualized_return:.4f}")
    print(f"夏普比率: {sharpe_ratio:.4f}")
    print(f"卡玛比率: {calmar_ratio:.4f}")

    # 绘制收益率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(df['cumulative_return'], label='累计收益率')
    plt.title('收益率曲线')
    plt.xlabel('周')
    plt.ylabel('累计收益率')
    plt.legend()
    plt.grid()
    plt.title(f"{factor}多空策略收益率曲线")

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(60))  

    # plt.show()
    folder_path = "./figures/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(f"{folder_path}{factor}.png")



if __name__ == "__main__":
    ### zero_investment_strategy test

    folder_path = "./data/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    cc_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    def cc_preprocess(cc):
        # 正则匹配
        pattern = r"^code(\d+)_(.+)\.csv$"
        match = re.match(pattern, cc)
        # 实例化
        number = match.group(1)
        name = match.group(2)
        test = crypto_currency(number, name, file_path=folder_path)
        test.get_daily_return()
        test.get_weekly_return()
        # 计算predictor
        test_predictor = predictor(test)
        test_predictor.get_predictor_MCAP()
        test_predictor.get_predictor_MAXDPRC()
        test_predictor.get_predictor_mom_r_1_0()
        test_predictor.get_predictor_mom_r_2_0()
        test_predictor.get_predictor_PRCVOL()

        return test_predictor.crypto_currency

    test_list = []
    for cc in tqdm(cc_list, desc='加密货币计算因子中...'):
        test_list.append(cc_preprocess(cc))
    
    # fetch_num = 50
    # test_list = test_list[:fetch_num]  # 只取50个
    # print(f"纳入分析的加密货币有{len(test_list)}个, \n 具体为{cc_list[:fetch_num]}")
    
    test_zis = zero_investment_strategy(test_list)
    test_zis.zero_investment_stragety(factor='MCAP')
    print(f"MCAP策略的周平均收益率为{test_zis.strategy_time_return.mean()}")
    get_eval_result(pd.DataFrame(test_zis.strategy_time_return, columns=['weekly_return']), factor='MCAP')

    test_zis.zero_investment_stragety(factor='MAXDPRC')
    print(f"MAXDPRC策略的周平均收益率为{test_zis.strategy_time_return.mean()}")
    get_eval_result(pd.DataFrame(test_zis.strategy_time_return, columns=['weekly_return']), factor='MAXDPRC')

    test_zis.zero_investment_stragety(factor='mom_r_1_0')
    print(f"mom_r_1_0策略的周平均收益率为{test_zis.strategy_time_return.mean()}")
    get_eval_result(pd.DataFrame(test_zis.strategy_time_return, columns=['weekly_return']), factor='mom_r_1_0')

    test_zis.zero_investment_stragety(factor='mom_r_2_0')
    print(f"mom_r_2_0策略的周平均收益率为{test_zis.strategy_time_return.mean()}")
    get_eval_result(pd.DataFrame(test_zis.strategy_time_return, columns=['weekly_return']), factor='mom_r_2_0')

    test_zis.zero_investment_stragety(factor='PRCVOL')
    print(f"PRCVOL策略的周平均收益率为{test_zis.strategy_time_return.mean()}")
    get_eval_result(pd.DataFrame(test_zis.strategy_time_return, columns=['weekly_return']), factor='PRCVOL')
    


