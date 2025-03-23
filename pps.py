import sys
sys.path.append('./')

from zis_strategy import *

__eps__ = 1e-10


def rolling_window(matrix_list, window_size=120):
    average_list = []

    for t in range(len(matrix_list)):
        if t < window_size:
            avg_matrix = np.zeros_like(matrix_list[0])
        else:
            matrix = np.array(matrix_list[t-window_size:t])
            avg_matrix = np.mean(matrix, axis=0)
        
        average_list.append(avg_matrix)

    return average_list



class pps():
    '''
    PPS策略: 实现pps优化策略和alpha，beta分解
    '''
    def __init__(self, crypto_currency_list):
        ## 初始化不进行数据转换，保留天数据
        self.cc_class_list = crypto_currency_list
        self.time_list = self.cc_class_list[0].df['timestamp'].unique()
        self.cc_list = [cc.name for cc in self.cc_class_list]

    def get_factor_matrix(self, factor='MCAP'):
        # 生成因子矩阵
        factor_matrix = [[float(c.df[c.df['timestamp']==t][factor].values) for c in self.cc_class_list] for t in self.time_list]
        factor_df = pd.DataFrame(factor_matrix, columns=self.cc_list, index=self.time_list).T
        self.factor_df = factor_df
        # return factor_df
    

    def get_return_matrix(self):
        # 生成收益率矩阵
        return_matrix = [[float(c.df[c.df['timestamp']==t]['daily_return'].values) for c in self.cc_class_list] for t in self.time_list]
        return_df = pd.DataFrame(return_matrix, columns=self.cc_list, index=self.time_list).T
        self.return_df = return_df

    def get_predictor_matrix(self, factor='MCAP'):
        # 预测矩阵由收益向量与因子向量相乘(ps: 收益矩阵计算时已经滞后一期，所以不需要再滞后)
        # 原始预测矩阵
        print('开始计算return matrix...')
        self.get_return_matrix()
        return_df = self.return_df.copy()
        print('开始计算factor matrix...')
        self.get_factor_matrix(factor)
        factor_df = self.factor_df.copy()

        # 用一个list存储每一天的预测矩阵
        print('开始计算rolling pred return matrix...')
        rolling_pred_return_matrix = []
        for i in range(len(return_df.columns)):
            # 每一天的预测矩阵
            R = return_df.iloc[:, i]
            S = factor_df.iloc[:, i]
            pred_return_matrix = np.outer(R, S)
            if np.all(np.isnan(pred_return_matrix)):
                # 如果全为 NaN，则赋值为 0
                pred_return_matrix = np.zeros_like(pred_return_matrix)

            rolling_pred_return_matrix.append(pred_return_matrix)
        
        ## 接下来需要滑窗120天，作为每一天的估计预测矩阵，即每一天的预测矩阵是由前120天的数据得到的

        ## 滑窗120天
        avg_rolling_list = rolling_window(rolling_pred_return_matrix, window_size=120)

        self.avg_rolling_list = avg_rolling_list
        return avg_rolling_list

    def get_svd_weight(self, st, matrix, k=3):
        
        U, Sigma, VT = np.linalg.svd(matrix) 
        weight = np.zeros_like(st)
        
        for i in range(k): ## 取前k个组合的和
            v_k = VT[:, i]
            u_k = U[i, :]
            if np.count_nonzero(matrix) == 0:
                ## 如果矩阵全为0，则返回全0
                weight_k = np.zeros_like(st.T@np.outer(v_k, u_k))
            else:
                weight_k = st.T@np.outer(v_k, u_k)
            
            weight = weight + weight_k

        if np.count_nonzero(weight) != 0:
            # 对权重缩放到和零投资策略的权重之和
            positive_numbers = [x for x in weight if x > 0]
            negative_numbers = [x for x in weight if x < 0]
            sum_positive = sum(positive_numbers)
            sum_negative = sum(negative_numbers)

            weight = [0.5*x/sum_positive if x>0 else 0.5*x/sum_negative for x in weight]

        return weight
    

    def get_weight_matrix(self, factor='MCAP'):
        rolling_pred_return_df = self.get_predictor_matrix(factor=factor)
        weight_matrix = []
        for st in range(len(self.factor_df.columns)):
            s = self.factor_df.iloc[:, st].values
            matrix = rolling_pred_return_df[st]
            weight_k = self.get_svd_weight(s, matrix, k=5)
            weight_matrix.append(weight_k)
        
        self.weight_matrix = pd.DataFrame(weight_matrix, columns=self.cc_list, index=self.time_list).T
        
    def zero_investment_stragety(self, factor='MCAP'):
        # 单因子零投资多空策略，计算其收益率
        self.get_weight_matrix(factor=factor)
        self.get_return_matrix()

        ## 周策略，要将每一周的权重设置为该周第一天
        
        df = self.weight_matrix.T.copy()
        df.index = pd.to_datetime(df.index)
        weekly_first = df.resample('W-MON').first()

        daily_filled = weekly_first.resample('D').ffill().reindex(df.index).ffill()
        result = daily_filled.T

        # 哈达玛积计算收益率
        self.strategy_return_df = result * self.return_df
        self.strategy_time_return = self.strategy_return_df.sum(axis=0)


    def get_weight_matrix_alpha(self, factor='MCAP'):
        rolling_pred_return_df = self.get_predictor_matrix(factor=factor)
        weight_matrix = []
        for st in range(len(self.factor_df.columns)):
            s = self.factor_df.iloc[:, st].values
            matrix = rolling_pred_return_df[st]
            matrix = 0.5*(matrix - matrix.T)
            weight_k = self.get_svd_weight(s, matrix, k=5)
            weight_matrix.append(weight_k)
        
        self.weight_matrix_alpha = pd.DataFrame(weight_matrix, columns=self.cc_list, index=self.time_list).T
    

    def alpha_investment(self, factor='MCAP'):
        # alpha分解
        self.get_return_matrix()
        self.get_weight_matrix_alpha(factor=factor)

        df = self.weight_matrix_alpha.T.copy()
        df.index = pd.to_datetime(df.index)
        weekly_first = df.resample('W-MON').first()

        daily_filled = weekly_first.resample('D').ffill().reindex(df.index).ffill()
        result = daily_filled.T

        # 哈达玛积计算收益率
        self.strategy_return_df_alpha = result * self.return_df
        self.strategy_time_return_alpha = self.strategy_return_df_alpha.sum(axis=0)


    def get_weight_matrix_beta(self, factor='MCAP'):
        rolling_pred_return_df = self.get_predictor_matrix(factor=factor)
        weight_matrix = []
        for st in range(len(self.factor_df.columns)):
            s = self.factor_df.iloc[:, st].values
            matrix = rolling_pred_return_df[st]
            matrix = 0.5*(matrix + matrix.T)
            weight_k = self.get_svd_weight(s, matrix, k=5)
            weight_matrix.append(weight_k)
        
        self.weight_matrix_beta = pd.DataFrame(weight_matrix, columns=self.cc_list, index=self.time_list).T
    
    def beta_investment(self, factor='MCAP'):
        # beta分解
        self.get_return_matrix()
        self.get_weight_matrix_beta(factor=factor)

        df = self.weight_matrix_beta.T.copy()
        df.index = pd.to_datetime(df.index)
        weekly_first = df.resample('W-MON').first()

        daily_filled = weekly_first.resample('D').ffill().reindex(df.index).ffill()
        result = daily_filled.T

        # 哈达玛积计算收益率
        self.strategy_return_df_beta = result * self.return_df
        self.strategy_time_return_beta = self.strategy_return_df_beta.sum(axis=0)

def get_eval_result_daily(df, factor='factor'):
    df['cumulative_return'] = (1 + df['daily_return']).cumprod()

    # 计算年化收益率
    annualized_return = (df['cumulative_return'].iloc[-1]) ** (365 / len(df)) - 1

    # 计算夏普比率（假设无风险利率为0）
    sharpe_ratio = np.sqrt(365) * (df['daily_return'].mean() / (df['daily_return'].std()+__eps__))

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
    plt.xlabel('天')
    plt.ylabel('累计收益率')
    plt.legend()
    plt.grid()
    plt.title(f"{factor}多空策略收益率曲线")

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(350))  

    # plt.show()
    folder_path = "./figures/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(f"{folder_path}{factor}.png")



if __name__ == '__main__':
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
    for cc in tqdm(cc_list[:], desc='加密货币计算因子中...'):
        test_list.append(cc_preprocess(cc))


    test_zis = pps(test_list)
    test_zis.zero_investment_stragety(factor='MCAP')
    print(f"MCAP策略的天平均收益率为{test_zis.strategy_time_return.mean()}")
    # factor指的是用于画图的title命名
    get_eval_result_daily(pd.DataFrame(test_zis.strategy_time_return, columns=['daily_return']), factor='MCAP_pps')

    test_zis.alpha_investment(factor='MCAP')
    print(f"MCAP策略-alpha的天平均收益率为{test_zis.strategy_time_return_alpha.mean()}")
    get_eval_result_daily(pd.DataFrame(test_zis.strategy_time_return_alpha, columns=['daily_return']), factor='MCAP_pps_alpha')

    test_zis.beta_investment(factor='MCAP')
    print(f"MCAP策略-beta的天平均收益率为{test_zis.strategy_time_return_beta.mean()}")
    get_eval_result_daily(pd.DataFrame(test_zis.strategy_time_return_beta, columns=['daily_return']), factor='MCAP_pps_beta')


    test_zis.zero_investment_stragety(factor='MAXDPRC')
    print(f"MAXDPRC策略的天平均收益率为{test_zis.strategy_time_return.mean()}")
    get_eval_result_daily(pd.DataFrame(test_zis.strategy_time_return, columns=['daily_return']), factor='MAXDPRC_pps')

    test_zis.alpha_investment(factor='MAXDPRC')
    print(f"MAXDPRC策略-alpha的天平均收益率为{test_zis.strategy_time_return_alpha.mean()}")
    get_eval_result_daily(pd.DataFrame(test_zis.strategy_time_return_alpha, columns=['daily_return']), factor='MAXDPRC_pps_alpha')

    test_zis.beta_investment(factor='MAXDPRC')
    print(f"MAXDPRC策略-beta的天平均收益率为{test_zis.strategy_time_return_beta.mean()}")
    get_eval_result_daily(pd.DataFrame(test_zis.strategy_time_return_beta, columns=['daily_return']), factor='MAXDPRC_pps_beta')

    test_zis.zero_investment_stragety(factor='mom_r_1_0')
    print(f"mom_r_1_0策略的天平均收益率为{test_zis.strategy_time_return.mean()}")
    get_eval_result_daily(pd.DataFrame(test_zis.strategy_time_return, columns=['daily_return']), factor='mom_r_1_0_pps')

    test_zis.alpha_investment(factor='mom_r_1_0')
    print(f"mom_r_1_0策略-alpha的天平均收益率为{test_zis.strategy_time_return_alpha.mean()}")
    get_eval_result_daily(pd.DataFrame(test_zis.strategy_time_return_alpha, columns=['daily_return']), factor='mom_r_1_0_pps_alpha')

    test_zis.beta_investment(factor='mom_r_1_0')
    print(f"mom_r_1_0策略-beta的天平均收益率为{test_zis.strategy_time_return_beta.mean()}")
    get_eval_result_daily(pd.DataFrame(test_zis.strategy_time_return_beta, columns=['daily_return']), factor='mom_r_1_0_pps_beta')

    test_zis.zero_investment_stragety(factor='mom_r_2_0')
    print(f"mom_r_2_0策略的天平均收益率为{test_zis.strategy_time_return.mean()}")
    get_eval_result_daily(pd.DataFrame(test_zis.strategy_time_return, columns=['daily_return']), factor='mom_r_2_0_pps')

    test_zis.alpha_investment(factor='mom_r_2_0')
    print(f"mom_r_2_0策略-alpha的天平均收益率为{test_zis.strategy_time_return_alpha.mean()}")
    get_eval_result_daily(pd.DataFrame(test_zis.strategy_time_return_alpha, columns=['daily_return']), factor='mom_r_2_0_pps_alpha')

    test_zis.beta_investment(factor='mom_r_2_0')
    print(f"mom_r_2_0策略-beta的天平均收益率为{test_zis.strategy_time_return_beta.mean()}")
    get_eval_result_daily(pd.DataFrame(test_zis.strategy_time_return_beta, columns=['daily_return']), factor='mom_r_2_0_pps_beta')

    test_zis.zero_investment_stragety(factor='PRCVOL')
    print(f"PRCVOL策略的天平均收益率为{test_zis.strategy_time_return.mean()}")
    get_eval_result_daily(pd.DataFrame(test_zis.strategy_time_return, columns=['daily_return']), factor='PRCVOL_pps')

    test_zis.alpha_investment(factor='PRCVOL')
    print(f"PRCVOL策略-alpha的天平均收益率为{test_zis.strategy_time_return_alpha.mean()}")
    get_eval_result_daily(pd.DataFrame(test_zis.strategy_time_return_alpha, columns=['daily_return']), factor='PRCVOL_pps_alpha')

    test_zis.beta_investment(factor='PRCVOL')
    print(f"PRCVOL策略-beta的天平均收益率为{test_zis.strategy_time_return_beta.mean()}")
    get_eval_result_daily(pd.DataFrame(test_zis.strategy_time_return_beta, columns=['daily_return']), factor='PRCVOL_pps_beta')


