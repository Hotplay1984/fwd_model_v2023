import os
import numpy as np

data_path = r'/Users/weiyaosun/Work/IFRS9/model_2023/data'

macro_file_name = os.path.join(data_path, '宏观指标/前瞻性模型20231115_宏观指标.xlsx')
npl_file_name = os.path.join(data_path, '系统重要性银行不良贷款率/系统重要性银行不良贷款数据20231115.xlsx')
ridge_variable_config_file = os.path.join(data_path, 'ridge_variable_config.xlsx')
npl_history_file = os.path.join(data_path, 'ind_npl_history.xlsx')
ratings_mapping_file = os.path.join(data_path, 'ratings_mapping.xlsx')
arima_param_config_file = os.path.join(data_path, 'ARIMA_PARAMS_20231231_20231116103539.xlsx')

config_file = os.path.join(data_path, '指标定义.xlsx')
last_dataset = '2023H1'
last_npl_folder = os.path.join(os.path.join(os.path.join(data_path, '0往期数据'), last_dataset),
							'上市银行不良贷款余额-分行业')
last_loan_folder = os.path.join(os.path.join(os.path.join(data_path, '0往期数据'), last_dataset),
								'上市银行贷款余额-分行业')

public_banks_loan_folder = os.path.join(data_path, '上市银行贷款余额-分行业')
public_banks_np_loan_folder = os.path.join(data_path, '上市银行不良贷款余额-分行业')

# coef_constrains = {'gdp': '<=0', 'expn': '<=0', 'earn': '<=0', 'pmi': '<=0', 'gov': '<=0', 'net_expo': '<=0',
# 				'inv_eff': '<=0', 'm2_eff': '<=0', 'loan_eff': '<=0', 'ints': '>=0'}

coef_constrains = {'gdp': '<=0', 'expn': '<=0', 'earn': '<=0', 'pmi': '<=0', 'gov': '<=0', 'net_expo': '<=0',
				'ints': '>=0'}

available_models = ['ridge', 'lasso', 'svm', 'forest', 'sgd']

min_npl, max_npl = 0.0003, 0.03/0.45

rating_bins = 10

distributions = ['skewnorm', 'lognorm', 'poisson']
para_step_dict = {'poisson': 0.001, 'skewnorm': 0.001, 'lognorm': 0.00001}
distribution_parameter_range = {'poisson': np.arange(1, 10, para_step_dict['poisson']),
								'skewnorm': np.arange(0.1, 30, para_step_dict['skewnorm']),
								'lognorm': np.arange(-1, 1, para_step_dict['lognorm'])}
deg_step = 0.0001
deg_range = np.arange(0.1, 5, deg_step)

arima_params_grid = {'ar_range': np.arange(1, 2), 'diff_range': np.arange(0, 2), 'ma_range': np.arange(0, 2),
			'enforce_stationarity': [True, False], 'enforce_invertibility': [True, False],
			'trend': ['n', 'c', 't', 'ct']}
