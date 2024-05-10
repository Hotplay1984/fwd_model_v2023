from . import dm_ts_funcs as ts_funcs
from scipy.stats import zscore
from collections import OrderedDict
import datetime as dt
import pandas as pd
import numpy as np


class dmTs:
	def __init__(self, bng_date, end_date=None):
		self.bng_date = bng_date
		if end_date is not None:
			self.end_date = end_date
		else:
			self.end_date = dt.datetime.now().strftime('%Y%m%d')
		self.variable_config = ts_funcs.read_config()
		self.df_raw_data = ts_funcs.read_raw_data(self.bng_date, self.end_date)
		self.exogs = []
		self.process_funcs = {
			'logit': ts_funcs.get_logit, 'mom': ts_funcs.chain_to_yoy, 'amt': ts_funcs.amt_to_yoy,
			'cmt': ts_funcs.cmt_to_yoy, 'acu': ts_funcs.acu_to_yoy, 'exp': ts_funcs.exp_to_yoy,
		}
		self.df_cycle, self.df_trend = pd.DataFrame(), pd.DataFrame()
		self.df_cycle_z = pd.DataFrame()
		self.df_lag = pd.DataFrame()
		self.df_data = pd.DataFrame()
		return

	def transform_to_cycle(self):
		cycle_dict, trend_dict = OrderedDict(), OrderedDict()
		for var_name_cn in self.variable_config.index:
			include = self.variable_config.loc[var_name_cn, 'include']
			if include == 'N':
				continue
			var_name = self.variable_config.loc[var_name_cn, 'var_name']
			if var_name != 'npl':
				self.exogs.append(var_name)
			freq = self.variable_config.loc[var_name_cn, 'freq']
			var_type = self.variable_config.loc[var_name_cn, 'var_type']
			seasonality = self.variable_config.loc[var_name_cn, 'seasonality']
			s = self.df_raw_data[var_name_cn].dropna().resample(freq).interpolate()
			if seasonality == 'Y':
				sd = ts_funcs.STL(s).fit()
				s = sd.trend
			cycle_s, trend = self.process_funcs[var_type](s, freq)
			cycle_dict[var_name] = cycle_s
			trend_dict[var_name] = trend
		df_cycle = pd.DataFrame(cycle_dict)
		df_trend = pd.DataFrame(trend_dict)

		df_cycle['net_expo'] = df_cycle['expo'] - df_cycle['impo']
		df_cycle['inv_eff'] = df_cycle['gdp'] / df_cycle['inv']
		df_cycle['m2_eff'] = df_cycle['gdp'] / df_cycle['m2']
		df_cycle['loan_eff'] = df_cycle['gdp'] / df_cycle['loan']
		df_cycle['cpi-ppi'] = df_cycle['cpi'] - df_cycle['ppi']

		for exog in ['inv', 'm2', 'cpi', 'ppi', 'expo', 'impo']:
			del df_cycle[exog]
		self.df_trend = df_trend
		self.df_cycle = df_cycle
		self.df_cycle_z = df_cycle.dropna().apply(zscore)
		return

	def find_exog_lag(self):
		# df_z = self.df_cycle_z
		df_cycle = self.df_cycle
		exogs = [exog for exog in self.exogs if exog in df_cycle.columns]
		periods = np.arange(1, 8)
		df_corr = pd.DataFrame()
		for exog in exogs:
			corr_series = [df_cycle['npl'].corr(df_cycle[exog].shift(p)) for p in periods]
			df_corr[exog] = corr_series
		df_corr.index = [-p for p in periods]

		df_lag = pd.DataFrame([[exog, df_corr[exog][df_corr[exog].abs() == df_corr[exog].abs().max()].index[0]]
							for exog in exogs], columns=['exog', 'lag']).set_index('exog')
		self.df_lag = df_lag
		return

	def process_input_data(self):
		self.transform_to_cycle()
		self.find_exog_lag()

		df_cycle = self.df_cycle.dropna()
		df_data = pd.DataFrame()
		df_data['npl'] = df_cycle['npl']
		# 将自变量转换为上期数
		for exog in df_cycle.columns:
			if exog == 'npl':
				continue
			s = df_cycle[exog].shift(1)
			df_data[exog] = s
		df_data.index = df_cycle.index
		df_data['npl_lag'] = df_data['npl'].shift(1)
		self.df_data = df_data.dropna()
		return
