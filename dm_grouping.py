import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import OrderedDict
import statsmodels.api as sm
from scipy.stats import lognorm, poisson
from scipy.stats import skewnorm
from fwd_model import dm_config as config
from fwd_model.dm_grouping_data import make_npl_history


def read_local_arima_config(file_path=None):
	if file_path is None:
		df_local = pd.read_excel(config.arima_param_config_file)
	else:
		df_local = pd.read_excel(file_path)
	ind_arima_params = {ind: eval(params) for ind, params in zip(df_local['industry'], df_local['arima_params'])}
	return ind_arima_params


def macro_rating_breakdown(macro_npl, ratings_mapping_file=config.ratings_mapping_file):
	df_map = pd.read_excel(ratings_mapping_file).set_index('INTL_RATING')
	return df_map['AVG_DFLT_RATE'] * (macro_npl / (df_map['AVG_DFLT_RATE'] * df_map['WEIGHT']).sum())


def get_distribution_poisson(para, buckets):
	return np.array([poisson.pmf(k=rating, mu=para) for rating in buckets])


def get_distribution_skewnorm(para, buckets):
	weight_s = []
	X = np.linspace(skewnorm.ppf(0.0001, para), skewnorm.ppf(0.9999, para), buckets[-1])
	for x in X:
		weight_s.append(skewnorm.pdf(x, para))
	return np.array(weight_s) / np.array(weight_s).sum()


def get_distribution_lognorm(para, buckets):
	weight_s = []
	X = np.linspace(lognorm.ppf(0.01, para), lognorm.ppf(0.99, para), buckets[-1])
	for x in X:
		weight_s.append(lognorm.pdf(x, para))
	return np.array(weight_s) / np.array(weight_s).sum()


def get_distribution(distribution, para, buckets, reverse=True):
	s = np.array([])
	if distribution == 'poisson':
		s = get_distribution_poisson(para, buckets)
	elif distribution == 'skewnorm':
		s = get_distribution_skewnorm(para, buckets)
	elif distribution == 'lognorm':
		s = get_distribution_lognorm(para, buckets)
	# return s
	if reverse:
		return np.array(list(reversed(s)))
	else:
		return s


def generate_param_candidates():
	params_grid = config.arima_params_grid
	param_candidates = []
	for ar in params_grid['ar_range']:
		for diff in params_grid['diff_range']:
			for ma in params_grid['ma_range']:
				for stationarity in params_grid['enforce_stationarity']:
					for invertibility in params_grid['enforce_invertibility']:
						for trend in params_grid['trend']:
							if trend in ['c', 'ct'] and diff > 0:
								continue
							dict_ = OrderedDict()
							dict_['order'] = (ar, diff, ma)
							dict_['enforce_stationarity'] = stationarity
							dict_['enforce_invertibility'] = invertibility
							dict_['trend'] = trend
							param_candidates.append(dict_)
	return param_candidates


def make_arima_reg(ind_nplr, macro_nplr, param_dict):
	reg = sm.tsa.arima.ARIMA(ind_nplr, exog=macro_nplr, order=param_dict['order'],
							enforce_stationarity=param_dict['enforce_stationarity'],
							enforce_invertibility=param_dict['enforce_invertibility'],
							trend=param_dict['trend']
							).fit(method_kwargs={"warn_convergence": False})
	reg_auto = sm.tsa.arima.ARIMA(ind_nplr, order=(1, 0, 0),
							enforce_stationarity=True,
							enforce_invertibility=False,
							trend='n'
							).fit(method_kwargs={"warn_convergence": False})
	return reg, reg_auto


def search_arima_reg(y, exog):
	def eval_candidates(candidates):
		values = []
		for reg_id, reg_info in candidates.items():
			reg = reg_info['reg']

			include = True
			for coef in ['drift', 'const', 'ar.L2', 'ma.L3', 'ma.L2', 'ma.L1']:
				if coef in reg.pvalues.index:
					if reg.pvalues.loc[coef] > 0.1:
						include = False
						break
			if include:
				values.append([reg_id, reg.pvalues.loc['All'], reg.aic, reg.bic])

		df_p = pd.DataFrame(values, columns=['para_id', 'exog_pvalue', 'aic', 'bic']).fillna(-1)
		df_p_sig = df_p[df_p['exog_pvalue'] <= 0.1]
		if len(df_p_sig) > 0:
			df_p_sig = df_p_sig.sort_values(by=['bic', 'aic']).reset_index(drop=True)
			return df_p_sig['para_id'].values[0]
		else:
			df_p = df_p.sort_values(by='exog_pvalue').reset_index(drop=True)
			return df_p['para_id'].values[0]

	candidates = {}
	param_candidates = generate_param_candidates()
	for para_id, param_dict in enumerate(param_candidates):
		reg, reg_auto = make_arima_reg(y, exog, param_dict)
		candidates[para_id] = {'reg': reg, 'param_dict': param_dict, 'reg_auto': reg_auto}
	best_reg_id = eval_candidates(candidates)
	return candidates[best_reg_id]


class dmRiskGrouping:
	def __init__(self, macro_npl=None, df_npls=None, forward_looking_factors=None, bng_date=None, end_date=None):
		self.bng_date, self.end_date = bng_date, end_date
		if df_npls is None:
			self.df_npl_history = make_npl_history(bng_date=self.bng_date, end_date=self.end_date)
		else:
			self.df_npl_history = df_npls

		if forward_looking_factors is None:
			self.forward_looking_factors = {'normal': 0, 'negative': 0, 'positive': 0.}
		else:
			self.forward_looking_factors = forward_looking_factors

		if macro_npl is None:
			self.macro_npl = self.df_npl_history['All'][-1] + self.forward_looking_factors['normal']
		else:
			self.macro_npl = macro_npl + self.forward_looking_factors['normal']

		self.local_arima_config = read_local_arima_config()
		self.distribution = 'skewnorm'
		self.distribution_parameter_range = config.distribution_parameter_range[self.distribution]
		self.min_npl, self.max_npl = config.min_npl, config.max_npl
		self.rating_bins = config.rating_bins
		self.n_deg_range = config.deg_range
		self.macro_npls, self.macro_weights, self.distribution_factor = None, None, None
		self.df_ind_npls = pd.DataFrame()
		self.df_stats = pd.DataFrame()
		self.df_proj = pd.DataFrame()
		self.df_ind_mapping = pd.DataFrame()
		self.df_ind_weights = pd.DataFrame()
		self.ind_search_deg_dict = {}
		self.ind_search_dis_dict = {}
		self.ind_fit_dict = {}

		return

	def run_grouping(self, param_search):
		self.df_ind_npls = self.get_ind_projections(param_search=param_search)

		self.df_proj = deepcopy(self.df_ind_npls)
		self.df_proj['applied_prediction'] = self.df_proj['prediction_normal'] * 0.6 + \
											self.df_proj['prediction_negative'] * 0.3 + \
											self.df_proj['prediction_positive'] * 0.1
		proj_columns = ['last_value', 'applied_prediction', 'coef_sig', 'prediction_normal', 'prediction_negative',
						'prediction_positive']
		self.df_proj = self.df_proj[proj_columns]
		self.df_stats = self.df_ind_npls[['coef_sig', 'params_info', 'arima_params', 'R2', 'mse', 'R2_auto', 'mse_auto']]
		self.macro_npls, self.macro_weights, self.distribution_factor = self.get_macro_npl_mapping(self.macro_npl,
																							distribution=self.distribution)
		self.df_ind_mapping = self.get_industry_mappings()
		self.get_ind_weights()
		self.get_ind_weights(reverse=False, ind_list=list(self.ind_search_dis_dict.keys()))
		return

	def get_ind_projection_ols(self, ind, y, X, X_auto, last_ind_nplr, macro_npl_projs):
		reg_macro = sm.OLS(y, X).fit()
		reg_auto = sm.OLS(y, X_auto).fit()

		r_macro, r_auto = round(reg_macro.rsquared_adj, 4), round(reg_auto.rsquared_adj, 4)
		params = reg_macro.params.round(4).tolist()
		pvalues = reg_macro.pvalues.round(4).tolist()
		coef_sig = True if pvalues[0] <= 0.1 else False

		macro_npl_normal = macro_npl_projs['normal']
		macro_npl_negative = macro_npl_projs['negative']
		macro_npl_positive = macro_npl_projs['positive']

		prediction_normal = round(reg_macro.predict(np.array([macro_npl_normal, last_ind_nplr]))[0], 6)
		prediction_negative = round(reg_macro.predict(np.array([macro_npl_negative, last_ind_nplr]))[0], 6)
		prediction_positive = round(reg_macro.predict(np.array([macro_npl_positive, last_ind_nplr]))[0], 6)

		prediction_info = OrderedDict()
		prediction_info['auto_reg_r2'] = r_auto
		prediction_info['r2'] = r_macro
		prediction_info['coefs'] = params
		prediction_info['pvalues'] = pvalues
		prediction_info['coef_sig'] = coef_sig
		prediction_info['last_value'] = last_ind_nplr
		prediction_info['prediction_normal'] = prediction_normal
		prediction_info['prediction_negative'] = prediction_negative
		prediction_info['prediction_positive'] = prediction_positive

		self.ind_fit_dict[ind] = reg_macro.predict()
		return prediction_info

	def get_ind_projection_arima(self, ind, y, X, last_ind_nplr, macro_npl_projs, param_search=False):
		y.index = pd.DatetimeIndex(y.index, freq='2Q-DEC')
		X.index = pd.DatetimeIndex(X.index, freq='2Q-DEC')
		if param_search:
			reg_macro_info = search_arima_reg(y, X)
			reg_macro = reg_macro_info['reg']
			param_dict = reg_macro_info['param_dict']
			reg_macro_auto = reg_macro_info['reg_auto']
		else:
			param_dict = self.local_arima_config[ind]
			reg_macro, reg_macro_auto = make_arima_reg(y, X, param_dict)

		params = reg_macro.params.round(4).tolist()
		param_names = reg_macro.param_names
		pvalues = reg_macro.pvalues.round(4)
		coef_sig = True if pvalues.loc['All'] <= 0.1 else False

		macro_npl_normal = macro_npl_projs['normal']
		macro_npl_negative = macro_npl_projs['negative']
		macro_npl_positive = macro_npl_projs['positive']

		prediction_0 = max(round(reg_macro.forecast(exog=macro_npl_normal).values[-1], 6), self.min_npl)
		prediction_1 = max(round(reg_macro.forecast(exog=macro_npl_negative).values[-1], 6), self.min_npl)
		prediction_2 = max(round(reg_macro.forecast(exog=macro_npl_positive).values[-1], 6), self.min_npl)
		predictions = list(sorted([prediction_0, prediction_1, prediction_2]))

		prediction_info = OrderedDict()
		prediction_info['coef_sig'] = coef_sig
		prediction_info['last_value'] = last_ind_nplr
		prediction_info['prediction_normal'] = predictions[1]
		prediction_info['prediction_negative'] = predictions[2]
		prediction_info['prediction_positive'] = predictions[0]
		prediction_info['params_info'] = ' | '.join(['%s: %s(%s)' % (name, '{:.6f}'.format(coef), '{:.0%}'.format(p))
										for name, coef, p in zip(param_names, params, pvalues)])
		prediction_info['arima_params'] = param_dict

		s_fit = reg_macro.predict()
		s_act = self.df_npl_history[ind]
		df_ = pd.DataFrame([s_fit, s_act]).transpose().dropna()
		df_.columns = ['fit', 'act']
		r_2 = 1 - np.sum((df_['fit'] - df_['act']) ** 2) / np.sum((df_['act'] - df_['act'].mean()) ** 2)
		mse = np.sum((df_['fit'] - df_['act']) ** 2) / len(df_)

		s_fit_auto = reg_macro_auto.predict()
		df_ = pd.DataFrame([s_fit_auto, s_act]).transpose().dropna()
		df_.columns = ['fit', 'act']
		r_2_auto = 1 - np.sum((df_['fit'] - df_['act']) ** 2) / np.sum((df_['act'] - df_['act'].mean()) ** 2)
		mse_auto = np.sum((df_['fit'] - df_['act']) ** 2) / len(df_)

		prediction_info['R2'] = r_2
		prediction_info['mse'] = mse
		prediction_info['R2_auto'] = r_2_auto
		prediction_info['mse_auto'] = mse_auto
		self.ind_fit_dict[ind] = s_fit
		return prediction_info

	def get_ind_projections(self, reg_method='arima', param_search=False):
		df_npls = self.df_npl_history
		last_macro = df_npls['All'].values[-1]
		industries = [c for c in df_npls.columns.tolist() if c != 'All']

		macro_npl_normal = last_macro * (1 + self.forward_looking_factors['normal'])
		macro_npl_negative = last_macro * (1 + self.forward_looking_factors['negative'])
		macro_npl_positive = last_macro * (1 + self.forward_looking_factors['positive'])
		macro_npl_projs = {'normal': macro_npl_normal, 'negative': macro_npl_negative, 'positive': macro_npl_positive}

		values = []
		ind_projection_info = OrderedDict()
		for ind in industries:
			s_ind = df_npls[ind].dropna()[1:]
			s_all = df_npls[df_npls.index.isin(s_ind.index)]['All']
			df_data = pd.DataFrame(np.c_[s_ind, s_all, df_npls[ind].shift(1).dropna()],
								columns=[ind, 'all', '%s_lag' % ind], index=s_ind.index).dropna()
			last_ind_nplr = round(df_npls[ind].values[-1], 6)
			y, X, X_auto = df_data[ind], df_data[['all', '%s_lag' % ind]], df_data['%s_lag' % ind]
			if reg_method == 'ols':
				ind_projection_info = self.get_ind_projection_ols(ind, y, X, X_auto, last_ind_nplr, macro_npl_projs)
			else:
				ind_projection_info = self.get_ind_projection_arima(ind, df_npls[ind], df_npls['All'], last_ind_nplr,
																	macro_npl_projs, param_search=param_search)
			values.append([ind] + [ind_projection_info[c] for c in list(ind_projection_info.keys())])

		columns = ['industry'] + list(ind_projection_info.keys())
		df_ind_projection = pd.DataFrame(values, columns=columns)
		df_ind_projection.set_index('industry', inplace=True)
		return df_ind_projection

	def get_macro_npl_mapping(self, macro_npl=None, distribution='lognorm'):
		if macro_npl is None:
			macro_npl = self.macro_npl
		macro_npls = macro_rating_breakdown(macro_npl)
		rating_arr = np.arange(1, self.rating_bins + 1)
		weight_s, distribution_factor = None, None
		para_range = self.distribution_parameter_range
		low, high = 0, len(para_range) - 1
		while low <= high:
			mid = low + (high - low) // 2
			para = para_range[mid]
			weight_s = get_distribution(distribution, para, rating_arr)

			wtd_npl = (weight_s * macro_npls).sum()
			if wtd_npl == macro_npl:
				break
			elif wtd_npl < macro_npl:
				low = mid + 1
			else:
				high = mid - 1

			distribution_factor = para
		return macro_npls, weight_s, distribution_factor

	def get_possible_curve(self, min_deg=None, max_deg=None, step=None):
		if min_deg is not None:
			n_deg_range = np.arange(min_deg, max_deg, step)
		else:
			n_deg_range = self.n_deg_range
		values = []
		for x in np.arange(self.rating_bins):
			rows = []
			for n_degree in n_deg_range:
				rows.append(((self.max_npl - self.min_npl) / ((self.rating_bins - 1) ** n_degree)) * (x ** n_degree) + self.min_npl)
			values.append(rows)
		df = pd.DataFrame(values, columns=[str(round(c, 3)) for c in n_deg_range])
		df_n = pd.DataFrame()
		for n_degree in df.columns:
			df_n[n_degree] = df[n_degree] * self.macro_weights
		possible_curve = df_n.sum()
		possible_curve.index = df.columns
		return df, possible_curve

	def search_ind_mapping(self, ind_npl):
		max_npl = max(self.max_npl, ind_npl)
		rating_range = np.arange(self.rating_bins)
		low, high = 0, len(self.n_deg_range) - 1
		values = []
		while low <= high:
			npl_curve = []
			mid = low + (high - low) // 2
			n_deg = self.n_deg_range[mid]
			for rating in rating_range:
				npl_curve.append(((max_npl - self.min_npl) / ((self.rating_bins - 1) ** n_deg)) * (rating ** n_deg) + self.min_npl)
			avg_npl = (np.array(npl_curve) * self.macro_weights).sum()
			values.append([ind_npl, n_deg, avg_npl])
			if avg_npl == ind_npl:
				break
			elif avg_npl > ind_npl:
				low = mid + 1
			else:
				high = mid - 1
		df_search = pd.DataFrame(values, columns=['ind_npl', 'n_deg', 'wtd_npl'])
		df_search.set_index('n_deg', inplace=True)
		return np.array(npl_curve), df_search

	def get_industry_mappings(self):
		df_ind_npls = self.df_ind_npls
		df_ind_mapping = pd.DataFrame()
		industries = df_ind_npls.index.tolist()
		df_normal = pd.DataFrame()
		for ind in industries:
			ind_npl = df_ind_npls.loc[ind, 'prediction_normal']
			ind_npls, df_search = self.search_ind_mapping(ind_npl)
			self.ind_search_deg_dict[ind] = df_search
			df_normal[ind] = ind_npls
		macro_npls, df_search_macro = self.search_ind_mapping(self.macro_npl * (1 + self.forward_looking_factors['normal']))
		df_normal['All'] = macro_npls
		df_normal['Scenario'] = ['normal'] * len(df_normal)
		df_normal.index = self.macro_npls.index
		self.ind_search_deg_dict['All'] = df_search_macro

		df_last = pd.DataFrame()
		for ind in industries:
			ind_npl = df_ind_npls.loc[ind, 'last_value']
			ind_npls, _ = self.search_ind_mapping(ind_npl)
			df_last[ind] = ind_npls
		macro_npls, _ = self.search_ind_mapping(self.df_npl_history['All'][-1])
		df_last['All'] = macro_npls
		df_last['Scenario'] = ['last'] * len(df_last)
		df_last.index = self.macro_npls.index

		df_scenario_adj = pd.DataFrame()
		for scenario in ['positive', 'negative']:
			df_ = pd.DataFrame()
			for ind in industries:
				nplr_chg_factor = df_ind_npls.loc[ind, 'prediction_%s' % scenario] / df_ind_npls.loc[ind, 'prediction_normal']
				df_[ind] = [max(self.min_npl, nplr) for nplr in df_normal[ind] * nplr_chg_factor]
			df_['All'] = [max(self.min_npl, nplr) for nplr in df_normal['All'] * (1 + self.forward_looking_factors[scenario])]
			df_['Scenario'] = [scenario] * len(df_)
			df_.index = self.macro_npls.index
			df_scenario_adj = pd.concat([df_scenario_adj, df_])

		for df_ in [df_last, df_normal, df_scenario_adj]:
			df_ind_mapping = pd.concat([df_ind_mapping, df_])
		df_ind_mapping.index.name = 'INTL_RATING'
		df_ind_mapping.reset_index(inplace=True)
		return df_ind_mapping

	def get_ind_weights(self, reverse=True, ind_list=None):
		df_ind_npls = self.df_ind_npls
		df_ind_mapping = deepcopy(self.df_ind_mapping[self.df_ind_mapping['Scenario'] == 'normal'])
		s_range = self.distribution_parameter_range
		rating_arr = np.arange(1, self.rating_bins + 1)
		if len(self.df_ind_weights) == 0:
			df_weight_refined = pd.DataFrame()
		else:
			df_weight_refined = self.df_ind_weights
		df_weight_refined['All'] = self.macro_weights
		if ind_list is None:
			ind_list = df_ind_npls.index.tolist()
		ind_search_dis_dict = {}
		for ind in ind_list:
			values = []
			act_npl = df_ind_npls.loc[ind, 'prediction_normal']
			npls = df_ind_mapping[ind]
			wtd_npl = round((npls * self.macro_weights).sum(), 6)
			if len(self.df_ind_weights) == 0:
				weight_s = self.macro_weights
			else:
				weight_s = np.array(self.df_ind_weights[ind].tolist())
			if abs(round(act_npl, 6) - round(wtd_npl, 6)) < 0.00001:
				df_weight_refined[ind] = self.macro_weights
				continue
			low, high = 0, len(s_range) - 1
			while low <= high:
				mid = low + (high - low) // 2
				s = s_range[mid]
				weight_s = get_distribution(distribution=self.distribution, para=s, buckets=rating_arr, reverse=reverse)
				wtd_npl = round((weight_s * npls).sum(), 6)
				values.append([s, act_npl, wtd_npl])
				if wtd_npl == act_npl:
					break
				elif wtd_npl < act_npl:
					if reverse:
						low = mid + 1
					else:
						high = mid - 1
				else:
					if reverse:
						high = mid - 1
					else:
						low = mid + 1
			df_weight_refined[ind] = weight_s
			df_search = pd.DataFrame(values, columns=['DistributionPara', 'ind_npl', 'wtd_npl'])
			df_search.set_index('DistributionPara', inplace=True)
			ind_search_dis_dict[ind] = df_search
		self.ind_search_dis_dict = ind_search_dis_dict
		df_weight_refined.index = df_ind_mapping.index
		self.df_ind_weights = df_weight_refined
		return

	def get_ind_data(self, ind):
		df_ind = pd.DataFrame()
		df_ind_mapping, df_ind_weights = deepcopy(self.df_ind_mapping), deepcopy(self.df_ind_weights)
		for scenario in ['positive', 'normal', 'negative', 'last']:
			df_ = df_ind_mapping[df_ind_mapping['Scenario'] == scenario][['INTL_RATING', ind]].set_index('INTL_RATING')
			df_.rename(columns={ind: scenario}, inplace=True)
			df_ind = df_ind.merge(df_, how='outer', left_index=True, right_index=True)
		df_ind['weights'] = df_ind_weights[ind].tolist()
		df_ind['diff_normal'] = df_ind['normal'] - df_ind['last']
		return df_ind

	def plot_ind_ts(self, ind, ax=None):
		df_proj = self.df_ind_npls
		df_history = self.df_npl_history
		s_ind = df_history[ind]
		s_fitted = None
		if ind in self.ind_fit_dict.keys():
			s_fitted = self.ind_fit_dict[ind]
			s_fitted = s_fitted[s_fitted.index.isin(pd.DatetimeIndex(s_ind.index))]
		time_grid = [pd.to_datetime(d) for d in s_ind.index]
		now = time_grid[-1]
		next_quarter_end = now + pd.tseries.offsets.QuarterEnd(startingMonth=now.month)

		if ind != 'All':
			p_normal = df_proj.loc[ind, 'prediction_normal']
			p_positive = df_proj.loc[ind, 'prediction_positive']
			p_negative = df_proj.loc[ind, 'prediction_negative']
		else:
			p_normal = self.forward_looking_factors['normal'] + s_ind[-1]
			p_positive = self.forward_looking_factors['positive'] + s_ind[-1]
			p_negative = self.forward_looking_factors['negative'] + s_ind[-1]
			print(p_normal, p_negative, p_positive, 0.6 * p_normal + 0.3 * p_negative + 0.1 * p_positive)
		if ax is None:
			ax = plt.figure(figsize=(20, 5)).add_subplot(111)
		ax.plot(s_ind.index, s_ind, '.-', label='历史值')
		if s_fitted is not None:
			ax.plot(s_fitted.index, s_fitted, '.-', label='拟合值', color='g')
			ax.plot([now, next_quarter_end], [s_fitted[-1], p_negative], '--.', color='r', label='悲观')
			ax.plot([now, next_quarter_end], [s_fitted[-1], p_normal], '--.', color='g', label='正常')
			ax.plot([now, next_quarter_end], [s_fitted[-1], p_positive], '--.', color='b', label='乐观')

		ax.fill_between(s_ind.index, s_ind - s_ind.rolling(8).std(), s_ind + s_ind.rolling(8).std(), alpha=0.4, color='r')
		ax.fill_between(s_ind.index, s_ind - 2 * s_ind.rolling(8).std(), s_ind + 2 * s_ind.rolling(8).std(), alpha=0.1,
						color='r')

		ax.set_ylim(0, )
		ax.grid(True)
		ax.legend(loc='lower left')
		if ax is None:
			plt.show()
		return

	def plot_ind_deg(self, ind, ax=None):
		df = self.ind_search_deg_dict[ind]
		if ax is None:
			ax = plt.figure(figsize=(20, 5)).add_subplot(111)
		df.plot(ax=ax, grid=True, style={'ind_npl': '-', 'wtd_npl': '.-'})
		if ax is None:
			plt.show()
		return

	def plot_ind_dis(self, ind, ax=None):
		df = self.ind_search_dis_dict[ind]
		if ax is None:
			ax = plt.figure(figsize=(20, 5)).add_subplot(111)
		df.plot(ax=ax, grid=True, style={'ind_npl': '-', 'wtd_npl': '.-'}, legend=False)
		if ax is None:
			plt.show()
		return
