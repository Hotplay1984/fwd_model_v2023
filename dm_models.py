import pandas as pd
import datetime as dt
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm_notebook as tqdm
# from regressors import stats
from sklearn.feature_selection import f_regression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import SGDRegressor, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from . import dm_ts_funcs
from . import dm_config
from . import dm_ts
import warnings
warnings.filterwarnings('ignore')


def get_exog_combinations(all_exogs, num, min_or_fixed='min', essential_exogs=None, exclude_exogs=None):
    if min_or_fixed == 'min':
        exog_combinations = []
        for n in range(num, len(all_exogs) + 1):
            exog_combinations += [list(c) for c in combinations(all_exogs, n)]
    else:
        exog_combinations = [list(c) for c in combinations(all_exogs, num)]

    if essential_exogs is None and exclude_exogs is None:
        return exog_combinations
    exog_combinations_new = []
    for exog_combination in exog_combinations:
        include = True
        for exog in essential_exogs:
            if exog not in exog_combination:
                include = False
                break
        if include:
            exog_combinations_new.append(exog_combination)
    return exog_combinations_new


def eval_model(model, X_train, X_val, y_train, y_val, print_info=False):
    y_train_predict = model.predict(X_train)
    y_val_predict = model.predict(X_val)
    r2_score_train = r2_score(y_train, y_train_predict)
    r2_score_val = r2_score(y_val, y_val_predict)
    mse_train = mean_squared_error(y_train_predict, y_train)
    mse_val = mean_squared_error(y_val_predict, y_val)
    # coef_pvalues = stats.coef_pval(model.best_estimator_.steps[-1][1], X_train, y_train)
    coefs = model.best_estimator_['ridge_reg'].coef_
    ols = sm.OLS(y_train, X_train).fit()
    coefs_ols = ols.params.tolist()
    coefs_pvalues_ols = ols.pvalues.tolist()
    # coef_pvalues_f = f_regression(X_train, y_train)[1]
    coef_infos = ', '.join(
        '%s/%s(%s)' % ('{:.6f}'.format(coef), '{:.6f}'.format(coef_ols), '{:.2%}'.format(pvalue_ols))
        for coef, coef_ols, pvalue_ols in zip(coefs, coefs_ols, coefs_pvalues_ols))
    eval_info = {'R2_Train': r2_score_train, 'MSE_Train': mse_train, 'R2_Val': r2_score_val, 'MSE_Val': mse_val,
                 'exogs': X_train.columns.tolist(), 'endog': y_train.name, 'coefs': coefs,
                 'coefs(pvalue)': coef_infos}
    if print_info:
        print('R2 Train: %s; R2 Val: %s\n' % ('{:.6%}'.format(r2_score_train), '{:.6%}'.format(r2_score_val)))
        print('MSE Train: %s; MSE Val: %s\n' % ('{:.6%}'.format(mse_train), '{:.6%}'.format(mse_val)))
    return eval_info


def fit_and_predict(model, X_train, X_val, y_train, y_val, fit=True, print_prams=False, n_delta=2, include_next=False):
    if fit:
        model.fit(X_train, y_train)
    if print_prams:
        params = {key: value for key, value in model.get_params().items() if 'estimator__' in key}
        print('Best Estimator Parameters:')
        print(params)
        print('\n')

    y_train_predict = model.predict(X_train)
    y_val_predict = model.predict(X_val)

    df_p_train = pd.DataFrame(np.c_[y_train, y_train_predict], columns=['actual', 'predict'])
    df_p_val = pd.DataFrame(np.c_[y_val, y_val_predict], columns=['actual', 'predict'])
    df_p = pd.concat([df_p_train, df_p_val])
    df_p = df_p.apply(dm_ts_funcs.reverse_logit)

    df_p['-1_STD'] = -1 * df_p['actual'].shift(1).rolling(8).std() + df_p['predict']
    df_p['1_STD'] = df_p['actual'].shift(1).rolling(8).std() + df_p['predict']

    df_p['-1.645_STD'] = -1.645 * df_p['actual'].shift(1).rolling(8).std() + df_p['predict']
    df_p['1.036_STD'] = 1.036 * df_p['actual'].shift(1).rolling(8).std() + df_p['predict']

    df_p['-2_STD'] = -2 * df_p['actual'].shift(1).rolling(8).std() + df_p['predict']
    df_p['2_STD'] = 2 * df_p['actual'].shift(1).rolling(8).std() + df_p['predict']
    df_p['out_of_sample'] = ['N'] * len(df_p_train) + ['Y'] * len(df_p_val)
    df_p.index = y_train.index.tolist() + y_val.index.tolist()

    df_predict_out = df_p[df_p['out_of_sample'] == 'Y']
    total_predictions = len(df_predict_out)
    within_1std = len(df_predict_out[(df_predict_out['actual'] <= df_predict_out['1_STD']) &
                                     (df_predict_out['actual'] >= df_predict_out['-1_STD'])])
    within_2std = len(df_predict_out[(df_predict_out['actual'] <= df_predict_out['2_STD']) &
                                     (df_predict_out['actual'] >= df_predict_out['-2_STD'])])

    pdt_normal = (df_p['predict'].values[-1] / df_p['actual'].values[-1]) - 1
    pdt_negative = (df_p['1.036_STD'].values[-1] / df_p['actual'].values[-1]) - 1
    pdt_positive = (df_p['-1.645_STD'].values[-1] / df_p['actual'].values[-1]) - 1

    # upper_bound = '2_STD' if n_delta == 2 else '1_STD'
    # lower_bound = '-2_STD' if n_delta == 2 else '-1_STD'
    # pdt_negative = (df_p[upper_bound].values[-1] / df_p['actual'].values[-1]) - 1
    # pdt_positive = (df_p[lower_bound].values[-1] / df_p['actual'].values[-1]) - 1

    forward_factors = {'normal': round(pdt_normal, 4), 'negative': round(pdt_negative, 4),
                       'positive': round(pdt_positive, 4)}

    if include_next:
        last_ix = df_p.index.tolist()[-1]
        df_p.loc[last_ix, 'actual'] = None

    prediction_info = {
        'data': df_p,
        'total_predictions': total_predictions,
        'within_1_std': within_1std,
        'within_2_std': within_2std,
        '1_std_score': within_1std / total_predictions,
        '2_std_score': within_2std / total_predictions,
        'forward_factors': forward_factors
    }
    return model, prediction_info


def generate_learning_curve(model, X, y):
    print('Generating Learning Curve...\n')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    y_train_rev = np.array([dm_ts_funcs.reverse_logit(logit) for logit in y_train])
    y_val_rev = np.array([dm_ts_funcs.reverse_logit(logit) for logit in y_val])
    train_errors, val_errors = [], []

    for m in range(3, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)

        y_train_rev_p = np.array([dm_ts_funcs.reverse_logit(logit) for logit in y_train_predict])
        y_val_rev_p = np.array([dm_ts_funcs.reverse_logit(logit) for logit in y_val_predict])

        train_errors.append(mean_squared_error(y_train_rev_p, y_train_rev[:m]))
        val_errors.append(mean_squared_error(y_val_rev_p, y_val_rev))

        train_list_ = train_errors + [0.] * (len(np.arange(3, len(X_train))) - len(train_errors))
        val_list_ = val_errors + [0.] * (len(np.arange(3, len(X_train))) - len(val_errors))
        df_tmp = pd.DataFrame(np.c_[np.sqrt(train_list_), np.sqrt(val_list_)], index=np.arange(3, len(X_train)),
                      columns=['Train MSE', 'Val MSE']) / 100.

    df = pd.DataFrame(np.c_[np.sqrt(train_errors), np.sqrt(val_errors)], index=np.arange(3, len(X_train)),
                      columns=['Train MSE', 'Val MSE']) / 100.

    return df


def plot_learning_curve(df, fig, title=None):
    if fig is None:
        ax = plt.figure().add_subplot(111)
    else:
        ax = fig.gca()
    ax.clear()
    if title:
        df.plot(grid=True, style='o-', ylim=(0, 0.005), title=title, ax=ax)
    else:
        df.plot(grid=True, style='o-', ylim=(0, 0.005), ax=ax)
    plt.legend()
    return


def plot_predict(df_predict, fig=None, title=None):
    if fig is not None:
        ax = fig.gca()
    else:
        ax = plt.figure().add_subplot(111)
    ax.clear()
    ax.plot(df_predict.index, df_predict['actual'], '.-', label='actual')
    ax.plot(df_predict.index, df_predict['predict'], '.-', label='predict')

    ax.fill_between(df_predict.index, df_predict['-1_STD'],
                    df_predict['1_STD'], color='b', alpha=0.2, label='1 STD')
    ax.fill_between(df_predict.index, df_predict['-2_STD'],
                    df_predict['2_STD'], color='b', alpha=0.1, label='2 STD')
    ax.axvline(df_predict[df_predict['out_of_sample'] == 'Y'].index[0], color='r')
    ax.set_ylim(0, )
    ax.grid(True)
    ax.legend(loc='best')
    if title:
        plt.title(title)
    return


class Predictor:
    def __init__(self, bng_date, end_date):
        self.dmData = dm_ts.dmTs(bng_date, end_date)
        self.dmData.process_input_data()
        self.df_data = self.dmData.df_data
        self.all_exogs = [c for c in self.df_data.columns if c != 'npl']
        self.model_constructor = None
        self.df_learning_curve = pd.DataFrame()
        self.prediction_info = {}
        self.best_model = None
        return

    def build_model(self, model='ridge', exog_num=1, min_or_fixed='min', essential_exogs=None, cpu_count=None):
        exog_combinations = get_exog_combinations(self.all_exogs, num=exog_num, min_or_fixed=min_or_fixed,
                                                  essential_exogs=essential_exogs)
        self.model_constructor = modelConstructor(self.df_data)
        if cpu_count is None:
            cpu_count = max(1, int(mp.cpu_count() * 0.8))
        self.model_constructor.fit_all_combinations_parallel(model, 'npl', exog_combinations,
                                                             core_num=cpu_count)
        self.model_constructor.eval_candidates()
        return

    def make_prediction(self, model_id=None, select_method='by_id', select_num=10, plot=True, fig=None):
        X, y = self.df_data[self.all_exogs], self.df_data['npl']

        new_X = self.dmData.df_cycle.dropna()[-1:]  # 用来做最新预测(df_data除npl字段外是每一期的上期数字，df_cycle都是每一期的本期数)
        new_X.rename(columns={'npl': 'npl_lag'}, inplace=True)
        new_X.index = new_X.index.shift(1)

        candidates = self.model_constructor.model_candidates
        df_score = self.model_constructor.df_score
        best_model_id = 0
        if select_method == 'by_id':
            if model_id is None:
                best_model_id = df_score.model_id[0]
            else:
                best_model_id = model_id
        elif select_method == 'most_exogs':
            df_top_score = df_score[:select_num]
            best_model_id = df_top_score[df_top_score['exog_count'] ==
                                         df_top_score['exog_count'].max()]['model_id'].values[0]

        best_model = candidates[best_model_id]['model']
        best_exogs = df_score[df_score['model_id'] == best_model_id]['exogs'].values[0].split(', ')
        X_train, X_val, y_train, y_val = train_test_split(X[best_exogs], y, test_size=0.2, shuffle=False)
        X_val = pd.concat([X_val, new_X[best_exogs]])

        new_y = y_val[-1:]
        new_y.index = new_y.index.shift(1)[-1:]
        y_val = pd.concat([y_val, new_y])

        self.best_model, self.prediction_info = fit_and_predict(best_model, X_train, X_val, y_train, y_val, fit=True,
                                                                print_prams=True, include_next=True)
        plot_predict(self.prediction_info['data'])
        if plot:
            self.df_learning_curve = generate_learning_curve(best_model, X[best_exogs], y)
            plot_learning_curve(self.df_learning_curve, fig=fig)
        return


class modelConstructor:
    def __init__(self, df_data):
        self.cv_num = 2
        self.common_stps = [('scaler', StandardScaler())]
        self.param_grids = {
            'ridge': {'ridge_reg__alpha': np.arange(0.1e-6, 1, 0.01),
                      'ridge_reg__fit_intercept': [True, False]
                      },
            'lasso': {'lasso_reg__alpha': np.arange(0.1e-6, 10, 0.01),
                      'ridge_reg__fit_intercept': [True, False], },
            'svm': {'svm_reg__C': [0.1, 0.5, 1, 10], 'svm_reg__degree': [2, 3, 5], },
            'forest': {},
            'sgd': {'poly_features__degree': [2, 10], 'sgd_reg__penalty': ['l2', 'l1', 'elasticnet']},
        }

        self.ridge = Pipeline(self.common_stps + [('ridge_reg', Ridge())])
        self.lasso = Pipeline(self.common_stps + [('lasso_reg', Lasso(max_iter=50000))])
        self.svm = Pipeline(self.common_stps + [('svm_reg', SVR(kernel='poly', epsilon=0.1))])
        self.forest = Pipeline(self.common_stps + [('forest_reg', RandomForestRegressor())])
        self.sgd = Pipeline(self.common_stps + [('poly_features', PolynomialFeatures()), ('sgd_reg', SGDRegressor(shuffle=False)), ])

        self.models = {'ridge': self.ridge, 'lasso': self.lasso, 'svm': self.svm, 'forest': self.forest,
                       'sgd': self.sgd}
        self.df_data = df_data
        self.all_exogs = []
        self.model_candidates = {}
        self.df_score = pd.DataFrame()
        self.df_std_score = pd.DataFrame()
        self.df_learning_curve = pd.DataFrame()
        self.df_coef = pd.DataFrame()
        self.sensible_models = []
        self.iter_counter = 0
        self.total_iter = 0
        return

    def fit_combinations_iter(self, kw):
        i, exogs, model_name, endog_name = kw['i'], kw['exogs'], kw['model_name'], kw['endog_name']
        X_train, X_val, y_train, y_val = train_test_split(self.df_data[exogs], self.df_data[endog_name],
                                                          test_size=0.2, shuffle=False)
        tscv = TimeSeriesSplit(n_splits=self.cv_num)
        model = GridSearchCV(self.models[model_name], self.param_grids[model_name], cv=tscv,
                             scoring='neg_mean_squared_error')
        
        model.fit(X_train, y_train)
        model, predict_info = fit_and_predict(model,  X_train, X_val, y_train, y_val, fit=False, print_prams=False)
        eval_info = eval_model(model, X_train, X_val, y_train, y_val, print_info=False)
        res = {'id': i,  'model': model, 'eval_info': eval_info, 'predict_info': predict_info}
        self.iter_counter = i
        print('Progress: %d/%d' % (self.iter_counter, self.total_iter))
        return res

    def fit_all_combinations_parallel(self, model_name, endog_name, exog_combinations, core_num=None):
        self.all_exogs = [c for c in self.df_data.columns if c != endog_name]
        kws = [{'i': i, 'exogs': exogs, 'model_name': model_name, 'endog_name': endog_name}
               for i, exogs in enumerate(exog_combinations)]
        self.total_iter = len(kws)
        if core_num is None:
            core_num = max(1, int(mp.cpu_count() * 0.8))
        model_num = self.get_model_num(model_name)
        desc = '共%s个变量组合，\n每个变量组合共%s个超参数组合，\n交叉验证%d次，\n合计需进行%s次拟合。' % ('{:,.0f}'.format(len(kws)),
                                                                 '{:,.0f}'.format(model_num), self.cv_num,
                                                                 '{:,.0f}'.format(len(kws)*model_num*self.cv_num))
        with mp.Pool(core_num) as p:
            res_list = list(tqdm(p.imap(self.fit_combinations_iter, kws), total=len(kws), desc=desc))
        for res in res_list:
            self.model_candidates[res['id']] = {'model': res['model'],
                                                'eval_info': res['eval_info'],
                                                'predict_info': res['predict_info']}
        return

    def get_model_num(self, model_name):
        param_grid = self.param_grids[model_name]
        model_num = 1
        for para, grid in param_grid.items():
            model_num = model_num * len(grid)
        return model_num

    def rank_models(self, eval_scoring='r2'):
        model_candidates = self.model_candidates
        model_ids = list(sorted(model_candidates.keys()))
        exog_list, r2_trains, r2_tests, mse_trains, mse_tests = [], [], [], [], []
        for ix in model_ids:
            model, eval_info = model_candidates[ix]['model'], model_candidates[ix]['eval_info']
            r2_trains.append(eval_info['R2_Train'])
            mse_trains.append(eval_info['MSE_Train'])
            r2_tests.append(eval_info['R2_Val'])
            mse_tests.append(eval_info['MSE_Val'])
            exog_list.append(', '.join(eval_info['exogs']))
            # coef_pvalues.append(', '.join(['%s' % str(round(coef, 6)) for coef in eval_info['coefs']]))

        num_columns = ['r2_train', 'mse_train', 'r2_test', 'mse_test']
        df_score = pd.DataFrame(np.c_[r2_trains, mse_trains, r2_tests, mse_tests], columns=num_columns)
        df_score['model_id'] = model_ids
        df_score['exogs'] = exog_list
        df_score['exog_count'] = [len(exogs.split(', ')) for exogs in exog_list]
        if eval_scoring == 'r2':
            df_score = df_score.sort_values(by='%s_test' % eval_scoring, ascending=False)
        else:
            df_score = df_score.sort_values(by='%s_test' % eval_scoring, ascending=True)
        columns = df_score.columns.tolist()
        # df_score[df_score['r2_test'].astype(float) > 0.5].reset_index()[['r2_test', 'mse_test']].plot(secondary_y='mse_test')
        df_score = df_score.reset_index()[columns]
        return df_score

    def eval_candidates(self, print_summary=False):
        self.df_score = self.rank_models(eval_scoring='mse')
        self.df_std_score = self.get_std_score()
        self.df_coef = self.get_candidate_coefs()
        self.sensible_models = self.get_sensible_models()
        if print_summary:
            self.print_summary()
        return

    def get_std_score(self):
        candidates = self.model_candidates
        df_score = self.df_score
        values = [[i, candidates[i]['predict_info']['1_std_score'], candidates[i]['predict_info']['2_std_score']]
                  for i in df_score.model_id]
        df_std_score = pd.DataFrame(values, columns=['model_id', '1_std_score', '2_std_score'])
        df_std_score['50%'] = [0.5] * len(df_std_score)
        df_std_score['90%'] = [0.9] * len(df_std_score)

        plot_columns = ['1_std_score', '2_std_score', '50%', '90%']
        df_std_score[plot_columns].plot(ylim=(0, 1), grid=True, style={'50%': 'r--', '90%': 'g--'})
        return df_std_score

    def get_candidate_coefs(self):
        candidates = self.model_candidates
        df_score = self.df_score

        values = []
        for ix in df_score.index:
            model_id = df_score.loc[ix, 'model_id']
            exogs = df_score.loc[ix, 'exogs'].split(', ')
            coefs = candidates[model_id]['model'].best_estimator_['ridge_reg'].coef_
            row_data = [model_id]
            for exog in self.all_exogs:
                coef = 0
                if exog in exogs:
                    coef = coefs[exogs.index(exog)]
                row_data.append(coef)
            values.append(row_data)
        df_coef = pd.DataFrame(values, columns=['model_id'] + self.all_exogs)
        return df_coef

    def get_sensible_models(self):
        df_coef = self.df_coef
        coef_constrains = dm_config.coef_constrains
        sensible_models = []
        for ix in df_coef.index:
            model_id = df_coef.loc[ix, 'model_id']
            sensible = True
            if False in [
                eval(''.join([str(df_coef.loc[ix, exog]), coef_constrains[exog]])) if exog in coef_constrains else True
                for exog in self.all_exogs]:
                sensible = False
            if sensible:
                sensible_models.append(model_id)
        return sensible_models

    def print_summary(self):
        candidates = self.model_candidates
        df_score = self.df_score
        total_cbms = len(df_score)
        sensible_models = self.sensible_models
        total_sensible_cbms = len(sensible_models)
        df_score = df_score[df_score['model_id'].isin(sensible_models)].reset_index()[df_score.columns.tolist()]

        best_model_id = df_score.model_id[0]
        best_exogs = df_score.exogs[0].split(', ')

        r2_train, mse_train = float(df_score.r2_train[0]), float(df_score.mse_train[0])
        r2_test, mse_test = float(df_score.r2_test[0]), float(df_score.mse_test[0])
        std_1_score = float(candidates[best_model_id]['predict_info']['1_std_score'])
        std_2_score = float(candidates[best_model_id]['predict_info']['2_std_score'])
        abv_9 = len(df_score[df_score['r2_test'].astype(float) > 0.9])
        abv_8 = len(df_score[df_score['r2_test'].astype(float) > 0.8])
        abv_5 = len(df_score[df_score['r2_test'].astype(float) > 0.5])

        summary = ''' 
        所有%d个变量组合中，共有%d个变量组合符合系数约束条件，其中Test R2 Score超过90pct/80pct/50pct的组合个数：%d/%d/%d;
        最佳变量组合：%s;
        R2 Score(Train/Test): %s/%s;
        MSE(Train/Test): %s/%s;
        样本外测试中，有%s落在1个标准差预测区间，%s落在2个标准差预测区间
        ''' % (total_cbms, total_sensible_cbms, abv_9, abv_8, abv_5,
               ', '.join(best_exogs),
               '{:.2%}'.format(r2_train), '{:.2%}'.format(r2_test),
               '{:.6f}'.format(mse_train), '{:.6f}'.format(mse_test),
               '{:.2%}'.format(std_1_score), '{:.2%}'.format(std_2_score))
        print(summary)
        return
