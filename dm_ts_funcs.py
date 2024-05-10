#
# Forward Looking Model Frame
# dm_ts_funcs.pys
#


import pandas as pd
import datetime as dt
import math
import numpy as np
from . import dm_config as config
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import STL


def read_config():
    print('Config File: %s\n' % config.config_file)
    return pd.read_excel(config.config_file).set_index('var_name_cn')


def read_npl_file(file_name):
    df = pd.read_excel(file_name, index_col=0)[:-4]
    rpt_dates = {'一季报': [3, 31], '中报': [6, 30], '三季报': [9, 30], '年报': [12, 31]}
    report_dates = [dt.date(int(rpt[:4]), rpt_dates[rpt.split('.')[0][4:]][0], rpt_dates[rpt.split('.')[0][4:]][1])
                    for rpt in df.columns]
    df.columns = report_dates
    df = df.transpose().reset_index()
    columns = ['RPT_DATE', 'BANK', 'NPL_R', 'NPL_V', 'TTL_LN']
    df.columns = columns
    for col in ['NPL_R', 'NPL_V', 'TTL_LN']:
        df[col] = df[col].astype(float)
    banks = df['BANK'].unique().tolist()
    df_npl_r_0 = pd.pivot_table(df, values='NPL_R', index=['RPT_DATE'], columns='BANK')[banks]
    df_npl_v = pd.pivot_table(df, values='NPL_V', index=['RPT_DATE'], columns='BANK')[banks]
    df_ttl_ln = pd.pivot_table(df, values='TTL_LN', index=['RPT_DATE'], columns='BANK')[banks]
    df_npl_r_1 = df_npl_v * 100 / df_ttl_ln
    df_npl_r = df_npl_r_1.fillna(df_npl_r_0)  # 不良贷款率公布值仅精确到0.01%，如果从不良贷款/总贷款计算不良贷款率缺失值可接受的话，可以获得更高精度。

    rpt_dates = df_npl_r.index
    wtd_nplr_list = []
    index_list = []
    for rpt_date in rpt_dates:
        s_nplr = df_npl_r.loc[rpt_date][~df_npl_r.loc[rpt_date].isnull()].sort_values()
        high = s_nplr.mean() + 2 * s_nplr.std()
        s_nplr = s_nplr[s_nplr < high]
        included_banks = s_nplr.index.tolist()
        weights = df_ttl_ln.loc[rpt_date, included_banks] / df_ttl_ln.loc[rpt_date, included_banks].sum()
        wtd_nplr = (s_nplr * weights).sum()
        wtd_nplr_list.append(wtd_nplr)
        index_list.append(rpt_date)
    df_wtd_nplr = pd.DataFrame(wtd_nplr_list, index=index_list, columns=['WTD_NPLR'])
    df_wtd_nplr.index.name = 'datadate'
    df_wtd_nplr.index = pd.to_datetime(df_wtd_nplr.index)
    return df_wtd_nplr


def read_macro_file(file_name):
    df_macro = pd.read_excel(file_name)[:-2]
    df_macro.set_index('指标名称', inplace=True)
    df_macro.index.name = 'datadate'
    df_macro.index = pd.to_datetime(df_macro.index)
    return df_macro


def read_raw_data(bng_date, end_date):
    macro_file, npl_file = config.macro_file_name, config.npl_file_name
    print('Data File: %s, %s\n' % (macro_file, npl_file))
    df_macro = read_macro_file(macro_file)
    df_nplr = read_npl_file(npl_file)
    # df_data = df_nplr.merge(df_macro, how='left', left_index=True, right_index=True)
    df_data = df_macro.merge(df_nplr, how='left', left_index=True, right_index=True)
    df_data = df_data[(df_data.index >= dt.datetime.strptime(bng_date, '%Y%m%d')) &
                      (df_data.index <= dt.datetime.strptime(end_date, '%Y%m%d'))]
    # df_data['npl_lag'] = df_data['WTD_NPLR'].shift(3)
    print(df_data.info())
    return df_data


def seasonal_adjust(s):
    sd = STL(s).fit()
    trend = sd.trend
    return trend.dropna()


def reverse_logit(logit):
    return 100 * np.exp(logit)/(1+np.exp(logit))


def logit(v):
    return math.log(v/(100-v))


def get_logit(s, freq=None, seasonality=False):
    s_new = pd.Series(logit(npl) for npl in s)
    s_new.index = s.index
    # s_new = (s_new.diff(4)/s_new.shift(4)).dropna().resample(freq).interpolate()
    # s_new = seasonal_adjust(s_new)
    # return get_cycle(s_new, freq)
    return s_new, None


def get_cycle(s, freq):
    lamb_quarterly = 1600
    lamb_monthly = 129600
    lamb_annually = 6.25
    if freq == 'M':
        lamb = lamb_monthly
    elif freq == 'Q':
        lamb = lamb_quarterly
    elif freq == 'Y':
        lamb = lamb_annually
    else:
        raise ValueError
    cycle, trend = hpfilter(s, lamb=lamb)
    return cycle, trend


def cycle_yoy(s, freq):
    s = s.dropna().resample(freq).interpolate()
    return get_cycle(s, freq)


def chain_to_yoy(s, freq):
    # 环比值转换为同比
    s = s.dropna()
    s = (s+100)/100
    s = s.cumprod().resample(freq).interpolate()
    cycle, trend = get_cycle(s, freq)
    if freq == 'M':
        d = 12
    elif freq == 'Q':
        d = 4
    else:
        raise ValueError
    cycle_yoy = cycle.diff(d).dropna()
    return cycle_yoy, trend


def mean_jan_feb(s):
    # 将1、2月数值平滑为两者均值
    for year in list(set([d.year for d in s.index])):
        s_tmp = s[s.index.year == year]
        if s_tmp.index[0].month == 1 and len(s_tmp) >= 2:
            mean = s_tmp[:2].mean()
            s.loc[s_tmp.index[0]] = mean
            s.loc[s_tmp.index[1]] = mean
    return s


def cmt_to_yoy(s, freq):
    # 当月/季值转换为同比
    s = (s/s.shift(1)-1).dropna().resample(freq).interpolate()
    cycle, trend = chain_to_yoy(s*100, freq)
    return cycle, trend


def amt_to_yoy(s, freq):
    s = s.dropna().resample(freq).interpolate()
    chain_s = ((s/s.shift(1)) - 1).dropna()
    return chain_to_yoy(chain_s*100, freq)


def acu_to_cmt(s, freq):
    # 当月累计值转换为当月值
    s = s.dropna().resample(freq).interpolate()
    s_new = [None, ]
    datadates = s.index.tolist()
    for i, datadate in enumerate(datadates[1:]):
        if datadate.year == datadates[i].year:
            s_new.append(s.loc[datadate] - s.loc[datadates[i]])
        else:
            s_new.append(s.loc[datadate])
    s_new = pd.Series(s_new)
    s_new.index = s.index
    s_new = s_new.dropna()
    s_new = mean_jan_feb(s_new)
    return s_new


def acu_to_yoy(s, freq):
    # 当月累计值序列转换为同比序列
    # cmt = acu_to_cmt(s, freq)
    # cmt = seasonal_adjust(cmt)
    # cmt = (cmt.diff(12)/cmt.shift(12)).dropna().resample(freq).interpolate()
    # cycle, trend = get_cycle(cmt, freq)
    cmt = acu_to_cmt(s, freq)
    s_prod = cmt / cmt[0]
    s_drop_season = seasonal_adjust(s_prod).dropna().resample(freq).interpolate()
    cycle, trend = get_cycle(s_drop_season, freq)
    s_yoy = cycle.diff(12).dropna().resample(freq).interpolate()
    return s_yoy, trend


def exp_to_yoy(s, freq):
    # 扩散指数转换为同比数据
    s = (1 + (s - 50) / 100).dropna()
    s = s.cumprod().resample(freq).interpolate()
    s = (s.diff(12) / s.shift(12)).dropna()
    cycle, trend = get_cycle(s, freq)
    return cycle, trend

