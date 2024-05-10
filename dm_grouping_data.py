import os
import datetime as dt
import numpy as np
import pandas as pd
from copy import deepcopy
from fwd_model import dm_config as config


def is_valid(df, industry):
    columns = df.columns.tolist()
    col_str = columns[3]
    is_valid = all([industry.replace('业', '') in col_str.replace('业', ''),
                    '[报表类型] 合并报表' in col_str,
                    '[数据类型] 本外币' in col_str,
                    '[单位] 亿元' in col_str])
    return is_valid


def read_file(folder, data_type):
    col = '贷款余额(按行业)' if data_type == 'loan' else '不良贷款余额(按行业)'
    for _, __, ___ in os.walk(folder):
        file_list = ___
    loan_dict = {}
    for file_name in file_list:
        if '.xlsx' not in file_name or '~' in file_name:
            continue
        industry = file_name.split('.')[0]
        full_file_name = os.path.join(folder, file_name)
        df = pd.read_excel(full_file_name).dropna(how='all', axis=0)[:-1]
        if not is_valid(df, industry):
            print('文件表头要素可能有误：%s' % full_file_name)
        df.columns = [c.split('[报表类型]')[0].replace(col, '').replace('[报告期] ', '').replace('\r\n', '').replace('贷款余额_个人消费贷款', '').replace('\n', '')
                   for c in df.columns]
        del df['证券代码']
        df.set_index('证券简称', inplace=True)
        df = df.transpose()
        df.index = [dt.datetime.strptime(d, '%Y%m%d').date() for d in df.index]
        df.columns.name = 'bank'
        df.index.name = 'datadate'
        loan_dict[industry] = df.round(2)
    return loan_dict


def check_raw_dict(raw_dict, last_raw_dict):
    def check_diff(current_s, last_s):
        current_s.name = 'current'
        last_s.name = 'last'
        df = pd.DataFrame()
        now = dt.datetime.now().date()
        if now.month in [1, 2, 3, 4, 5, 6]:
            end_date = dt.date(now.year - 1, 12, 31)
        else:
            end_date = dt.date(now.year, 6, 30)
        df['current'] = current_s[current_s.index < end_date]
        df['last'] = last_s[last_s.index < end_date]
        # df.dropna(how='all', inplace=True)
        # if len(df) == 0:
        #     return pd.DataFrame()
        # 排除两期均为空值情况【1】后，当期值不等于上期值 &【2】，并且剔除最新一期数据 &【3】

        df_diff = df[((~df['last'].isna()) | (~df['current'].isna()))
                     & (df['last'] != df['current'])
                     & (df.index != df.index.values[-1])]
        return df_diff
    industries = list(last_raw_dict.keys())
    banks = last_raw_dict[industries[0]].columns.tolist()
    raw_diff_dict = {}
    bank_diff_dict = {}
    for industry in industries:
        bank_diff_dict = {}
        for bank in banks:
            current_s = raw_dict[industry][bank]
            last_s = last_raw_dict[industry][bank]
            df_diff = check_diff(current_s, last_s)
            if len(df_diff) > 0:
                bank_diff_dict[bank] = df_diff
        if len(bank_diff_dict) > 0:
            raw_diff_dict[industry] = bank_diff_dict
    return raw_diff_dict


def modify_data_dict(data_dict, diff_dict):
    modified_dict = deepcopy(data_dict)
    for industry, bank_diff_dict in diff_dict.items():
        for bank, df_diff in bank_diff_dict.items():
            for ix in df_diff.index:
                if np.isnan(df_diff.loc[ix, 'current']):
                    print(bank, df_diff.loc[ix])
                    modified_dict[industry].loc[ix, bank] = df_diff.loc[ix, 'last']
    return modified_dict


def make_npl_history(bng_date=None, end_date=None):
    folder_loan = config.public_banks_loan_folder
    folder_npl = config.public_banks_np_loan_folder
    last_npl_folder = config.last_npl_folder
    last_loan_folder = config.last_loan_folder

    loan_dict = read_file(folder_loan, 'loan')
    npl_dict = read_file(folder_npl, 'npl')
    last_npl_dict = read_file(last_npl_folder, 'npl')
    last_loan_dict = read_file(last_loan_folder, 'loan')

    # 将本期数据集中历史数据点与上期数据集中同一数据点进行比较，将本期与上期不一致的数据点修正为上期数据点：
    npl_diff_dict = check_raw_dict(npl_dict, last_npl_dict)
    loan_diff_dict = check_raw_dict(loan_dict, last_loan_dict)
    npl_dict, loan_dict = modify_data_dict(npl_dict, npl_diff_dict), modify_data_dict(loan_dict, loan_diff_dict)

    ind_nplr_dict = {}
    for ind in npl_dict.keys():
        df_loan_raw = loan_dict[ind]
        df_npl_raw = npl_dict[ind]

        if bng_date is not None and end_date is not None:
            df_loan_raw = df_loan_raw[(df_loan_raw.index >= dt.datetime.strptime(bng_date, '%Y%m%d').date()) &
                                      (df_loan_raw.index <= dt.datetime.strptime(end_date, '%Y%m%d').date())]
            df_npl_raw = df_npl_raw[(df_npl_raw.index >= dt.datetime.strptime(bng_date, '%Y%m%d').date()) &
                                    (df_npl_raw.index <= dt.datetime.strptime(end_date, '%Y%m%d').date())]

        df_loan = df_loan_raw.dropna(how='all', axis=1)
        df_npl = df_npl_raw.dropna(how='all', axis=1)

        npl_list, loan_list = [], []
        for report_date in df_npl.index:
            npl, loan = 0, 0
            if len(df_npl.loc[report_date][~df_npl.loc[report_date].isna()]) == 0:
                npl_list.append(np.nan)
                loan_list.append(np.nan)
            else:
                for bank in df_npl.columns:
                    if bank not in df_loan.columns:
                        continue
                    if not np.isnan(df_loan.loc[report_date, bank]):
                        loan += df_loan.loc[report_date, bank]
                        if not np.isnan(df_npl.loc[report_date, bank]):
                            npl += df_npl.loc[report_date, bank]
                npl_list.append(npl)
                loan_list.append(loan)

        df_ind_nplr = pd.DataFrame(np.c_[npl_list, loan_list],
                                   index=df_npl.index,
                                   columns=['npl', 'loan'])
        df_ind_nplr['nplr'] = df_ind_nplr['npl'] / df_ind_nplr['loan']
        df_ind_nplr.dropna(subset=['nplr'], inplace=True)
        ind_nplr_dict[ind] = df_ind_nplr

    df_nplr = pd.DataFrame()
    df_total_loan, df_total_npl = pd.DataFrame(), pd.DataFrame()
    columns = []
    for ind, df_ind in ind_nplr_dict.items():
        columns.append(ind)
        df_nplr = df_nplr.merge(df_ind['nplr'], how='outer', left_index=True, right_index=True)
        df_nplr.columns = columns

        df_ind = df_ind.rename(columns={'loan': '%s_loan' % ind, 'npl': '%s_npl' % ind})
        df_total_loan = df_total_loan.merge(df_ind['%s_loan' % ind], how='outer', left_index=True, right_index=True)
        df_total_npl = df_total_npl.merge(df_ind['%s_npl' % ind], how='outer', left_index=True, right_index=True)
    s_total_loan = df_total_loan.sum(axis=1)
    s_total_npl = df_total_npl.sum(axis=1)
    s_total_nplr = s_total_npl / s_total_loan
    s_total_nplr.name = 'All'
    df_nplr = df_nplr.merge(s_total_nplr, how='outer', left_index=True, right_index=True)
    df_nplr.interpolate(inplace=True)
    return df_nplr
