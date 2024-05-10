import numpy as np
import pandas as pd
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import datetime as dt
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as figureCanvas
from dm_grouping import dmRiskGrouping, read_local_arima_config
from fwd_model.qtm_table import QtmTable
from fwd_model import dm_config as config
from fwd_model.utils import set_frame, set_table
from fwd_model.dm_grouping_data import read_file

plt.rcParams['font.sans-serif'] = ['SimHei']


class guiGrouping(QWidget):
	def __init__(self):
		super().__init__()
		self.dm_fwd = None
		self.loan_dict, self.npl_dict = None, None
		self.df_current_ind = pd.DataFrame()
		self.dm_grouping = None

		self.spin_min_npl, self.spin_max_npl = QDoubleSpinBox(), QDoubleSpinBox()
		self.ck_search_arima_param = QCheckBox('搜索最优ARIMA参数')
		self.bt_import_arima_param = QPushButton('导入ARIMA最优参数')
		self.bt_run_grouping = QPushButton('运行分组模型')
		self.bt_export_grouping = QPushButton('导出分组数据')
		self.bt_export_arima_param = QPushButton('导出ARIMA最优参数')
		self.bt_export_nplr_history = QPushButton('导出NPLR历史序列')
		self.bt_read_raw = QPushButton('读取原始数据')
		self.bt_export_raw = QPushButton('导出原始数据')
		self.bt_plot_curve = QPushButton('绘制违约曲线集')
		self.bt_export_proj_res = QPushButton('导出拟合信息')
		self.bt_export_proj_ts = QPushButton('导出拟合序列')

		self.table_nplr_history = QTableView()
		self.table_nplr_proj = QTableView()
		self.table_nplr_proj_stats = QTableView()

		self.combo_ind_nplr = QComboBox()
		self.label_wtd_nplr = QLabel()
		self.bt_export_ind_nplr = QPushButton('导出')
		self.table_ind_nplr = QTableView()
		self.nplr_stat_tab = QTabWidget()

		self.combo_distribution = QComboBox()
		self.line_para_min = QLineEdit()
		self.line_para_max = QLineEdit()
		self.line_para_step = QLineEdit()
		self.line_current_para = QLineEdit()

		self.line_deg_min = QLineEdit()
		self.line_deg_max = QLineEdit()
		self.line_deg_step = QLineEdit()
		self.line_deg_range = QLineEdit()

		self.combo_ind_raw = QComboBox()
		self.table_loan, self.table_npl = QTableView(), QTableView()

		self.tab = QTabWidget()

		self.figure = plt.figure(tight_layout=True)
		self.canvas = figureCanvas(self.figure)
		self.figure_curve = plt.figure(tight_layout=True)
		self.canvas_curve = figureCanvas(self.figure_curve)
		self.figure_ts = plt.figure(tight_layout=True)
		self.canvas_ts = figureCanvas(self.figure_ts)
		self.figure_deg = plt.figure(tight_layout=True)
		self.canvas_deg = figureCanvas(self.figure_deg)
		self.table_deg = QTableView()
		self.figure_dis = plt.figure(tight_layout=True)
		self.canvas_dis = figureCanvas(self.figure_dis)
		self.table_dis = QTableView()
		self.widget_setting()
		self.layout()
		return

	def widget_setting(self):
		self.spin_min_npl.setDecimals(6)
		self.spin_max_npl.setDecimals(6)
		self.spin_min_npl.setValue(config.min_npl * 100)
		self.spin_max_npl.setValue(config.max_npl * 100)
		self.spin_min_npl.setSuffix('%')
		self.spin_max_npl.setSuffix('%')
		self.spin_max_npl.setSingleStep(0.01)
		self.spin_min_npl.setSingleStep(0.01)
		self.combo_distribution.addItems(config.distributions)
		self.update_distribution_parameters()
		self.combo_distribution.currentIndexChanged.connect(self.update_distribution_parameters)
		self.ck_search_arima_param.setChecked(False)
		self.ck_search_arima_param.stateChanged.connect(self.change_import_arima_param_bt_state)
		self.bt_import_arima_param.clicked.connect(self.import_arima_param)
		self.bt_run_grouping.clicked.connect(self.run_grouping)
		self.bt_read_raw.clicked.connect(self.load_ind_data)
		self.bt_export_arima_param.clicked.connect(self.export_arima_params)
		self.bt_export_nplr_history.clicked.connect(self.export_nplr_history)
		self.bt_plot_curve.clicked.connect(self.plot_curve)
		self.combo_ind_nplr.currentIndexChanged.connect(self.update_ind_nplr)
		self.combo_ind_nplr.currentIndexChanged.connect(self.highlight_stat)
		self.combo_ind_raw.currentIndexChanged.connect(self.update_ind_table)
		self.bt_export_grouping.clicked.connect(self.export_nplr)
		self.bt_export_raw.clicked.connect(self.export_raw)
		self.line_current_para.setEnabled(False)
		self.line_deg_range.setEnabled(False)
		self.table_nplr_proj.clicked.connect(self.update_combo_nplr)
		self.table_nplr_proj_stats.clicked.connect(self.update_combo_nplr)
		self.bt_export_ind_nplr.clicked.connect(self.export_ind_nplr)
		self.bt_export_proj_res.clicked.connect(self.export_proj_res)
		self.bt_export_proj_ts.clicked.connect(self.export_proj_ts)

		return

	def update_widget(self, dm):
		self.dm_grouping = dmRiskGrouping(bng_date=dm.dmData.bng_date, end_date=dm.dmData.end_date)
		self.dm_grouping.forward_looking_factors = dm.prediction_info['forward_factors']
		return

	def update_distribution_parameters(self):
		min_v = config.distribution_parameter_range[self.combo_distribution.currentText()].min()
		max_v = config.distribution_parameter_range[self.combo_distribution.currentText()].max()
		step = config.para_step_dict[self.combo_distribution.currentText()]
		min_deg = config.deg_range.min()
		max_deg = config.deg_range.max()
		deg_step = config.deg_step
		self.line_para_min.setText('{:.4f}'.format(min_v))
		self.line_para_max.setText('{:.4f}'.format(max_v))
		self.line_para_step.setText('{:.6f}'.format(step))
		self.line_deg_min.setText('{:.4f}'.format(min_deg))
		self.line_deg_max.setText('{:.4f}'.format(max_deg))
		self.line_deg_step.setText('{:.6f}'.format(deg_step))
		return

	def update_model_parameters(self):
		self.dm_grouping.min_npl = self.spin_min_npl.value() * 0.01
		self.dm_grouping.max_npl = self.spin_max_npl.value() * 0.01
		self.dm_grouping.distribution = self.combo_distribution.currentText()
		self.dm_grouping.distribution_parameter_range = self.update_distribution_parameter_range()
		self.dm_grouping.n_deg_range = self.update_deg_range()
		return

	def update_distribution_parameter_range(self):
		min_v = float(self.line_para_min.text())
		max_v = float(self.line_para_max.text())
		step = float(self.line_para_step.text())
		return np.arange(min_v, max_v, step)

	def update_deg_range(self):
		min_v = float(self.line_deg_min.text())
		max_v = float(self.line_deg_max.text())
		step = float(self.line_deg_step.text())
		return np.arange(min_v, max_v, step)

	def import_arima_param(self):
		file_name = QFileDialog.getOpenFileName()[0]
		if not file_name:
			return
		self.dm_grouping.local_arima_config = read_local_arima_config(file_name)
		return

	def run_grouping(self):
		self.update_model_parameters()
		param_search = True if self.ck_search_arima_param.isChecked() else False
		self.dm_grouping.run_grouping(param_search=param_search)
		self.update_ind_nplr_combo()
		self.update_ind_projs()
		self.update_ind_proj_stats()
		self.update_npl_history()
		self.highlight_stat()
		self.line_current_para.setText(str(self.dm_grouping.distribution_factor.round(6)))
		return

	def update_ind_nplr_combo(self):
		self.combo_ind_nplr.clear()
		industries = self.dm_grouping.df_ind_weights.columns.tolist()
		self.combo_ind_nplr.addItems(list(reversed(industries)))
		self.update_ind_nplr()
		return

	def update_ind_nplr(self):
		ind = self.combo_ind_nplr.currentText()
		if len(ind) == 0:
			return
		df_ind = self.dm_grouping.get_ind_data(ind).reset_index()
		self.df_current_ind = df_ind
		headers = ['INTL_RATING', 'Positive', 'Normal', 'Negative', 'LAST', 'IssuerWeights', 'DIFF_NORMAL']
		format_dict = {c: '{:.4%}' for c in df_ind.columns}
		model = QtmTable(df_ind, header_labels=headers, format_dict=format_dict)
		set_table(self.table_ind_nplr, model)

		wtd_nplr = (df_ind['normal'] * df_ind['weights']).sum()
		self.update_wtd_nplr(wtd_nplr)
		self.plot_ind_nplr()
		self.plot_ind_ts()
		self.plot_ind_deg()
		self.update_deg_table()
		self.plot_ind_dis()
		self.update_dis_table()
		return

	def plot_ind_nplr(self):
		self.figure.clear()
		df_ind = self.df_current_ind
		if len(df_ind) == 0:
			return
		ax = self.figure.add_subplot(121)
		ax_1 = self.figure.add_subplot(122)
		df_ind['normal'].plot(style='.-', ax=ax, legend=True, label='NPL: Normal', sharex=True, grid=True)
		df_ind['positive'].plot(style='-', ax=ax, legend=True, label='NPL: Positive', sharex=True, grid=True, color='b')
		df_ind['negative'].plot(style='-', ax=ax, legend=True, label='NPL: Negative', sharex=True, grid=True, color='r')
		df_ind['last'].plot(style='-.', ax=ax, legend=True, label='NPL: LAST', sharex=True, grid=True, color='grey')
		df_ind['weights'].plot(kind='bar', ax=ax_1, legend=True, label='IssuerWeights', grid=True)
		ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.2%}'))
		ax_1.yaxis.set_major_formatter(StrMethodFormatter('{x:.2%}'))
		self.canvas.draw()
		return

	def plot_ind_ts(self):
		self.figure_ts.clear()
		ind = self.combo_ind_nplr.currentText()
		ax = self.figure_ts.add_subplot(111)
		self.dm_grouping.plot_ind_ts(ind, ax)
		ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.2%}'))
		self.canvas_ts.draw()
		return

	def plot_ind_deg(self):
		self.figure_deg.clear()
		ind = self.combo_ind_nplr.currentText()
		if ind == 'All':
			self.canvas_deg.draw()
			return
		ax = self.figure_deg.add_subplot(111)
		self.dm_grouping.plot_ind_deg(ind, ax)
		ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.2%}'))
		self.canvas_deg.draw()
		return

	def update_deg_table(self):
		ind = self.combo_ind_nplr.currentText()
		if len(ind) == 0 or ind == 'All':
			df = pd.DataFrame()
			model = QtmTable(df)
		else:
			df = self.dm_grouping.ind_search_deg_dict[ind].reset_index()
			format_dict = {'ind_npl': '{:.4%}', 'wtd_npl': '{:.4%}', 'n_deg': '{:.4f}'}
			headers = ['阶数', '目标NPL', '搜索NPL']
			model = QtmTable(df, format_dict=format_dict, header_labels=headers)
		set_table(self.table_deg, model)
		return

	def plot_ind_dis(self):
		self.figure_dis.clear()
		ind = self.combo_ind_nplr.currentText()
		if ind not in self.dm_grouping.ind_search_dis_dict.keys() or len(ind) == 0:
			self.canvas_dis.draw()
			return
		ax = self.figure_dis.add_subplot(111)
		self.dm_grouping.plot_ind_dis(ind, ax)
		ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.2%}'))
		self.canvas_dis.draw()
		return

	def update_dis_table(self):
		ind = self.combo_ind_nplr.currentText()
		if ind not in self.dm_grouping.ind_search_dis_dict.keys() or len(ind) == 0:
			df = pd.DataFrame()
			model = QtmTable(df)
		else:
			df = self.dm_grouping.ind_search_dis_dict[ind].reset_index()
			format_dict = {'ind_npl': '{:.4%}', 'wtd_npl': '{:.4%}', 'DistributionPara': '{:.4f}'}
			headers = ['分布参数', '目标NPL', '搜索NPL']
			model = QtmTable(df, header_labels=headers, format_dict=format_dict)
		set_table(self.table_dis, model)
		return

	def update_wtd_nplr(self, wtd_nplr):
		val = '{:.4%}'.format(wtd_nplr)
		self.label_wtd_nplr.setText('加权平均NPLR(Normal):%s' % val)
		return

	def highlight_stat(self):
		ind = self.combo_ind_nplr.currentText()
		if len(ind) == 0:
			return
		if ind == 'All':
			return
		df = self.dm_grouping.df_ind_npls
		row = df.index.tolist().index(ind)
		self.table_nplr_proj.selectRow(row)
		return

	def update_combo_nplr(self):
		row = self.table_nplr_proj.currentIndex().row()
		ind = self.dm_grouping.df_ind_npls.index.tolist()[row]
		industries = list(reversed(self.dm_grouping.df_ind_weights.columns.tolist()))
		ix = industries.index(ind)
		self.combo_ind_nplr.setCurrentIndex(ix)
		return

	def update_ind_projs(self):
		df = self.dm_grouping.df_proj.reset_index()
		format_dict = {'last_value': '{:.4%}', 'applied_prediction': '{:.4%}', 'prediction_normal': '{:.4%}', 'prediction_positive': '{:.4%}',
					'prediction_negative': '{:.4%}', }
		model = QtmTable(df, format_dict=format_dict)
		set_table(self.table_nplr_proj, model)
		return

	def update_ind_proj_stats(self):
		df = self.dm_grouping.df_stats.reset_index()
		format_dict = {'r2': '{:.2%}', 'auto_reg_r2': '{:.2%}', 'R2': '{:.2%}',
					'R2_auto': '{:.2%}', 'mse': '{:.6f}', 'mse_auto': '{:.6f}'}
		model = QtmTable(df, format_dict=format_dict)
		set_table(self.table_nplr_proj_stats, model)
		return

	def update_npl_history(self):
		df = self.dm_grouping.df_npl_history.reset_index()
		format_dict = {ind: '{:.4%}' for ind in df.columns if ind != 'datadate'}
		model = QtmTable(df, format_dict=format_dict)
		set_table(self.table_nplr_history, model)
		return

	def plot_curve(self):
		deg_min, deg_max = float(self.line_deg_min.text()), float(self.line_deg_max.text())
		step = 0.01
		df_bound, s_curve = self.dm_grouping.get_possible_curve(min_deg=deg_min, max_deg=deg_max, step=step)

		self.figure_curve.clear()
		ax = self.figure_curve.add_subplot(121)
		ax_1 = self.figure_curve.add_subplot(122)
		df_bound.plot(ax=ax, grid=True, legend=False, title='NPLR分布范围')
		s_curve.plot(ax=ax_1, grid=True, legend=False, title='NPLR边界')
		ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.2%}'))
		ax_1.yaxis.set_major_formatter(StrMethodFormatter('{x:.2%}'))
		self.canvas_curve.draw()
		self.nplr_stat_tab.setCurrentIndex(1)

		min_deg, max_deg = s_curve.min(), s_curve.max()
		range_info = '%s-%s' % ('{:.4%}'.format(min_deg), '{:.4%}'.format(max_deg))
		self.line_deg_range.setText(range_info)
		return

	def load_ind_data(self):
		folder_loan = config.public_banks_loan_folder
		folder_npl = config.public_banks_np_loan_folder
		self.loan_dict = read_file(folder_loan, 'loan')
		self.npl_dict = read_file(folder_npl, 'npl')
		self.combo_ind_raw.clear()
		self.combo_ind_raw.addItems(list(self.loan_dict.keys()))
		self.update_ind_table()
		return

	def update_ind_table(self):
		ind = self.combo_ind_raw.currentText()
		df_loan = self.loan_dict[ind]
		df_npl = self.npl_dict[ind]
		format_dict = {c: '{:,.2f}' for c in df_loan.columns}
		model_loan, model_npl = QtmTable(df_loan.reset_index(), format_dict=format_dict),\
							QtmTable(df_npl.reset_index(), format_dict=format_dict)
		set_table(self.table_loan, model_loan)
		set_table(self.table_npl, model_npl)
		return

	def export_nplr(self):
		df = self.dm_grouping.df_ind_mapping
		end_date = self.dm_grouping.end_date
		if len(df) == 0:
			return
		time_stp = dt.datetime.now().strftime('%Y%m%d%H%M%S')
		file_name = 'NPLR_PROJECTION_%s_%s.xlsx' % (end_date, time_stp)
		file_dlg = QFileDialog.getSaveFileName(None, '导出Excel', file_name, 'Excel files (*.xlsx)')[0]
		if file_dlg:
			df.to_excel(file_dlg, index=False)
		return

	def export_arima_params(self):
		df = self.dm_grouping.df_ind_npls.reset_index()[['industry', 'arima_params']]
		end_date = self.dm_grouping.end_date
		if len(df) == 0:
			return
		time_stp = dt.datetime.now().strftime('%Y%m%d%H%M%S')
		file_name = 'ARIMA_PARAMS_%s_%s' % (end_date, time_stp)
		file_dlg = QFileDialog.getSaveFileName(None, '导出EXCEL', file_name, 'Excel files (*.xlsx)')[0]
		if file_dlg:
			df.to_excel(file_dlg, index=False)
		return

	def export_ind_nplr(self):
		df = self.df_current_ind
		ind = self.combo_ind_nplr.currentText()
		if len(df) == 0:
			return
		time_stp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
		file_name = '%s_NPLR_PROJECTION_%s.xlsx' % (ind, time_stp)
		file_dlg = QFileDialog.getSaveFileName(None, '导出Excel', file_name, 'Excel files (*.xlsx)')[0]
		if file_dlg:
			df.to_excel(file_dlg, index=False)
		return

	def export_nplr_history(self):
		df = self.dm_grouping.df_npl_history.reset_index()
		if len(df) == 0:
			return
		time_stp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
		file_name = 'NPLR_HISTORY_%s.xlsx' % time_stp
		file_dlg = QFileDialog.getSaveFileName(None, '导出Excel', file_name, 'Excel files (*.xlsx)')[0]
		if file_dlg:
			df.to_excel(file_dlg, index=False)
		return

	def export_proj_res(self):
		df = self.dm_grouping.df_ind_npls.reset_index()
		time_stp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
		file_name = 'INF_REG_INFO%s.xlsx' % time_stp
		file_dlg = QFileDialog.getSaveFileName(None, '导出Excel', file_name, 'Excel files (*.xlsx)')[0]
		if file_dlg:
			df.to_excel(file_dlg, index=False)
		return

	def export_proj_ts(self):
		df_his = self.dm_grouping.df_npl_history
		dlg = QFileDialog()
		dlg.setFileMode(QFileDialog.Directory)
		file_addr = dlg.getExistingDirectory()
		if len(file_addr) == 0:
			return
		datetime = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
		for ind in df_his.columns:
			if ind == 'All':
				continue
			ts_his = df_his[ind]
			ts_fit = self.dm_grouping.ind_fit_dict[ind]
			df_ = pd.DataFrame([ts_fit, ts_his]).transpose().dropna()
			df_.columns = ['fit', 'act']
			file_name = 'FIT_RES_%s_%s.xlsx' % (ind, datetime)
			file_full_name = os.path.join(file_addr, file_name)
			df_.to_excel(file_full_name)
		return

	def export_raw(self):
		ind = self.combo_ind_raw.currentText()
		df_loan = self.loan_dict[ind].reset_index()
		df_npl = self.npl_dict[ind].reset_index()
		time_stp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
		file_name_loan = '贷款余额-%s-%s.xlsx' % (ind, time_stp)
		file_name_npl = '不良贷款余额-%s-%s.xlsx' % (ind, time_stp)

		dlg = QFileDialog()
		dlg.setFileMode(QFileDialog.Directory)
		file_addr = dlg.getExistingDirectory()
		if len(file_addr) == 0:
			return
		for df, file_name in zip([df_loan, df_npl], [file_name_loan, file_name_npl]):
			file = os.path.join(file_addr, file_name)
			df.to_excel(file, index=False)
		return

	def change_import_arima_param_bt_state(self):
		if self.ck_search_arima_param.isChecked():
			self.bt_import_arima_param.setEnabled(False)
		else:
			self.bt_import_arima_param.setEnabled(True)
		return

	def layout(self):
		dist_frame, dist_layout = QFrame(), QFormLayout()
		dist_layout.addRow(QLabel('H1 NPLR：'), self.spin_min_npl)
		dist_layout.addRow(QLabel('H10 NPLR：'), self.spin_max_npl)
		dist_layout.addRow('假设分布：', self.combo_distribution)
		dist_layout.addRow('分布参数(Min)：', self.line_para_min)
		dist_layout.addRow('分布参数(Max)：', self.line_para_max)
		dist_layout.addRow('分布搜索步长：', self.line_para_step)
		dist_layout.addRow('分布参数：', self.line_current_para)
		dist_layout.addRow('曲线参数(Min)：', self.line_deg_min)
		dist_layout.addRow('曲线参数(Max)：', self.line_deg_max)
		dist_layout.addRow('曲线搜索步长：', self.line_deg_step)
		dist_layout.addRow('NPLR均值边界：', self.line_deg_range)
		set_frame(dist_frame, dist_layout)
		dist_frame.setFixedWidth(280)

		bt_frame, bt_layout = QFrame(), QVBoxLayout()
		bt_layout.addWidget(self.ck_search_arima_param)
		bt_layout.addWidget(self.bt_import_arima_param)
		bt_layout.addWidget(self.bt_run_grouping)
		bt_layout.addWidget(self.bt_export_grouping)
		bt_layout.addWidget(self.bt_export_arima_param)
		bt_layout.addWidget(self.bt_export_nplr_history)
		bt_layout.addWidget(self.bt_read_raw)
		bt_layout.addWidget(self.bt_export_raw)
		bt_layout.addWidget(self.bt_plot_curve)
		bt_layout.addWidget(self.bt_export_proj_res)
		bt_layout.addWidget(self.bt_export_proj_ts)
		set_frame(bt_frame, bt_layout)

		option_right_frame, option_right_layout = QFrame(), QVBoxLayout()
		option_right_layout.addWidget(bt_frame)
		option_right_layout.addWidget(QFrame())
		set_frame(option_right_frame, option_right_layout)

		option_frame, option_layout = QFrame(), QHBoxLayout()
		option_layout.addWidget(dist_frame)
		option_layout.addWidget(option_right_frame)
		set_frame(option_frame, option_layout)
		option_frame.setFixedWidth(500)

		self.nplr_stat_tab.addTab(self.table_nplr_proj, '预测结果')
		self.nplr_stat_tab.addTab(self.table_nplr_proj_stats, '稳健性指标')
		self.nplr_stat_tab.addTab(self.canvas_curve, '边界分析')
		self.nplr_stat_tab.addTab(self.table_nplr_history, '历史数据')

		nplr_top_frame, nplr_top_layout = QFrame(), QHBoxLayout()
		nplr_top_layout.addWidget(option_frame)
		nplr_top_layout.addWidget(self.nplr_stat_tab)
		set_frame(nplr_top_frame, nplr_top_layout)
		nplr_top_frame.setMaximumHeight(450)

		nplr_combo_frame, nplr_combo_layout = QFrame(), QHBoxLayout()
		nplr_combo_layout.addWidget(QLabel('选择行业：'))
		nplr_combo_layout.addWidget(self.combo_ind_nplr)
		nplr_combo_layout.addWidget(self.label_wtd_nplr)
		nplr_combo_layout.addWidget(self.bt_export_ind_nplr)
		set_frame(nplr_combo_frame, nplr_combo_layout)

		nplr_table_frame, nplr_table_layout = QFrame(), QVBoxLayout()
		nplr_table_layout.addWidget(nplr_combo_frame)
		nplr_table_layout.addWidget(self.table_ind_nplr)
		set_frame(nplr_table_frame, nplr_table_layout)
		nplr_table_frame.setMaximumWidth(550)

		canvas_tab = QTabWidget()
		canvas_tab.addTab(self.canvas, 'NPLR曲线')
		canvas_tab.addTab(self.canvas_ts, 'NPLR时间序列')
		deg_frame, deg_layout = QFrame(), QHBoxLayout()
		deg_layout.addWidget(self.canvas_deg)
		deg_layout.addWidget(self.table_deg)
		set_frame(deg_frame, deg_layout)
		canvas_tab.addTab(deg_frame, '曲线搜索')
		dis_frame, dis_layout = QFrame(), QHBoxLayout()
		dis_layout.addWidget(self.canvas_dis)
		dis_layout.addWidget(self.table_dis)
		set_frame(dis_frame, dis_layout)
		canvas_tab.addTab(dis_frame, '分布搜索')

		nplr_bottom_frame, nplr_bottom_layout = QFrame(), QHBoxLayout()
		nplr_bottom_layout.addWidget(nplr_table_frame)
		nplr_bottom_layout.addWidget(canvas_tab)
		set_frame(nplr_bottom_frame, nplr_bottom_layout)
		nplr_bottom_frame.setMaximumHeight(350)

		nplr_splitter = QSplitter(Qt.Vertical)
		nplr_splitter.addWidget(nplr_top_frame)
		nplr_splitter.addWidget(nplr_bottom_frame)

		raw_loan_frame, raw_loan_layout = QFrame(), QVBoxLayout()
		raw_loan_layout.addWidget(self.combo_ind_raw)
		raw_loan_layout.addWidget(self.table_loan)
		set_frame(raw_loan_frame, raw_loan_layout)

		raw_splitter = QSplitter(Qt.Horizontal)
		raw_splitter.addWidget(raw_loan_frame)
		raw_splitter.addWidget(self.table_npl)

		self.tab.addTab(nplr_splitter, '分组预测')
		self.tab.addTab(raw_splitter, '原始数据')

		main_layout = QVBoxLayout()
		main_layout.addWidget(self.tab)
		self.setLayout(main_layout)
		return
