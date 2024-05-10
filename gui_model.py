import multiprocessing

import pandas as pd
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as figureCanvas
import fwd_model.dm_config as config
from fwd_model import dm_models
from fwd_model.qtm_table import QtmTable
from fwd_model.utils import set_frame, set_table

plt.rcParams['font.sans-serif'] = ['SimHei']


class guiModel(QWidget):
	signal_forecast_made = pyqtSignal(name='forecastMade')

	def __init__(self):
		super().__init__()
		self.dm = None
		self.all_exogs = []
		self.preset_exogs = []
		self.ck_load_variable_preset = QCheckBox('预设最优变量')
		self.list_var = QListWidget()
		self.combo_var_num = QComboBox()
		self.spin_var_num = QSpinBox()
		self.spin_cpu = QSpinBox()
		self.combo_models = QComboBox()
		self.progress = QProgressBar()
		self.bt_run = QPushButton('运行')

		self.label_model_count = QLabel()
		self.combo_top_score = QComboBox()
		self.line_select_model = QLineEdit()
		self.spin_test_size = QSpinBox()
		self.bt_forecast = QPushButton('执行预测')
		self.bt_backtest = QPushButton('执行回测')
		self.bt_export_proj = QPushButton('导出拟合序列')
		self.bt_export_backtest = QPushButton('导出学习曲线')
		self.bt_export_candidates = QPushButton('导出候选模型')

		self.model_tab = QTabWidget()
		self.line_positive, self.line_normal, self.line_negative = QLineEdit(), QLineEdit(), QLineEdit()
		self.table_score = QTableView()
		self.table_forecast = QTableView()
		self.figure_forecast = plt.figure(tight_layout=True)
		self.canvas_forecast = figureCanvas(self.figure_forecast)
		self.figure_learning_curve = plt.figure(tight_layout=True)
		self.canvas_learning_curve = figureCanvas(self.figure_learning_curve)

		self.widget_setting()
		self.layout()
		return

	def widget_setting(self):
		self.ck_load_variable_preset.setChecked(True)
		self.combo_var_num.addItems(['固定值', '至少'])
		self.spin_var_num.setValue(1)
		self.spin_var_num.setSingleStep(1)
		cpu_count = multiprocessing.cpu_count()
		self.spin_cpu.setValue(max(cpu_count-2, 1))
		self.spin_cpu.setSingleStep(1)
		self.spin_cpu.setRange(1, mp.cpu_count())
		self.combo_models.addItems(config.available_models)

		self.combo_top_score.addItems(['10', '20', '50', '-50'])
		self.combo_top_score.currentIndexChanged.connect(self.update_score_table)
		self.bt_run.clicked.connect(self.run_model)

		self.spin_test_size.setValue(20)
		self.spin_test_size.setSingleStep(5)
		self.spin_test_size.setRange(10, 50)
		self.spin_test_size.setSuffix('%')

		self.table_score.doubleClicked.connect(self.get_model_id_from_table)
		self.bt_forecast.setEnabled(False)
		self.bt_backtest.setEnabled(False)
		self.bt_forecast.clicked.connect(lambda: self.run_forecast(plot_learning_curve=False))
		self.bt_backtest.clicked.connect(lambda: self.run_forecast(plot_learning_curve=True))

		self.line_positive.setEnabled(False)
		self.line_negative.setEnabled(False)
		self.line_normal.setEnabled(False)
		self.ck_load_variable_preset.stateChanged.connect(self.load_variable_presets)

		self.bt_export_proj.clicked.connect(self.export_forecast)
		self.bt_export_backtest.clicked.connect(self.export_backtest)
		self.bt_export_candidates.clicked.connect(self.export_candidates)
		return

	def update_widget(self, dm):
		self.dm = dm
		self.all_exogs = self.dm.all_exogs
		self.spin_var_num.setRange(1, len(self.all_exogs))
		self.load_variable_presets()
		self.update_var_list()
		return

	def load_variable_presets(self):
		if self.ck_load_variable_preset.isChecked():
			self.preset_exogs = pd.read_excel(config.ridge_variable_config_file)['variable'].tolist()
			print(self.preset_exogs)
			self.spin_var_num.setValue(len(self.preset_exogs))
			self.combo_var_num.setCurrentIndex(0)
		else:
			self.spin_var_num.setValue(5)
			self.preset_exogs = []
		self.update_var_list()
		return

	def update_var_list(self):
		self.list_var.clear()
		for ix, exog in enumerate(self.all_exogs):
			ck = QListWidgetItem(exog)
			if exog == 'npl_lag':
				ck.setCheckState(Qt.Checked)
			elif exog in self.preset_exogs:
				ck.setCheckState(Qt.Checked)
			else:
				ck.setCheckState(Qt.Unchecked)
			self.list_var.addItem(ck)
		return

	def get_essential_exogs(self):
		list_ = []
		for ix, exog in enumerate(self.all_exogs):
			if self.list_var.item(ix).checkState() == 2:
				list_.append(exog)
		return list_

	def run_model(self):
		self.build_constructor()
		self.update_model_count()
		self.update_score_table()
		self.bt_forecast.setEnabled(True)
		self.bt_backtest.setEnabled(True)
		return

	def build_constructor(self):
		if self.combo_var_num.currentText() == '至少':
			min_or_fixed = 'min'
		else:
			min_or_fixed = 'fixed'
		exog_num = self.spin_var_num.value()
		essential_exogs = self.get_essential_exogs()
		model = self.combo_models.currentText()
		cpu_num = self.spin_cpu.value()

		self.dm.build_model(model=model, exog_num=exog_num, min_or_fixed=min_or_fixed, essential_exogs=essential_exogs,
							cpu_count=cpu_num)
		return

	def update_model_count(self):
		model_count = len(self.dm.model_constructor.df_score)
		info = '共%s个模型，显示前N个：' % '{:,.0f}'.format(model_count)
		self.label_model_count.setText(info)
		top_id = self.dm.model_constructor.df_score.model_id[0]
		self.line_select_model.setText(str(top_id))
		return

	def update_score_table(self):
		top_n = int(self.combo_top_score.currentText())
		columns = ['model_id', 'exogs', 'exog_count', 'r2_train', 'mse_train', 'r2_test', 'mse_test']
		if top_n > 0:
			df = self.dm.model_constructor.df_score[:top_n][columns]
		else:
			df = self.dm.model_constructor.df_score[top_n:][columns]
		format_dict = {'r2_train': '{:.2f}', 'r2_test': '{:2f}', 'mse_train': '{:.6f}', 'mse_test': '{:.6f}'}

		model = QtmTable(df, format_dict=format_dict)
		self.table_score.setModel(model)
		self.table_score.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
		self.table_score.verticalHeader().setVisible(False)
		self.table_score.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
		return

	def get_model_id_from_table(self):
		row = self.table_score.currentIndex().row()
		model_id = self.dm.model_constructor.df_score.loc[row, 'model_id']
		self.line_select_model.setText(str(model_id))
		return

	def run_forecast(self, plot_learning_curve=False):
		model_id = int(self.line_select_model.text())
		self.dm.make_prediction(model_id=model_id, plot=plot_learning_curve,
								fig=self.figure_learning_curve)
		self.update_forecast_table()
		self.plot_forecast()
		if plot_learning_curve:
			self.canvas_learning_curve.draw()
		self.update_factors()
		self.signal_forecast_made.emit()
		return

	def update_forecast_table(self):
		self.model_tab.setCurrentIndex(1)
		df_forecast = self.dm.prediction_info['data'].reset_index()
		df_forecast['index'] = [d.strftime('%Y-%m-%d') for d in df_forecast['index']]
		df_forecast.rename(columns={'index': 'datadate'}, inplace=True)
		format_dict = {'actual': '{:.4f}', 'predict': '{:.4f}',
					'1_STD': '{:.4f}', '-1_STD': '{:.4f}',
					'2_STD': '{:.4f}', '-2_STD': '{:.4f}',
					'-1.645_STD': '{:.4f}', '1.036_STD': '{:.4f}'}
		model = QtmTable(df_forecast, format_dict=format_dict)
		set_table(self.table_forecast, model)
		return

	def export_forecast(self):
		df_forecast = self.dm.prediction_info['data'].reset_index().dropna()
		file_name = 'model_fit_result.xlsx'
		file_dlg = QFileDialog.getSaveFileName(None, '导出EXCEL', file_name, 'Excel files (*.xlsx)')[0]
		if file_dlg:
			df_forecast.to_excel(file_dlg, index=False)
		return

	def export_backtest(self):
		df = self.dm.df_learning_curve.reset_index()
		file_name = 'learning_curve.xlsx'
		file_dlg = QFileDialog.getSaveFileName(None, '导出EXCEL', file_name, 'Excel files (*.xlsx)')[0]
		if file_dlg:
			df.to_excel(file_dlg, index=False)
		return

	def export_candidates(self):
		columns = ['model_id', 'exogs', 'exog_count', 'r2_train', 'mse_train', 'r2_test', 'mse_test']
		df = self.dm.model_constructor.df_score[columns]
		file_name = 'fwd_model_candidates.xlsx'
		file_dlg = QFileDialog.getSaveFileName(None, '导出EXCEL', file_name, 'Excel files (*.xlsx)')[0]
		if file_dlg:
			df.to_excel(file_dlg, index=False)
		return

	def plot_forecast(self):
		df_forecast = self.dm.prediction_info['data']
		dm_models.plot_predict(df_forecast, fig=self.figure_forecast)
		self.canvas_forecast.draw()
		return

	def update_factors(self):
		factor_dict = self.dm.prediction_info['forward_factors']
		normal = '{:.4%}'.format(factor_dict['normal'])
		positive = '{:.4%}'.format(factor_dict['positive'])
		negative = '{:.4%}'.format(factor_dict['negative'])
		self.line_normal.setText(str(normal))
		self.line_positive.setText(str(positive))
		self.line_negative.setText(str(negative))
		return

	def layout(self):
		setting_frame, setting_layout = QFrame(), QFormLayout()
		setting_layout.addRow('选择模型：', self.combo_models)
		var_frame, var_layout = QFrame(), QFormLayout()
		var_layout.addWidget(self.ck_load_variable_preset)
		var_layout.addRow('变量组合中：', self.combo_var_num)
		var_layout.addRow('变量数：', self.spin_var_num)
		var_frame.setLayout(var_layout)
		var_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
		setting_layout.addWidget(var_frame)
		setting_layout.addRow('并发进程数：', self.spin_cpu)
		setting_layout.addRow('必须包含变量：', self.list_var)
		self.list_var.setFixedHeight(400)
		setting_layout.addWidget(self.bt_run)
		setting_layout.addWidget(self.progress)
		set_frame(setting_frame, setting_layout)
		setting_frame.setFixedWidth(300)

		score_info_frame, score_info_layout = QFrame(), QHBoxLayout()
		score_info_layout.addWidget(self.label_model_count)
		score_info_layout.addWidget(self.combo_top_score)
		set_frame(score_info_frame, score_info_layout)

		label_model_selection = QLabel('选择模型（model id）:')
		self.line_select_model.setFixedWidth(60)
		self.spin_test_size.setFixedWidth(60)
		model_id_frame, model_id_layout = QFrame(), QHBoxLayout()
		model_id_layout.addWidget(label_model_selection)
		model_id_layout.addWidget(self.line_select_model)
		set_frame(model_id_frame, model_id_layout)

		test_size_frame, test_size_layout = QFrame(), QHBoxLayout()
		test_size_layout.addWidget(QLabel('Test Size:'))
		test_size_layout.addWidget(self.spin_test_size)
		set_frame(test_size_frame, test_size_layout)

		bt_frame, bt_layout = QFrame(), QHBoxLayout()
		bt_layout.addWidget(self.bt_forecast)
		bt_layout.addWidget(self.bt_backtest)
		bt_layout.addWidget(self.bt_export_proj)
		bt_layout.addWidget(self.bt_export_backtest)
		bt_layout.addWidget(self.bt_export_candidates)
		set_frame(bt_frame, bt_layout)

		score_info_frame.setFixedWidth(250)
		model_id_frame.setFixedWidth(240)
		test_size_frame.setFixedWidth(160)
		bt_frame.setFixedWidth(500)

		model_option_frame, model_option_layout = QFrame(), QHBoxLayout()
		model_option_layout.addWidget(score_info_frame)
		model_option_layout.addWidget(model_id_frame)
		model_option_layout.addWidget(test_size_frame)
		model_option_layout.addWidget(bt_frame)
		model_option_layout.addWidget(QFrame())
		set_frame(model_option_frame, model_option_layout)

		factor_frame, factor_layout = QFrame(), QFormLayout()
		factor_layout.addRow('前瞻性因子-乐观：', self.line_positive)
		factor_layout.addRow('前瞻性因子-正常：', self.line_normal)
		factor_layout.addRow('前瞻性因子-悲观：', self.line_negative)
		set_frame(factor_frame, factor_layout)

		table_frame, table_layout = QFrame(), QVBoxLayout()
		table_layout.addWidget(factor_frame)
		table_layout.addWidget(self.table_forecast)
		set_frame(table_frame, table_layout)
		table_frame.setMinimumWidth(680)

		canvas_splitter = QSplitter(Qt.Vertical)
		canvas_splitter.addWidget(self.canvas_forecast)
		canvas_splitter.addWidget(self.canvas_learning_curve)

		forecast_splitter = QSplitter(Qt.Horizontal)
		forecast_splitter.addWidget(table_frame)
		forecast_splitter.addWidget(canvas_splitter)

		self.model_tab.addTab(self.table_score, '候选模型')
		self.model_tab.addTab(forecast_splitter, '模型预测')

		model_frame, model_layout = QFrame(), QVBoxLayout()
		model_layout.addWidget(model_option_frame)
		model_layout.addWidget(self.model_tab)
		model_frame.setLayout(model_layout)
		model_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)

		main_layout = QHBoxLayout()
		main_layout.addWidget(setting_frame)
		main_layout.addWidget(model_frame)
		self.setLayout(main_layout)
		return
