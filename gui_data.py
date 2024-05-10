import pandas as pd
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as figureCanvas
import fwd_model.dm_config as config
from fwd_model.qtm_table import QtmTable


class guiData(QWidget):
	def __init__(self):
		super().__init__()
		self.setAutoFillBackground(True)

		self.line_config_source = QLabel()
		self.line_macro = QLabel()
		self.line_npl = QLabel()
		self.table_data_config = QTableView()
		self.table_variable = QTableView()
		self.table_data = QTableView()
		self.combo_ts_type = QComboBox()
		self.combo_variable = QComboBox()
		self.figure = plt.figure(tight_layout=True)
		self.canvas = figureCanvas(self.figure)
		self.bt_export = QPushButton('导出全部数据')
		self.dm = None

		self.widget_setting()
		self.layout()
		return

	def widget_setting(self):
		self.combo_ts_type.addItems(['预处理前', '预处理后'])
		self.combo_ts_type.currentIndexChanged.connect(self.update_variable_table)
		self.combo_variable.currentIndexChanged.connect(self.update_data_table)
		return

	def update_file_source(self):
		config_source = config.config_file
		macro_source = config.macro_file_name
		npl_source = config.npl_file_name
		self.line_config_source.setText(config_source)
		self.line_macro.setText(macro_source)
		self.line_npl.setText(npl_source)
		return

	def update_widget(self, dm):
		self.dm = dm.dmData
		self.update_file_source()
		self.update_data_config()
		self.update_variable_table()
		return

	def update_data_config(self):
		df_var_config = self.dm.variable_config.reset_index()
		model = QtmTable(df_var_config)
		self.table_data_config.setModel(model)
		self.table_data_config.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
		self.table_data_config.verticalHeader().setVisible(False)
		self.table_data_config.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
		return

	def update_variable_table(self):
		data_type = self.combo_ts_type.currentText()
		if data_type == '预处理前':
			df_data = self.dm.df_raw_data
		else:
			df_data = self.dm.df_data
		self.combo_variable.clear()
		self.combo_variable.addItems(df_data.columns.tolist())
		self.update_data_table()
		return

	def update_data_table(self):
		data_type = self.combo_ts_type.currentText()
		data_col = self.combo_variable.currentText()
		if len(data_col) == 0 or len(data_type) == 0:
			return
		if data_type == '预处理前':
			df_data = self.dm.df_raw_data[data_col].astype(float)
		else:
			df_data = self.dm.df_data[data_col].astype(float)
		self.update_canvas(data_col, df_data)
		df_data = df_data.reset_index()
		df_data['datadate'] = [d.strftime('%Y%m%d') for d in df_data['datadate']]
		format_dict = {data_col: '{:,.4f}'}
		model = QtmTable(df_data, format_dict=format_dict)
		self.table_data.setModel(model)
		self.table_data.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
		self.table_data.verticalHeader().setVisible(False)
		self.table_data.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

		return

	def update_canvas(self, title, df_data):
		self.figure.clear()
		ax = self.figure.add_subplot(111)
		df_data.plot(grid=True, ax=ax, title=title, color='g', style='.-')
		self.canvas.draw()
		return

	def layout(self):
		source_frame, source_layout = QFrame(), QFormLayout()
		source_layout.addRow('指标定义文件：', self.line_config_source)
		source_layout.addRow('宏观数据：', self.line_macro)
		source_layout.addRow('不良贷款数据：', self.line_npl)
		source_frame.setLayout(source_layout)
		source_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
		source_frame.setFixedWidth(900)
		top_frame, top_layout = QFrame(), QHBoxLayout()
		top_layout.addWidget(source_frame)
		top_layout.addWidget(QFrame())
		top_frame.setLayout(top_layout)
		top_frame.setFixedHeight(100)

		pool_frame, pool_layout = QFrame(), QVBoxLayout()
		pool_layout.addWidget(QLabel('指标池：'))
		pool_layout.addWidget(self.table_data_config)
		pool_frame.setLayout(pool_layout)
		pool_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
		pool_frame.setMinimumWidth(500)

		data_frame, data_layout = QFrame(), QHBoxLayout()
		data_table_frame, data_table_layout = QFrame(), QVBoxLayout()
		data_table_layout.addWidget(self.combo_ts_type)
		data_table_layout.addWidget(self.combo_variable)
		data_table_layout.addWidget(self.table_data)
		data_table_layout.addWidget(self.bt_export)
		data_table_frame.setLayout(data_table_layout)
		data_table_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
		data_table_frame.setMaximumWidth(300)

		canvas_splitter = QSplitter(Qt.Vertical)
		canvas_splitter.addWidget(self.canvas)
		canvas_splitter.addWidget(QFrame())

		data_layout.addWidget(data_table_frame)
		data_layout.addWidget(canvas_splitter)
		data_frame.setLayout(data_layout)
		data_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)

		main_splitter = QSplitter(Qt.Horizontal)
		main_splitter.addWidget(pool_frame)
		main_splitter.addWidget(data_frame)

		main_layout = QVBoxLayout()
		main_layout.addWidget(top_frame)
		main_layout.addWidget(main_splitter)
		self.setLayout(main_layout)
		return
