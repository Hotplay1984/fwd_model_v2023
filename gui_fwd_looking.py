from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import datetime as dt
import sys
from fwd_model import dm_ts_funcs
from fwd_model import dm_models
from fwd_model import dm_config as config
from fwd_model.gui_data import guiData
from fwd_model.gui_model import guiModel
from fwd_model.gui_grouping import guiGrouping
from fwd_model.utils import set_frame


class guiFWDMain(QWidget):
	def __init__(self):
		super().__init__()
		self.setAutoFillBackground(True)

		self.date_edit_bng, self.date_edit_end = QDateEdit(), QDateEdit()
		self.bt_load_data = QPushButton('加载数据')
		self.status = QStatusBar()

		self.guiData = guiData()
		self.guiModel = guiModel()
		self.guiGrouping = guiGrouping()
		self.tab_main = QTabWidget()
		self.tab_main.addTab(self.guiData, '数据')
		self.tab_main.addTab(self.guiModel, '模型')
		self.tab_main.addTab(self.guiGrouping, '分组')

		self.dm = None

		self.widget_setting()
		self.layout()
		return

	def widget_setting(self):
		self.status.showMessage('就绪')
		self.date_edit_bng.setDate(QDate(2001, 1, 31))
		self.date_edit_end.setDate(QDate(dt.datetime.now().year, 12, 31))
		self.date_edit_bng.setCalendarPopup(True)
		self.date_edit_end.setCalendarPopup(True)
		self.bt_load_data.clicked.connect(self.load_data)
		self.guiModel.signal_forecast_made.connect(self.update_grouping_widget)
		return

	def load_data(self):
		bng_date = self.date_edit_bng.date().toString('yyyyMMdd')
		end_date = self.date_edit_end.date().toString('yyyyMMdd')
		self.dm = dm_models.Predictor(bng_date, end_date)

		self.update_widgets()
		return

	def update_widgets(self):
		self.guiData.update_widget(self.dm)
		self.guiModel.update_widget(self.dm)
		return

	def update_grouping_widget(self):
		dm = self.guiModel.dm
		self.guiGrouping.update_widget(dm)
		return

	def layout(self):
		date_frame, date_layout = QFrame(), QHBoxLayout()
		date_layout.addWidget(QLabel('数据区间：'))
		date_layout.addWidget(self.date_edit_bng)
		date_layout.addWidget(QLabel('至：'))
		date_layout.addWidget(self.date_edit_end)
		set_frame(date_frame, date_layout)

		self.bt_load_data.setFixedHeight(60)

		top_frame, top_layout = QFrame(), QHBoxLayout()
		top_layout.addWidget(date_frame)
		top_layout.addWidget(self.bt_load_data)
		top_layout.addWidget(self.status)
		set_frame(top_frame, top_layout)

		main_layout = QVBoxLayout()
		main_layout.addWidget(top_frame)
		main_layout.addWidget(self.tab_main)
		self.setLayout(main_layout)
		return


def main():
	app = QApplication(sys.argv)
	gui = guiFWDMain()
	gui.show()
	sys.exit(app.exec_())


if __name__ == '__main__':
	main()
