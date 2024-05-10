from PyQt5.QtCore import *
from PyQt5 import QtGui, QtWidgets


class QtmTable(QAbstractTableModel):
	def __init__(self, data, format_dict=None, header_labels=None, apply_warning=False, warning_config=None):
		super().__init__()
		self._data = data
		if format_dict is not None:
			self.format_dict = format_dict
		else:
			self.format_dict = {}
		self.header_labels = header_labels
		self.apply_warning = apply_warning
		if warning_config is None:
			self.warning_config = {}
		else:
			self.warning_config = warning_config
		return

	def data(self, index, role):
		value = self._data.iloc[index.row(), index.column()]
		column = self._data.columns.tolist()[index.column()]
		if role == Qt.DisplayRole:
			if column not in self.format_dict.keys():
				return str(value)
			else:
				try:
					return self.format_dict[column].format(float(value))
				except:
					return str(value)
		if self.apply_warning:
			show_warning = False
			if column in ['diff', '异常个数']:
				if int(value) != 0:
					show_warning = True
			elif column in ['limit_used', ]:
				if float(value) > 1:
					show_warning = True
			if len(self.warning_config) != 0:
				fields = self.warning_config.keys()
				if column in fields:
					for kw in self.warning_config[column]['keywords']:
						if kw in value:
							show_warning = True
			if show_warning:
				if role == Qt.BackgroundRole:
					return QtGui.QColor(255, 0, 0)
				if role == Qt.ForegroundRole:
					return QtGui.QColor(255, 255, 255)
			else:
				if role == Qt.ForegroundRole:
					return QtGui.QColor(0, 255, 255)
		# else:
		# 	if role == Qt.ForegroundRole:
		# 		return QtGui.QColor(0, 255, 255)

	def rowCount(self, index):
		return self._data.shape[0]

	def columnCount(self, index):
		return self._data.shape[1]

	def headerData(self, section, orientation, role):
		if role == Qt.DisplayRole:
			if orientation == Qt.Horizontal:
				if self.header_labels is None:
					try:
						return str(self._data.columns[section])
					except:
						print(section)
						print(self._data)
				else:
					return self.header_labels[section]
			if orientation == Qt.Vertical:
				return str(self._data.index[section])


class ProgressDelegateLimit(QtWidgets.QStyledItemDelegate):
	def paint(self, painter, option, index):
		progress_origin = float(index.data())
		progress = int(round(progress_origin * 100, 0))
		opt = QtWidgets.QStyleOptionProgressBar()
		opt.rect = option.rect
		opt.minimum = 0
		opt.maximum = 100.01
		if hasattr(progress, 'toPyObject'):
			progress = progress.toPyObject()
		opt.progress = progress
		if progress_origin <= 1:
			opt.text = '{}%'.format(progress)
			rect_color = QtGui.QColor("transparent")
		else:
			opt.text = '超限'
			rect_color = QtGui.QColor(255, 0, 0)
		text_color = QtGui.QColor(255, 255, 255)
		opt.textVisible = True

		painter.fillRect(opt.rect, rect_color)
		# painter.setBrush(rect_color)
		painter.setPen(text_color)
		QtWidgets.QApplication.style().drawControl(QtWidgets.QStyle.CE_ProgressBar,
											   opt, painter)

