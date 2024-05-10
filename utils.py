from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


def set_frame(frame, layout):
	frame.setLayout(layout)
	frame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
	return


def set_table(table, model):
	table.setModel(model)
	table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
	table.verticalHeader().setVisible(False)
	table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
	return
