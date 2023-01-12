import os
from pathlib import Path

from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWidgets import QDialog, QGridLayout, QComboBox, QSizePolicy, QPushButton


class SelectUserWindow(QDialog):
    def __init__(self, drive: Path, parent=None):
        super(SelectUserWindow, self).__init__(parent)
        self.drive = drive

        self.setGeometry(0, 0, 256, 128)
        self.center()
        self.setWindowTitle('Select User')

        self.user_list = self.get_user_list(drive)

        self.selecting_layout = QGridLayout()

        self.selecting_field = QComboBox()
        self.selecting_field.addItems(self.user_list)
        self.selecting_field.setSizePolicy(QSizePolicy.Expanding,
                                           QSizePolicy.Preferred)
        self.selecting_layout.addWidget(self.selecting_field)

        self.selecting_button = QPushButton('Ok')
        self.selecting_button.clicked.connect(self.accept)
        self.selecting_button.setSizePolicy(QSizePolicy.Expanding,
                                            QSizePolicy.Preferred)
        self.selecting_layout.addWidget(self.selecting_button)

        self.setLayout(self.selecting_layout)

    @staticmethod
    def get_user_list(drive):
        user_list = sorted(os.listdir(drive / 'pose/data/user'))
        return user_list

    def center(self):
        qr = self.frameGeometry()
        cp = QGuiApplication.primaryScreen().geometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def get_user(self):
        user_id = self.selecting_field.currentIndex()
        user = self.user_list[user_id]
        return user

    @staticmethod
    def start(drive, parent=None):
        selecting = SelectUserWindow(drive=drive, parent=parent)
        exit_sel = selecting.exec_()
        user = selecting.get_user()
        return user, exit_sel == QDialog.Accepted
