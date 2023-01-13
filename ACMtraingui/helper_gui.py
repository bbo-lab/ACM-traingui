from PyQt5.QtWidgets import QPushButton


def get_button_status(button: QPushButton):
    return button.isChecked()


def update_button_stylesheet(button: QPushButton):
    if get_button_status(button):
        button.setStyleSheet("background-color: green;")
    else:
        button.setStyleSheet("")


def toggle_button(button: QPushButton):
    button.setChecked(not get_button_status(button))
    update_button_stylesheet(button)


def disable_button(button: QPushButton):
    if get_button_status(button):
        toggle_button(button)


def enable_button(button: QPushButton):
    if not get_button_status(button):
        toggle_button(button)
