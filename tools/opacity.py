from PyQt5 import QtWidgets, QtCore
import sys
from PyQt5.QtCore import *
import time

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myshow = QtWidgets.QPushButton('Button')

    myshow.setStyleSheet("""
        padding-left: 10px;
        padding-right: 10px;
        padding-top: 1px;
        padding-bottom: 1px;
        border:1px solid #0073df;
        border-radius:5px;
        background: #167ce9;
        color: #fff;
    """)


    def changeOpacity(_):
        op = QtWidgets.QGraphicsOpacityEffect()
        op.setOpacity(0.2)
        myshow.setGraphicsEffect(op)
        myshow.setAutoFillBackground(True)


    myshow.clicked.connect(changeOpacity)

    layout = QtWidgets.QVBoxLayout()
    layout.addWidget(myshow)

    main = QtWidgets.QWidget()
    main.setLayout(layout)
    main.show()
    sys.exit(app.exec_())