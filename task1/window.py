import sys
from PyQt5.QtWidgets import QApplication
from handwrite_window import MyMnistWindow


def getnum():
    app = QApplication(sys.argv)
    mymnist = MyMnistWindow()
    mymnist.show()
    app.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mymnist = MyMnistWindow()
    mymnist.show()
    app.exec_()
