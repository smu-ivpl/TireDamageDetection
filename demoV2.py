import sys
from PyQt5 import QtWidgets
from window import Window

app = QtWidgets.QApplication(sys.argv)
DemoGUI = Window()
DemoGUI.setWindowTitle('Demo')
DemoGUI.show()
app.exec_()
