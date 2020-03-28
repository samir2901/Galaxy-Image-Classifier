from PyQt5 import QtCore, QtGui, QtWidgets
from tensorflow import keras 
import cv2
import sys
import numpy as np


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.image = np.array([])
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(447, 339)
        MainWindow.setMaximumSize(QtCore.QSize(447, 600))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Image = QtWidgets.QLabel(self.centralwidget)
        self.Image.setGeometry(QtCore.QRect(20, 30, 200, 200))
        self.Image.setText("")
        self.Image.setObjectName("Image")


        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40,260,361,41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")


        self.browseImageBtn = QtWidgets.QPushButton(self.centralwidget)
        self.browseImageBtn.setGeometry(QtCore.QRect(250, 20, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(11)
        self.browseImageBtn.setFont(font)
        self.browseImageBtn.setObjectName("browseImageBtn")
        self.browseImageBtn.clicked.connect(self.browseImage)


        self.predictBtn = QtWidgets.QPushButton(self.centralwidget)
        self.predictBtn.setGeometry(QtCore.QRect(250, 80, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(11)
        self.predictBtn.setFont(font)
        self.predictBtn.setObjectName("predictBtn")
        self.predictBtn.clicked.connect(self.prediction)


        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Galaxy Image Classifier"))
        self.label.setText(_translate("MainWindow", "PREDICTION:"))
        self.browseImageBtn.setText(_translate("MainWindow", "Browse Image"))
        self.predictBtn.setText(_translate("MainWindow", "Analyse"))

    def browseImage(self):        
        fm = QtWidgets.QFileDialog.getOpenFileName(None,"Open File")
        filename = fm[0]
        self.image = cv2.imread(filename)                
        self.Image.setPixmap(QtGui.QPixmap(filename))
        self.Image.setScaledContents(True)
           
        
    
    def prediction(self):        
        CATEGORIES = ["elliptical","spiral"]
        IMAGE_SIZE = 100        
        self.image = cv2.resize(self.image, (IMAGE_SIZE,IMAGE_SIZE))
        model = keras.models.load_model("CNN-Classifier.h5")
        try:
            pred = model.predict(self.image.reshape(-1,IMAGE_SIZE,IMAGE_SIZE,3))
            x = np.argmax(pred)
            string = "PREDICTION: " + CATEGORIES[x].upper() + " GALAXY"
            self.label.setText(string)
            msgDone = QtWidgets.QMessageBox()
            msgDone.setIcon(QtWidgets.QMessageBox.Information)
            msgDone.setWindowTitle("Done")
            msgDone.setText("Prediction Done")
            msgDone.exec_()
        except:
            msgError = QtWidgets.QMessageBox()
            msgError.setIcon(QtWidgets.QMessageBox.Critical)
            msgError.setWindowTitle("Error")
            msgError.setText("Oops!! Error")
            msgError.exec_()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
