from ast import arguments
from operator import itemgetter
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5.QtCore import QDir, QFile, QTextStream, Qt
from PyQt5.uic import loadUi
from distutils import cmd
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        loadUi("main.ui",self)
        self.comboBox.setCurrentIndex(-1)
        self.argument.setCurrentIndex(-1)
        self.browse.clicked.connect(self.browsefiles)
        self.generate.clicked.connect(self.generatecd)
        self.generate_2.clicked.connect(self.runcmd)
        self.download.clicked.connect(self.saveAs)
        self.clear.clicked.connect(self.clearall)

    def browsefiles(self):
        fname=QFileDialog.getOpenFileName(self)
        self.filename.setText(fname[0])
        self.loaddata()

    def browsefiles1(self):
        fname=QFileDialog.getOpenFileName(self)
        self.filename.setText(fname[0])
        self.loaddata()

    def filecsv(item):
        temp = True
        i = len(item)
        filename = item
        while i > 0 and temp:
            if item[i-1] == "/":
                temp = False
                filename = item[i:]
            i -= 1
        return filename

    def loaddata(self):
        # item = MainWindow.filecsv(self.filename.text())
        item = self.filename.text()
        # self.listWidget.addItem(item)
        return item
    
    def readcombobox(self):
        item = self.comboBox.currentText()
        argument = self.argument.currentText()
        return item, argument

    def clearall(self):
        self.plainTextEdit.clear()
        self.filename.clear()
        self.filename_2.clear()
        self.comboBox.setCurrentIndex(-1)
        self.argument.setCurrentIndex(-1)
        self.statusBar().showMessage("Data berhasil di hapus !!!")

    def saveAs(self):
        fileName, _ = QFileDialog.getSaveFileName(self, QDir.homePath() , "output.csv" )
        if fileName:
            file = QFile(fileName)
            file.open(QFile.WriteOnly)
            outfile = QTextStream(file)
            outfile << self.plainTextEdit.toPlainText()
            self.statusBar().showMessage("Data berhasil di download !!!")
        else:
            self.statusBar().showMessage("Data gagal di download !!!")

    def generatecd(self):
        self.plainTextEdit.clear()

        item1,item2 = self.readcombobox()
        item3 = self.loaddata()
        item4 = ''
        item5 = ''
        if self.checkBox.isChecked():
            count = len(item3)
            find = True
            while count != 0 and find :
                if item3[count-1] == '/' :
                    find = False
                count -= 1    

            item4 = "-o "+item3[:count]

        if self.checkBox_2.isChecked():
            item5 = "--dump --pid <PID>"
            
        cmd="python vol.py "+item1+" "+item3+" "+item4+" "+item2+" "+item5
        # cmd = 'python --version'
        self.filename_2.setText(cmd)
        
        # p = os.popen (cmd)
        # if p:
        #     self.plainTextEdit.clear()
        #     output = p.read()
        #     self.plainTextEdit.insertPlainText(output)
            
        # self.plainTextEdit.setPlainText(os.system(cmd))
        # self.textEdit.setText("\n".join(data))
        # self.download.clicked.connect(self.saveAs)
        # self.clear.clicked.connect(self.clearall)

        # print('returned value:', cmd)
    def runcmd(self):
        item = self.filename_2.text()
        self.plainTextEdit.clear()
        # self.plainTextEdit.insertPlainText(item)
        p = os.popen(item)
        if p:
            self.plainTextEdit.clear()
            output = p.read()
            self.plainTextEdit.insertPlainText(output)

app=QApplication(sys.argv)
mainwindow=MainWindow()
widget=QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.setFixedWidth(1000)
widget.setFixedHeight(700)
widget.show()
sys.exit(app.exec_())