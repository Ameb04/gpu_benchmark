import sys
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QTableWidget, QTableWidgetItem,
    QFileDialog, QLabel, QMessageBox
)
from PyQt5.QtGui import QFont

import proj


def show_alert(message):
    alert = QMessageBox()
    alert.setWindowTitle("Alert")
    alert.setText(message)
    alert.setIcon(QMessageBox.Warning)
    alert.setStandardButtons(QMessageBox.Ok)
    alert.exec_()


class CSVApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Table App")
        self.resize(800, 500)

        # Apply stylesheet
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: Arial, sans-serif;
                font-size: 14px;
            }
            QLabel {
                color: #333;
                font-weight: bold;
            }
            QLineEdit {
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 5px;
                background-color: #fff;
            }
            QPushButton {
                background-color: #0078d7;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QTableWidget {
                border: 1px solid #ccc;
                background-color: #fff;
            }
            QTableWidgetItem {
                padding: 5px;
            }
        """)

        # Layouts
        main_layout = QVBoxLayout()
        input_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        # Inputs
        self.input1 = QLineEdit()
        self.input3 = QLineEdit()
        self.input4 = QLineEdit()
        self.input5 = QLineEdit()
        input_layout.addWidget(QLabel("Matrix Size:"))
        input_layout.addWidget(self.input1)
        input_layout.addWidget(QLabel("Vector Size 1:"))
        input_layout.addWidget(self.input3)
        input_layout.addWidget(QLabel("Vector Size 2:"))
        input_layout.addWidget(self.input4)
        input_layout.addWidget(QLabel("Vector Size 3:"))
        input_layout.addWidget(self.input5)

        # Buttons
        load_button = QPushButton("Load CSV")
        mat_mul = QPushButton("Matrix Multiplication")
        mat_inv = QPushButton("Matrix Inverse")
        distance = QPushButton("Distance of Vectors")
        analyze_button = QPushButton("Analyze Results")
        button_layout.addWidget(load_button)
        button_layout.addWidget(mat_mul)
        button_layout.addWidget(mat_inv)
        button_layout.addWidget(distance)
        button_layout.addWidget(analyze_button)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Name", "N", "Time", "Memory", "Flops"])

        # Assemble layouts
        main_layout.addLayout(input_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.table)
        self.setLayout(main_layout)

        # Connect actions
        load_button.clicked.connect(self.load_csv)
        mat_mul.clicked.connect(self.mat_mul)
        mat_inv.clicked.connect(self.mat_inv)
        distance.clicked.connect(self.distance)
        analyze_button.clicked.connect(self.goto_analyse_window)

    def load_csv(self):
        with open('results.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            self.table.setRowCount(len(rows))
        for row_idx, row_data in enumerate(rows):
            # if row_idx == 0:
            #     continue
            for col_idx in range(min(5, len(row_data))):
                self.table.setItem(row_idx, col_idx, QTableWidgetItem(row_data[col_idx]))

    def mat_mul(self):
        if self.input1.text() == "":
            show_alert("Please enter a value for Matrix Size.")
            return
        val1 = self.input1.text()
        proj.matmul(int(val1))

    def mat_inv(self):
        if self.input1.text() == "":
            show_alert("Please enter a value for Matrix Size.")
            return
        val2 = self.input1.text()
        proj.inverse(int(val2))

    def distance(self):
        if self.input3.text() == "" or self.input4.text() == "" or self.input5.text() == "":
            show_alert("Please enter values for all vector sizes.")
            return
        val1 = self.input3.text()
        val2 = self.input4.text()
        val3 = self.input5.text()
        proj.distance(int(val1), int(val2), int(val3))

    def goto_analyse_window(self):
        from analyse import AnalyseWindow
        self.analyse_window = AnalyseWindow()
        self.analyse_window.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CSVApp()
    win.show()
    sys.exit(app.exec_())
