import sys
import csv
import subprocess
import time
import signal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QTableWidget, QTableWidgetItem,
    QLabel, QMessageBox
)
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
        self.setWindowTitle("CSV & GPU Table App")
        self.resize(1000, 600)

        # Stylesheet
        self.setStyleSheet("""
            QWidget { background-color: #f0f0f0; font-family: Arial, sans-serif; font-size: 14px; }
            QLabel { color: #333; font-weight: bold; }
            QLineEdit { border: 1px solid #ccc; border-radius: 5px; padding: 5px; background-color: #fff; }
            QPushButton { background-color: #0078d7; color: white; border: none; border-radius: 5px; padding: 8px 15px; }
            QPushButton:hover { background-color: #005a9e; }
            QTableWidget { border: 1px solid #ccc; background-color: #fff; }
            QTableWidgetItem { padding: 5px; }
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
        load_button = QPushButton("Load CSV & GPU Averages")
        mat_mul = QPushButton("Matrix Multiplication")
        mat_inv = QPushButton("Matrix Inverse")
        distance = QPushButton("Distance of Vectors")
        analyze = QPushButton("Analyze Results")
        analyze_button = QPushButton("Analyze Results 2")

        button_layout.addWidget(load_button)
        button_layout.addWidget(mat_mul)
        button_layout.addWidget(mat_inv)
        button_layout.addWidget(distance)
        button_layout.addWidget(analyze)
        button_layout.addWidget(analyze_button)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "Name", "N" ,"Time(s)", "Mem(MiB)", "FLOPS", 
            "Avg Temp(C)", "Avg Power(W)", "Avg GPU Util(%)", "Avg MemUsed(MiB)"
        ])

        # Assemble
        main_layout.addLayout(input_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.table)
        self.setLayout(main_layout)

        # Connect
        load_button.clicked.connect(self.load_csv)
        mat_mul.clicked.connect(self.run_mat_mul)
        mat_inv.clicked.connect(self.run_mat_inv)
        distance.clicked.connect(self.run_distance)
        analyze.clicked.connect(self.goto_analyze_window)
        analyze_button.clicked.connect(self.goto_analyse_window)


    def load_csv(self):
        try:
            with open('results.csv', newline='') as csvfile:
                rows = list(csv.reader(csvfile))
        except FileNotFoundError:
            show_alert("results.csv not found.")
            return

        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                self.table.setItem(i, j, QTableWidgetItem(val))
                
    def run_with_monitor(self, func, *args):
        # Start gpu_logger with CREATE_NEW_PROCESS_GROUP so we can send CTRL_BREAK_EVENT
        gpu_logger = subprocess.Popen(
            [sys.executable, 'gpu_monitor.py'],
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
        try:
            func(*args)
        finally:
            time.sleep(0.01)
            try:
                gpu_logger.send_signal(signal.CTRL_BREAK_EVENT)
            except Exception as e:
                print(f"Failed to send signal: {e}")
            gpu_logger.wait()

    # def run_with_monitor(self, func, *args):
    #     # start gpu monitor
    #     proc = subprocess.Popen([sys.executable, 'gpu_monitor.py'])
    #     try:
    #         func(*args)
    #     finally:
    #         time.sleep(0.1)
    #         proc.send_signal(signal.SIGINT)
    #         proc.wait()

    def run_mat_mul(self):
        if not self.input1.text(): show_alert("Enter Matrix Size."); return
        n = int(self.input1.text())
        self.run_with_monitor(proj.matmul, n)

    def run_mat_inv(self):
        if not self.input1.text(): show_alert("Enter Matrix Size."); return
        n = int(self.input1.text())
        self.run_with_monitor(proj.inverse, n)

    def run_distance(self):
        if not (self.input3.text() and self.input4.text() and self.input5.text()):
            show_alert("Enter all vector sizes."); return
        N, M, D = map(int, (self.input3.text(), self.input4.text(), self.input5.text()))
        self.run_with_monitor(proj.distance, N, M, D)
    def goto_analyze_window(self):
        from analyze import AnalyzeWindow
        self.analyze_window = AnalyzeWindow()
        self.analyze_window.show()
    def goto_analyse_window(self):
        from analyse import AnalyseWindow
        self.analyse_window = AnalyseWindow()
        self.analyse_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CSVApp()
    win.show()
    sys.exit(app.exec_())
