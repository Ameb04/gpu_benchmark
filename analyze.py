### analyze.py
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QScrollArea
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import pandas as pd

class AnalyzeWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Benchmark & GPU Analysis")
        self.resize(1000, 700)

        # Load data once
        df = pd.read_csv("results.csv", header=None)
        df.columns = [
            "Name","N","Time","Memory","Flops",
            "Temp","Power","Utilization","MemUsedByGPU",
            "Category","Method"
        ]
        df["N"] = df["N"].astype(int)
        df["Time"]   = df["Time"].astype(float)
        df["Memory"] = df["Memory"].astype(float)
        df["Flops"]  = df["Flops"].apply(lambda x: float(str(x).replace('e','E')))
        for c in ["Temp","Power","Utilization","MemUsedByGPU"]:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        self.df = df

        # Controls: choose metric and method
        metrics = ["Time","Memory","Flops","Temp","Power","Utilization","MemUsedByGPU"]
        methods = self.df["Method"].unique().tolist()

        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Metric:"))
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(metrics)
        control_layout.addWidget(self.metric_combo)

        control_layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(methods)
        control_layout.addWidget(self.method_combo)

        self.plot_btn = QPushButton("Plot")
        control_layout.addWidget(self.plot_btn)
        control_layout.addStretch()

        # Figure & canvas
        self.figure = plt.figure(figsize=(6,4))
        self.canvas = FigureCanvas(self.figure)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.canvas)
        self.setLayout(main_layout)

        # Connect
        self.plot_btn.clicked.connect(self.update_plot)
        self.update_plot()

    def update_plot(self):
        metric = self.metric_combo.currentText()
        method = self.method_combo.currentText()

        df_sub = self.df[self.df['Method'] == method]
        categories = df_sub['Category'].unique()

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        for cat in categories:
            cat_df = df_sub[df_sub['Category']==cat]
            ax.plot(cat_df['N'], cat_df[metric], marker='o', label=cat)

        ax.set_xlabel('N')
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs N  [Method: {method}]")
        if metric in ['Time','Flops']:
            ax.set_yscale('log')
        ax.legend(title='Category')
        ax.grid(True)
        self.canvas.draw()
