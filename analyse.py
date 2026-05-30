from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QScrollArea
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class AnalyseWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Statistical Analysis")
        self.resize(1800, 900)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content.setMinimumHeight(2000)

        df = pd.read_csv("results.csv", header=None)
        df.columns = ["Name", "N", "Time", "Memory", "Flops", "GPU_Temp","GPU_Power","GPU_Util","GPU_mem", "Category", "Method"]

        # convert Flops to number
        df["Flops"] = df["Flops"].apply(lambda x: float(str(x).replace("e", "E")))

        # --- Execution Time per Category ---
        row_layout = QHBoxLayout()
        for category in df["Category"].unique():
            fig = plt.figure(figsize=(10, 5))  # Create a figure for each category

            # Filter data just for current category (matmul, matinv, dist)
            cat_df = df[df["Category"] == category]

            # Draw the execution timeline plot
            sns.lineplot(data=cat_df, x="N", y="Time", hue="Name", marker="o")

            # Set plot titles and labels
            plt.title(f"Execution Time vs N for {category}")
            plt.xlabel("Matrix Size (N)")
            plt.ylabel("Execution Time (s)")
            plt.legend(title="Method")

            # There's a big difference in times
            plt.yscale("log")

            # Add to layout
            canvas = FigureCanvas(fig)
            row_layout.addWidget(canvas)
        content_layout.addLayout(row_layout)


        # --- Memory Usage per Category ---
        row_layout = QHBoxLayout()
        for category in df["Category"].unique():
            fig = plt.figure(figsize=(10, 4))
            sns.lineplot(data=df[df["Category"] == category], x="N", y="Memory", hue="Name", marker="o")
            plt.title(f"Memory Usage vs Matrix Size - {category}")
            plt.xlabel("Matrix Size (N)")
            plt.ylabel("Memory (MB)")
            canvas = FigureCanvas(fig)
            row_layout.addWidget(canvas)
        content_layout.addLayout(row_layout)


        # --- FLOPS per Category ---
        row_layout = QHBoxLayout()
        for category in df["Category"].unique():
            fig = plt.figure(figsize=(10, 4))
            sns.lineplot(data=df[df["Category"] == category], x="N", y="Flops", hue="Name", marker="s")
            plt.title(f"FLOPS vs Matrix Size - {category}")
            plt.xlabel("Matrix Size (N)")
            plt.ylabel("FLOPS")
            plt.yscale("log")
            canvas = FigureCanvas(fig)
            row_layout.addWidget(canvas)
        content_layout.addLayout(row_layout)

        # Auto-generated analysis summary
        desc = QTextEdit()
        desc.setReadOnly(True)
        desc.setFont(QFont("Courier", 10))
        summary = self.exec_time_best_per_n(df)
        summary = summary + (self.memory_least_per_n(df))
        summary = summary + (self.flops_best_per_n(df))
        summary = summary + (self.combined_score_per_n(df))
        desc.setText(summary)
        content_layout.addWidget(desc)


        scroll.setWidget(content)
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)


    def exec_time_best_per_n(self, df):
        best_per_n = {}

        df_exec = df.copy()

        df_exec = df_exec[["Name", "N", "Time"]]

        summary = "🔍 Analysis based on metric 'Execution Time'\n"

        for n_val in sorted(df_exec["N"].unique()):
            sub = df_exec[df_exec["N"] == n_val]

            if len(sub) < 2:
                continue

            # Inverse Normalising (less time -> better)
            sub["Score"] = 1 - (sub["Time"] - sub["Time"].min()) / (sub["Time"].max() - sub["Time"].min() + 1e-9)

            best_row = sub.loc[sub["Score"].idxmax()]
            best_name = best_row["Name"]
            if best_name.startswith("NumPy"):
                best_name = "NumPy"
            elif best_name.startswith("Naive"):
                best_name = "Naive"
            elif best_name.startswith("PyTorch"):
                best_name = "PyTorch"

            best_per_n[n_val] = best_name

        ranges = self.compress_best_ranges(best_per_n)
        summary += self.format_ranges(ranges)
        # summary += f"🔹 In N = {n_val}: {best_name} worked best\n"

        return summary

    def memory_least_per_n(self, df):
        best_per_n = {}

        df_mem = df.copy()
        df_mem = df_mem[["Name", "N", "Memory"]]

        summary = "\n\n🔍 Analysis based on metric 'Memory Usage'\n"

        for n_val in sorted(df_mem["N"].unique()):
            sub = df_mem[df_mem["N"] == n_val]

            if len(sub) < 2:
                continue

            best_row = sub.loc[sub["Memory"].idxmin()]  # least memory
            best_name = best_row["Name"]

            if best_name.startswith("NumPy"):
                best_name = "NumPy"
            elif best_name.startswith("Naive"):
                best_name = "Naive"
            elif best_name.startswith("PyTorch"):
                best_name = "PyTorch"

            best_per_n[n_val] = best_name

        ranges = self.compress_best_ranges(best_per_n)
        summary += self.format_ranges(ranges)

        # summary += f"🔹 In N = {n_val}: {best_name} used the least memory\n"

        return summary

    def flops_best_per_n(self, df):
        best_per_n = {}

        df_flops = df.copy()
        df_flops = df_flops[["Name", "N", "Flops"]]

        summary = "\n\n🔍 Analysis based on metric 'FLOPS'\n"

        for n_val in sorted(df_flops["N"].unique()):
            sub = df_flops[df_flops["N"] == n_val]

            if len(sub) < 2:
                continue

            best_row = sub.loc[sub["Flops"].idxmax()]  # highest FLOPS
            best_name = best_row["Name"]

            if best_name.startswith("NumPy"):
                best_name = "NumPy"
            elif best_name.startswith("Naive"):
                best_name = "Naive"
            elif best_name.startswith("PyTorch"):
                best_name = "PyTorch"

            best_per_n[n_val] = best_name

        ranges = self.compress_best_ranges(best_per_n)
        summary += self.format_ranges(ranges)
            # summary += f"🔹 In N = {n_val}: {best_name} achieved the highest FLOPS\n"

        return summary

    def combined_score_per_n(self, df):

        best_per_n = {}
        # just necessary columns
        df_score = df[["Name", "N", "Time", "Memory", "Flops"]].copy()

        summary = "\n\n\n🔍 Combined Analysis (Execution Time, Memory, FLOPS)\n"

        for n_val in sorted(df_score["N"].unique()):
            sub = df_score[df_score["N"] == n_val].copy()

            if len(sub) < 2:
                continue


            time_min, time_max = sub["Time"].min(), sub["Time"].max()
            mem_min, mem_max = sub["Memory"].min(), sub["Memory"].max()
            flops_min, flops_max = sub["Flops"].min(), sub["Flops"].max()

            # For avoiding division by zero
            eps = 1e-9

            def normalize_reverse(x, min_val, max_val):
                return 1 - (x - min_val) / (max_val - min_val + eps)

            def normalize(x, min_val, max_val):
                return (x - min_val) / (max_val - min_val + eps)

            sub = sub.copy()
            sub["Time_norm"] = sub["Time"].apply(lambda x: normalize_reverse(x, time_min, time_max))
            sub["Memory_norm"] = sub["Memory"].apply(lambda x: normalize_reverse(x, mem_min, mem_max))
            sub["Flops_norm"] = sub["Flops"].apply(lambda x: normalize(x, flops_min, flops_max))

            w_time, w_mem, w_flops = 1, 1, 1

            sub["CombinedScore"] = w_time * sub["Time_norm"] + w_mem * sub["Memory_norm"] + w_flops * sub["Flops_norm"]

            best_row = sub.loc[sub["CombinedScore"].idxmax()]
            best_name = best_row["Name"]

            if best_name.startswith("NumPy"):
                best_name = "NumPy"
            elif best_name.startswith("Naive"):
                best_name = "Naive"
            elif best_name.startswith("PyTorch"):
                best_name = "PyTorch"

            best_per_n[n_val] = best_name

        ranges = self.compress_best_ranges(best_per_n)
        summary += self.format_ranges(ranges)
        # summary += f"🔹 In N = {n_val}: {best_name} is the best overall method\n"

        return summary

    @staticmethod
    def compress_best_ranges(best_per_n):
        result = []
        n_list = sorted(best_per_n.keys())

        if not n_list:
            return result

        start_n = n_list[0]
        current_method = best_per_n[start_n]

        for i in range(1, len(n_list)):
            n = n_list[i]
            method = best_per_n[n]

            if method != current_method:
                end_n = n_list[i - 1]
                result.append((start_n, end_n, current_method))
                start_n = n
                current_method = method

        result.append((start_n, n_list[-1], current_method))
        return result

    @staticmethod
    def format_ranges(ranges):
        output = "\n"
        for start, end, method in ranges:
            output += f"🔸 From N = {start} to N = {end}: {method} is best overall method.\n"
        return output
