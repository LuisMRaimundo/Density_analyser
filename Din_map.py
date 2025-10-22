import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern
from xgboost import XGBRegressor
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.widgets import CheckButtons

# Define pitches and initial values as empty
pitches = ["C4", "Db4", "D4", "Eb4", "E4", "F4", "F#4", "G4", "Ab4", "A4", "Bb4", "B4",
           "C5", "Db5", "D5", "Eb5", "E5", "F5", "F#5", "G5", "Ab5", "A5", "Bb5", "B5",
           "C6", "Db6", "D6", "Eb6", "E6", "F6", "F#6", "G6", "Ab6", "A6", "Bb6", "B6", "C7", "Db7"]
pp_values = []
mf_values = []
ff_values = []

# Load data from Excel
def load_excel_data():
    global pp_values, mf_values, ff_values
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls")])
    if file_path:
        try:
            df = pd.read_excel(file_path, header=0)
            if 'Notes' in df.columns:
                notes = df['Notes'].tolist()
                corrected_notes = [note.strip() for note in notes]
                if corrected_notes != pitches:
                    differences = [f"Arquivo: '{n}', Esperado: '{p}'" for n, p in zip(corrected_notes, pitches) if n != p]
                    messagebox.showerror("Erro", f"As notas no arquivo não correspondem às esperadas. Diferenças: {differences}")
                    return

            pp_values = df['pp'].fillna(0).tolist()
            mf_values = df['mf'].fillna(0).tolist()
            ff_values = df['ff'].fillna(0).tolist()
            pp_values, mf_values, ff_values = pp_values[:len(pitches)], mf_values[:len(pitches)], ff_values[:len(pitches)]

            messagebox.showinfo("Sucesso", "Dados carregados com sucesso!")
            start_analysis()
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar o arquivo: {e}")

# Main analysis function
def start_analysis():
    if not (pp_values and mf_values and ff_values):
        messagebox.showerror("Erro", "Por favor, insira ou carregue os valores para pp, mf e ff antes de iniciar a análise.")
        return

    dynamic_levels = {"pppp": 1, "ppp": 2, "pp": 3, "p": 4, "mf": 5, "f": 6, "ff": 7, "fff": 8, "ffff": 9}
    existing_dynamics = ["pp", "mf", "ff"]
    intermediate_predictions = {"ppp": [], "p": [], "f": [], "fff": []}
    extreme_predictions = {"pppp": [], "ffff": []}
    existing_levels = np.array([dynamic_levels[d] for d in existing_dynamics]).reshape(-1, 1)
    combined_values = np.array([pp_values, mf_values, ff_values])

    # GPR with Matern kernel for intermediate dynamics
    matern_kernel = C(1.0, (1e-4, 1e4)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e4), nu=1.5)
    gpr = GaussianProcessRegressor(kernel=matern_kernel, n_restarts_optimizer=10, alpha=1e-1)

    # Predictions for each pitch
    for i in range(len(pitches)):
        y_train = combined_values[:, i]
        
        # Fit GPR with Matern kernel for intermediate levels
        gpr.fit(existing_levels, y_train)
        intermediate_levels = np.array([dynamic_levels[d] for d in ["ppp", "p", "f", "fff"]]).reshape(-1, 1)
        y_intermediate = gpr.predict(intermediate_levels)
        for level, value in zip(["ppp", "p", "f", "fff"], y_intermediate):
            intermediate_predictions[level].append(value)

        # XGBoost model without monotonic constraints
        xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42, 
                                 learning_rate=0.1, max_depth=3, n_estimators=100)
        
        # Fit model and predict pppp and ffff without constraints
        xgb_model.fit(existing_levels, y_train)
        y_extreme_pppp = xgb_model.predict([[dynamic_levels["pppp"]]])
        y_extreme_ffff = xgb_model.predict([[dynamic_levels["ffff"]]])

        # Manual adjustments for monotonic constraints
        pppp_value = min(y_extreme_pppp[0], y_intermediate[0] - 1e-6)  # Ensure pppp < ppp
        ffff_value = max(y_extreme_ffff[0], y_intermediate[3] + 1e-6)  # Ensure ffff > fff

        extreme_predictions["pppp"].append(pppp_value)
        extreme_predictions["ffff"].append(ffff_value)

    dynamic_values = {**{"pp": pp_values, "mf": mf_values, "ff": ff_values}, **intermediate_predictions, **extreme_predictions}
    plot_results(dynamic_values)

# Plotting function with interactive features
def plot_results(dynamic_values):
    pitch_numeric = list(range(len(pitches)))
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['darkviolet', 'deeppink', 'royalblue', 'dodgerblue', 'darkorange', 'crimson', 'dimgray', 'black', 'maroon']
    lines = []
    all_dynamics = ["pppp", "ppp", "pp", "p", "mf", "f", "ff", "fff", "ffff"]

    for i, dynamic in enumerate(all_dynamics):
        line, = ax.plot(pitch_numeric, dynamic_values[dynamic], label=dynamic, color=colors[i % len(colors)], linestyle='-')
        lines.append(line)

    ax.set_xlabel("Pitches (Representação Numérica)")
    ax.set_ylabel("Densidade Espectral")
    ax.set_xticks(pitch_numeric)
    ax.set_xticklabels(pitches, rotation=90)
    ax.set_ylim(0, max(max(dynamic_values[dynamic]) for dynamic in dynamic_values) * 1.1)
    ax.set_title("Densidade Espectral em Diferentes Dinâmicas")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Interactive CheckButtons
    rax = plt.axes([0.02, 0.4, 0.15, 0.4])
    check = CheckButtons(rax, all_dynamics, [True] * len(all_dynamics))

    def func(label):
        index = all_dynamics.index(label)
        lines[index].set_visible(not lines[index].get_visible())
        plt.draw()

    check.on_clicked(func)
    plt.show()

# Main Tkinter GUI
def main():
    root = tk.Tk()
    root.title("Análise de Densidade Espectral")
    load_button = tk.Button(root, text="Carregar Dados do Excel", command=load_excel_data)
    load_button.pack(pady=10)
    analyze_button = tk.Button(root, text="Iniciar Análise", command=start_analysis)
    analyze_button.pack(pady=10)
    root.mainloop()

if __name__ == "__main__":
    main()

