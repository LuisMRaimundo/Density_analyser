# gui_calibration.py
"""
Complemento para a GUI principal com widgets específicos para calibração
do parâmetro lambda.
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging

# Configurar logging
logger = logging.getLogger('gui_calibration')

class CalibrationWindow:
    """
    Janela dedicada à calibração do parâmetro lambda.
    """

    def __init__(self, parent, on_calibrate=None):
        """
        Inicializa a janela de calibração.

        Args:
            parent: Widget pai (normalmente a janela principal)
            on_calibrate: Callback a ser chamado após calibração bem-sucedida
        """
        self.parent = parent
        self.on_calibrate = on_calibrate

        # Importar funções de calibração somente quando necessário
        try:
            from calibration import (
                obter_lambda_atual,
                realizar_calibracao,
                visualizar_funcao_exponencial,
                testar_modelo_calibrado,
                analisar_consonancia_vs_lambda,
                CONSONANCE_RATINGS,
                salvar_parametros_calibrados
            )
            self.obter_lambda_atual = obter_lambda_atual
            self.realizar_calibracao = realizar_calibracao
            self.visualizar_funcao_exponencial = visualizar_funcao_exponencial
            self.testar_modelo_calibrado = testar_modelo_calibrado
            self.analisar_consonancia_vs_lambda = analisar_consonancia_vs_lambda
            self.CONSONANCE_RATINGS = CONSONANCE_RATINGS
            self.salvar_parametros_calibrados = salvar_parametros_calibrados
        except ImportError as e:
            logger.error(f"Erro ao importar módulos de calibração: {e}")
            messagebox.showerror("Erro", f"Não foi possível carregar os módulos de calibração: {e}")
            return

        # Criar a janela
        self.window = tk.Toplevel(parent)
        self.window.title("Calibração de Lambda")
        self.window.geometry("800x600")
        self.window.transient(parent)
        self.window.grab_set()

        # Dividir em duas áreas: controles à esquerda, visualização à direita
        self.paned = ttk.PanedWindow(self.window, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Painel de controles
        self.control_frame = ttk.LabelFrame(self.paned, text="Configurações de Calibração")
        self.paned.add(self.control_frame, weight=1)

        # Painel de visualização
        self.viz_frame = ttk.LabelFrame(self.paned, text="Visualização")
        self.paned.add(self.viz_frame, weight=3)

        # Configurar controles
        self._setup_controls()

        # Configurar área de visualização
        self._setup_visualization()

        # Centralizar a janela
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    def _setup_controls(self):
        """Configura os controles no painel esquerdo."""
        frame = self.control_frame

        # Obter valor atual de lambda
        lambda_atual = self.obter_lambda_atual()

        # Mostrar valor atual
        valor_frame = ttk.Frame(frame)
        valor_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(valor_frame, text="Valor atual de lambda:").pack(side=tk.LEFT)

        self.lambda_var = tk.StringVar(value=f"{lambda_atual:.4f}")
        lambda_label = ttk.Label(valor_frame, textvariable=self.lambda_var, font=("Arial", 12, "bold"))
        lambda_label.pack(side=tk.LEFT, padx=5)

        # Separador
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=5)

        # Botões de ação
        action_frame = ttk.LabelFrame(frame, text="Ações")
        action_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(action_frame, text="Calibrar com Dados Padrão",
                  command=self._calibrar_automaticamente).pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(action_frame, text="Visualizar Função Atual",
                  command=self._visualizar_funcao).pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(action_frame, text="Testar Modelo Atual",
                  command=self._testar_modelo).pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(action_frame, text="Analisar Sensibilidade",
                  command=self._analisar_sensibilidade).pack(fill=tk.X, padx=5, pady=5)

        # Separador
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=5)

        # Ajuste manual
        manual_frame = ttk.LabelFrame(frame, text="Ajuste Manual")
        manual_frame.pack(fill=tk.X, padx=10, pady=10)

        # Slider para ajuste manual
        slider_frame = ttk.Frame(manual_frame)
        slider_frame.pack(fill=tk.X, padx=5, pady=5)

        self.lambda_slider = ttk.Scale(
            slider_frame, from_=0.01, to=1.0,
            orient=tk.HORIZONTAL,
            value=lambda_atual,
            command=self._on_slider_change
        )
        self.lambda_slider.pack(fill=tk.X, padx=5, pady=5)

        # Mostrar valores extremos do slider
        limits_frame = ttk.Frame(slider_frame)
        limits_frame.pack(fill=tk.X)

        ttk.Label(limits_frame, text="0.01").pack(side=tk.LEFT)
        ttk.Label(limits_frame, text="1.0").pack(side=tk.RIGHT)

        # Valor atual do slider
        slider_value_frame = ttk.Frame(manual_frame)
        slider_value_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(slider_value_frame, text="Valor:").pack(side=tk.LEFT)

        self.slider_value_var = tk.StringVar(value=f"{lambda_atual:.4f}")
        slider_value_label = ttk.Label(slider_value_frame, textvariable=self.slider_value_var)
        slider_value_label.pack(side=tk.LEFT, padx=5)

        # Botão de aplicar ajuste manual
        ttk.Button(manual_frame, text="Aplicar Valor Manual",
                  command=self._aplicar_valor_manual).pack(fill=tk.X, padx=5, pady=5)

        # Separador
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=5)

        # Botões de controle
        control_buttons = ttk.Frame(frame)
        control_buttons.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(control_buttons, text="OK", command=self.window.destroy).pack(side=tk.RIGHT, padx=5)

    def _setup_visualization(self):
        """Configura a área de visualização no painel direito."""
        # Mensagem inicial
        self.viz_msg = tk.Label(self.viz_frame, text="Selecione uma opção para visualizar", font=("Arial", 12))
        self.viz_msg.pack(expand=True, fill=tk.BOTH)

        # Figura
        self.fig = None
        self.canvas = None

    def _on_slider_change(self, value):
        """Atualiza o valor exibido quando o slider é movido."""
        value_float = float(value)
        self.slider_value_var.set(f"{value_float:.4f}")

        # Atualização em tempo real (opcional)
        # self._visualizar_funcao(value_float)

    def _aplicar_valor_manual(self):
        """Aplica o valor manual de lambda definido pelo slider."""
        try:
            valor = float(self.lambda_slider.get())
            if 0.01 <= valor <= 1.0:
                self.salvar_parametros_calibrados({'lambda': valor})
                self.lambda_var.set(f"{valor:.4f}")
                messagebox.showinfo("Lambda Definido", f"Valor de lambda definido manualmente: {valor:.4f}")

                # Visualizar com o novo valor
                self._visualizar_funcao(valor)

                # Chamar callback se fornecido
                if self.on_calibrate:
                    self.on_calibrate(valor)
            else:
                messagebox.showerror("Erro", "O valor deve estar entre 0.01 e 1.0")
        except ValueError:
            messagebox.showerror("Erro", "Digite um número válido")

    def _calibrar_automaticamente(self):
        """Executa a calibração automática com dados padrão."""
        try:
            lambda_otimizado = self.realizar_calibracao()
            self.lambda_var.set(f"{lambda_otimizado:.4f}")
            self.lambda_slider.set(lambda_otimizado)
            self.slider_value_var.set(f"{lambda_otimizado:.4f}")

            messagebox.showinfo("Calibração Concluída",
                               f"Calibração realizada com sucesso!\nNovo valor de lambda: {lambda_otimizado:.4f}")

            # Visualizar resultado da calibração
            self._testar_modelo()

            # Chamar callback se fornecido
            if self.on_calibrate:
                self.on_calibrate(lambda_otimizado)
        except Exception as e:
            logger.error(f"Erro durante calibração automática: {e}")
            messagebox.showerror("Erro", f"Erro durante calibração: {e}")

    def _visualizar_funcao(self, lambda_valor=None):
        """Visualiza a função exponencial com o valor atual de lambda."""
        # Limpar visualização anterior
        self._clear_visualization()

        # Se não foi fornecido um valor, usar o atual
        if lambda_valor is None:
            lambda_valor = self.obter_lambda_atual()

        # Criar figura
        self.fig, ax = plt.subplots(figsize=(6, 4))

        # Plotar função
        steps = np.linspace(0, 48, 100)  # 0 a 48 microtons (~2 oitavas)

        # Calcular valores da função
        valores = []
        for s in steps:
            if s == 0:
                valores.append(0)  # Uníssono = 0
            else:
                valores.append(np.exp(-lambda_valor * s))

        # Plotar
        ax.plot(steps, valores, label=f"Decaimento: e^(-{lambda_valor:.4f}*delta)")

        # Configurar eixos
        ax.set_title(f"Função de Decaimento (λ={lambda_valor:.4f})")
        ax.set_xlabel("Distância (microtons)")
        ax.set_ylabel("Peso")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Ajustar layout
        plt.tight_layout()

        # Mostrar no canvas
        self._show_figure()

    def _testar_modelo(self):
        """Testa o modelo calibrado comparando com dados experimentais."""
        # Limpar visualização anterior
        self._clear_visualization()

        # Obter valor atual de lambda
        lambda_atual = self.obter_lambda_atual()

        # Criar figura
        self.fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

        # Calcular dados para comparação
        intervalos = []
        valores_exp = []
        valores_modelo = []
        erros = []

        for intervalo, valor_exp in self.CONSONANCE_RATINGS.items():
            intervalos.append(str(intervalo))
            valores_exp.append(valor_exp)

            # Calcular valor do modelo com o lambda atual
            if intervalo == 0:  # Uníssono
                densidade = 0
            else:
                delta = intervalo * 2  # Convertendo para escala microtonal
                densidade = np.exp(-lambda_atual * delta) if delta > 0 else 0

            # Normalizar para comparação
            max_valor = max(self.CONSONANCE_RATINGS.values())
            densidade_norm = 2 * (densidade / max_valor) - 1

            valores_modelo.append(densidade_norm)
            erros.append(abs(densidade_norm - valor_exp))

        # Plotar comparação
        x = np.arange(len(intervalos))
        width = 0.35

        ax1.bar(x - width/2, valores_exp, width, label='Experimental', alpha=0.7)
        ax1.bar(x + width/2, valores_modelo, width, label='Modelo', alpha=0.7)

        ax1.set_title(f"Comparação (λ={lambda_atual:.4f})")
        ax1.set_xlabel("Intervalo (semitons)")
        ax1.set_ylabel("Consonância")
        ax1.set_xticks(x)
        ax1.set_xticklabels(intervalos)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plotar erro
        ax2.bar(x, erros, alpha=0.7)
        ax2.set_title("Erro Absoluto")
        ax2.set_xlabel("Intervalo (semitons)")
        ax2.set_ylabel("Erro")
        ax2.set_xticks(x)
        ax2.set_xticklabels(intervalos)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Mostrar no canvas
        self._show_figure()

    def _analisar_sensibilidade(self):
        """Analisa como diferentes valores de lambda afetam a consonância."""
        # Limpar visualização anterior
        self._clear_visualization()

        # Criar figura
        self.fig, ax = plt.subplots(figsize=(6, 4))

        # Intervalos padrão para teste
        intervalos_teste = [
            ("Unísono", 0),
            ("Segunda menor", 1),
            ("Segunda maior", 2),
            ("Terça menor", 3),
            ("Terça maior", 4),
            ("Quarta justa", 5),
            ("Trítono", 6),
            ("Quinta justa", 7),
        ]

        # Valores de lambda para testar
        lambdas = np.arange(0.01, 1.01, 0.05)

        # Calcular densidades para cada intervalo e cada lambda
        for nome, intervalo in intervalos_teste:
            densidades = []

            for lamb in lambdas:
                # Calcular densidade
                if intervalo == 0:  # Uníssono
                    densidade = 0
                else:
                    delta = intervalo * 2  # Convertendo para escala microtonal
                    densidade = np.exp(-lamb * delta)

                densidades.append(densidade)

            # Plotar linha para este intervalo
            ax.plot(lambdas, densidades, label=nome)

        # Configurar eixos
        ax.set_title("Densidade vs. Lambda por Intervalo")
        ax.set_xlabel("Lambda (λ)")
        ax.set_ylabel("Densidade")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Mostrar valor atual de lambda como linha vertical
        lambda_atual = self.obter_lambda_atual()
        ax.axvline(x=lambda_atual, color='black', linestyle='--', alpha=0.5)
        ax.text(lambda_atual + 0.02, 0.9, f"λ atual = {lambda_atual:.4f}",
               transform=ax.get_xaxis_transform(), fontsize=9)

        plt.tight_layout()

        # Mostrar no canvas
        self._show_figure()

    def _clear_visualization(self):
        """Limpa a área de visualização."""
        # Remover mensagem
        if hasattr(self, 'viz_msg') and self.viz_msg:
            self.viz_msg.pack_forget()
            self.viz_msg = None

        # Remover figura anterior
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
            self.canvas = None

        if self.fig:
            plt.close(self.fig)
            self.fig = None

    def _show_figure(self):
        """Exibe a figura atual no canvas."""
        if self.fig:
            # Criar canvas
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
            self.canvas.draw()

            # Adicionar ao frame
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def adicionar_menu_calibracao(root, menu_principal, callback=None):
    """
    Adiciona opções de calibração ao menu principal.

    Args:
        root: A janela raiz tkinter
        menu_principal: Menu principal ao qual adicionar as opções
        callback: Função a ser chamada quando a calibração for concluída
    """
    # Menu de ferramentas
    if not menu_principal:
        # Criar menu principal se não existir
        menu_principal = tk.Menu(root)
        root.config(menu=menu_principal)

    # Verificar se já existe menu de Ferramentas
    tools_menu = None
    for i in range(menu_principal.index('end') + 1):
        if menu_principal.entrycget(i, 'label') == 'Ferramentas':
            tools_menu = menu_principal.nametowidget(menu_principal.entrycget(i, 'menu'))
            break

    if not tools_menu:
        # Criar menu de ferramentas
        tools_menu = tk.Menu(menu_principal, tearoff=0)
        menu_principal.add_cascade(label="Ferramentas", menu=tools_menu)

    # Adicionar comando de calibração
    tools_menu.add_command(
        label="Calibração de Lambda",
        command=lambda: abrir_janela_calibracao(root, callback)
    )


def abrir_janela_calibracao(root, callback=None):
    """
    Abre a janela de calibração.

    Args:
        root: A janela raiz tkinter
        callback: Função a ser chamada quando a calibração for concluída
    """
    # Verificar se as bibliotecas necessárias estão disponíveis
    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except ImportError as e:
        messagebox.showerror("Erro", f"Bibliotecas necessárias não disponíveis: {e}")
        return

    # Abrir janela de calibração
    CalibrationWindow(root, callback)


if __name__ == "__main__":
    # Teste rápido do componente
    root = tk.Tk()
    root.title("Teste da Janela de Calibração")
    root.geometry("300x200")

    def on_calibrate(value):
        print(f"Novo valor de lambda: {value}")

    btn = ttk.Button(root, text="Abrir Calibração",
                    command=lambda: abrir_janela_calibracao(root, on_calibrate))
    btn.pack(expand=True)

    # Criar menu para teste
    menubar = tk.Menu(root)
    root.config(menu=menubar)

    adicionar_menu_calibracao(root, menubar, on_calibrate)

    root.mainloop()
