# gui_components.py

# Versão simplificada para corrigir problemas

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import numpy as np
import pandas as pd
import os
from datetime import datetime

# CENTRALIZAÇÃO ABSOLUTA DE NOTAS:
from utils.notes import (
    normalize_note_string, note_to_midi, midi_to_note_name,
    extract_cents, is_valid_note, QUARTO_TOM_ACIMA, QUARTO_TOM_ABAIXO
)

# Importe as funções de densidade_intervalar.py
from densidade_intervalar import (
    analisar_consonancia_vs_lambda,
    carregar_parametros_calibrados,
    CONSONANCE_RATINGS
)

print("\n=== CARREGANDO GUI_COMPONENTS.PY MODIFICADO COM SUPORTE A CENTS E FIGURAS MUSICAIS ===\n")


class DensityCalculatorGUI:

    """

    Classe que encapsula a interface gráfica da Calculadora de Densidade.

    """



    def __init__(self, root, callbacks):
        """
        Inicializa a interface gráfica.
        Args:
            root: O widget raiz tkinter
            callbacks: Dicionário com callbacks para ações da interface
        """
        self.root = root
        self.root.title("Calculadora de Densidade Integrada")

        # Armazenar callbacks
        self.callbacks = callbacks

        # Variáveis da interface
        self.weight_factor_slider = None
        self.note_vars = []
        self.octave_vars = []
        self.cents_vars = []  # NOVO: Lista para variáveis de cents
        self.dynamic_vars = []
        self.instrument_vars = []
        self.num_instruments_vars = []
        self.state_vars = []
        self.note_menus = []
        self.octave_menus = []
        self.cents_menus = []  # NOVO: Lista para menus de cents
        self.dynamic_menus = []
        self.instrument_menus = []
        self.num_instruments_menus = []

        # Variáveis para opções adicionais
        self.var_save_results = tk.BooleanVar(value=False)
        self.var_show_graphs = tk.BooleanVar(value=True)

        # Variáveis para a Lei de Stevens
        self.var_use_stevens = tk.BooleanVar(value=True)  # Ativar Lei de Stevens por padrão
        self.alpha_var = tk.DoubleVar(value=0.7)  # Expoente alpha padrão
        self.beta_var = tk.DoubleVar(value=0.4)   # Expoente beta padrão

        # Variável para correções psicoacústicas
        self.var_use_psychoacoustic = tk.BooleanVar(value=True)

        # NOVA: Variável para ponderação perceptual nos intervalos
        self.var_perceptual_weighting = tk.BooleanVar(value=False)

        # Widgets importantes
        self.result_text = None
        self.tree = None
        self.validation_text = None
        self.embedded_graphs_frame = None

        self.var_use_psychoacoustic = tk.BooleanVar(value=True)
        self.var_perceptual_weighting = tk.BooleanVar(value=False)

        # Criar a interface
        self._create_interface()





    def get_input_data(self):
        """
        Coleta todos os dados de entrada da interface.
        Returns:
            dict: Dicionário com dados de entrada
        """
        import logging
        logger = logging.getLogger('gui_components')

        active_indices = [i for i in range(len(self.state_vars)) if self.state_vars[i].get() == 1]

        # Correção para garantir que as notas estão no formato correto
        complete_notes = []
        for i in active_indices:
            note_part = self.note_vars[i].get()
            octave_part = self.octave_vars[i].get()
            cents_part = self.cents_vars[i].get() if hasattr(self, 'cents_vars') else None

            # Verificar se ambas as partes estão preenchidas
            if note_part and octave_part:
                # Construir a nota completa com cents se não for zero
                if cents_part and cents_part != '0':
                    complete_note = f"{note_part}{octave_part}{cents_part}c"  # Adiciona "c" para indicar cents
                else:
                    complete_note = f"{note_part}{octave_part}"
                complete_notes.append(complete_note)
            else:
                logger.warning(f"Nota incompleta: nota={note_part}, oitava={octave_part}")

                # Usar uma nota padrão se alguma parte estiver faltando
                if note_part and not octave_part:
                    complete_notes.append(f"{note_part}4")  # Usar oitava 4 como padrão
                elif not octave_part and note_part:
                    complete_notes.append(f"C{octave_part}")  # Usar C como nota padrão
                else:
                    complete_notes.append("C4")  # Nota padrão

        return {
            'notes': complete_notes,
            'dynamics': [self.dynamic_vars[i].get() for i in active_indices],
            'instruments': [self.instrument_vars[i].get() for i in active_indices],
            'num_instruments': [int(self.num_instruments_vars[i].get()) for i in active_indices],
            'weight_factor': self.weight_factor_slider.get(),
            'save_results': self.var_save_results.get() if hasattr(self, 'var_save_results') else False,
            'show_graphs': self.var_show_graphs.get() if hasattr(self, 'var_show_graphs') else True,
            'use_stevens': self.var_use_stevens.get() if hasattr(self, 'var_use_stevens') else True,
            'alpha': self.alpha_var.get() if hasattr(self, 'alpha_var') else 0.7,
            'beta': self.beta_var.get() if hasattr(self, 'beta_var') else 0.4,
            'use_psychoacoustic': self.var_use_psychoacoustic.get() if hasattr(self, 'var_use_psychoacoustic') else True,
            'use_perceptual_weighting': self.var_perceptual_weighting.get() if hasattr(self, 'var_perceptual_weighting') else False,  # ADD THIS LINE
        }

    def _create_interface(self):
        """Cria a interface gráfica completa."""

        # Criar canvas com barra de rolagem
        canvas = tk.Canvas(self.root)
        scroll_y = tk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        self.input_frame = tk.Frame(canvas)

        canvas.create_window((0, 0), window=self.input_frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll_y.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll_y.pack(side="right", fill="y")

        self.input_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Slider para ajuste de ponderação com legenda clara
        slider_frame = tk.Frame(self.root)
        slider_frame.pack(pady=(10, 5), fill='x', padx=20)

        slider_title = tk.Label(slider_frame, text="Ajustar Balanceamento entre Componentes:", font=("Arial", 10, "bold"))
        slider_title.pack(pady=(5, 3))

        # Frame para o slider e suas legendas
        balance_frame = tk.Frame(slider_frame)
        balance_frame.pack(fill='x')

        # Legenda do lado esquerdo
        left_label = tk.Label(balance_frame, text="100% Densidade Intervalar", anchor="w", fg="blue")
        left_label.pack(side=tk.LEFT)

        # Legenda do lado direito
        right_label = tk.Label(balance_frame, text="100% Densidade do Instrumento", anchor="e", fg="red")
        right_label.pack(side=tk.RIGHT)

        # Slider centralizado com cores nas extremidades
        self.weight_factor_slider = tk.Scale(slider_frame, from_=0, to=1, orient="horizontal",
                                            resolution=0.01, length=400, showvalue=True,
                                            tickinterval=0.25, label="Fator de Peso")
        self.weight_factor_slider.set(0.5)  # Valor padrão
        self.weight_factor_slider.pack(pady=(0, 10))

        # Adicionar texto explicativo
        help_text = "Mova o slider para a esquerda para aumentar a importância da densidade intervalar, ou\n" \
        "para a direita para aumentar a importância da densidade produzida pelo instrumento."
        slider_help = tk.Label(slider_frame, text=help_text, font=("Arial", 8, "italic"))
        slider_help.pack(pady=(0, 5))

        # Criar campos de entrada para 60 notas
        self._create_note_inputs()

        # Opções adicionais
        options_frame = tk.Frame(self.root)
        options_frame.pack(pady=5)

        save_check = tk.Checkbutton(options_frame, text="Salvar resultados em arquivo", variable=self.var_save_results)
        save_check.pack(side=tk.LEFT, padx=5)

        graph_check = tk.Checkbutton(options_frame, text="Mostrar gráficos detalhados", variable=self.var_show_graphs)
        graph_check.pack(side=tk.LEFT, padx=5)

        # Criar frame para Lei de Stevens - COM BORDA VISÍVEL
        stevens_frame = tk.LabelFrame(self.root, text="Lei de Stevens")
        stevens_frame.pack(pady=5, fill="x", padx=10)

        # Controles dentro do frame
        stevens_check = tk.Checkbutton(stevens_frame, text="Ativar", variable=self.var_use_stevens)
        stevens_check.pack(side=tk.LEFT, padx=10, pady=5)

        tk.Label(stevens_frame, text="a:").pack(side=tk.LEFT, padx=5, pady=5)
        alpha_entry = tk.Entry(stevens_frame, textvariable=self.alpha_var, width=6)
        alpha_entry.pack(side=tk.LEFT, padx=5, pady=5)

        tk.Label(stevens_frame, text="ß:").pack(side=tk.LEFT, padx=5, pady=5)
        beta_entry = tk.Entry(stevens_frame, textvariable=self.beta_var, width=6)
        beta_entry.pack(side=tk.LEFT, padx=5, pady=5)

        # Botão de ajuda
        help_btn = tk.Button(stevens_frame, text="?", command=lambda: messagebox.showinfo(
            "Lei de Stevens",
            "Modifica a percepção usando uma função de potência.\n" +
            "a: expoente para densidade do instrumento\n" +
            "ß: expoente para densidade intervalar\n\n" +
            "Valores menores (0.1-0.3) produzem resposta mais linear."
        ))
        help_btn.pack(side=tk.RIGHT, padx=10, pady=5)

        # Criar frame para correções psicoacústicas
        psycho_frame = tk.LabelFrame(self.root, text="Correções Psicoacústicas")
        psycho_frame.pack(pady=5, fill="x", padx=10)

        psycho_check = tk.Checkbutton(
            psycho_frame,
            text="Ativar correções psicoacústicas (mascaramento, roughness, loudness)",
            variable=self.var_use_psychoacoustic
        )
        psycho_check.pack(side=tk.LEFT, padx=10, pady=5)

        # Botão de informação para psicoacústica
        psycho_info_btn = tk.Button(
            psycho_frame,
            text="?",
            command=lambda: messagebox.showinfo(
                "Correções Psicoacústicas",
                "Aplica modelos psicoacústicos para tornar as medições mais perceptualmente precisas:\n\n" +
                "• Mascaramento de banda crítica\n" +
                "• Cálculo de roughness (aspereza)\n" +
                "• Correção de equal loudness\n\n" +
                "Recomendado para análises mais precisas."
            )
        )
        psycho_info_btn.pack(side=tk.RIGHT, padx=10, pady=5)

        # NOVA SEÇÃO: Frame para ponderação perceptual de intervalos
        perceptual_frame = tk.LabelFrame(self.root, text="Configurações de Densidade Intervalar")
        perceptual_frame.pack(pady=5, fill="x", padx=10)

        # Checkbox para ponderação perceptual
        self.perceptual_weighting_cb = tk.Checkbutton(
            perceptual_frame,
            text="Usar ponderação perceptual nos intervalos",
            variable=self.var_perceptual_weighting,
            command=self._on_perceptual_weighting_changed
        )
        self.perceptual_weighting_cb.pack(side=tk.LEFT, padx=10, pady=5)

        # Label informativo sobre quando usar
        info_label = tk.Label(
            perceptual_frame,
            text="(Recomendado quando peso dos intervalos > 70%)",
            font=("Arial", 8),
            fg="gray"
        )
        info_label.pack(side=tk.LEFT, padx=5, pady=5)

        # Botão de informação para ponderação perceptual
        perceptual_info_btn = tk.Button(
            perceptual_frame,
            text="?",
            command=lambda: messagebox.showinfo(
                "Ponderação Perceptual",
                "Aplica ponderação baseada em registro ao calcular densidade intervalar.\n\n" +
                "Características:\n" +
                "• Registros agudos = maior peso perceptual\n" +
                "• Intervalos pequenos = maior peso (2ªs menores/maiores)\n" +
                "• Intervalos grandes = menor peso (8ªs ou maiores)\n\n" +
                "Melhora a precisão perceptual quando a densidade intervalar " +
                "é o componente dominante da análise."
            )
        )
        perceptual_info_btn.pack(side=tk.RIGHT, padx=10, pady=5)

        # Notebook (sistema de abas) para resultados
        self._create_notebook()

        # Botões principais
        self._create_main_buttons()

    def _on_perceptual_weighting_changed(self):
        """Callback quando a ponderação perceptual é alterada"""
        if self.var_perceptual_weighting.get():
            # Opcional: mostrar aviso na primeira ativação
            if not hasattr(self, '_perceptual_warning_shown'):
                messagebox.showinfo(
                    "Ponderação Perceptual Ativada",
                    "A ponderação perceptual foi ativada.\n\n" +
                    "Isso aplicará pesos diferentes aos intervalos baseados no registro " +
                    "e tamanho dos intervalos, tornando o cálculo mais preciso perceptualmente."
                )
                self._perceptual_warning_shown = True

        # Se houver callback definido, chama ele
        if self.callbacks and 'on_perceptual_weighting_changed' in self.callbacks:
            self.callbacks['on_perceptual_weighting_changed'](self.var_perceptual_weighting.get())

    def get_perceptual_weighting_status(self):
        """Retorna o status da ponderação perceptual"""
        return self.var_perceptual_weighting.get()

    def set_perceptual_weighting(self, enabled):
        """Define o status da ponderação perceptual programaticamente"""
        self.var_perceptual_weighting.set(enabled)

    def check_and_suggest_perceptual_weighting(self, interval_weight_percentage):
        """
        Verifica se deve sugerir ativar a ponderação perceptual
        Args:
            interval_weight_percentage: Percentual do peso dos intervalos
        """
        if (interval_weight_percentage > 70 and
            not self.var_perceptual_weighting.get() and
            not hasattr(self, '_perceptual_suggestion_shown')):

            result = messagebox.askyesno(
                "Sugestão de Ponderação Perceptual",
                f"O peso dos intervalos é {interval_weight_percentage:.1f}%.\n\n" +
                "Para uma análise mais precisa perceptualmente, " +
                "recomenda-se ativar a ponderação perceptual.\n\n" +
                "Deseja ativar agora?"
            )

            if result:
                self.var_perceptual_weighting.set(True)
                messagebox.showinfo(
                    "Ponderação Ativada",
                    "A ponderação perceptual foi ativada. " +
                    "Os próximos cálculos usarão pesos baseados no registro."
                )

            # Marcar que a sugestão já foi mostrada
            self._perceptual_suggestion_shown = True



    def _create_note_inputs(self):
        """
        Cria os campos de entrada para notas, oitavas, dinâmicas, etc.
        Utiliza símbolos Unicode consistentes para notação musical.
        """
        # Listas para menus dropdown
        octave_list = [str(i) for i in range(10)]
        dynamic_levels = ['pppp', 'ppp', 'pp', 'p', 'mf', 'f', 'ff', 'fff', 'ffff']
        instruments = [
            'Flautim', 'Flauta', 'Oboe', 'Corne_ingles', 'Clarinete', 'Clarinete baixo',
            'Fagote', 'Contrafagote', 'Trompa', 'Trompete', 'Trombone', 'Tuba',
            'Violino', 'Viola', 'Violoncelo', 'Contrabaixo'
        ]

        # Definir símbolos musicais Unicode
        SUSTENIDO_MUSICAL = "♯"  # U+266F - Símbolo musical para sustenido
        QUARTO_TOM_ABAIXO = "↓"  # U+2193 - Consistente com o dicionário spectral_data

        # Lista de valores de cents (+ e - cents)
        cents_values = ['0'] + [f"+{i}" for i in range(1, 51)] + [f"-{i}" for i in range(1, 51)]

        # Lista de notas base com notação consistente
        # Usa o símbolo SUSTENIDO_MUSICAL (♯) em vez do símbolo # normal
        # Usa QUARTO_TOM_ABAIXO (↓) para indicar quartos de tom
        notas_base = [
            'C', f'C{QUARTO_TOM_ABAIXO}', f'C{SUSTENIDO_MUSICAL}', f'C{SUSTENIDO_MUSICAL}{QUARTO_TOM_ABAIXO}',
            'D', f'D{QUARTO_TOM_ABAIXO}', f'D{SUSTENIDO_MUSICAL}', f'D{SUSTENIDO_MUSICAL}{QUARTO_TOM_ABAIXO}',
            'E', f'E{QUARTO_TOM_ABAIXO}',
            'F', f'F{QUARTO_TOM_ABAIXO}', f'F{SUSTENIDO_MUSICAL}', f'F{SUSTENIDO_MUSICAL}{QUARTO_TOM_ABAIXO}',
            'G', f'G{QUARTO_TOM_ABAIXO}', f'G{SUSTENIDO_MUSICAL}', f'G{SUSTENIDO_MUSICAL}{QUARTO_TOM_ABAIXO}',
            'A', f'A{QUARTO_TOM_ABAIXO}', f'A{SUSTENIDO_MUSICAL}', f'A{SUSTENIDO_MUSICAL}{QUARTO_TOM_ABAIXO}',
            'B', f'B{QUARTO_TOM_ABAIXO}'
        ]

        # Adicionar cabeçalhos para melhor identificação
        tk.Label(self.input_frame, text="Ativar").grid(row=0, column=0, padx=5, pady=5)
        tk.Label(self.input_frame, text="Nota").grid(row=0, column=1, padx=5, pady=5)
        tk.Label(self.input_frame, text="Oitava").grid(row=0, column=2, padx=5, pady=5)
        tk.Label(self.input_frame, text="Cents").grid(row=0, column=3, padx=5, pady=5)
        tk.Label(self.input_frame, text="Dinâmica").grid(row=0, column=4, padx=5, pady=5)
        tk.Label(self.input_frame, text="Instrumento").grid(row=0, column=5, padx=5, pady=5)
        tk.Label(self.input_frame, text="Qtd").grid(row=0, column=6, padx=5, pady=5)

        # Criar 60 linhas para entrada de notas
        for i in range(60):
            row_offset = i + 1  # +1 devido à linha de cabeçalho

            # Checkbox para ativar a linha
            state_var = tk.IntVar(value=0)
            self.state_vars.append(state_var)
            checkbutton = tk.Checkbutton(self.input_frame, variable=state_var,
                                         command=lambda i=i: self.toggle_state(i))
            checkbutton.grid(row=row_offset, column=0, padx=5, pady=2)

            # Menu de notas
            note_var = tk.StringVar()
            self.note_vars.append(note_var)
            note_menu = ttk.Combobox(self.input_frame, textvariable=note_var,
                                    values=notas_base, width=5, state='disabled')
            note_menu.grid(row=row_offset, column=1, padx=5, pady=2)
            self.note_menus.append(note_menu)

            # Menu de oitavas
            octave_var = tk.StringVar(value='4')
            self.octave_vars.append(octave_var)
            octave_menu = ttk.Combobox(self.input_frame, textvariable=octave_var,
                                      values=octave_list, width=5, state='disabled')
            octave_menu.grid(row=row_offset, column=2, padx=5, pady=2)
            self.octave_menus.append(octave_menu)

            # Menu de cents
            cents_var = tk.StringVar(value='0')
            self.cents_vars.append(cents_var)
            cents_menu = ttk.Combobox(self.input_frame, textvariable=cents_var,
                                     values=cents_values, width=5, state='disabled')
            cents_menu.grid(row=row_offset, column=3, padx=5, pady=2)
            self.cents_menus.append(cents_menu)

            # Menu de dinâmicas
            dynamic_var = tk.StringVar(value='mf')
            self.dynamic_vars.append(dynamic_var)
            dynamic_menu = ttk.Combobox(self.input_frame, textvariable=dynamic_var,
                                       values=dynamic_levels, width=5, state='disabled')
            dynamic_menu.grid(row=row_offset, column=4, padx=5, pady=2)
            self.dynamic_menus.append(dynamic_menu)

            # Menu de instrumentos
            instrument_var = tk.StringVar(value='Flauta')
            self.instrument_vars.append(instrument_var)
            instrument_menu = ttk.Combobox(self.input_frame, textvariable=instrument_var,
                                          values=instruments, width=10, state='disabled')
            instrument_menu.grid(row=row_offset, column=5, padx=5, pady=2)
            self.instrument_menus.append(instrument_menu)

            # Menu de quantidade de instrumentos
            num_instruments_var = tk.StringVar(value='1')
            self.num_instruments_vars.append(num_instruments_var)
            num_values = [str(j) for j in range(1, 21)]
            num_instruments_menu = ttk.Combobox(self.input_frame, textvariable=num_instruments_var,
                                               values=num_values, width=5, state='disabled')
            num_instruments_menu.grid(row=row_offset, column=6, padx=5, pady=2)
            self.num_instruments_menus.append(num_instruments_menu)




    def _create_notebook(self):

        """Cria o notebook com abas para diferentes visualizações de resultados."""

        notebook = ttk.Notebook(self.root)

        notebook.pack(fill='both', expand=True, pady=10)



        # Aba para resultados de texto

        text_frame = tk.Frame(notebook)

        notebook.add(text_frame, text="Resultados Numéricos")



        self.result_text = tk.Text(text_frame, height=15, width=60)

        self.result_text.pack(pady=10, padx=10, fill='both', expand=True)



        text_scrollbar = tk.Scrollbar(text_frame, command=self.result_text.yview)

        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.result_text.config(yscrollcommand=text_scrollbar.set)



        # Aba para gráficos embutidos

        self.embedded_graphs_frame = tk.Frame(notebook)

        notebook.add(self.embedded_graphs_frame, text="Gráficos Rápidos")



        # Aba para métricas avançadas

        metrics_frame = tk.Frame(notebook)

        notebook.add(metrics_frame, text="Métricas Avançadas")



        # Treeview para métricas avançadas

        self.tree = ttk.Treeview(metrics_frame)

        self.tree.pack(fill='both', expand=True, padx=10, pady=10)



        # Configurar colunas do treeview

        self.tree["columns"] = ("valor")

        self.tree.column("#0", width=250, minwidth=250)

        self.tree.column("valor", width=150, minwidth=150)



        # Criar cabeçalhos

        self.tree.heading("#0", text="Métrica")

        self.tree.heading("valor", text="Valor")



        # Aba para validação estatística

        validation_frame = tk.Frame(notebook)

        notebook.add(validation_frame, text="Validação Estatística")



        # Label para explicação

        validation_label = tk.Label(validation_frame, text="Esta seção permite validar a confiabilidade das métricas calculadas através de métodos estatísticos.")

        validation_label.pack(pady=10)



        # Botão para executar análise de validação

        validation_button = tk.Button(validation_frame, text="Executar Validação Estatística",

                                    command=lambda: self.callbacks['execute_validation']())

        validation_button.pack(pady=5)



        # Campo de texto para resultados de validação

        self.validation_text = tk.Text(validation_frame, height=15, width=60)

        self.validation_text.pack(pady=10, padx=10, fill='both', expand=True)



        # Armazenar o notebook para referência

        self.notebook = notebook



    def _create_main_buttons(self):

        """Cria os botões principais da interface."""

        buttons_frame = tk.Frame(self.root)

        buttons_frame.pack(pady=10)



        calculate_button = tk.Button(buttons_frame, text="Calcular", command=self.callbacks['calculate'])

        calculate_button.pack(side=tk.LEFT, padx=5, pady=5)



        clear_button = tk.Button(buttons_frame, text="Limpar", command=self.callbacks['clear'])

        clear_button.pack(side=tk.LEFT, padx=5, pady=5)



        report_button = tk.Button(buttons_frame, text="Gerar Relatório Científico", command=self.callbacks['generate_report'])

        report_button.pack(side=tk.LEFT, padx=5, pady=5)

     # ==========================================
    # ADD THE HELPER METHODS HERE (after your existing methods):
    # ==========================================

    def _on_perceptual_weighting_changed(self):
        """Callback quando a ponderação perceptual é alterada"""
        if self.var_perceptual_weighting.get():
            # Opcional: mostrar aviso na primeira ativação
            if not hasattr(self, '_perceptual_warning_shown'):
                messagebox.showinfo(
                    "Ponderação Perceptual Ativada",
                    "A ponderação perceptual foi ativada.\n\n" +
                    "Isso aplicará pesos diferentes aos intervalos baseados no registro " +
                    "e tamanho dos intervalos, tornando o cálculo mais preciso perceptualmente."
                )
                self._perceptual_warning_shown = True

        # Se houver callback definido, chama ele
        if self.callbacks and 'on_perceptual_weighting_changed' in self.callbacks:
            self.callbacks['on_perceptual_weighting_changed'](self.var_perceptual_weighting.get())

    def get_perceptual_weighting_status(self):
        """Retorna o status da ponderação perceptual"""
        return self.var_perceptual_weighting.get()

    def set_perceptual_weighting(self, enabled):
        """Define o status da ponderação perceptual programaticamente"""
        self.var_perceptual_weighting.set(enabled)

    def check_and_suggest_perceptual_weighting(self, interval_weight_percentage):
        """
        Verifica se deve sugerir ativar a ponderação perceptual
        Args:
            interval_weight_percentage: Percentual do peso dos intervalos
        """
        if (interval_weight_percentage > 70 and
            not self.var_perceptual_weighting.get() and
            not hasattr(self, '_perceptual_suggestion_shown')):

            result = messagebox.askyesno(
                "Sugestão de Ponderação Perceptual",
                f"O peso dos intervalos é {interval_weight_percentage:.1f}%.\n\n" +
                "Para uma análise mais precisa perceptualmente, " +
                "recomenda-se ativar a ponderação perceptual.\n\n" +
                "Deseja ativar agora?"
            )

            if result:
                self.var_perceptual_weighting.set(True)
                messagebox.showinfo(
                    "Ponderação Ativada",
                    "A ponderação perceptual foi ativada. " +
                    "Os próximos cálculos usarão pesos baseados no registro."
                )

            # Marcar que a sugestão já foi mostrada
            self._perceptual_suggestion_shown = True

    # ==========================================
    # Your other existing methods continue here:
    # ==========================================

    def get_input_notes(self):
        """Obtém as notas dos campos de entrada."""
        # ... your existing method ...
        pass

    def show_error(self, message):
        """Mostra mensagem de erro."""
        # ... your existing method ...
        pass


    def toggle_state(self, index):
        """
        Ativa/desativa os controles de entrada com base no estado do checkbox.

        Args:
            index (int): Índice do conjunto de controles a ser alternado
        """
        state = 'normal' if self.state_vars[index].get() == 1 else 'disabled'

        # Configurar todos os widgets
        self.note_menus[index].config(state=state)
        self.octave_menus[index].config(state=state)

        if hasattr(self, 'cents_menus') and len(self.cents_menus) > index:
            self.cents_menus[index].config(state=state)

        self.dynamic_menus[index].config(state=state)
        self.instrument_menus[index].config(state=state)
        self.num_instruments_menus[index].config(state=state)



    def clear_inputs(self):

        """Limpa todos os campos de entrada."""

        for var in self.note_vars:

            var.set('')

        for var in self.octave_vars:

            var.set('4')

        for var in self.cents_vars:

            var.set('0')

        for var in self.dynamic_vars:

            var.set('mf')

        for var in self.instrument_vars:

            var.set('Flauta')

        for var in self.num_instruments_vars:

            var.set('1')

        for var in self.state_vars:

            var.set(0)


        # Limpar a árvore de métricas

        for item in self.tree.get_children():

            self.tree.delete(item)



        # Limpar o frame de gráficos embutidos

        for widget in self.embedded_graphs_frame.winfo_children():

            widget.destroy()



    def show_results(self, result_text):

        """

        Mostra os resultados na área de texto.



        Args:

            result_text (str): Texto de resultados para exibir

        """

        self.result_text.delete(1.0, tk.END)

        self.result_text.insert(tk.END, result_text)



    def show_validation_results(self, validation_text):

        """

        Mostra os resultados da validação estatística.



        Args:

            validation_text (str): Texto de validação para exibir

        """

        self.validation_text.delete(1.0, tk.END)

        self.validation_text.insert(tk.END, validation_text)



    def update_metrics_tree(self, results):

        """

        Atualiza o treeview com as métricas calculadas.



        Args:

            results (dict): Resultados completos da análise

        """

        # Limpar árvore

        for item in self.tree.get_children():

            self.tree.delete(item)



        # Adicionar categorias principais

        density_id = self.tree.insert("", "end", text="Densidade", open=True)

        moments_id = self.tree.insert("", "end", text="Momentos Espectrais", open=True)

        additional_id = self.tree.insert("", "end", text="Métricas Adicionais", open=True)

        texture_id = self.tree.insert("", "end", text="Textura", open=True)

        timbre_id = self.tree.insert("", "end", text="Timbre", open=True)

        orchestration_id = self.tree.insert("", "end", text="Orquestração", open=True)



        # Adicionar métricas de densidade

        for k, v in results["densidade"].items():

            self.tree.insert(density_id, "end", text=k.capitalize(), values=(f"{v:.4f}",))



        # Adicionar momentos espectrais

        for k, v in results["momentos_espectrais"].items():

            if k == "Centróide":

                self.tree.insert(moments_id, "end", text="Centróide", values=(f"{v['frequency']:.2f} Hz ({v['note']})",))

            elif k == "Dispersão":

                self.tree.insert(moments_id, "end", text="Dispersão", values=(f"±{v['deviation']:.2f} Hz",))

            else:

                if isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v):

                    self.tree.insert(moments_id, "end", text=k.replace("spectral_", "").capitalize(), values=(f"{v:.4f}",))



        # Adicionar métricas adicionais

        for k, v in results["metricas_adicionais"].items():

            if k != "chroma_vector":  # Tratar vetor de croma separadamente

                if isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v):

                    self.tree.insert(additional_id, "end", text=k.capitalize(), values=(f"{v:.4f}",))



        # Adicionar textura

        for k, v in results["textura"].items():

            if isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v):

                self.tree.insert(texture_id, "end", text=k.capitalize(), values=(f"{v:.4f}",))



        # Adicionar timbre

        for k, v in results["timbre"].items():

            if k != "family_contributions" and isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v):

                self.tree.insert(timbre_id, "end", text=k.capitalize(), values=(f"{v:.4f}",))



        # Adicionar orquestração

        for k, v in results["orquestracao"].items():

            if k != "register_distribution" and isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v):

                self.tree.insert(orchestration_id, "end", text=k.capitalize(), values=(f"{v:.4f}",))



    def create_embedded_graphs(self, pitches, densities):

        """

        Cria um gráfico claro e informativo embutido na interface.



        Args:

            pitches (array-like): Valores de pitch (MIDI)

            densities (array-like): Densidades correspondentes

        """

        # Limpar frame

        for widget in self.embedded_graphs_frame.winfo_children():

            widget.destroy()



        # Converter MIDI para nomes de notas

        from utils.notes import midi_to_note_name


        note_names = [midi_to_note_name(p) for p in pitches]



        # Criar figura única mas clara

        fig = plt.Figure(figsize=(10, 6), dpi=100)

        ax = fig.add_subplot(111)



        # Usar cores mais vibrantes e pontos maiores

        bars = ax.bar(range(len(pitches)), densities, color='royalblue', alpha=0.8)



        # Adicionar valores exatos no topo das barras

        for bar in bars:

            height = bar.get_height()

            ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,

                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)



        # Configurar eixos com nomes de notas

        ax.set_title('Densidade Espectral por Nota', fontsize=14)

        ax.set_xlabel('Notas', fontsize=12)

        ax.set_ylabel('Densidade', fontsize=12)



        # Usar nomes de notas no eixo x

        ax.set_xticks(range(len(pitches)))

        ax.set_xticklabels(note_names, rotation=45, ha='right')



        ax.grid(axis='y', linestyle='--', alpha=0.3)



        # Garantir que tudo se encaixa bem

        fig.tight_layout()



        # Adicionar a figura ao frame

        canvas = FigureCanvasTkAgg(fig, self.embedded_graphs_frame)

        canvas.draw()

        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)



        # Adicionar toolbar para interatividade

        toolbar = NavigationToolbar2Tk(canvas, self.embedded_graphs_frame)

        toolbar.update()



    def show_report_config_dialog(self, on_generate):

        """

        Exibe o diálogo de configuração para geração de relatório.



        Args:

            on_generate (callable): Função a ser chamada quando o usuário confirmar a geração

        """

        # Obter diretório para salvar relatórios

        directory = filedialog.askdirectory(title="Selecione o diretório para salvar os relatórios")

        if not directory:

            return  # Usuário cancelou a seleção



        # Janela para configuração do relatório

        config_window = tk.Toplevel(self.root)

        config_window.title("Configurações do Relatório")

        config_window.geometry("500x500")



        # Campos de configuração

        config_frame = tk.Frame(config_window, padx=10, pady=10)

        config_frame.pack(fill="both", expand=True)



        # Título do relatório

        tk.Label(config_frame, text="Título do Relatório:").grid(row=0, column=0, sticky="w", pady=5)

        title_var = tk.StringVar(value="Análise Espectral Avançada de Composição Musical")

        tk.Entry(config_frame, textvariable=title_var, width=50).grid(row=0, column=1, pady=5)



        # Autores

        tk.Label(config_frame, text="Autores:").grid(row=1, column=0, sticky="w", pady=5)

        authors_var = tk.StringVar(value="")

        tk.Entry(config_frame, textvariable=authors_var, width=50).grid(row=1, column=1, pady=5)



        # Instituição

        tk.Label(config_frame, text="Instituição:").grid(row=2, column=0, sticky="w", pady=5)

        institution_var = tk.StringVar(value="")

        tk.Entry(config_frame, textvariable=institution_var, width=50).grid(row=2, column=1, pady=5)



        # Resumo

        tk.Label(config_frame, text="Resumo:").grid(row=3, column=0, sticky="w", pady=5)

        summary_text = tk.Text(config_frame, width=48, height=5)

        summary_text.grid(row=3, column=1, pady=5)

        summary_text.insert("1.0", "Este relatório apresenta uma análise detalhada das propriedades espectrais, texturais e de timbre de um conjunto de notas musicais, utilizando métodos quantitativos avançados.")



        # Conclusões

        tk.Label(config_frame, text="Conclusões:").grid(row=4, column=0, sticky="w", pady=5)

        conclusions_text = tk.Text(config_frame, width=48, height=5)

        conclusions_text.grid(row=4, column=1, pady=5)

        conclusions_text.insert("1.0", "As análises realizadas demonstram a eficácia das métricas espectrais e de textura para a caracterização objetiva de material musical. Os resultados podem ser aplicados em contextos de análise musical, compositiva e de síntese sonora.")



        # Opções de relatório

        tk.Label(config_frame, text="Formatos de Relatório:").grid(row=5, column=0, sticky="w", pady=5)



        pdf_var = tk.BooleanVar(value=True)

        tk.Checkbutton(config_frame, text="Relatório em PDF", variable=pdf_var).grid(row=5, column=1, sticky="w")



        paper_var = tk.BooleanVar(value=True)

        tk.Checkbutton(config_frame, text="Artigo Científico", variable=paper_var).grid(row=6, column=1, sticky="w")



        figures_var = tk.BooleanVar(value=True)

        tk.Checkbutton(config_frame, text="Figuras para Publicação", variable=figures_var).grid(row=7, column=1, sticky="w")



        tables_var = tk.BooleanVar(value=True)

        tk.Checkbutton(config_frame, text="Tabelas de Dados", variable=tables_var).grid(row=8, column=1, sticky="w")



        # Status

        status_label = tk.Label(config_frame, text="", font=("Arial", 10, "italic"))

        status_label.grid(row=10, column=0, columnspan=2, pady=5)



        # Função para gerar os relatórios com as configurações escolhidas

        def execute_generation():

            config = {

                'title': title_var.get(),

                'authors': authors_var.get(),

                'institution': institution_var.get(),

                'abstract': summary_text.get("1.0", "end-1c"),

                'conclusions': conclusions_text.get("1.0", "end-1c"),

                'date': datetime.now().strftime("%d de %B de %Y"),

                'formats': {

                    'pdf': pdf_var.get(),

                    'paper': paper_var.get(),

                    'figures': figures_var.get(),

                    'tables': tables_var.get()

                },

                'output_directory': directory

            }



            # Mostrar mensagem de processamento

            status_label.config(text="Gerando relatórios. Aguarde...")

            config_window.update()



            # Chamar a função de geração

            on_generate(config)



            # Fechar janela de configuração

            config_window.destroy()



        # Botões

        buttons_frame = tk.Frame(config_frame)

        buttons_frame.grid(row=9, column=0, columnspan=2, pady=10)



        generate_btn = tk.Button(buttons_frame, text="Gerar Relatórios", command=execute_generation)

        generate_btn.pack(side=tk.LEFT, padx=5)



        cancel_btn = tk.Button(buttons_frame, text="Cancelar", command=config_window.destroy)

        cancel_btn.pack(side=tk.LEFT, padx=5)



        # Centralizar a janela

        config_window.update_idletasks()

        width = config_window.winfo_width()

        height = config_window.winfo_height()

        x = (config_window.winfo_screenwidth() // 2) - (width // 2)

        y = (config_window.winfo_screenheight() // 2) - (height // 2)

        config_window.geometry('{}x{}+{}+{}'.format(width, height, x, y))



# Adicione esta função para criar uma janela de calibração
def abrir_janela_calibracao(root):
    """
    Abre uma janela para calibrar o parâmetro lambda com base em dados experimentais.
    """
    calibration_window = tk.Toplevel(root)
    calibration_window.title("Calibração de Parâmetros")
    calibration_window.geometry("800x600")

    # Frame para controles
    control_frame = ttk.Frame(calibration_window)
    control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    # Mostrar valor atual
    lambda_atual = obter_lambda_atual()
    lambda_label = ttk.Label(control_frame, text=f"Valor atual de lambda: {lambda_atual:.4f}")
    lambda_label.pack(side=tk.LEFT, padx=5)

    # Frame para o gráfico
    plot_frame = ttk.Frame(calibration_window)
    plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Criar figura inicial
    fig, ax = plt.subplots(figsize=(8, 5))
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Atualizar o gráfico com valores de consonância experimentais vs. calculados
    def atualizar_grafico():
        ax.clear()
        lambda_atual = obter_lambda_atual()

        # Preparar dados para o gráfico
        intervalos = []
        valores_exp = []
        valores_calc = []

        for intervalo, valor_exp in CONSONANCE_RATINGS.items():
            intervalos.append(str(intervalo))
            valores_exp.append(valor_exp)

            # Calcular valores com lambda atual (código simplificado)
            if intervalo == 0:
                densidade = 0  # uníssono é tratado especialmente
            else:
                delta = intervalo * 2  # Convertendo para escala microtonal
                import math
                densidade = math.exp(-lambda_atual * delta) if delta > 0 else 0

            # Normalizar para comparação
            max_valor = max(CONSONANCE_RATINGS.values())
            densidade_norm = 2 * (densidade / max_valor) - 1
            valores_calc.append(densidade_norm)

        # Plotar barras comparativas
        bar_width = 0.35
        x = np.arange(len(intervalos))
        ax.bar(x - bar_width/2, valores_exp, bar_width, label='Experimental', alpha=0.7)
        ax.bar(x + bar_width/2, valores_calc, bar_width, label='Modelo (λ={:.4f})'.format(lambda_atual), alpha=0.7)

        ax.set_xlabel('Intervalo (semitons)')
        ax.set_ylabel('Consonância Normalizada')
        ax.set_title('Consonância Experimental vs. Modelo')
        ax.set_xticks(x)
        ax.set_xticklabels(intervalos)
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        canvas.draw()

    # Inicializar o gráfico
    atualizar_grafico()

    # Função para executar calibração com dados padrão
    def calibrar_com_dados_padrao():
        try:
            realizar_calibracao()
            lambda_atual = obter_lambda_atual()
            lambda_label.config(text=f"Valor atual de lambda: {lambda_atual:.4f}")
            atualizar_grafico()
            messagebox.showinfo("Calibração", f"Calibração concluída. Novo lambda: {lambda_atual:.4f}")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro durante a calibração: {e}")

    # Função para inserir um valor de lambda manualmente
    def definir_lambda_manual():
        try:
            valor = simpledialog.askfloat("Definir Lambda",
                                          "Digite o valor de lambda (0.01-1.0):",
                                          minvalue=0.01, maxvalue=1.0)
            if valor is not None:
                # Salvar o valor manualmente
                from densidade_intervalar import salvar_parametros_calibrados
                salvar_parametros_calibrados({"lambda": valor})
                lambda_atual = obter_lambda_atual()
                lambda_label.config(text=f"Valor atual de lambda: {lambda_atual:.4f}")
                atualizar_grafico()
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao definir lambda: {e}")

    # Função para coletar dados experimentais personalizados
    def coletar_dados_experimentais():
        try:
            # Esta função abrirá uma nova janela para coletar avaliações de consonância
            data_window = tk.Toplevel(calibration_window)
            data_window.title("Coleta de Dados Experimentais")
            data_window.geometry("500x400")

            # Armazenar valores inseridos
            dados_coletados = {}

            # Criar widgets para cada intervalo
            frame = ttk.Frame(data_window)
            frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            ttk.Label(frame, text="Avalie a consonância de cada intervalo (-1 a 1):").grid(
                row=0, column=0, columnspan=3, pady=10)

            # Definir intervalos para avaliação
            intervalos = [
                (0, "Unísono (C-C)"),
                (2, "Segunda menor (C-C#/Db)"),
                (4, "Segunda maior (C-D)"),
                (6, "Terça menor (C-D#/Eb)"),
                (8, "Terça maior (C-E)"),
                (10, "Quarta justa (C-F)"),
                (12, "Trítono (C-F#/Gb)"),
                (14, "Quinta justa (C-G)")
            ]

            # Criar sliders para cada intervalo
            sliders = {}
            row = 1
            for intervalo, descricao in intervalos:
                ttk.Label(frame, text=descricao).grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
                slider = ttk.Scale(frame, from_=-1.0, to=1.0, orient=tk.HORIZONTAL, length=200)
                slider.grid(row=row, column=1, padx=5, pady=5)
                slider.set(CONSONANCE_RATINGS.get(intervalo, 0))  # Valor padrão da literatura

                # Mostrar valor numérico
                var = tk.StringVar()
                var.set(f"{slider.get():.2f}")
                slider.configure(command=lambda v, var=var: var.set(f"{float(v):.2f}"))
                ttk.Label(frame, textvariable=var).grid(row=row, column=2, padx=5, pady=5)

                sliders[intervalo] = (slider, var)
                row += 1

            # Função para capturar os valores e calibrar
            def calibrar_com_dados_coletados():
                dados = {}
                for intervalo, (slider, _) in sliders.items():
                    dados[intervalo] = slider.get()

                # Fechar janela
                data_window.destroy()

                # Executar calibração
                realizar_calibracao(dados)
                lambda_atual = obter_lambda_atual()
                lambda_label.config(text=f"Valor atual de lambda: {lambda_atual:.4f}")
                atualizar_grafico()
                messagebox.showinfo("Calibração", f"Calibração concluída. Novo lambda: {lambda_atual:.4f}")

            # Botões
            button_frame = ttk.Frame(data_window)
            button_frame.pack(fill=tk.X, padx=10, pady=10)

            ttk.Button(button_frame, text="Calibrar", command=calibrar_com_dados_coletados).pack(side=tk.RIGHT, padx=5)
            ttk.Button(button_frame, text="Cancelar", command=data_window.destroy).pack(side=tk.RIGHT, padx=5)

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao coletar dados: {e}")

    # Função para visualizar o efeito de diferentes lambdas
    def analisar_efeito_lambda():
        try:
            # Esta função abrirá uma nova janela com o gráfico de análise
            analisar_consonancia_vs_lambda()
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na análise: {e}")

    # Adicionar botões
    button_frame = ttk.Frame(calibration_window)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

    ttk.Button(button_frame, text="Calibrar (Dados da Literatura)", command=calibrar_com_dados_padrao).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Inserir Dados Experimentais", command=coletar_dados_experimentais).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Definir Lambda Manualmente", command=definir_lambda_manual).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Analisar Efeito Lambda", command=analisar_efeito_lambda).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Fechar", command=calibration_window.destroy).pack(side=tk.RIGHT, padx=5)








# Adicione um botão ou item de menu na sua GUI principal para abrir a janela de calibração
# Exemplo:
def adicionar_opcao_calibracao(root, menu_principal=None):
    """
    Adiciona a opção de calibração ao menu principal ou como botão
    """
    if menu_principal:
        # Se houver um menu, adicionar como item de menu
        menu_principal.add_command(
            label="Calibrar Parâmetros",
            command=lambda: abrir_janela_calibracao(root)
        )
    else:
        # Caso contrário, adicionar como botão
        calibration_button = ttk.Button(
            root,
            text="Calibrar Parâmetros",
            command=lambda: abrir_janela_calibracao(root)
        )
        # Posicione o botão de acordo com o layout da sua GUI
        calibration_button.pack(pady=10)  # ou .grid() dependendo do seu layout

# No seu código de inicialização da GUI, chame:
# adicionar_opcao_calibracao(root, menu_principal)

        # Tornar modal

        config_window.transient(self.root)

        config_window.grab_set()

        self.root.wait_window(config_window)
