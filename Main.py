# Main.py
"""
Versão integrada do arquivo principal com suporte
para calibração do parâmetro lambda.
"""

import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import pandas as pd
import json

# Importar módulos para calibração de lambda
from calibration import (
    realizar_calibracao,
    obter_lambda_atual,
    analisar_consonancia_vs_lambda,
    visualizar_funcao_exponencial,
    testar_modelo_calibrado,
    calcular_densidade_intervalar
)

# Importações originais mantidas
from gui_components import DensityCalculatorGUI
from data_processor import (
    calcular_metricas, 
    salvar_resultados, 
    format_output_string, 
    generate_validation_text
)
from plot_metr_espectrais import extract_and_plot_metrics
from timbre_texture_analysis import plot_orchestration_analysis
from statistical_validation import (
    validate_metrics_reliability, 
    create_metrics_profile, 
    plot_metrics_comparison
)
import logging_config  # noqa: F401  (mantém o side-effect)

# Importar módulos de utilidades
from utils.serialize_utils import safe_operation, ensure_directory_exists, log_execution_time, safe_show_figure

# Importar tratamento de erros (ajusta para o módulo correto, se diferente)
from error_handler import (
    handle_exceptions, 
    log_and_show_error, 
    init_global_exception_hook,
    InputError, 
    CalculationError, 
    FileOperationError
)

# Importar configurações
from config import DEFAULT_OUTPUT_DIRECTORY, LOG_LEVEL, LOG_FORMAT

# Configurar logging
logger = logging.getLogger('main')

class DensityAnalyzerApp:
    """
    Classe principal da aplicação de análise de densidade musical.
    Integra interface gráfica com funcionalidades de processamento e calibração.
    """
    
    def __init__(self, root):
        """
        Inicializa a aplicação.
        
        Args:
            root: O widget raiz tkinter
        """
        self.root = root
        
        # Inicializar tratamento global de exceções
        init_global_exception_hook()
        
        # Garantir que o diretório de saída existe
        ensure_directory_exists(DEFAULT_OUTPUT_DIRECTORY)
        
        # Lista para armazenar histórico de resultados para validação
        self.resultados_historicos = []
        
        # Armazenar resultados completos para geração de relatórios
        self.resultados_completos = None
        
        # Criar callbacks para a interface
        callbacks = {
            'calculate': self.calcular,
            'clear': self.limpar,
            'generate_report': self.gerar_relatorio_cientifico,
            'execute_validation': self.executar_validacao,
            'calibrate': self.calibrar_lambda  # Novo callback para calibração
        }
        
        # Inicializar a interface
        self.gui = DensityCalculatorGUI(root, callbacks)
        
        # Adicionar opção de calibração ao menu
        self._adicionar_menu_calibracao()
    
    def _adicionar_menu_calibracao(self):
        """Adiciona um menu para as funções de calibração."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Menu de arquivo
        filemenu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Arquivo", menu=filemenu)
        filemenu.add_command(label="Sair", command=self.root.quit)
        
        # Menu de ferramentas
        toolsmenu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Ferramentas", menu=toolsmenu)
        
        # Submenu de calibração
        calibmenu = tk.Menu(toolsmenu, tearoff=0)
        toolsmenu.add_cascade(label="Calibração", menu=calibmenu)
        
        # Opções de calibração
        calibmenu.add_command(label="Calibrar Lambda", command=self.calibrar_lambda)
        calibmenu.add_command(label="Visualizar Função Exponencial", 
                              command=lambda: visualizar_funcao_exponencial(obter_lambda_atual()))
        calibmenu.add_command(label="Testar Modelo Calibrado", 
                              command=testar_modelo_calibrado)
        calibmenu.add_command(label="Analisar Efeito Lambda", 
                              command=lambda: analisar_consonancia_vs_lambda())
    
    @handle_exceptions(show_dialog=True)
    def calibrar_lambda(self):
        """
        Abre uma janela de diálogo para calibrar o parâmetro lambda.
        """
        # Criar janela de diálogo
        dialog = tk.Toplevel(self.root)
        dialog.title("Calibração de Lambda")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Mostrar valor atual
        lambda_atual = obter_lambda_atual()
        tk.Label(dialog, text=f"Valor atual de lambda: {lambda_atual:.4f}", 
                font=("Arial", 12)).pack(pady=10)
        
        # Opções de calibração
        tk.Button(dialog, text="Calibrar com Dados Padrão", 
                 command=lambda: self._executar_calibracao(dialog)).pack(pady=5)
        
        tk.Button(dialog, text="Visualizar Função Atual", 
                 command=lambda: visualizar_funcao_exponencial(lambda_atual)).pack(pady=5)
        
        tk.Button(dialog, text="Testar Modelo Atual", 
                 command=testar_modelo_calibrado).pack(pady=5)
        
        tk.Button(dialog, text="Análise de Sensibilidade", 
                 command=lambda: analisar_consonancia_vs_lambda()).pack(pady=5)
        
        # Opção para inserir lambda manualmente
        tk.Label(dialog, text="Ou insira um valor manualmente:").pack(pady=5)
        
        entry_frame = tk.Frame(dialog)
        entry_frame.pack(pady=5)
        
        lambda_entry = tk.Entry(entry_frame, width=10)
        lambda_entry.pack(side=tk.LEFT, padx=5)
        lambda_entry.insert(0, str(lambda_atual))
        
        tk.Button(entry_frame, text="Definir", 
                 command=lambda: self._definir_lambda_manual(dialog, lambda_entry.get())).pack(side=tk.LEFT)
        
        # Botão de fechar
        tk.Button(dialog, text="Fechar", command=dialog.destroy).pack(pady=10)
    
    def _executar_calibracao(self, dialog):
        """Executa a calibração e fecha o diálogo."""
        lambda_otimizado = realizar_calibracao()
        dialog.destroy()
        tk.messagebox.showinfo("Calibração Concluída", 
                              f"Calibração realizada com sucesso!\nNovo valor de lambda: {lambda_otimizado:.4f}")
    
    def _definir_lambda_manual(self, dialog, valor_texto):
        """Define o valor de lambda manualmente."""
        try:
            valor = float(valor_texto)
            if 0.01 <= valor <= 1.0:
                from calibration import salvar_parametros_calibrados
                salvar_parametros_calibrados({'lambda': valor})
                dialog.destroy()
                tk.messagebox.showinfo("Lambda Definido", 
                                      f"Valor de lambda definido manualmente: {valor:.4f}")
            else:
                tk.messagebox.showerror("Erro", "O valor deve estar entre 0.01 e 1.0")
        except ValueError:
            tk.messagebox.showerror("Erro", "Digite um número válido")
    
    @handle_exceptions(show_dialog=True)
    def _plot_detailed_visualizations(self, notas, duracoes, instrumentos, numeros_instrumentos, densidades_instrumento, pitches):
        """
        Plota visualizações detalhadas dos resultados.
        
        Args:
            notas: Lista de notas
            duracoes: Lista de durações
            instrumentos: Lista de instrumentos
            numeros_instrumentos: Lista de números de instrumentos
            densidades_instrumento: Lista de densidades calculadas
            pitches: Lista de valores MIDI
        """
        # Extrair e plotar métricas espectrais
        extract_and_plot_metrics(
            notas, duracoes, instrumentos, numeros_instrumentos, 
            densidades_instrumento
        )
        
        # Plotar análise de orquestração
        plot_orchestration_analysis(pitches, densidades_instrumento, instrumentos)
        
        # Criar DataFrame para análise e visualização se houver resultados completos
        if self.resultados_completos:
            profile_df = create_metrics_profile(
                self.resultados_completos['momentos_espectrais'], 
                self.resultados_completos['textura'], 
                self.resultados_completos['timbre']
            )
            
            # Plotar comparação de métricas se houver dados suficientes
            if len(profile_df.columns) > 2:
                plot_metrics_comparison(profile_df, "Análise de Métricas Musicais")
    
    @handle_exceptions(show_dialog=True)
    def limpar(self):
        """Limpa todos os campos de entrada e resultados."""
        self.gui.clear_inputs()
        self.resultados_completos = None
    
    @handle_exceptions(show_dialog=True)
    @log_execution_time
    def calcular(self):
        """
        Executa os cálculos de métricas com base nos dados de entrada.
        """
        # Obter dados de entrada
        input_data = self.gui.get_input_data()
    
        # Validar entrada básica
        if not input_data['notes']:
            raise InputError("Nenhuma nota selecionada. Marque pelo menos uma caixa de seleção.")
    
        # Log para diagnóstico
        logger.info(f"Iniciando cálculo com notas: {input_data['notes']}")
    
        # Executar cálculos
        resultados, densidades_instrumento, pitches = calcular_metricas(input_data)
    
        # Armazenar resultados para histórico e relatórios
        self.resultados_completos = resultados
        self.resultados_historicos.append(resultados)
    
        # Formatar e exibir resultados
        output_text = format_output_string(resultados)
        self.gui.show_results(output_text)
    
        # Atualizar a árvore de métricas
        self.gui.update_metrics_tree(resultados)
    
        # Criar gráficos embutidos
        self.gui.create_embedded_graphs(pitches, densidades_instrumento)
    
        # Se solicitado, mostrar gráficos detalhados
        if input_data['show_graphs']:
            self._plot_detailed_visualizations(
                input_data['notes'],
                input_data.get('durations', None),
                input_data['instruments'],
                input_data['num_instruments'],
                densidades_instrumento,
                pitches
            )
    
        # Se solicitado, salvar resultados
        if input_data['save_results']:
            try:
                # Criar nome de arquivo com timestamp para evitar sobrescrita
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
                # Garantir que o diretório existe
                import os
                from config import DEFAULT_OUTPUT_DIRECTORY
                if not os.path.exists(DEFAULT_OUTPUT_DIRECTORY):
                    os.makedirs(DEFAULT_OUTPUT_DIRECTORY, exist_ok=True)
                
                # Gerar caminho completo
                output_file = os.path.join(DEFAULT_OUTPUT_DIRECTORY, f"resultados_{timestamp}.json")
            
                # Salvar usando a função centralizada
                from data_processor import salvar_resultados
                arquivo_salvo = salvar_resultados(resultados, output_file)
            
                if arquivo_salvo:
                    from tkinter import messagebox
                    messagebox.showinfo("Informação", f"Resultados salvos com sucesso em:\n{arquivo_salvo}")
                    logger.info(f"Arquivo salvo com sucesso em: {arquivo_salvo}")
                else:
                    logger.warning("Função salvar_resultados não retornou caminho válido")
            except Exception as e:
                logger.error(f"Erro ao salvar resultados: {e}", exc_info=True)
                from tkinter import messagebox
                messagebox.showerror("Erro", f"Erro ao salvar resultados: {str(e)}")
    
        return resultados
    
    @handle_exceptions(show_dialog=True)
    @log_execution_time
    def executar_validacao(self):
        """Executa análise de validação estatística nos resultados históricos."""
        if len(self.resultados_historicos) < 5:  # Verificar quantidade mínima de amostras
            raise InputError(
                "São necessários pelo menos 5 conjuntos de resultados para validação estatística.",
                f"Atualmente possui {len(self.resultados_historicos)} conjuntos."
            )
        
        # Extrair métricas do histórico
        metricas_extraidas = {}
        
        # Usar conjunto de primeira ordem (métricas diretas)
        for resultado in self.resultados_historicos:
            for categoria, valores in resultado.items():
                if categoria not in ["dados_entrada"]:  # Ignorar dados de entrada
                    for metrica, valor in valores.items():
                        if isinstance(valor, (int, float)) and not np.isnan(valor) and not np.isinf(valor):
                            chave = f"{categoria}.{metrica}"
                            if chave not in metricas_extraidas:
                                metricas_extraidas[chave] = []
                            metricas_extraidas[chave].append(valor)
        
        # Analisar correlações entre métricas
        df_metricas = pd.DataFrame({k: v for k, v in metricas_extraidas.items() 
                                   if len(v) == len(self.resultados_historicos)})
        
        if df_metricas.shape[1] < 2:
            raise InputError("Não há métricas suficientes para análise de correlação.")
        
        # Executar validação
        resultados_validacao = validate_metrics_reliability(df_metricas)
        
        # Gerar texto de validação
        texto_validacao = generate_validation_text(resultados_validacao, len(self.resultados_historicos))
        
        # Exibir resultados
        self.gui.show_validation_results(texto_validacao)
        
        # Exibir gráfico da matriz de correlação
        plt.figure(figsize=(10, 8))
        import seaborn as sns
        sns.heatmap(resultados_validacao['correlation_matrix'], annot=True, cmap='coolwarm', center=0)
        plt.title('Matriz de Correlação entre Métricas')
        plt.tight_layout()
        plt.show()
    
    @handle_exceptions(show_dialog=True)
    @log_execution_time
    def gerar_relatorio_cientifico(self):
        """
        Gera relatórios científicos com os resultados da análise.
        Deve ser chamado após o cálculo das métricas.
        """
        # Verificar se há resultados para gerar relatório
        if self.resultados_completos is None:
            raise InputError(
                "É necessário calcular as métricas primeiro antes de gerar um relatório."
            )
        
        # Mostrar diálogo de configuração para o relatório
        self.gui.show_report_config_dialog(self._generate_report_with_config)
    
    @handle_exceptions(show_dialog=True)
    @log_execution_time
    def _generate_report_with_config(self, config):
        """
        Gera relatórios científicos com base na configuração fornecida.
        
        Args:
            config (dict): Configuração para o relatório
        """
        from scientific_report_generator import ScientificReportGenerator
        
        # Garantir que o diretório de saída existe
        output_dir = config.get('output_directory', DEFAULT_OUTPUT_DIRECTORY)
        ensure_directory_exists(output_dir)
        
        # Inicializar gerador de relatórios
        generator = ScientificReportGenerator(output_dir)
        
        report_paths = {}
        
        # Gerar relatórios selecionados
        if config['formats']['pdf']:
            pdf_path = generator.generate_pdf_report(self.resultados_completos, config)
            report_paths['pdf'] = pdf_path
        
        if config['formats']['paper']:
            paper_path = generator.generate_scientific_paper(self.resultados_completos, config)
            report_paths['paper'] = paper_path
        
        if config['formats']['figures']:
            figures_dir = generator.generate_publication_figures(self.resultados_completos, config)
            report_paths['figures'] = figures_dir
        
        if config['formats']['tables']:
            tables_path = generator.generate_data_tables(self.resultados_completos, config)
            report_paths['tables'] = tables_path
        
        # Mostrar resultados
        result_text = "Relatórios gerados:\n\n"
        for tipo, caminho in report_paths.items():
            if caminho:
                if tipo == 'pdf':
                    result_text += f"Relatório PDF: {os.path.basename(caminho)}\n"
                elif tipo == 'paper':
                    result_text += f"Artigo Científico: {os.path.basename(caminho)}\n"
                elif tipo == 'figures':
                    result_text += f"Figuras para Publicação: {os.path.basename(caminho)}\n"
                elif tipo == 'tables':
                    result_text += f"Tabelas de Dados: {os.path.basename(caminho)}\n"
        
        result_text += f"\nSalvos em: {output_dir}"
        
        # Mostrar mensagem com resultados
        tk.messagebox.showinfo("Relatórios Gerados", result_text)


# Ponto de entrada principal
if __name__ == "__main__":
    # Criar a janela principal
    root = tk.Tk()
    
    # Definir um tamanho inicial maior para garantir que todos os campos sejam visíveis
    root.geometry("1200x700")
    
    app = DensityAnalyzerApp(root)
    
    # Iniciar o loop principal da aplicação
    root.mainloop()

