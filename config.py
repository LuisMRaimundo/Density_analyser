# config.py
# Configurações centralizadas para toda a aplicação

from typing import Dict, List, Tuple, Any
import os



# -------------------------------------------------------------------
# Coloque no topo do ficheiro (ou em config.py) um valor global:
MAX_DENS_GLOBAL = 20.0          # calibrar com o vosso corpus

USE_LOG_COMPRESSION = True
# -------------------------------------------------------------------



# ===================================================================
# CONFIGURAÇÕES GERAIS
# ===================================================================

# Diretório padrão para salvar arquivos
DEFAULT_OUTPUT_DIRECTORY = os.path.join(os.path.expanduser("~"), "Densidade_Espectral_Output")

# Definição de níveis de log
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ===================================================================
# CONFIGURAÇÕES MUSICAIS
# ===================================================================

# Notação e intervalos
TAMANHO_OITAVA_MICROTONAL = 24
NOTAS_CROMATICAS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Mapeamento de "nota base" -> posição (1..24)
ESCALA_MICROTONAL = {
    'C': 1, 'C#-': 2, 'C#': 3, 'C#+': 4,
    'D': 5, 'D#-': 6, 'D#': 7, 'D#+': 8,
    'E': 9, 'E#-': 10,
    'F': 11, 'F#-': 12, 'F#': 13, 'F#+': 14,
    'G': 15, 'G#-': 16, 'G#': 17, 'G#+': 18,
    'A': 19, 'A#-': 20, 'A#': 21, 'A#+': 22,
    'B': 23, 'B#-': 24,
    'Cb+': 24,
    'Db+': 2, 'Db': 3, 'Db-': 4,
    'Eb+': 6, 'Eb': 7, 'Eb-': 8,
    'Fb+': 10,
    'Gb+': 12, 'Gb': 13, 'Gb-': 14,
    'Ab+': 16, 'Ab': 17, 'Ab-': 18,
    'Bb+': 20, 'Bb': 21, 'Bb-': 22,
}

# Equivalências entre notas com bemol e sustenido
EQUIVALENCIAS_NOTAS = {
    'Cb': 'B', 'Db': 'C#', 'Eb': 'D#', 'Fb': 'E', 'Gb': 'F#',
    'Ab': 'G#', 'Bb': 'A#',
    'C-': 'B#', 'D-': 'C#+', 'E-': 'D#+', 'F-': 'E#+', 'G-': 'F#+',
    'A-': 'G#+', 'B-': 'A#+',
    'C+': 'B-', 'D+': 'C#-', 'E+': 'D#-', 'F+': 'E-', 'G+': 'F#-',
    'A+': 'G#-', 'B+': 'A#-',
}

# Mapeamento de nota para valor MIDI
NOTA_PARA_MIDI_BASE = {
    'C': 0, 'C#': 1, 'Db': 1, 'C#-': 0.5, 'C#+': 1.5,
    'D': 2, 'D#': 3, 'Eb': 3, 'D#-': 2.5, 'D#+': 3.5,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'F#-': 5.5, 'F#+': 6.5,
    'G': 7, 'G#': 8, 'Ab': 8, 'G#-': 7.5, 'G#+': 8.5,
    'A': 9, 'A#': 10, 'Bb': 10, 'A#-': 9.5, 'A#+': 10.5,
    'B': 11, 'C-': 11.5, 'B#': 12, 'B#-': 11.5
}

# ===================================================================
# CONFIGURAÇÕES DA INTERFACE GRÁFICA
# ===================================================================

# Tamanhos e estilos da interface
UI_FONT_FAMILY = "Arial"
UI_DEFAULT_FONT_SIZE = 10
UI_TITLE_FONT_SIZE = 12
UI_PADDING = 10
UI_MARGIN = 5

# Cores
UI_COLORS = {
    "primary": "#4472C4",
    "secondary": "#5B9BD5",
    "background": "#F2F2F2",
    "text": "#333333",
    "warning": "#FFC000",
    "error": "#C00000",
    "success": "#70AD47",
}

# Opções para os menus dropdown
DYNAMIC_LEVELS = ['pppp', 'ppp', 'pp', 'p', 'mf', 'f', 'ff', 'fff', 'ffff']
INSTRUMENT_LIST = ['flautim', 'flauta', 'Oboe', 'Corne_ingles', 'clarinete',
                   'clarinete baixo', 'fagote', 'contrafagote', 'violino']
OCTAVE_LIST = [str(i) for i in range(10)]
QUANTITY_LIST = [str(i) for i in range(1, 21)]
DURATION_LIST = [str(i) for i in range(1, 17)]

# ===================================================================
# CONFIGURAÇÕES DE ANÁLISE E PROCESSAMENTO
# ===================================================================

# Parâmetros de análise espectral
DEFAULT_LAMBDA = 0.05  # Parâmetro lambda para função de decaimento exponencial
MIDI_BASE_FREQUENCY = 440.0  # Frequência referência A4
MIDI_BASE_NOTE = 69  # Valor MIDI para A4

# Valores padrão para pesos na análise
DEFAULT_WEIGHT_FACTOR = 0.5

# Configurações para validação estatística
MIN_SAMPLES_FOR_VALIDATION = 5
HIGH_CORRELATION_THRESHOLD = 0.7

# ===================================================================
# CONFIGURAÇÕES DE RELATÓRIOS
# ===================================================================

# Formatos de saída suportados
REPORT_FORMATS = {
    "pdf": "PDF Report",
    "txt": "Text Report",
    "json": "JSON Data",
    "csv": "CSV Data",
}

# Configurações de figuras para relatórios
FIGURE_DPI = 300
DEFAULT_FIGURE_FORMAT = "png"
FIGURE_SIZES = {
    "small": (6, 4),
    "medium": (8, 6),
    "large": (12, 8),
}

# ===================================================================
# MENSAGENS E TEXTOS
# ===================================================================

# Textos de erro comuns
ERROR_MESSAGES = {
    "invalid_note": "Nota musical inválida",
    "calculation_error": "Erro durante o cálculo das métricas",
    "missing_inputs": "Preencha todos os campos obrigatórios",
    "file_save_error": "Erro ao salvar o arquivo",
    "insufficient_samples": "Número insuficiente de amostras para validação estatística",
    "module_not_found": "Módulo não encontrado",
}

# Textos informativos
INFO_MESSAGES = {
    "calculation_success": "Cálculos realizados com sucesso",
    "report_saved": "Relatório salvo com sucesso",
    "validation_success": "Validação estatística concluída",
}
