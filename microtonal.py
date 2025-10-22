# microtonal.py
"""
Módulo central para lógica de notação microtonal.

Este módulo contém todas as constantes, mapeamentos e funções relacionadas
com notação microtonal, eliminando a duplicação em vários módulos do sistema.
Serve como interface comum para diferentes representações de notas microtonais.
"""

import re
import logging
import math
from typing import Dict, Tuple, List, Optional, Union

logger = logging.getLogger(__name__)


def normalizar_simbolos_nota(nota: str) -> str:
    """
    Normaliza todos os símbolos musicais especiais para um formato padrão internamente.
    
    Args:
        nota (str): Nota musical com qualquer tipo de símbolo
        
    Returns:
        str: Nota com símbolos normalizados para processamento interno
    """
    if not nota or not isinstance(nota, str):
        return nota
        
    # 1. Converter sustenido musical para # padrão (para processamento interno)
    nota = nota.replace("♯", "#")
    
    # 2. Normalizar símbolos de quartos de tom
    # Converter as setas para o formato +/- para consistência interna
    if QUARTO_TOM_ACIMA in nota:
        nota = nota.replace(QUARTO_TOM_ACIMA, "-")  # Nota mais baixa
    if QUARTO_TOM_ABAIXO in nota:
        nota = nota.replace(QUARTO_TOM_ABAIXO, "+")  # Nota mais alta
    
    return nota


# -----------------------------------------------------------------------------
# Constantes
# -----------------------------------------------------------------------------

# Símbolos Unicode para indicar quarto-de-tom
QUARTO_TOM_ACIMA = '↑'   # U+2191 - Seta para cima
QUARTO_TOM_ABAIXO = '↓'  # U+2193 - Seta para baixo
SUSTENIDO_MUSICAL = "♯"  # U+266F - Símbolo musical para sustenido
SIMBOLO_CENTS = "¢"      # U+00A2 - Símbolo de cents (opcional)

# Constantes relacionadas a cents
CENTS_POR_SEMITOM = 100  # 100 cents = 1 semitom
CENTS_POR_OITAVA = 1200  # 12 semitons × 100 cents
CENTS_PER_SEMITONE = CENTS_POR_SEMITOM  # Alias em inglês
CENTS_PER_OCTAVE = CENTS_POR_OITAVA    # Alias em inglês

# Oitava microtonal (24 passos por oitava)
TAMANHO_OITAVA_MICROTONAL = 24

# Frequência de referência para conversão MIDI <-> Hz
A4_FREQ = 440.0  # Hz
A4_MIDI = 69     # Valor MIDI para A4

# Lista de notas cromáticas (12 por oitava)
NOTAS_CROMATICAS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# -----------------------------------------------------------------------------
# Mapeamentos
# -----------------------------------------------------------------------------

# Mapeamento de nota para índice na escala cromática
ESCALA_CROMATICA = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11,
}

# Mapeamento de nota base para valor MIDI dentro da oitava (de microtonal_utils.py)
NOTE_BASE_MIDI = {
    # Naturais/sustenidos
    "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4,
    "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11,
    # Bemóis
    "Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10, "Cb": 11, "Fb": 4,
    # Quartos de tom com código +/-
    "C+": 0.5, "C#+": 1.5, "D+": 2.5, "D#+": 3.5, "E+": 4.5,
    "F+": 5.5, "F#+": 6.5, "G+": 7.5, "G#+": 8.5, "A+": 9.5, "A#+": 10.5, "B+": 11.5,
    "C-": 11.5, "C#-": 0.5, "D-": 1.5, "D#-": 2.5, "E-": 3.5,
    "F-": 4.5, "F#-": 5.5, "G-": 6.5, "G#-": 7.5, "A-": 8.5, "A#-": 9.5, "B-": 10.5,
    # Com símbolos musicais
    f"C{SUSTENIDO_MUSICAL}": 1, f"D{SUSTENIDO_MUSICAL}": 3, f"F{SUSTENIDO_MUSICAL}": 6,
    f"G{SUSTENIDO_MUSICAL}": 8, f"A{SUSTENIDO_MUSICAL}": 10,
    # Quartos de tom com símbolos Unicode
    f"C{QUARTO_TOM_ACIMA}": 0.5, f"C{QUARTO_TOM_ABAIXO}": 11.5,
    f"C{SUSTENIDO_MUSICAL}{QUARTO_TOM_ACIMA}": 1.5, f"C{SUSTENIDO_MUSICAL}{QUARTO_TOM_ABAIXO}": 0.5,
    f"D{QUARTO_TOM_ACIMA}": 2.5, f"D{QUARTO_TOM_ABAIXO}": 1.5,
    f"D{SUSTENIDO_MUSICAL}{QUARTO_TOM_ACIMA}": 3.5, f"D{SUSTENIDO_MUSICAL}{QUARTO_TOM_ABAIXO}": 2.5,
    f"E{QUARTO_TOM_ACIMA}": 4.5, f"E{QUARTO_TOM_ABAIXO}": 3.5,
    f"F{QUARTO_TOM_ACIMA}": 5.5, f"F{QUARTO_TOM_ABAIXO}": 4.5,
    f"F{SUSTENIDO_MUSICAL}{QUARTO_TOM_ACIMA}": 6.5, f"F{SUSTENIDO_MUSICAL}{QUARTO_TOM_ABAIXO}": 5.5,
    f"G{QUARTO_TOM_ACIMA}": 7.5, f"G{QUARTO_TOM_ABAIXO}": 6.5,
    f"G{SUSTENIDO_MUSICAL}{QUARTO_TOM_ACIMA}": 8.5, f"G{SUSTENIDO_MUSICAL}{QUARTO_TOM_ABAIXO}": 7.5,
    f"A{QUARTO_TOM_ACIMA}": 9.5, f"A{QUARTO_TOM_ABAIXO}": 8.5,
    f"A{SUSTENIDO_MUSICAL}{QUARTO_TOM_ACIMA}": 10.5, f"A{SUSTENIDO_MUSICAL}{QUARTO_TOM_ABAIXO}": 9.5,
    f"B{QUARTO_TOM_ACIMA}": 11.5, f"B{QUARTO_TOM_ABAIXO}": 10.5,
}

# Mapeamento para a escala microtonal - 24 passos por oitava
ESCALA_MICROTONAL = {
    'C': 1,  'C#-': 2, 'C#': 3,  'C#+': 4,
    'D': 5,  'D#-': 6, 'D#': 7,  'D#+': 8,
    'E': 9,  'E#-': 10,
    'F': 11, 'F#-': 12, 'F#': 13, 'F#+': 14,
    'G': 15, 'G#-': 16, 'G#': 17, 'G#+': 18,
    'A': 19, 'A#-': 20, 'A#': 21, 'A#+': 22,
    'B': 23, 'B#-': 24,
    # bemóis ↔ sustenidos + símbolos de ¼-tom
    'Cb+': 24, 'Db+': 2, 'Db': 3, 'Db-': 4,
    'Eb+': 6,  'Eb': 7, 'Eb-': 8,
    'Fb+': 10,
    'Gb+': 12, 'Gb': 13, 'Gb-': 14,
    'Ab+': 16, 'Ab': 17, 'Ab-': 18,
    'Bb+': 20, 'Bb': 21, 'Bb-': 22,
    # Notação simples de quartos de tom
    'C+': 2,  # C quarter-tone sharp
    'D+': 6,  # D quarter-tone sharp
    'E+': 10, # E quarter-tone sharp
    'F+': 12, # F quarter-tone sharp
    'G+': 16, # G quarter-tone sharp
    'A+': 20, # A quarter-tone sharp
    'B+': 24, # B quarter-tone sharp
    'C-': 24, # C quarter-tone flat (= B)
    'D-': 4,  # D quarter-tone flat
    'E-': 8,  # E quarter-tone flat
    'F-': 10, # F quarter-tone flat
    'G-': 14, # G quarter-tone flat
    'A-': 18, # A quarter-tone flat
    'B-': 22, # B quarter-tone flat
}

# Atribuir alias para compatibilidade
escala_microtonal = ESCALA_MICROTONAL

# Adicionar entradas com símbolos Unicode
for nota in list(ESCALA_MICROTONAL.keys()):
    if '+' in nota and len(nota) <= 3:
        nota_base = nota.replace('+', '')
        ESCALA_MICROTONAL[f"{nota_base}{QUARTO_TOM_ABAIXO}"] = ESCALA_MICROTONAL[nota]
    elif '-' in nota and len(nota) <= 3:
        nota_base = nota.replace('-', '')
        ESCALA_MICROTONAL[f"{nota_base}{QUARTO_TOM_ACIMA}"] = ESCALA_MICROTONAL[nota]

# Mapeamento para conversão entre notações
NOTACAO_QUARTOS_TOM = {
    f'C{QUARTO_TOM_ACIMA}': 'C-', f'C{QUARTO_TOM_ABAIXO}': 'C+',
    f'D{QUARTO_TOM_ACIMA}': 'D-', f'D{QUARTO_TOM_ABAIXO}': 'D+',
    f'E{QUARTO_TOM_ACIMA}': 'E-', f'E{QUARTO_TOM_ABAIXO}': 'E+',
    f'F{QUARTO_TOM_ACIMA}': 'F-', f'F{QUARTO_TOM_ABAIXO}': 'F+',
    f'G{QUARTO_TOM_ACIMA}': 'G-', f'G{QUARTO_TOM_ABAIXO}': 'G+',
    f'A{QUARTO_TOM_ACIMA}': 'A-', f'A{QUARTO_TOM_ABAIXO}': 'A+',
    f'B{QUARTO_TOM_ACIMA}': 'B-', f'B{QUARTO_TOM_ABAIXO}': 'B+',
    # Adicionar também o inverso
    'C-': f'C{QUARTO_TOM_ACIMA}', 'C+': f'C{QUARTO_TOM_ABAIXO}',
    'D-': f'D{QUARTO_TOM_ACIMA}', 'D+': f'D{QUARTO_TOM_ABAIXO}',
    'E-': f'E{QUARTO_TOM_ACIMA}', 'E+': f'E{QUARTO_TOM_ABAIXO}',
    'F-': f'F{QUARTO_TOM_ACIMA}', 'F+': f'F{QUARTO_TOM_ABAIXO}',
    'G-': f'G{QUARTO_TOM_ACIMA}', 'G+': f'G{QUARTO_TOM_ABAIXO}',
    'A-': f'A{QUARTO_TOM_ACIMA}', 'A+': f'A{QUARTO_TOM_ABAIXO}',
    'B-': f'B{QUARTO_TOM_ACIMA}', 'B+': f'B{QUARTO_TOM_ABAIXO}',
}

# Equivalências entre notas com bemol e sustenido
EQUIVALENCIAS_NOTAS = {
    # Bemol → sustenido
    'Cb': 'B', 'Db': 'C#', 'Eb': 'D#', 'Fb': 'E', 'Gb': 'F#',
    'Ab': 'G#', 'Bb': 'A#',
    # Modificadores especiais (¼-tom em código + / -)
    'C-': 'B#',  'C+': 'B-',
    'D-': 'C#+', 'D+': 'C#-',
    'E-': 'D#+', 'E+': 'D#-',
    'F-': 'E#+', 'F+': 'E-',
    'G-': 'F#+', 'G+': 'F#-',
    'A-': 'G#+', 'A+': 'G#-',
    'B-': 'A#+', 'B+': 'A#-',
}

# Padrões de expressão regular para diferentes formatos de nota (do microtonal_utils.py)
_RE_STANDARD = re.compile(r"^([A-Ga-g][#♯b]?)(\d)$")
_RE_QUARTER = re.compile(r"^([A-Ga-g][#♯b]?)([+-])(\d)$")
_RE_ARROW = re.compile(fr"^([A-Ga-g][#♯b]?)([{QUARTO_TOM_ACIMA}{QUARTO_TOM_ABAIXO}])(\d)$")
_RE_CENTS = re.compile(r"^([A-Ga-g][#♯b]?\d)([+-]\d{1,2})c$")

# -----------------------------------------------------------------------------
# Funções de manipulação microtonal
# -----------------------------------------------------------------------------

def is_valid_note(nota: str) -> bool:
    """
    Verifica se uma string representa uma nota musical válida.
    Aceita notação microtonal em cents, símbolos, ou combinações.
    
    Args:
        nota (str): String de nota para validar
        
    Returns:
        bool: True se válida, False caso contrário
    """
    if not isinstance(nota, str) or not nota:
        return False
    
    # Normalizar símbolos para processamento consistente
    nota_normalizada = nota.replace("#", SUSTENIDO_MUSICAL)
    
    # Verificar usando expressões regulares compiladas (mais eficiente)
    if (_RE_STANDARD.match(nota_normalizada) or 
        _RE_QUARTER.match(nota_normalizada) or 
        _RE_ARROW.match(nota_normalizada) or
        _RE_CENTS.match(nota_normalizada)):
        return True
    
    # Padrões adicionais para casos complexos
    patterns = [
        # Nota com seta/quarto de tom + cents
        f'^[A-Ga-g][#♯b]?[{QUARTO_TOM_ACIMA}{QUARTO_TOM_ABAIXO}][0-9][+-][0-9]{{1,2}}c$',
        # Nota com modificador +/- e oitava + cents
        r'^[A-Ga-g][#♯b]?[-+][0-9][+-][0-9]{1,2}c$'
    ]
    
    # Verificar padrões adicionais
    for pattern in patterns:
        if re.match(pattern, nota) or re.match(pattern, nota_normalizada):
            return True
    
    return False


def extract_cents(nota: str) -> Tuple[str, int]:
    """
    Extrai o componente cents de uma nota, se presente.
    Suporta notas com símbolos microtonais, incluindo combinações complexas.
    
    Args:
        nota (str): String de nota possivelmente com cents
        
    Returns:
        Tuple[str, int]: (nota base sem cents, valor em cents)
    """
    if not isinstance(nota, str) or 'c' not in nota:
        return nota, 0
        
    try:
        # Usar a expressão regular compilada primeiro
        match = _RE_CENTS.match(nota)
        if match:
            base_note, cents_part = match.groups()
            cents_value = int(cents_part)
            return base_note, cents_value
        
        # Normalizar símbolos para consistência
        nota_processada = nota.replace("♯", "#")
        
        # Padrões para extrair cents em diferentes formatos de nota
        patterns = [
            # Nota simples com cents: C4+50c
            r'^([A-Ga-g][#b]?[0-9])([+-][0-9]{1,2})c$',
            # Nota com símbolo de quarto de tom e cents: C↓4+50c
            f'^([A-Ga-g][#b]?[{QUARTO_TOM_ACIMA}{QUARTO_TOM_ABAIXO}][0-9])([+-][0-9]{{1,2}})c$',
            # Nota com modificador +/- e cents: C+4-50c
            r'^([A-Ga-g][#b]?[-+][0-9])([+-][0-9]{1,2})c$'
        ]
        
        # Tentar cada padrão
        for pattern in patterns:
            match = re.match(pattern, nota_processada)
            if match:
                base_note, cents_part = match.groups()
                cents_value = int(cents_part)
                
                # Restaurar símbolos Unicode se necessário
                if "♯" in nota and "#" in base_note:
                    base_note = base_note.replace("#", "♯")
                
                return base_note, cents_value
        
        # Se nenhum padrão corresponder
        return nota, 0
    except Exception as e:
        logger.error(f"Erro ao extrair cents da nota {nota}: {e}")
        return nota, 0


def converter_para_sustenido(nota: str) -> str:
    """
    Converte uma nota com bemol para equivalente com sustenido.
    Suporta notação microtonal com símbolos e cents.
    Alias: to_sharp()
    
    Args:
        nota (str): String de nota em qualquer formato
        
    Returns:
        str: Nota com sustenidos em vez de bemóis
    """
    if not is_valid_note(nota):
        logger.warning(f"Formato de nota inválido: {nota}")
        return nota
    
    try:
        # Extrair cents primeiro, se presentes
        base_note, cents = extract_cents(nota)
        
        # Verificar se a nota tem símbolo microtonal
        has_microtonal_symbol = QUARTO_TOM_ACIMA in base_note or QUARTO_TOM_ABAIXO in base_note
        
        if has_microtonal_symbol:
            # Para notas com símbolos microtonais, extrair parte da nota, símbolo e oitava
            pattern = f'^([A-Ga-g][#b]?)([{QUARTO_TOM_ACIMA}{QUARTO_TOM_ABAIXO}])([0-9])$'
            match = re.match(pattern, base_note)
            if match:
                nota_parte, simbolo, oitava = match.groups()
                
                # Converter a parte da nota se necessário
                if nota_parte in EQUIVALENCIAS_NOTAS:
                    nota_parte = EQUIVALENCIAS_NOTAS[nota_parte]
                    
                # Reconstruir a nota com símbolo e oitava
                base_convertida = f"{nota_parte}{simbolo}{oitava}"
            else:
                # Se não conseguir extrair o padrão, manter como está
                base_convertida = base_note
        else:
            # Processamento para notas sem símbolos microtonais
            nota_parte = base_note[:-1]  # Tudo exceto o último caractere (oitava)
            oitava = base_note[-1]      # Último caractere (oitava)
            
            # Converter nota se existir no dicionário de equivalências
            if nota_parte in EQUIVALENCIAS_NOTAS:
                nota_parte = EQUIVALENCIAS_NOTAS[nota_parte]
                
            base_convertida = f"{nota_parte}{oitava}"
        
        # Recolocar os cents, se presentes
        if cents != 0:
            cents_str = f"+{cents}" if cents > 0 else f"{cents}"
            return f"{base_convertida}{cents_str}c"
        else:
            return base_convertida
            
    except Exception as e:
        logger.error(f"Erro ao converter nota: {e}")
        return nota  # Retornar a nota original em caso de erro


# Alias para compatibilidade com microtonal_utils.py
to_sharp = converter_para_sustenido


def converter_notacao_microtonal(nota: str) -> str:
    """
    Converte entre notações de quartos de tom (código e símbolos musicais).
    
    Args:
        nota (str): Nota musical com possível modificador de quarto de tom
        
    Returns:
        str: Nota convertida para a notação alternativa
    """
    # Separar a nota base da oitava
    if len(nota) < 2:
        return nota
        
    # Verificar se a última posição é um número (oitava)
    if nota[-1].isdigit():
        nota_base = nota[:-1]
        oitava = nota[-1]
    else:
        nota_base = nota
        oitava = ""
    
    # Converter a notação da nota base
    nova_nota_base = NOTACAO_QUARTOS_TOM.get(nota_base, nota_base)
    
    # Reconstruir a nota completa com oitava
    return f"{nova_nota_base}{oitava}"


def preprocess_nota(nota: str) -> str:
    """
    Preprocessa uma nota musical para garantir compatibilidade com diferentes notações.
    Converte símbolos Unicode (↑/↓) para notação +/-.
    
    A convenção adotada é:
    - ↑ (seta para cima) representa uma nota mais baixa e é convertida para '-'
    - ↓ (seta para baixo) representa uma nota mais alta e é convertida para '+'
    
    Args:
        nota (str): Nota musical em qualquer formato
        
    Returns:
        str: Nota processada em formato padronizado
    """
    if not nota:
        return nota
        
    # Extrair e preservar componente de cents, se presente
    base_note, cents = extract_cents(nota)
    
    # Converter símbolos Unicode para notação +/-
    # IMPORTANTE: A lógica foi invertida para corresponder à convenção visual
    processed_note = base_note
    if QUARTO_TOM_ACIMA in processed_note:
        # Seta para cima (↑) indica nota mais baixa, portanto '-'
        processed_note = processed_note.replace(QUARTO_TOM_ACIMA, '-')
    if QUARTO_TOM_ABAIXO in processed_note:
        # Seta para baixo (↓) indica nota mais alta, portanto '+'
        processed_note = processed_note.replace(QUARTO_TOM_ABAIXO, '+')
    
    # Recolocar cents, se presentes
    if cents != 0:
        cents_str = f"+{cents}" if cents > 0 else f"{cents}"
        return f"{processed_note}{cents_str}c"
    
    return processed_note


def nota_para_posicao(nota: str) -> float:
    """
    Converte strings como 'C4', 'F#-3', 'G#+5' ou 'D4+50c' etc.
    para um valor float: (oitava * 24) + deslocamento [1..24] + fração de cents.
    
    Args:
        nota (str): String da nota
        
    Returns:
        float: Valor da posição
        
    Raises:
        ValueError: Se a nota for inválida ou não reconhecida
    """
    # Normalizar o símbolo Unicode ♯ para # para processamento interno
    if isinstance(nota, str):
        nota = nota.replace("♯", "#")
    
    # Extrair cents se presente
    base_note, cents = extract_cents(nota)
    cents_fraction = cents / CENTS_POR_SEMITOM  # Converter para fração de semitom
    
    # Pré-processar para formato uniforme
    base_note = preprocess_nota(base_note)
    
    # Processar a nota base - atualizar o padrão para incluir ambos os símbolos
    padrao = r'([A-Ga-g][#♯b]?[-+]?)(\d+)'
    match = re.match(padrao, base_note)
    if not match:
        raise ValueError(f"Nota '{base_note}' não corresponde ao padrão esperado.")

    nota_base, oitava_str = match.groups()
    nota_base = nota_base.upper()  # Padronizar para maiúsculas
    
    # Garantir que qualquer ♯ remanescente seja convertido para #
    nota_base = nota_base.replace("♯", "#")
    
    oitava = int(oitava_str)
    
    # Verificar se a nota existe na escala microtonal
    if nota_base not in ESCALA_MICROTONAL:
        # Tentar uma variante com # em vez de ♯
        nota_base_alt = nota_base.replace("♯", "#")
        if nota_base_alt in ESCALA_MICROTONAL:
            nota_base = nota_base_alt
        else:
            raise ValueError(f"Nota '{nota_base}' não está definida na escala microtonal.")
    
    posicao_na_oitava = ESCALA_MICROTONAL[nota_base]  # de 1..24
    posicao = posicao_na_oitava + (TAMANHO_OITAVA_MICROTONAL * oitava)
    
    # Adicionar a fração de cents
    posicao_cents = posicao + (cents_fraction * 2)  # 2 = fator de conversão para manter proporção
    
    return posicao_cents


def note_to_midi(note: str) -> float:
    """Converte uma nota textual para o número MIDI (float)."""
    if not isinstance(note, str) or not note:
        return 60.0

    # Normalizar símbolos: converter ♯ para # para processamento interno
    note = note.replace("♯", "#")
    
    # Extrair cents se presentes
    base_note, cents_value = extract_cents(note)
    base_midi = 0
    
    # -----------------------------------------------------------------
    # 1) nota com cents já extraídos
    # -----------------------------------------------------------------
    if cents_value != 0:
        return note_to_midi(base_note) + (cents_value / 100.0)
    
    # -----------------------------------------------------------------
    # 2) flecha ↑ - nota mais baixa
    # -----------------------------------------------------------------
    m = re.fullmatch(r"([A-Ga-g][#b]?)" + QUARTO_TOM_ACIMA + r"(\d)", note)
    if m:
        base, octave = m.groups()
        base = base.capitalize()
        if base in ESCALA_CROMATICA:
            return (int(octave) + 1) * 12 + ESCALA_CROMATICA[base] - 0.5  # SUBTRAIR 0.5

    # -----------------------------------------------------------------
    # 3) flecha ↓ - nota mais alta
    # -----------------------------------------------------------------
    m = re.fullmatch(r"([A-Ga-g][#b]?)" + QUARTO_TOM_ABAIXO + r"(\d)", note)
    if m:
        base, octave = m.groups()
        base = base.capitalize()
        if base in ESCALA_CROMATICA:
            return (int(octave) + 1) * 12 + ESCALA_CROMATICA[base] + 0.5  # ADICIONAR 0.5

    # -----------------------------------------------------------------
    # 4) código de ¼-tom (+ / -)
    # -----------------------------------------------------------------
    m = re.fullmatch(r"([A-Ga-g][#b]?)([+-])(\d)", note)
    if m:
        base, sign, octave = m.groups()
        base = base.capitalize()
        if base in ESCALA_CROMATICA:
            delta = 0.5 if sign == "+" else -0.5
            return (int(octave) + 1) * 12 + ESCALA_CROMATICA[base] + delta

    # -----------------------------------------------------------------
    # 5) nota padrão
    # -----------------------------------------------------------------
    m = re.fullmatch(r"([A-Ga-g][#b]?)(\d)", note)
    if m:
        base, octave = m.groups()
        base = base.capitalize()
        if base in ESCALA_CROMATICA:
            return (int(octave) + 1) * 12 + ESCALA_CROMATICA[base]

    # Falhou? Avisa e devolve C4
    logger.warning(f"Formato de nota não reconhecido: {note}")
    return 60.0


def midi_to_note_name(midi_number: float, include_cents: bool = False, use_symbols: bool = False) -> str:
    """
    Converte um número MIDI para um nome de nota, com exibição opcional de cents ou símbolos microtonais.
    
    Args:
        midi_number (float): Número da nota MIDI
        include_cents (bool): Incluir cents na saída para valores MIDI não inteiros
        use_symbols (bool): Usar símbolos microtonais para quartos de tom
        
    Returns:
        str: Nome da nota (ex: 'C4', 'G#5', 'A4+50c' ou 'D↑4')
    """
    if midi_number < 0 or midi_number > 127:
        return "N/A"
    
    # Separar a parte inteira e fracionária do valor MIDI
    midi_int = int(midi_number)
    midi_frac = midi_number - midi_int
    
    # Calcular oitava e índice da nota
    octave = (midi_int // 12) - 1
    note_index = midi_int % 12
    
    # Nome básico da nota sem cents
    basic_note = f"{NOTAS_CROMATICAS[note_index]}{octave}"
    
    # Se queremos usar símbolos microtonais para quartos de tom
    if use_symbols and abs(midi_frac) > 0.001:
        # Converter para cents para determinar o símbolo correto
        cents = round(midi_frac * CENTS_POR_SEMITOM)
        
        # Verificar se está próximo de um quarto de tom
        if 15 <= cents <= 35:  # ~quarto de tom acima
            note_base = NOTAS_CROMATICAS[note_index]
            return f"{note_base}{QUARTO_TOM_ABAIXO}{octave}"  # Usa seta para baixo
        elif -35 <= cents <= -15:  # ~quarto de tom abaixo
            note_base = NOTAS_CROMATICAS[note_index]
            return f"{note_base}{QUARTO_TOM_ACIMA}{octave}"  # Usa seta para cima
    
    # Adicionar cents se solicitado e se houver uma parte fracionária
    if include_cents and abs(midi_frac) > 0.001:  # Pequena margem para evitar problemas de ponto flutuante
        cents = round(midi_frac * CENTS_POR_SEMITOM)
        if cents > 0:
            return f"{basic_note}+{cents}c"
        else:
            return f"{basic_note}{cents}c"  # O sinal negativo já está incluído
    
    return basic_note


def midi_to_hz(midi_pitch: float) -> float:
    """
    Converte altura MIDI para frequência em Hz.
    
    Args:
        midi_pitch (float): Valor de altura MIDI
        
    Returns:
        float: Frequência em Hertz
    """
    return A4_FREQ * (2 ** ((midi_pitch - A4_MIDI) / 12))


def hz_to_midi(frequency: float) -> float:
    """
    Converte frequência em Hz para altura MIDI.
    
    Args:
        frequency (float): Frequência em Hertz
        
    Returns:
        float: Valor de altura MIDI
    """
    if frequency <= 0:
        return 0
    return A4_MIDI + 12 * math.log2(frequency / A4_FREQ)


def frequency_to_note_name(frequency: float, include_cents: bool = False, use_symbols: bool = False) -> str:
    """
    Converte frequência para o nome da nota musical mais próxima, com cents ou símbolos opcionais.
    
    Args:
        frequency (float): Frequência em Hertz
        include_cents (bool): Incluir cents na saída
        use_symbols (bool): Usar símbolos microtonais quando apropriado
        
    Returns:
        str: Nome da nota (ex: 'A4', 'C5', 'G3+35c' ou 'F↑4')
    """
    if frequency <= 0:
        return "Inválida"
    
    try:
        # Converter frequência para MIDI
        midi_number = hz_to_midi(frequency)
        
        # Converter MIDI para nome da nota
        return midi_to_note_name(midi_number, include_cents, use_symbols)
    except Exception as e:
        logger.error(f"Erro ao converter frequência {frequency} para nome da nota: {e}")
        return "Inválida"


# -----------------------------------------------------------------------------
# Funções para debug e teste
# -----------------------------------------------------------------------------

def debug_note_conversion(nota: str) -> None:
    """
    Função para debugging da conversão de notas. Mostra todas as representações.
    
    Args:
        nota (str): Nota para depurar
    """
    print(f"Nota original: {nota}")
    
    try:
        # Testar cada função de conversão
        midi_value = note_to_midi(nota)
        print(f"  MIDI: {midi_value:.2f}")
        
        posicao = nota_para_posicao(nota)
        print(f"  Posição na escala microtonal: {posicao:.2f}")
        
        preprocessed = preprocess_nota(nota)
        print(f"  Pré-processada: {preprocessed}")
        
        with_sharp = converter_para_sustenido(nota)
        print(f"  Convertida para sustenido: {with_sharp}")
        
        freq = midi_to_hz(midi_value)
        print(f"  Frequência: {freq:.2f} Hz")
        
        back_to_note = midi_to_note_name(midi_value, include_cents=True)
        print(f"  MIDI → Nota (com cents): {back_to_note}")
        
        back_to_note_symbols = midi_to_note_name(midi_value, use_symbols=True)
        print(f"  MIDI → Nota (com símbolos): {back_to_note_symbols}")
        
    except Exception as e:
        print(f"  ERRO: {str(e)}")


def test_microtonal_functions():
    """Função para testar as conversões de nota microtonal."""
    test_notes = [
        "C4", "C#4", "Db4", "D4",                     # Notas padrão
        "C4+50c", "D4-25c", "F#4+75c",                # Notas com cents
        f"C{QUARTO_TOM_ACIMA}4", f"D{QUARTO_TOM_ABAIXO}5",  # Notas com símbolos
        "C+4", "D-5", "F#-3", "Bb+2"                  # Notas com códigos +/-
    ]
    
    print("TESTE DE FUNÇÕES MICROTONAIS:")
    print("=" * 50)
    
    for note in test_notes:
        debug_note_conversion(note)
        print("-" * 50)


# -----------------------------------------------------------------------------
# Lista de exportações públicas
# -----------------------------------------------------------------------------

__all__ = [
    # Constantes
    "QUARTO_TOM_ACIMA", "QUARTO_TOM_ABAIXO", "SUSTENIDO_MUSICAL", "SIMBOLO_CENTS",
    "CENTS_POR_SEMITOM", "CENTS_POR_OITAVA", "CENTS_PER_SEMITONE", "CENTS_PER_OCTAVE",
    "NOTAS_CROMATICAS", "TAMANHO_OITAVA_MICROTONAL", "A4_FREQ", "A4_MIDI",
    
    # Mapeamentos
    "ESCALA_CROMATICA", "ESCALA_MICROTONAL", "escala_microtonal", "NOTE_BASE_MIDI",
    "NOTACAO_QUARTOS_TOM", "EQUIVALENCIAS_NOTAS",
    
    # Funções principais
    "normalizar_simbolos_nota", "is_valid_note", "extract_cents", 
    "converter_para_sustenido", "to_sharp", "converter_notacao_microtonal", 
    "preprocess_nota", "nota_para_posicao", "note_to_midi", "midi_to_note_name", 
    "midi_to_hz", "hz_to_midi", "frequency_to_note_name",
    
    # Funções de debug
    "debug_note_conversion", "test_microtonal_functions"
]


# -----------------------------------------------------------------------------
# Execução principal para testes
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Configurar logging básico
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    
    # Executar testes
    test_microtonal_functions()

