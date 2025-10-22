# utils.py
# Essential constants and microtonal utils section

import re
import os
import logging
from typing import Dict, List, Tuple, Union, Optional, Any, Callable, TypeVar, cast
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import io
from PIL import Image
from contextlib import contextmanager
from config import TAMANHO_OITAVA_MICROTONAL   # << linha nova
from utils.notes import note_to_midi, normalize_note_string

import logging
logger = logging.getLogger(__name__)
from microtonal import (
    QUARTO_TOM_ACIMA, QUARTO_TOM_ABAIXO, CENTS_POR_SEMITOM, CENTS_POR_OITAVA,
    NOTAS_CROMATICAS, ESCALA_MICROTONAL as escala_microtonal, NOTACAO_QUARTOS_TOM, EQUIVALENCIAS_NOTAS,
    converter_notacao_microtonal, is_valid_note, extract_cents, converter_para_sustenido,
    nota_para_posicao, note_to_midi, midi_to_note_name, midi_to_hz, frequency_to_note_name
)

# Símbolos Unicode para indicar quarto-de-tom
QUARTO_TOM_ACIMA  = '↑'   # U+2191
QUARTO_TOM_ABAIXO = '↓'   # U+2193

# ------------------------------------------------------------
#   Constantes de cents  (faltavam depois da última limpeza)
# ------------------------------------------------------------
CENTS_POR_SEMITOM  = 100          # 100 cents = 1 semitom
CENTS_POR_OITAVA   = 1200         # 12 semitons × 100 cents

# Lista simplificada com apenas notas cromáticas padrão (12 por oitava)
NOTAS_CROMATICAS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# --------------------------------------------------------------------------
#  ESCALA MICROTONAL – 24 passos por oitava
#  1 = C  •  2 = C#-  •  3 = C#  •  4 = C#+  •  …  •  24 = B#-
# --------------------------------------------------------------------------
escala_microtonal = {
    'C': 1,  'C#-': 2, 'C#': 3,  'C#+': 4,
    'D': 5,  'D#-': 6, 'D#': 7,  'D#+': 8,
    'E': 9,  'E#-': 10,
    'F': 11, 'F#-': 12, 'F#': 13, 'F#+': 14,
    'G': 15, 'G#-': 16, 'G#': 17, 'G#+': 18,
    'A': 19, 'A#-': 20, 'A#': 21, 'A#+': 22,
    'B': 23, 'B#-': 24,
    # bemóis ↔ sustenidos + símbolos de ¼-tom, se quiser incluir
    'Cb+': 24, 'Db+': 2, 'Db': 3, 'Db-': 4,
    'Eb+': 6,  'Eb': 7, 'Eb-': 8,
    'Fb+': 10,
    'Gb+': 12, 'Gb': 13, 'Gb-': 14,
    'Ab+': 16, 'Ab': 17, 'Ab-': 18,
    'Bb+': 20, 'Bb': 21, 'Bb-': 22,
    # equivalentes com símbolos
    f'C{QUARTO_TOM_ACIMA}': 2, f'C#{QUARTO_TOM_ABAIXO}': 2,
    f'C#{QUARTO_TOM_ACIMA}': 4, f'D{QUARTO_TOM_ABAIXO}': 4,
    f'D{QUARTO_TOM_ACIMA}': 6, f'D#{QUARTO_TOM_ABAIXO}': 6,
    f'D#{QUARTO_TOM_ACIMA}': 8, f'E{QUARTO_TOM_ABAIXO}': 8,
    f'E{QUARTO_TOM_ACIMA}': 10, f'F{QUARTO_TOM_ABAIXO}': 10,
    f'F{QUARTO_TOM_ACIMA}': 12, f'F#{QUARTO_TOM_ABAIXO}': 12,
    f'F#{QUARTO_TOM_ACIMA}': 14, f'G{QUARTO_TOM_ABAIXO}': 14,
    f'G{QUARTO_TOM_ACIMA}': 16, f'G#{QUARTO_TOM_ABAIXO}': 16,
    f'G#{QUARTO_TOM_ACIMA}': 18, f'A{QUARTO_TOM_ABAIXO}': 18,
    f'A{QUARTO_TOM_ACIMA}': 20, f'A#{QUARTO_TOM_ABAIXO}': 20,
    f'A#{QUARTO_TOM_ACIMA}': 22, f'B{QUARTO_TOM_ABAIXO}': 22,
    f'B{QUARTO_TOM_ACIMA}': 24,
    # Adicionando as notações simples faltantes
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

# Mapping for converting between notation systems
NOTACAO_QUARTOS_TOM = {
    f'C{QUARTO_TOM_ACIMA}': 'C+',
    f'D{QUARTO_TOM_ACIMA}': 'D+',
    f'E{QUARTO_TOM_ACIMA}': 'E+',
    f'F{QUARTO_TOM_ACIMA}': 'F+',
    f'G{QUARTO_TOM_ACIMA}': 'G+',
    f'A{QUARTO_TOM_ACIMA}': 'A+',
    f'B{QUARTO_TOM_ACIMA}': 'B+',
    f'C{QUARTO_TOM_ABAIXO}': 'C-',
    f'D{QUARTO_TOM_ABAIXO}': 'D-',
    f'E{QUARTO_TOM_ABAIXO}': 'E-',
    f'F{QUARTO_TOM_ABAIXO}': 'F-',
    f'G{QUARTO_TOM_ABAIXO}': 'G-',
    f'A{QUARTO_TOM_ABAIXO}': 'A-',
    f'B{QUARTO_TOM_ABAIXO}': 'B-',
}




# Variável de tipo para funções genéricas
T = TypeVar('T')



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

def is_valid_note(nota: str) -> bool:
    """
    Validates if a string represents a valid note.
    Aceita notação microtonal em cents, símbolos, ou combinações.
    
    Args:
        nota (str): Note string to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Basic note pattern: letter (A-G), optional accidental (#, b, +, -), and octave (number)
    pattern_standard = r'^[A-Ga-g][#b]?[-+]?[0-9]$'
    
    # Pattern for cents notation: letter, optional accidental, octave, +/- sign, 1-2 digits, 'c'
    pattern_cents = r'^[A-Ga-g][#b]?[0-9][+-][0-9]{1,2}c$'
    
    # Pattern for arrow symbols: letter, optional accidental, arrow symbol, octave
    pattern_arrows = f'^[A-Ga-g][#b]?[{QUARTO_TOM_ACIMA}{QUARTO_TOM_ABAIXO}][0-9]$'
    
    # Pattern for combined notation: letter, optional accidental, arrow symbol, octave, cents
    pattern_combined = f'^[A-Ga-g][#b]?[{QUARTO_TOM_ACIMA}{QUARTO_TOM_ABAIXO}][0-9][+-][0-9]{{1,2}}c$'
    
    return bool(re.match(pattern_standard, nota) or 
               re.match(pattern_cents, nota) or 
               re.match(pattern_arrows, nota) or
               re.match(pattern_combined, nota))

def extract_cents(nota: str) -> Tuple[str, int]:
    """
    Extrai o componente cents de uma nota, se presente.
    Suporta notas com símbolos microtonais.
    
    Args:
        nota (str): String de nota possivelmente com cents (ex: C4+50c, C?4+30c)
        
    Returns:
        Tuple[str, int]: (nota base sem cents, valor em cents)
    """
    if 'c' not in nota:
        return nota, 0
        
    try:
        # Padrão para notas regulares com cents
        pattern_cents = r'^([A-Ga-g][#b]?[0-9])([+-][0-9]{1,2})c$'
        
        # Padrão para notas com símbolos microtonais e cents
        pattern_combined = f'^([A-Ga-g][#b]?[{QUARTO_TOM_ACIMA}{QUARTO_TOM_ABAIXO}][0-9])([+-][0-9]{{1,2}})c$'
        
        match = re.match(pattern_cents, nota) or re.match(pattern_combined, nota)
        
        if not match:
            return nota, 0
            
        base_note, cents_part = match.groups()
        cents_value = int(cents_part)
        
        return base_note, cents_value
    except Exception as e:
        logger.error(f"Erro ao extrair cents da nota {nota}: {e}")
        return nota, 0

def converter_para_sustenido(nota: str) -> str:
    """
    Converts a note with flat (bemol) to its equivalent with sharp (sustenido).
    Supports microtonal notation with symbols and cents.
    
    Args:
        nota (str): Note string in any format
        
    Returns:
        str: Note with sharps instead of flats
        
    Raises:
        ValueError: If the note is invalid
    """
    if not is_valid_note(nota):
        raise ValueError(f"Formato de nota inválido: {nota}")
    
    try:
        # Extrair cents primeiro, se presentes
        base_note, cents = extract_cents(nota)
        
        # Agora base_note é a nota sem os cents
        
        # Verificar se a nota tem símbolo microtonal
        has_microtonal_symbol = QUARTO_TOM_ACIMA in base_note or QUARTO_TOM_ABAIXO in base_note
        
        if has_microtonal_symbol:
            # Para notas com símbolos microtonais, precisamos encontrar a parte da nota, o símbolo e a oitava
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
                # Se não conseguir extrair o padrão, mantenha como está
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
        raise ValueError(f"Erro ao converter nota {nota}: {str(e)}")





import re
import logging

log = logging.getLogger(__name__)

# símbolos unicode de ¼-tom
QUARTO_TOM_ACIMA  = "↑"
QUARTO_TOM_ABAIXO = "↓"

ESCALA_CROMATICA = {
    "C": 0,  "C#": 1,  "Db": 1,
    "D": 2,  "D#": 3,  "Eb": 3,
    "E": 4,
    "F": 5,  "F#": 6,  "Gb": 6,
    "G": 7,  "G#": 8,  "Ab": 8,
    "A": 9,  "A#": 10, "Bb": 10,
    "B": 11,
}

def note_to_midi(note: str) -> float:
    """
    Converte uma nota textual para o número MIDI (float).
    Suporta cents (+-XXc), códigos de ¼-tom (+ / -) e símbolos ↑ / ↓.
    Retorna 60.0 (C4) se falhar.
    """
    if not isinstance(note, str) or not note:
        return 60.0

    note = note.strip()

    # -----------------------------------------------------------------
    # 1) cents  --------  ex.:  C4+37c   C#5-12c
    # -----------------------------------------------------------------
    m = re.fullmatch(r"([A-Ga-g][#b]?\d+)([+-]\d{1,2})c", note)
    if m:
        base, cents = m.groups()
        return note_to_midi(base) + int(cents) / 100.0     # 100 cents = 1 semitom

    # -----------------------------------------------------------------
    # 2) flecha ↑  --------  C↑4   D#↑5
    # -----------------------------------------------------------------
    m = re.fullmatch(r"([A-Ga-g][#b]?)" + QUARTO_TOM_ACIMA + r"(\d)", note)
    if m:
        base, octave = m.groups()
        base = base.capitalize()
        if base in ESCALA_CROMATICA:
            return (int(octave) + 1) * 12 + ESCALA_CROMATICA[base] + 0.5

    # 3) flecha ↓  --------  C↓4   F#↓3
    m = re.fullmatch(r"([A-Ga-g][#b]?)" + QUARTO_TOM_ABAIXO + r"(\d)", note)
    if m:
        base, octave = m.groups()
        base = base.capitalize()
        if base in ESCALA_CROMATICA:
            return (int(octave) + 1) * 12 + ESCALA_CROMATICA[base] - 0.5

    # -----------------------------------------------------------------
    # 4) código de ¼-tom (+ / -)  --------  C+4   D#-5
    # -----------------------------------------------------------------
    m = re.fullmatch(r"([A-Ga-g][#b]?)([+-])(\d)", note)
    if m:
        base, sign, octave = m.groups()
        base = base.capitalize()
        if base in ESCALA_CROMATICA:
            delta = 0.5 if sign == "+" else -0.5
            return (int(octave) + 1) * 12 + ESCALA_CROMATICA[base] + delta

    # -----------------------------------------------------------------
    # 5) nota padrão  --------  C4   F#5   Bb3
    # -----------------------------------------------------------------
    m = re.fullmatch(r"([A-Ga-g][#b]?)(\d)", note)
    if m:
        base, octave = m.groups()
        base = base.capitalize()
        if base in ESCALA_CROMATICA:
            return (int(octave) + 1) * 12 + ESCALA_CROMATICA[base]

    # Falhou?  avisa e devolve C4
    log.warning("Formato de nota não reconhecido: %s", note)
    return 60.0


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
    # Extrair cents se presente
    base_note, cents = extract_cents(nota)
    cents_fraction = cents / CENTS_POR_SEMITOM  # Converter para fração de semitom
    
    # Lidar com símbolos unicode de quarto de tom
    if QUARTO_TOM_ACIMA in base_note or QUARTO_TOM_ABAIXO in base_note:
        # Converter para notação +/- para processamento uniforme
        if QUARTO_TOM_ACIMA in base_note:
            base_note = base_note.replace(QUARTO_TOM_ACIMA, '+')
        if QUARTO_TOM_ABAIXO in base_note:
            base_note = base_note.replace(QUARTO_TOM_ABAIXO, '-')
    
    # Processar a nota base
    padrao = r'([A-Ga-g][#b]?[-+]?)(\d+)'
    match = re.match(padrao, base_note)
    if not match:
        raise ValueError(f"Nota '{base_note}' não corresponde ao padrão esperado.")

    nota_base, oitava_str = match.groups()
    nota_base = nota_base.upper()  # Padronizar para maiúsculas
    oitava = int(oitava_str)

    if nota_base not in escala_microtonal:
        raise ValueError(f"Nota '{nota_base}' não está definida na escala microtonal.")

    posicao_na_oitava = escala_microtonal[nota_base]  # de 1..24
    posicao = posicao_na_oitava + (TAMANHO_OITAVA_MICROTONAL * oitava)
    
    # Adicionar a fração de cents
    posicao_cents = posicao + (cents_fraction * 2)  # 2 = fator de conversão para manter proporção
    

def calcular_densidade_intervalar_com_cents(notas, lamb=0.05):
    """
    Versão atualizada da função calcular_densidade_intervalar que suporta notação de cents.
    Esta função calcula a densidade intervalar considerando distâncias microtonais precisas.
    
    Args:
        notas (list): Lista de strings de notas, possivelmente com notação de cents
        lamb (float): Parâmetro lambda para o decaimento exponencial
        
    Returns:
        float: Densidade total calculada
    """
    from densidade_intervalar import decaimento_exponencial_modificado
    import logging
    
    # Inicializar logger
    logger = logging.getLogger('data_processor')
    
    # Validação da entrada
    if not notas or len(notas) < 2:
        logger.info("Menos de duas notas para calcular densidade intervalar")
        return 0.0
    
    # Lista para armazenar os valores MIDI válidos e suas notas originais correspondentes
    valid_pitches = []
    valid_notas = []
  

    # Converter notas para valores MIDI para maior precisão
    for nota in notas:
        if not nota:  # Ignorar notas vazias
            continue

        try:
            nota_norm = normalize_note_string(nota)
            midi_value = note_to_midi(nota_norm)

            # Garantir que temos um valor válido (evitar fallback silencioso)
            if midi_value is not None and midi_value != 60.0:  # Se não é o fallback C4
                valid_pitches.append(midi_value)
                valid_notas.append(nota_norm)
            else:
                # Se midi_value é o fallback 60.0, vamos verificar se a nota é realmente C4
                if nota_norm.upper().startswith('C4'):
                    valid_pitches.append(midi_value)
                    valid_notas.append(nota_norm)
                else:
                    logger.warning(f"Nota ignorada (convertida para fallback): {nota_norm}")
        except Exception as e:
            logger.error(f"Erro ao converter nota para MIDI: {nota} — {e}")

    
    # Se não temos notas suficientes para calcular intervalos
    if len(valid_pitches) < 2:
        logger.warning(f"Menos de duas notas válidas para densidade intervalar: {len(valid_pitches)}")
        return 0.0
    
    densidade_total = 0.0
    n = len(valid_pitches)
    
    # Calcular densidade par a par
    for i in range(n):
        for j in range(i+1, n):
            # Verificar se ambos os valores são válidos
            if valid_pitches[i] is None or valid_pitches[j] is None:
                continue
                
            # Calcular a diferença em semitons
            delta_semitons = abs(valid_pitches[i] - valid_pitches[j])
            
            # Se o intervalo for muito pequeno mas as notas são diferentes,
            # forçamos um valor mínimo para garantir que o intervalo seja contabilizado
            if delta_semitons < 0.01 and valid_notas[i] != valid_notas[j]:
                delta_semitons = 0.25  # Forçar pelo menos um quarto de tom
                logger.debug(f"Forçando intervalo mínimo entre {valid_notas[i]} e {valid_notas[j]}")
            
            # Transformar para escala microtonal (para manter a mesma escala da função original)
            delta = delta_semitons * 2  # Fator 2 para manter proporção com a escala original
            densidade_parcial = decaimento_exponencial_modificado(delta, lamb)
            densidade_total += densidade_parcial
            
            # Debug para rastrear o cálculo
            logger.debug(f"Intervalo entre {valid_notas[i]} ({valid_pitches[i]:.2f}) e {valid_notas[j]} ({valid_pitches[j]:.2f}): delta={delta_semitons:.2f}, densidade={densidade_parcial:.6f}")
    
    logger.debug(f"Densidade total calculada: {densidade_total:.6f}")
    return densidade_total

def midi_to_hz(midi_pitch: float) -> float:
    """
    Converte altura MIDI para frequência em Hz.
    
    Args:
        midi_pitch (float): Valor de altura MIDI
        
    Returns:
        float: Frequência em Hertz
    """
    # MIDI 69 = A4 = 440Hz
    return 440.0 * (2 ** ((midi_pitch - 69) / 12))

# Add this to the constants section at the top of utils.py
# This should go with the other constants like QUARTO_TOM_ACIMA and QUARTO_TOM_ABAIXO

# Lista simplificada com apenas notas cromáticas padrão (12 por oitava)
NOTAS_CROMATICAS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# The error occurs in the midi_to_note_name function, so make sure it looks like this:
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
        
        # Verificar se está próximo de um quarto de tom (25 cents) ou três quartos de tom (75 cents)
        if 15 <= cents <= 35:  # Cerca de um quarto de tom acima
            note_base = NOTAS_CROMATICAS[note_index]
            return f"{note_base}{QUARTO_TOM_ACIMA}{octave}"
        elif -35 <= cents <= -15:  # Cerca de um quarto de tom abaixo
            note_base = NOTAS_CROMATICAS[note_index]
            return f"{note_base}{QUARTO_TOM_ABAIXO}{octave}"
        # Casos extremos vão para o fallback de cents
    
    # Adicionar cents se solicitado e se houver uma parte fracionária
    if include_cents and abs(midi_frac) > 0.001:  # Pequena margem para evitar problemas de ponto flutuante
        cents = round(midi_frac * CENTS_POR_SEMITOM)
        if cents > 0:
            return f"{basic_note}+{cents}c"
        else:
            return f"{basic_note}{cents}c"  # O sinal negativo já está incluído
    
    return basic_note

# Equivalências entre notas com bemol e sustenido
EQUIVALENCIAS_NOTAS: Dict[str, str] = {
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

    # Símbolos de ¼-tom
    f'C{QUARTO_TOM_ACIMA}': 'C+',
    f'C#{QUARTO_TOM_ABAIXO}': 'C#-',
 
}

def frequency_to_note_name(frequency: float, include_cents: bool = False, use_symbols: bool = False) -> str:
    """
    Converte frequência para o nome da nota musical mais próxima, com cents ou símbolos opcionais.
    
    Args:
        frequency (float): Frequência em Hertz
        include_cents (bool): Incluir cents na saída
        use_symbols (bool): Usar símbolos microtonais quando apropriado
        
    Returns:
        str: Nome da nota (ex: 'A4', 'C5', 'G3+35c' ou 'F?4')
    """
    if frequency <= 0 or np.isnan(frequency) or np.isinf(frequency):
        logger.warning(f"Frequência inválida: {frequency}")
        return "Inválida"

    try:
        # Calcular número da nota MIDI a partir da frequência
        # A4 = 440Hz = MIDI 69
        midi_number = 69 + 12 * np.log2(frequency / 440.0)
        
        # Verificar se o resultado está em um intervalo razoável
        if midi_number < 0 or midi_number > 127:
            logger.warning(f"Frequência {frequency}Hz mapeia para MIDI fora do intervalo: {midi_number}")
            return "Fora do intervalo"
        
        # Converter para nome da nota com cents ou símbolos se solicitado
        return midi_to_note_name(midi_number, include_cents, use_symbols)
    except Exception as e:
        logger.error(f"Erro ao converter frequência {frequency} para nome da nota: {e}")
        return "Inválida"

# ===================================================================
# UTILIDADES DE MANIPULAÇÃO E FORMATAÇÃO DE DADOS
# ===================================================================

def serialize_for_json(obj: Any) -> Any:
    """
    Converte tipos numpy para tipos Python nativos para serialização JSON.
    
    Args:
        obj: Objeto a converter
        
    Returns:
        Objeto com tipos serializáveis
    """
    try:
        if obj is None:
            return None
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [serialize_for_json(i) for i in obj]
        else:
            return obj
    except Exception as e:
        logger.error(f"Erro ao serializar objeto para JSON: {e}")
        return None

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Aplana um dicionário aninhado.
    
    Args:
        d (dict): Dicionário aninhado
        parent_key (str): Chave do dicionário pai
        sep (str): Separador para chaves aninhadas
        
    Returns:
        dict: Dicionário aplanado
    """
    if not d:
        return {}
        
    items: List[Tuple[str, Any]] = []
    
    for k, v in d.items():
        if k is None:
            continue
            
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            # Filtrar apenas valores numéricos que não são NaN ou inf
            if isinstance(v, (int, float)) and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                items.append((new_key, v))
    
    return dict(items)

# ===================================================================
# UTILIDADES DE VISUALIZAÇÃO
# ===================================================================

def safe_show_figure(fig: Optional[plt.Figure] = None) -> None:
    """
    Mostra uma figura matplotlib com segurança, tratando erros potenciais.
    
    Args:
        fig: objeto figura matplotlib, ou None para usar a figura atual
    """
    try:
        if fig is not None:
            plt.figure(fig.number)
        plt.show()
    except Exception as e:
        logger.error(f"Erro ao exibir figura: {e}")
        print(f"Erro ao exibir figura: {e}")

@contextmanager
def figure_to_image(format: str = 'png', dpi: int = 300):
    """
    Gerenciador de contexto para criar e fechar uma figura adequadamente.
    
    Args:
        format (str): Formato da imagem
        dpi (int): Resolução
        
    Yields:
        função para converter a figura atual para uma Image PIL
    """
    try:
        # Criar uma função para converter a figura atual em uma imagem
        def convert_to_image():
            buf = io.BytesIO()
            plt.savefig(buf, format=format, dpi=dpi)
            buf.seek(0)
            return Image.open(buf)
        
        yield convert_to_image
    finally:
        plt.close()

def save_figure_as_image(fig: plt.Figure, format: str = 'png', dpi: int = 300) -> Optional[Image.Image]:
    """
    Salva uma figura matplotlib como uma imagem na memória.
    
    Args:
        fig: objeto figura matplotlib
        format (str): Formato da imagem
        dpi (int): Resolução
        
    Returns:
        PIL.Image: Objeto Image ou None se ocorrer erro
    """
    if fig is None:
        logger.error("Não é possível salvar figura None como imagem")
        return None
        
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format=format, dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    except Exception as e:
        logger.error(f"Erro ao salvar figura como imagem: {e}")
        if fig is not None:
            plt.close(fig)
        return None

# ===================================================================
# UTILIDADES DE ARQUIVO E CAMINHO
# ===================================================================

def ensure_directory_exists(directory_path: str) -> str:
    """
    Cria um diretório se ele não existir.
    
    Args:
        directory_path (str): Caminho para o diretório
        
    Returns:
        str: Caminho para o diretório
        
    Raises:
        OSError: Se o diretório não puder ser criado
    """
    if not directory_path:
        raise ValueError("O caminho do diretório não pode estar vazio")
        
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
        return directory_path
    except OSError as e:
        logger.error(f"Erro ao criar diretório {directory_path}: {e}")
        raise

def generate_timestamp_filename(prefix: str, extension: str) -> str:
    """
    Gera um nome de arquivo com timestamp.
    
    Args:
        prefix (str): Prefixo do nome do arquivo
        extension (str): Extensão do arquivo
        
    Returns:
        str: Nome do arquivo com timestamp
    """
    if not prefix:
        prefix = "arquivo"
    if not extension:
        extension = "txt"
        
    # Remover ponto inicial da extensão se presente
    if extension.startswith('.'):
        extension = extension[1:]
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"

# ===================================================================
# UTILIDADES DE TRATAMENTO DE ERROS
# ===================================================================

def safe_operation(func: Callable[..., T], *args: Any, 
                  fallback_value: Optional[T] = None, 
                  error_message: str = "Operação falhou", 
                  **kwargs: Any) -> Optional[T]:
    """
    Executa uma função com segurança, tratando exceções.
    
    Args:
        func: Função a executar
        *args: Argumentos para func
        fallback_value: Valor a retornar se func falhar
        error_message: Mensagem para registrar se func falhar
        **kwargs: Argumentos de palavra-chave para func
        
    Returns:
        Resultado de func ou fallback_value se func falhar
    """
    if func is None:
        logger.error("Não é possível executar função None")
        return fallback_value
        
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"{error_message}: {e}")
        return fallback_value

def log_execution_time(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorador para registrar o tempo de execução da função.
    
    Args:
        func: Função a decorar
        
    Returns:
        Função encapsulada
    """
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Função {func.__name__} executada em {duration:.2f} segundos")
        return result
    return wrapper


def test_microtonal_system():
    """
    Função para testar o sistema microtonal atualizado.
    Execute este teste chamando utils.py diretamente como script:
    python utils.py
    """
    print("Testes do sistema microtonal com suporte a cents:")
    
    # Testar conversão de notas para MIDI
    test_notes = [
        "C4", "C#4", "D4", "Eb4", "E4",  # Notas padrão
        "C4+50c", "C#4-30c", "D4+25c",   # Notas com cents positivos e negativos
        "E5+99c", "F#3-12c", "G2+75c",   # Variações de oitavas e cents
        f"C{QUARTO_TOM_ACIMA}4", f"D{QUARTO_TOM_ABAIXO}5"  # Notas com símbolos microtonais
    ]
    
    print("\n1. Conversão nota -> MIDI:")
    for note in test_notes:
        midi_value = note_to_midi(note)
        print(f"  {note:8} -> MIDI: {midi_value:.2f}")
    
    # Testar conversão de MIDI para notas
    test_midi_values = [
        60.0, 61.0, 62.0,           # Valores MIDI inteiros (C4, C#4, D4)
        60.5, 61.25, 62.75,         # Valores MIDI fracionários
        72.33, 73.66, 57.42, 58.91  # Mais variações
    ]
    
    print("\n2. Conversão MIDI -> nota (com diferentes opções):")
    for midi_value in test_midi_values:
        note_basic = midi_to_note_name(midi_value, include_cents=False, use_symbols=False)
        note_with_cents = midi_to_note_name(midi_value, include_cents=True, use_symbols=False)
        note_with_symbols = midi_to_note_name(midi_value, include_cents=False, use_symbols=True)
        print(f"  MIDI: {midi_value:.2f} -> Básica: {note_basic:5} | Com cents: {note_with_cents:10} | Com símbolos: {note_with_symbols}")
    
    # Testar notação microtonal
    print("\n3. Conversão entre notações microtonais:")
    microtonal_notations = [
        "C+4", "C#-4", "D+4", "D#-4", "E+4",
        f"C{QUARTO_TOM_ACIMA}4", f"C#{QUARTO_TOM_ABAIXO}4", f"D{QUARTO_TOM_ACIMA}4"
    ]
    
    for notation in microtonal_notations:
        converted = converter_notacao_microtonal(notation)
        print(f"  {notation:8} -> {converted}")
    
    # Testar conversão frequência -> nota
    test_freqs = [
        440.0,    # A4 exato
        466.16,   # A#4/Bb4 exato
        452.89,   # A4+50c aproximado
        430.54,   # A4-38c aproximado
        523.25    # C5 exato
    ]

    print("\n4. Conversão frequência -> nota (com diferentes opções):")
    for freq in test_freqs:
        note_basic = frequency_to_note_name(freq, include_cents=False, use_symbols=False)
        note_with_cents = frequency_to_note_name(freq, include_cents=True, use_symbols=False)
        note_with_symbols = frequency_to_note_name(freq, include_cents=False, use_symbols=True)
        print(f"  {freq:.2f} Hz -> Básica: {note_basic:5} | Com cents: {note_with_cents:10} | Com símbolos: {note_with_symbols}")

# Executar os testes quando o arquivo for chamado diretamente
if __name__ == "__main__":
    print("\nExecutando testes do sistema microtonal...")
    test_microtonal_system()
