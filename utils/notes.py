from __future__ import annotations
"""
Utilities for musical pitch conversion and validation.

Public API
----------
- NOTE_BASE_MIDI: dict[str, float]
- note_to_midi(note: str) -> float
- midi_to_frequency(midi: float) -> float
- frequency_to_midi(freq: float) -> float
- midi_to_note_name(midi: float) -> str
- is_valid_note(note: str) -> bool
- extract_cents(note: str) -> tuple[str, int]
- to_sharp(note: str) -> str
- normalize_note_string(note: str) -> str
- QUARTO_TOM_ACIMA, QUARTO_TOM_ABAIXO: microtonal Unicode symbols
"""

from math import log2
import logging
import re
from typing import Dict, Tuple
import unicodedata

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------

CENTS_PER_SEMITONE = 100
CENTS_PER_OCTAVE = 1200
A4_MIDI = 69
A4_FREQ = 440.0

# Quarter-tone symbols (public for GUI etc.)
QUARTO_TOM_ACIMA = '↑'   # U+2191
QUARTO_TOM_ABAIXO = '↓'  # U+2193

# Chromatic scale – semitone index inside octave.
# Quarter-tone positions expressed with ±0.5.
NOTE_BASE_MIDI: Dict[str, float] = {
    # Naturals / sharps / flats
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "E#": 5,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "B#": 0,

    # Microtonais (- = quarto tom abaixo, + = quarto tom acima)
    "C-": 0.5, "C+": -0.5,
    "C#-": 1.5, "C#+": 0.5,
    "D-": 2.5, "D+": 1.5,
    "D#-": 3.5, "D#+": 2.5,
    "E-": 4.5, "E+": 3.5,
    "F-": 5.5, "F+": 4.5,
    "F#-": 6.5, "F#+": 5.5,
    "G-": 7.5, "G+": 6.5,
    "G#-": 8.5, "G#+": 7.5,
    "A-": 9.5, "A+": 8.5,
    "A#-": 10.5, "A#+": 9.5,
    "B-": 11.5, "B+": 10.5,

    # Alternativas enharmónicas para evitar falhas
    "Cb": 11, "Cb-": 11.5, "Cb+": 10.5,
    "Fb": 4, "Fb-": 4.5, "Fb+": 3.5,
    "E#-": 5.5, "B#-": 0.5,
    "E#+": 4.5, "B#+": -0.5,
}

# --------------------------------------------------------------------------------
# Regex patterns (compiled once)
# --------------------------------------------------------------------------------
_RE_STANDARD = re.compile(r"^([A-Ga-g][#b]?)(\d)$")
_RE_QUARTER = re.compile(r"^([A-Ga-g][#b]?)([+-])(\d)$")
_RE_ARROW = re.compile(rf"^([A-Ga-g][#b]?)([{QUARTO_TOM_ACIMA}{QUARTO_TOM_ABAIXO}])(\d)$")
_RE_CENTS = re.compile(r"^([A-Ga-g][#b]?\d)([+-]\d{1,2})c$")

# --------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------

def midi_to_hz(midi_value):
    """Converte valor MIDI em frequência em Hz (A4=440 Hz, MIDI 69)"""
    return 440.0 * (2 ** ((midi_value - 69) / 12))

def normalize_note_string(note: str) -> str:
    """
    Normaliza qualquer string de nota musical para a forma canónica:
    - Remove espaços e normaliza Unicode
    - Converte flats Unicode (♭) para 'b' e sharps Unicode (♯) para '#'
    - Converte setas microtonais (↑/↓) para +/- 
    - Aplica flats→sharps (ex: Eb4 → D#4) via to_sharp()
    - Capitaliza a letra da nota
    """
    if not isinstance(note, str):
        raise ValueError(f"Nota não é uma string: {note}")

    # Remove espaços e normaliza Unicode
    note = unicodedata.normalize("NFKC", note.replace(" ", "").strip())

    # Troca flats/sharps Unicode por ASCII
    note = note.replace("♭", "b").replace("♯", "#")

    # Troca setas por +/- para microtonalidade
    note = note.replace(QUARTO_TOM_ACIMA, "+").replace(QUARTO_TOM_ABAIXO, "-")

    # Converte para forma sharp/canónica
    note = to_sharp(note)

    # Capitaliza primeira letra (nota)
    if len(note) >= 2:
        note = note[0].upper() + note[1:]

    return note

def _note_base_to_semitone(base: str) -> float:
    if base not in NOTE_BASE_MIDI:
        raise ValueError(f"Nota base desconhecida: {base}")
    return NOTE_BASE_MIDI[base]

def to_sharp(note: str) -> str:
    """Convert flats / arrow notation to the canonical sharp/quarter-code."""
    # First deal with arrow symbols (Unicode) as +/- form
    m = _RE_ARROW.match(note)
    if m:
        name, arrow, octave = m.groups()
        sign = '+' if arrow == QUARTO_TOM_ACIMA else '-'
        return f"{name}{sign}{octave}"

    # Replace flats with equivalent sharps (including quarter codes)
    flats = {
        'Cb': 'B', 'Db': 'C#', 'Eb': 'D#', 'Fb': 'E', 'Gb': 'F#',
        'Ab': 'G#', 'Bb': 'A#',
    }
    for fl, sh in flats.items():
        if note.startswith(fl):
            note = note.replace(fl, sh, 1)
            break
    return note

# --------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------

def is_valid_note(note: str) -> bool:
    return bool(
        _RE_STANDARD.match(note) or
        _RE_QUARTER.match(note) or
        _RE_ARROW.match(note) or
        _RE_CENTS.match(note)
    )

def extract_cents(note: str) -> Tuple[str, int]:
    m = _RE_CENTS.match(note)
    if not m:
        return note, 0
    base, cents = m.groups()
    return base, int(cents)

def note_to_midi(note: str) -> float:
    """Converte uma string de nota para float MIDI (suporta centésimos e microtonais canónicos e Unicode)."""
    # Normaliza a nota para garantir forma canónica (resolve Unicode, setas, ♯, ♭, etc.)
    note = normalize_note_string(note)

    note, cents = extract_cents(note)
    note = to_sharp(note)

    m = _RE_STANDARD.match(note)
    mq = _RE_QUARTER.match(note)

    if m:
        base, octave = m.groups()
        semitone = _note_base_to_semitone(base)
    elif mq:
        base, sign, octave = mq.groups()
        semitone = _note_base_to_semitone(f"{base}{sign}")
    else:
        raise ValueError(f"Formato de nota inválido: {note}")

    midi = semitone + (int(octave) + 1) * 12 + cents / 100
    return midi


def midi_to_frequency(m: float) -> float:
    return A4_FREQ * 2 ** ((m - A4_MIDI) / 12)

def frequency_to_midi(f: float) -> float:
    return 12 * log2(f / A4_FREQ) + A4_MIDI

def midi_to_note_name(m: float) -> str:
    m_round = int(round(m))
    octave = m_round // 12 - 1
    semitone = m_round % 12
    rev = {v: k for k, v in NOTE_BASE_MIDI.items() if v == int(v)}
    base = rev.get(semitone, '?')
    cents = round((m - m_round) * 100)
    if cents:
        sign = '+' if cents > 0 else ''
        return f"{base}{octave}{sign}{cents}c"
    return f"{base}{octave}"








