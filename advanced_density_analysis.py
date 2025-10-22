# advanced_density_analysis.py
"""Ferramentas autocontidas para análise espectral avançada.

Este módulo **não** depende de nenhum outro módulo do seu projecto –
assim evitamos *imports* circulares.  Ele exporta:

* ``calculate_spectral_moments``  – centroid, spread, skewness.
* ``calculate_extended_spectral_moments``  – acrescenta kurtosis, flatness,
  roll‑off, entropy, etc.
* ``calculate_spectral_complexity``         – alias usado noutros ficheiros.

Todas as funções aceitam **pitches em MIDI** + **densidades/amplitudes**
e devolvem um ``dict`` pronto a serializar.
"""
from __future__ import annotations

from typing import Dict, List, Sequence
import numpy as np
import logging

from spectral_analysis import (
    calculate_spectral_moments,
    calculate_extended_spectral_moments,
    calculate_chroma_vector,
    robust_gaussian_kde
)

__all__: List[str] = [
    "calculate_spectral_moments",
    "calculate_extended_spectral_moments",
    "calculate_spectral_complexity",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# helpers -------------------------------------------------------------------

_A4_FREQ = 440.0
from microtonal import midi_to_hz as _midi_to_hz, frequency_to_note_name as _hz_to_note_name

# ---------------------------------------------------------------------------
# main API -------------------------------------------------------------------

def calculate_spectral_moments(pitches: Sequence[float], amplitudes: Sequence[float]) -> Dict[str, Dict[str, float]]:
    """Centroid, spread (desvio‑padrão) e skewness.

    ``pitches`` – valores MIDI (floats);
    ``amplitudes`` – mesma dimensão, não‑negativos.
    """
    p = np.asarray(pitches, dtype=float)
    a = np.nan_to_num(amplitudes, nan=0.0, posinf=0.0, neginf=0.0)
    if p.size == 0 or np.sum(a) == 0:
        logger.warning("spectral_moments: entradas vazias")
        return {
            "Centróide": {"frequency": 0.0, "note": "Invalid"},
            "Dispersão": {"deviation": 0.0},
            "spectral_skewness" : 0.0,
        }

    w = a / np.sum(a)  # normalizar pesos
    centroid_midi = np.sum(p * w)
    spread_midi = np.sqrt(np.sum(((p - centroid_midi) ** 2) * w))
    skew_num = np.sum(((p - centroid_midi) ** 3) * w)
    skew = skew_num / (spread_midi ** 3) if spread_midi else 0.0

    centroid_hz = _midi_to_hz(centroid_midi)
    spread_hz = _midi_to_hz(centroid_midi + spread_midi) - centroid_hz

    return {
        "Centróide": {"frequency": float(centroid_hz), "note": _hz_to_note_name(centroid_hz)},
        "Dispersão": {"deviation": float(spread_hz)},
        "spectral_skewness" : float(skew),
    }


def calculate_extended_spectral_moments(pitches: Sequence[float], amplitudes: Sequence[float]) -> Dict[str, float | Dict[str, float]]:
    """Versão estendida com kurtosis, flatness, roll‑off 85 % e entropy."""
    base = calculate_spectral_moments(pitches, amplitudes)
    p = np.asarray(pitches, dtype=float)
    a = np.nan_to_num(amplitudes, nan=0.0, posinf=0.0, neginf=0.0)
    if p.size == 0 or np.sum(a) == 0:
        # já tratado em base; basta devolver complemento zerado
        ext = {
            "spectral_kurtosis": 0.0,
            "spectral_flatness": 0.0,
            "spectral_rolloff": 0.0,
            "spectral_entropy": 0.0,
        }
        base.update(ext)
        return base

    w = a / np.sum(a)
    centroid = np.sum(p * w)
    spread = np.sqrt(np.sum(((p - centroid) ** 2) * w)) or 1e-9
    kurt = (np.sum(((p - centroid) ** 4) * w) / (spread ** 4)) - 3

    # flatness (razão geom./arit.)
    nonzero = a[a > 0]
    flatness = float(np.exp(np.mean(np.log(nonzero))) / np.mean(nonzero)) if nonzero.size else 0.0

    # roll‑off 85 %
    cumsum = np.cumsum(a)
    idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
    roll_midi = p[min(idx, len(p) - 1)]

    # entropy de Shannon
    entropy = -float(np.sum(w[w > 0] * np.log2(w[w > 0])))

    base.update({
        "spectral_kurtosis": float(kurt),
        "spectral_flatness": flatness,
        "spectral_rolloff": float(roll_midi),
        "spectral_entropy": entropy,
    })
    return base

# Alias utilizado noutros ficheiros para evitar ter de refatorizar tudo.
calculate_spectral_complexity = calculate_extended_spectral_moments


