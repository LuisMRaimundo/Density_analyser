"""spectral_analysis.py

Módulo autónomo (sem dependências circulares) com utilitários para
análise espectral – centroid, spread, skewness, kurtosis, flatness,
roll‑off e vetor de chroma. Substitui a antiga lógica que estava em
``advanced_density_analysis.py`` e evita a auto‑importação desse ficheiro.

Depois de gravar este ficheiro basta:
    import importlib, spectral_analysis as sa
    importlib.reload(sa)
    # ... ou simplesmente reiniciar a aplicação se preferir.

ATENÇÃO: nos outros módulos mude:
    from advanced_density_analysis import …
para:
    from spectral_analysis import …
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
import logging
import numpy as np
from scipy.stats import gaussian_kde
from microtonal import midi_to_hz as midi_to_frequency, frequency_to_note_name

LOGGER = logging.getLogger(__name__)

################################################################################
# 1. UTILITÁRIOS BÁSICOS                                                       #
################################################################################

_A4 = 440.0  # Hz

_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
               'F#', 'G', 'G#', 'A', 'A#', 'B']


def midi_to_frequency(midi: float | int) -> float:
    """Converte número MIDI (float permite microtons) → frequência em Hz."""
    return _A4 * 2 ** ((midi - 69) / 12)


def frequency_to_note_name(freq: float) -> str:
    """Nota aproximada mais próxima (± sem cents)."""
    if freq <= 0 or np.isnan(freq) or np.isinf(freq):
        return "Invalid"
    midi = 69 + 12 * np.log2(freq / _A4)
    name = _NOTE_NAMES[int(round(midi)) % 12]
    octave = int(round(midi)) // 12 - 1
    return f"{name}{octave}"

################################################################################
# 2. KERNEL DENSITY ESTIMATION ROBUSTO                                        #
################################################################################

def robust_gaussian_kde(data: np.ndarray,
                        weights: np.ndarray | None = None,
                        bw_method: str | float | None = None):
    """Versão à prova de *LinAlgError* quando a matriz de covariância é singular."""
    try:
        return gaussian_kde(data, weights=weights, bw_method=bw_method)
    except np.linalg.LinAlgError:
        LOGGER.warning("Singular covariance detected – adding jitter…")
        noise = np.random.normal(0, 1e-6, size=data.shape)
        return gaussian_kde(data + noise, weights=weights, bw_method=bw_method)

################################################################################
# 3. MOMENTOS ESPECTRAIS                                                      #
################################################################################

def _safe_array(a):
    """Converte entrada para array numpy, substituindo NaN por 0."""
    return np.nan_to_num(np.asarray(a, dtype=float))


def calculate_spectral_moments(pitches: List[float],
                               amplitudes: List[float]) -> Dict[str, Dict[str, float] | float]:
    """Centroid, spread e skewness – equivalem à antiga *calculate_spectral_moments*.

    * ``pitches``  – alturas em MIDI (aceita floats para microtons)
    * ``amplitudes`` – quaisquer pesos (densidade, RMS, etc.)
    """
    pitches = _safe_array(pitches)
    amps = _safe_array(amplitudes)
    
    # Verificar por entrada vazia ou inválida
    total = amps.sum()
    if total <= 0 or len(pitches) == 0 or len(amps) == 0:
        return {
            "Centróide": {"frequency": 0.0, "note": "Invalid"},
            "Dispersão": {"deviation": 0.0},
            "spectral_skewness": 0.0,
        }

    # Calcular centróide (média ponderada)
    centroid_midi = (pitches * amps).sum() / total
    
    # Calcular dispersão (desvio padrão ponderado)
    spread_midi = np.sqrt(np.maximum(0, ((pitches - centroid_midi) ** 2 * amps).sum() / total))
    
    # Calcular assimetria (skewness)
    if spread_midi > 0:
        skew_num = ((pitches - centroid_midi) ** 3 * amps).sum() / total
        skewness = skew_num / (spread_midi ** 3)
    else:
        skewness = 0.0

    # Converter para frequência
    centroid_freq = midi_to_frequency(centroid_midi)
    spread_freq = midi_to_frequency(centroid_midi + spread_midi) - centroid_freq if spread_midi > 0 else 0.0

    return {
        "Centróide": {"frequency": centroid_freq, "note": frequency_to_note_name(centroid_freq)},
        "Dispersão": {"deviation": spread_freq},
        "spectral_skewness": skewness,
    }


def calculate_extended_spectral_moments(pitches: List[float],
                                        amplitudes: List[float]) -> Dict[str, float | Dict[str, float]]:
    """Versão estendida (kurtosis, flatness, roll‑off e entropia)."""
    # Obter resultados básicos
    base = calculate_spectral_moments(pitches, amplitudes)
    
    # Preparar arrays
    pitches = _safe_array(pitches)
    amps = _safe_array(amplitudes)
    
    # Verificar por entrada vazia ou inválida
    total = amps.sum()
    if total <= 0 or len(pitches) == 0 or len(amps) == 0:
        # Adicionar métricas estendidas como zeros
        base.update({
            "spectral_kurtosis": 0.0,
            "spectral_flatness": 0.0,
            "spectral_rolloff": 0.0,
            "spectral_entropy": 0.0,
        })
        return base

    # Recuperar centróide já calculado (MIDI)
    centroid_midi = (pitches * amps).sum() / total
    
    # Calcular dispersão (spread)
    spread_midi = np.sqrt(np.maximum(0, ((pitches - centroid_midi) ** 2 * amps).sum() / total))
    
    # Calcular curtose (kurtosis)
    if spread_midi > 0:
        kurt_num = ((pitches - centroid_midi) ** 4 * amps).sum() / total
        kurtosis = kurt_num / (spread_midi ** 4) - 3
    else:
        kurtosis = 0.0

    # Calcular planura espectral (flatness) - razão entre média geométrica e média aritmética
    nz_amps = amps[amps > 1e-10]  # Usar apenas amplitudes não-zero
    if len(nz_amps) > 0:
        # Média geométrica / média aritmética
        flatness = np.exp(np.log(nz_amps).mean()) / nz_amps.mean()
    else:
        flatness = 0.0

    # Calcular roll-off (85%)
    if len(pitches) > 0:
        cumsum = np.cumsum(amps)
        threshold = 0.85 * cumsum[-1]
        idx = np.searchsorted(cumsum, threshold)
        rolloff_midi = pitches[min(idx, len(pitches)-1)]
        rolloff_freq = midi_to_frequency(rolloff_midi)
    else:
        rolloff_freq = 0.0

    # Calcular entropia espectral
    # Normalizar amplitudes para formar uma distribuição de probabilidade
    prob = amps / total
    
    # Filtrar probabilidades muito pequenas que causariam problemas no log
    valid_mask = prob > 1e-10
    if np.any(valid_mask):
        valid_probs = prob[valid_mask]
        # Calcular entropia apenas com probabilidades válidas
        entropy = -np.sum(valid_probs * np.log2(valid_probs))
    else:
        entropy = 0.0

    # Adicionar métricas estendidas ao dicionário de resultados
    base.update({
        "spectral_kurtosis": kurtosis,
        "spectral_flatness": flatness,
        "spectral_rolloff": rolloff_freq,
        "spectral_entropy": entropy,
    })
    return base

################################################################################
# 4. VETOR DE CHROMA                                                          #
################################################################################

def calculate_chroma_vector(pitches: List[float],
                            amplitudes: List[float] | None = None) -> List[float]:
    """Devolve vetor de 12 posições (C..B) normalizado."""
    pitches = _safe_array(pitches)
    amps = _safe_array(amplitudes) if amplitudes is not None else np.ones_like(pitches)
    
    # Inicializar vetor de chroma
    chroma = np.zeros(12)
    
    # Acumular energia em cada classe de alturas
    for p, a in zip(pitches, amps):
        if not np.isnan(p) and not np.isinf(p):
            chroma[int(round(p)) % 12] += a
    
    # Normalizar se houver valores não-zero
    total = chroma.sum()
    if total > 0:
        chroma /= total
        
    return chroma.tolist()

################################################################################
# 5. RAZÃO HARMÔNICA                                                          #
################################################################################

def calculate_harmonic_ratio(pitches, amplitudes=None, fundamental=None):
    """
    Razão harmónica simples: quanta energia está em harmónicos vs. fundamentais.
    • pitches – lista de valores MIDI
    • amplitudes – lista de amplitudes (opcional, default = 1)
    • fundamental – MIDI da fundamental (opcional: procura-se o mais grave)
    Devolve um float entre 0 e 1 (≈ mais harmónicos → valor maior).
    """
    # Verificar entrada vazia
    if not pitches:
        return 0.0

    # Preparar arrays
    pitches = _safe_array(pitches)
    amps = np.ones_like(pitches) if amplitudes is None else _safe_array(amplitudes)
    
    # Remover valores NaN ou Inf
    valid_mask = ~(np.isnan(pitches) | np.isinf(pitches))
    pitches = pitches[valid_mask]
    amps = amps[valid_mask]
    
    # Verificar se ainda temos dados válidos
    if len(pitches) == 0:
        return 0.0

    # Determinar fundamental se não fornecida
    if fundamental is None:
        fundamental = pitches.min()

    # Calcular distâncias (em semitons) à fundamental
    intervals = pitches - fundamental
    
    # Identificar harmônicos (intervalos próximos a múltiplos de 12 semitons)
    harmonic_mask = np.isclose(intervals % 12, 0, atol=0.25)

    # Calcular razão de energia
    harm_energy = amps[harmonic_mask].sum()
    total_energy = amps.sum()
    
    return float(harm_energy / total_energy) if total_energy > 0 else 0.0

################################################################################
# API pública                                                                  #
################################################################################
__all__ = [
    "calculate_spectral_moments",
    "calculate_extended_spectral_moments",
    "calculate_chroma_vector",
    "robust_gaussian_kde",
    "calculate_harmonic_ratio",
]