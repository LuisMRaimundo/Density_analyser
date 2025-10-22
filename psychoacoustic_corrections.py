# psychoacoustic_corrections.py
"""
Correções psicoacústicas para análise de densidade musical.
Implementa modelos simplificados de mascaramento, roughness e loudness.
"""

import numpy as np
import logging
from typing import List, Tuple, Union
from microtonal import midi_to_hz

logger = logging.getLogger(__name__)

# Constants
BARK_SCALE_FACTOR = 13.0
BARK_SCALE_FREQ1 = 0.00076
BARK_SCALE_FREQ2 = 7500.0
ROUGHNESS_PEAK_HZ = 35.0
ROUGHNESS_CUTOFF_HZ = 150.0

def frequency_to_bark(freq: float) -> float:
    """
    Converte frequência (Hz) para escala Bark (critical band rate).
    
    Args:
        freq (float): Frequência em Hz
        
    Returns:
        float: Valor na escala Bark
    """
    if freq <= 0:
        return 0.0
    
    # Fórmula de Zwicker & Terhardt (1980)
    return BARK_SCALE_FACTOR * np.arctan(BARK_SCALE_FREQ1 * freq) + \
           3.5 * np.arctan((freq / BARK_SCALE_FREQ2) ** 2)

def critical_band_masking(pitches: List[float], amplitudes: List[float], 
                         masking_slope: float = 0.25) -> np.ndarray:
    """
    Aplica mascaramento de banda crítica simplificado.
    Frequências próximas dentro de bandas críticas se mascaram parcialmente.
    
    Args:
        pitches (List[float]): Lista de valores MIDI
        amplitudes (List[float]): Lista de amplitudes correspondentes
        masking_slope (float): Inclinação da curva de mascaramento (0-1)
        
    Returns:
        np.ndarray: Amplitudes ajustadas após mascaramento
    """
    if len(pitches) == 0 or len(amplitudes) == 0:
        return np.array([])
    
    # Converter MIDI para frequências
    freqs = np.array([midi_to_hz(p) for p in pitches])
    amps = np.array(amplitudes)
    
    # Converter para escala Bark
    barks = np.array([frequency_to_bark(f) for f in freqs])
    
    # Calcular mascaramento entre todos os pares
    masked_amps = amps.copy()
    
    for i in range(len(freqs)):
        for j in range(len(freqs)):
            if i != j:
                # Distância em bandas críticas
                bark_dist = abs(barks[i] - barks[j])
                
                # Dentro de 1 Bark = mascaramento significativo
                if bark_dist < 1.0:
                    # Amplitude maior mascara menor
                    if amps[j] > amps[i]:
                        masking_factor = (1 - bark_dist) * masking_slope
                        masked_amps[i] *= (1 - masking_factor)
    
    return masked_amps

# In psychoacoustic_corrections.py, update the calculate_roughness function:

# Replace the calculate_roughness function in psychoacoustic_corrections.py:

def calculate_roughness(pitches: List[float], amplitudes: List[float]) -> float:
    """
    Calcula roughness (aspereza) usando modelo simplificado de Plomp-Levelt.
    Versão melhorada com maior sensibilidade para intervalos pequenos.
    
    Args:
        pitches (List[float]): Lista de valores MIDI
        amplitudes (List[float]): Lista de amplitudes
        
    Returns:
        float: Valor total de roughness
    """
    if len(pitches) < 2:
        return 0.0
    
    freqs = np.array([midi_to_hz(p) for p in pitches])
    amps = np.array(amplitudes)
    
    roughness_total = 0.0
    
    # Calcular roughness para cada par de frequências
    for i in range(len(freqs)):
        for j in range(i + 1, len(freqs)):
            freq_diff = abs(freqs[i] - freqs[j])
            freq_mean = (freqs[i] + freqs[j]) / 2
            
            # Evitar divisão por zero
            if freq_mean == 0:
                continue
                
            # Normalizar pela frequência média (roughness é relativo)
            relative_diff = freq_diff / freq_mean * 100
            
            # Roughness máximo em torno de 5-8% da frequência média
            if relative_diff < 30:  # Só calcular para diferenças pequenas
                # Pico em torno de 6.5% (baseado em Sethares)
                x = relative_diff / 6.5
                
                # Função mais pronunciada para intervalos pequenos
                if x < 1:
                    # Subida mais íngreme antes do pico
                    roughness_contribution = x * np.exp(1 - x)
                else:
                    # Descida mais gradual após o pico
                    roughness_contribution = np.exp(-(x - 1) * 0.5)
                
                # Ponderar pela amplitude mínima do par
                weight = min(amps[i], amps[j])
                roughness_total += roughness_contribution * weight
    
    return roughness_total

def equal_loudness_correction(frequency: float, reference_spl: float = 60.0) -> float:
    """
    Correção simplificada da curva de equal loudness (ISO 226:2003).
    Compensa a percepção dependente de frequência.
    
    Args:
        frequency (float): Frequência em Hz
        reference_spl (float): Nível de pressão sonora de referência em dB
        
    Returns:
        float: Fator de correção (multiplicador)
    """
    if frequency <= 0:
        return 1.0
    
    # Correção simplificada baseada nas curvas de Fletcher-Munson
    # Para uma implementação completa, usar tabelas ISO 226
    
    if frequency < 200:
        # Frequências graves precisam de boost
        correction = 1.0 + (200 - frequency) / 200 * 0.5
    elif frequency < 1000:
        # Região de transição
        correction = 1.0 + (1000 - frequency) / 800 * 0.2
    elif frequency > 4000:
        # Altas frequências são percebidas como mais altas
        correction = 1.0 + (frequency - 4000) / 4000 * 0.3
        if frequency > 10000:
            # Rolloff para frequências muito altas
            correction *= (20000 - frequency) / 10000
    else:
        # Região de referência (1-4 kHz)
        correction = 1.0
    
    return max(0.1, correction)  # Evitar valores negativos ou muito pequenos

def apply_loudness_correction(pitches: List[float], amplitudes: List[float]) -> List[float]:
    """
    Aplica correção de equal loudness a uma lista de pitches e amplitudes.
    
    Args:
        pitches (List[float]): Lista de valores MIDI
        amplitudes (List[float]): Lista de amplitudes
        
    Returns:
        List[float]: Amplitudes corrigidas
    """
    if len(pitches) == 0:
        return []
    
    freqs = [midi_to_hz(p) for p in pitches]
    corrections = [equal_loudness_correction(f) for f in freqs]
    
    return [a * c for a, c in zip(amplitudes, corrections)]

def combination_tones_simple(pitches: List[float], amplitudes: List[float], 
                           threshold: float = 0.1) -> Tuple[List[float], List[float]]:
    """
    Calcula tons de combinação simples (diferença).
    Versão simplificada que considera apenas tons de diferença de primeira ordem.
    
    Args:
        pitches (List[float]): Lista de valores MIDI
        amplitudes (List[float]): Lista de amplitudes
        threshold (float): Limiar mínimo de amplitude para incluir tom de combinação
        
    Returns:
        Tuple[List[float], List[float]]: (pitches_combinados, amplitudes_combinadas)
    """
    if len(pitches) < 2:
        return [], []
    
    freqs = np.array([midi_to_hz(p) for p in pitches])
    combination_pitches = []
    combination_amps = []
    
    for i in range(len(freqs)):
        for j in range(i + 1, len(freqs)):
            # Tom de diferença
            diff_freq = abs(freqs[i] - freqs[j])
            
            # Apenas dentro da faixa audível e acima do limiar
            if 20 < diff_freq < 2000:
                # Amplitude proporcional ao produto das originais
                comb_amp = amplitudes[i] * amplitudes[j] * 0.1
                
                if comb_amp > threshold:
                    # Converter de volta para MIDI
                    from microtonal import hz_to_midi
                    midi_value = hz_to_midi(diff_freq)
                    
                    combination_pitches.append(midi_value)
                    combination_amps.append(comb_amp)
    
    return combination_pitches, combination_amps
