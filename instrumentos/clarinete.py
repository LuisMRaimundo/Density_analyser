# instrumentos/clarinete.py

import re
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern
import logging
from utils.notes import (
    note_to_midi,
    midi_to_note_name,
    to_sharp,
    extract_cents,
    is_valid_note,
    normalize_note_string,
)

logger = logging.getLogger('clarinete')

# Define the spectral data for each note and dynamic level for clarinete
# Estes valores são adaptados do clarinete - se precisar dos valores reais, 
# substitua estes dados por medições específicas para o clarinete

spectral_data_unicode = {
    'D3': {'pp': 11.258, 'mf': 47.852, 'ff': 44.549},
    'D↓3': {'pp': 10.977, 'mf': 45.983, 'ff': 44.174},
    'D♯3': {'pp': 10.703, 'mf': 44.755, 'ff': 44.537},
    'D♯↓3': {'pp': 10.466, 'mf': 43.205, 'ff': 44.547},
    'E3': {'pp': 10.193, 'mf': 41.833, 'ff': 44.475},
    'E↓3': {'pp': 9.973, 'mf': 40.519, 'ff': 44.636},
    'F3': {'pp': 9.724, 'mf': 39.081, 'ff': 44.294},
    'F↓3': {'pp': 9.511, 'mf': 37.93, 'ff': 44.395},
    'F♯3': {'pp': 9.297, 'mf': 36.493, 'ff': 43.908},
    'F♯↓3': {'pp': 9.087, 'mf': 35.447, 'ff': 43.792},
    'G3': {'pp': 8.907, 'mf': 34.063, 'ff': 43.237},
    'G↓3': {'pp': 8.705, 'mf': 33.074, 'ff': 42.816},
    'G♯3': {'pp': 8.554, 'mf': 31.786, 'ff': 42.22},
    'G♯↓3': {'pp': 8.366, 'mf': 30.816, 'ff': 41.476},
    'A3': {'pp': 8.235, 'mf': 29.655, 'ff': 40.834},
    'A↓3': {'pp': 8.068, 'mf': 28.678, 'ff': 39.803},
    'A♯3': {'pp': 7.949, 'mf': 27.666, 'ff': 39.091},
    'A♯↓3': {'pp': 7.806, 'mf': 26.661, 'ff': 37.851},
    'B3': {'pp': 7.692, 'mf': 25.812, 'ff': 37.041},
    'B↓3': {'pp': 7.575, 'mf': 24.768, 'ff': 35.688},
    'C4': {'pp': 7.464, 'mf': 23.778, 'ff': 34.762},
    'C↓4': {'pp': 7.372, 'mf': 22.999, 'ff': 33.394},
    'C♯4': {'pp': 7.261, 'mf': 24.229, 'ff': 32.352},
    'C♯↓4': {'pp': 7.19, 'mf': 21.354, 'ff': 31.051},
    'D4': {'pp': 7.083, 'mf': 19.956, 'ff': 29.911},
    'D↓4': {'pp': 7.026, 'mf': 19.832, 'ff': 28.736},
    'D♯4': {'pp': 6.926, 'mf': 20.569, 'ff': 27.532},
    'D♯↓4': {'pp': 6.878, 'mf': 18.43, 'ff': 26.512},
    'E4': {'pp': 6.79, 'mf': 21.447, 'ff': 25.292},
    'E↓4': {'pp': 6.744, 'mf': 17.147, 'ff': 24.422},
    'F4': {'pp': 6.671, 'mf': 13.15, 'ff': 23.239},
    'F↓4': {'pp': 6.624, 'mf': 15.977, 'ff': 22.492},
    'F♯4': {'pp': 6.568, 'mf': 9.543, 'ff': 21.394},
    'F♯↓4': {'pp': 6.517, 'mf': 14.916, 'ff': 20.723},
    'G4': {'pp': 6.479, 'mf': 20.043, 'ff': 19.75},
    'G↓4': {'pp': 6.424, 'mf': 13.959, 'ff': 19.101},
    'G♯4': {'pp': 6.401, 'mf': 15.777, 'ff': 18.276},
    'G♯↓4': {'pp': 6.344, 'mf': 13.1, 'ff': 17.597},
    'A4': {'pp': 6.333, 'mf': 12.515, 'ff': 16.926},
    'A↓4': {'pp': 6.278, 'mf': 12.332, 'ff': 16.182},
    'A♯4': {'pp': 6.273, 'mf': 15.069, 'ff': 15.648},
    'A♯↓4': {'pp': 6.223, 'mf': 11.648, 'ff': 14.825},
    'B4': {'pp': 6.218, 'mf': 10.409, 'ff': 14.395},
    'B↓4': {'pp': 6.176, 'mf': 11.04, 'ff': 13.507},
    'C5': {'pp': 6.167, 'mf': 10.842, 'ff': 13.134},
    'C↓5': {'pp': 6.136, 'mf': 10.501, 'ff': 12.224},
    'C♯5': {'pp': 6.117, 'mf': 11.535, 'ff': 11.852},
    'C♯↓5': {'pp': 6.096, 'mf': 10.022, 'ff': 10.987},
    'D5': {'pp': 6.066, 'mf': 9.518, 'ff': 10.562},
    'D↓5': {'pp': 6.055, 'mf': 9.595, 'ff': 9.822},
    'D♯5': {'pp': 6.013, 'mf': 8.958, 'ff': 9.301},
    'D♯↓5': {'pp': 6.006, 'mf': 9.212, 'ff': 8.763},
    'E5': {'pp': 5.955, 'mf': 8.981, 'ff': 8.128},
    'E↓5': {'pp': 5.948, 'mf': 8.865, 'ff': 7.849},
    'F5': {'pp': 5.89, 'mf': 9.965, 'ff': 7.112},
    'F↓5': {'pp': 5.877, 'mf': 8.544, 'ff': 7.116},
    'F♯5': {'pp': 5.816, 'mf': 7.804, 'ff': 6.323},
    'F♯↓5': {'pp': 5.791, 'mf': 8.242, 'ff': 6.585},
    'G5': {'pp': 5.732, 'mf': 7.973, 'ff': 5.816},
    'G↓5': {'pp': 5.691, 'mf': 7.952, 'ff': 6.261},
    'G♯5': {'pp': 5.634, 'mf': 12.861, 'ff': 5.617},
    'G♯↓5': {'pp': 5.575, 'mf': 7.665, 'ff': 6.127},
    'A5': {'pp': 5.522, 'mf': 8.499, 'ff': 5.714},
    'A↓5': {'pp': 5.444, 'mf': 7.374, 'ff': 6.145},
    'A♯5': {'pp': 5.392, 'mf': 7.108, 'ff': 6.045},
    'A♯↓5': {'pp': 5.298, 'mf': 7.074, 'ff': 6.259},
    'B5': {'pp': 5.244, 'mf': 11.004, 'ff': 6.501},
    'B↓5': {'pp': 5.137, 'mf': 6.757, 'ff': 6.4},
    'C6': {'pp': 5.074, 'mf': 0.226, 'ff': 6.935},
    'C↓6': {'pp': 4.959, 'mf': 6.419, 'ff': 6.499},
    'C♯6': {'pp': 4.881, 'mf': 7.014, 'ff': 7.183},
    'C♯↓6': {'pp': 4.762, 'mf': 6.054, 'ff': 6.49},
    'D6': {'pp': 4.663, 'mf': 9.555, 'ff': 7.091},
    'D↓6': {'pp': 4.542, 'mf': 5.659, 'ff': 6.323},
    'D♯6': {'pp': 4.418, 'mf': 3.303, 'ff': 6.559},
    'D♯↓6': {'pp': 4.294, 'mf': 5.229, 'ff': 5.97},
    'E6': {'pp': 4.143, 'mf': 3.304, 'ff': 5.577},
    'E↓6': {'pp': 4.014, 'mf': 4.763, 'ff': 5.429},
    'F6': {'pp': 3.837, 'mf': 0.803, 'ff': 4.261},
    'F↓6': {'pp': 3.695, 'mf': 4.257, 'ff': 4.727},
    'F♯6': {'pp': 3.497, 'mf': 6.521, 'ff': 2.875},
    'F♯↓6': {'pp': 3.335, 'mf': 3.711, 'ff': 3.914},
    'G6': {'pp': 3.122, 'mf': 4.49, 'ff': 1.798},
    'G↓6': {'pp': 2.93, 'mf': 3.124, 'ff': 3.062},
    'G♯6': {'pp': 2.709, 'mf': 0.843, 'ff': 1.44},
    'G♯↓6': {'pp': 2.482, 'mf': 2.496, 'ff': 2.251},
    'A6': {'pp': 2.257, 'mf': 7.172, 'ff': 2.042},
    'A↓6': {'pp': 1.995, 'mf': 1.829, 'ff': 1.569},
    'A♯6': {'pp': 1.763, 'mf': 1.553, 'ff': 3.337},
    'A♯↓6': {'pp': 1.476, 'mf': 1.123, 'ff': 1.093},
    'B6': {'pp': 1.225, 'mf': 0.153, 'ff': 3.989},
    'B↓6': {'pp': 0.937, 'mf': 0.381, 'ff': 0.888},
    'C7': {'pp': 0.642, 'mf': 0.118, 'ff': 0.759},
}

# 2. Converte todas as chaves para a forma canónica (ASCII, sem Unicode)
spectral_data = {
    normalize_note_string(nota): valores
    for nota, valores in spectral_data_unicode.items()
}

def nota_para_int(nota):
    """Converte a notação de altura para um inteiro (MIDI-like)."""
    try:
        midi = note_to_midi(normalize_note_string(nota))
        return int(round(midi))
    except Exception as e:
        logger.error(f"Erro ao converter nota '{nota}' para inteiro: {e}")
        return 60  # C4 como fallback

def calcular_densidade(nota, dinamica):
    """
    Calcula a densidade com base nos dados espectrais, com fallback robusto para notas não encontradas.
    """
    try:
        # Extrair cents e normalizar notação
        base_nota, cents = extract_cents(nota)
        nota_norm = normalize_note_string(base_nota)

        # Verificar existência direta
        if nota_norm in spectral_data and dinamica in spectral_data[nota_norm]:
            return spectral_data[nota_norm][dinamica]

        # Ajuste de dinâmica para as três básicas
        dinamica_proxima = dinamica
        if dinamica not in ['pp', 'mf', 'ff']:
            if dinamica in ['pppp', 'ppp', 'p']:
                dinamica_proxima = 'pp'
            elif dinamica in ['f', 'fff', 'ffff']:
                dinamica_proxima = 'ff'
            else:
                dinamica_proxima = 'mf'

        if nota_norm in spectral_data and dinamica_proxima in spectral_data[nota_norm]:
            return spectral_data[nota_norm][dinamica_proxima]

        # Busca nota mais próxima por base e oitava
        match = re.match(r'([A-Ga-g][#b]?)(\d+)', nota_norm)
        if not match:
            logger.warning(f"Formato de nota inválido após normalização: {nota_norm}, usando C4 como fallback")
            return spectral_data.get("C4", {}).get(dinamica_proxima, 5.0)

        nota_base, oitava = match.groups()
        oitava = int(oitava)
        candidatos = []
        for n_existente in spectral_data:
            m = re.match(r'([A-Ga-g][#b]?)(\d+)', n_existente)
            if m:
                base_existente, oitava_existente = m.groups()
                oitava_existente = int(oitava_existente)
                if base_existente.lower() == nota_base.lower():
                    distancia = abs(oitava - oitava_existente)
                    candidatos.append((n_existente, distancia))

        if candidatos:
            candidatos.sort(key=lambda x: x[1])
            nota_proxima = candidatos[0][0]
            logger.info(f"Nota {nota} não encontrada, usando {nota_proxima} como aproximação")
            if dinamica_proxima in spectral_data[nota_proxima]:
                return spectral_data[nota_proxima][dinamica_proxima]
            elif 'mf' in spectral_data[nota_proxima]:
                return spectral_data[nota_proxima]['mf']
            else:
                return sum(spectral_data[nota_proxima].values()) / len(spectral_data[nota_proxima])

        # Fallback final: C4
        logger.warning(f"Nenhuma nota similar a {nota} encontrada, usando fallback")
        if "C4" in spectral_data and dinamica_proxima in spectral_data["C4"]:
            return spectral_data["C4"][dinamica_proxima]
        else:
            valores = [val for nota_dict in spectral_data.values() for val in nota_dict.values()]
            return sum(valores) / len(valores) if valores else 5.0

    except Exception as e:
        logger.error(f"Erro ao calcular densidade para {nota}/{dinamica}: {e}")
        return 5.0

def predict_intermediate_dynamics(pitches, pp_values, mf_values, ff_values):
    """
    Prevê dinâmicas intermediárias usando Gaussian Process Regression.
    """
    dynamic_levels = {"pppp": 1, "ppp": 2, "pp": 3, "p": 4, "mf": 5, "f": 6, "ff": 7, "fff": 8, "ffff": 9}
    all_dynamics = list(dynamic_levels.keys())
    predictions = {dynamic: [] for dynamic in all_dynamics}

    existing_levels = np.array([dynamic_levels[d] for d in ["pp", "mf", "ff"]]).reshape(-1, 1)
    all_levels = np.array([dynamic_levels[d] for d in all_dynamics]).reshape(-1, 1)

    try:
        y_train = np.array([pp_values, mf_values, ff_values]).T
        if y_train.size == 0 or np.isnan(y_train).any():
            logger.warning("Dados de treinamento insuficientes ou inválidos para GPR")
            return {d: np.zeros_like(pp_values) for d in all_dynamics}

        matern_kernel = C(1.0) * Matern(length_scale=1.0, nu=1.5)
        gpr = GaussianProcessRegressor(kernel=matern_kernel, n_restarts_optimizer=10, alpha=1e-1)

        for i, y in enumerate(y_train):
            try:
                gpr.fit(existing_levels, y)
                y_pred = gpr.predict(all_levels)
                for j, dynamic in enumerate(all_dynamics):
                    predictions[dynamic].append(y_pred[j])
            except Exception as e:
                logger.error(f"Erro no GPR para pitch {i}: {e}")
                for j, dynamic in enumerate(all_dynamics):
                    if dynamic == "pp":
                        predictions[dynamic].append(y[0])
                    elif dynamic == "mf":
                        predictions[dynamic].append(y[1])
                    elif dynamic == "ff":
                        predictions[dynamic].append(y[2])
                    elif dynamic in ["pppp", "ppp", "p"]:
                        predictions[dynamic].append(y[0])
                    else:
                        predictions[dynamic].append(y[2])

        return {k: np.array(v) for k, v in predictions.items()}
    except Exception as e:
        logger.error(f"Erro ao prever dinâmicas intermediárias: {e}")
        return {d: np.zeros_like(pp_values) for d in all_dynamics}

def get_max_note_density(nota, num):
    """
    Retorna a densidade máxima da nota multiplicada pela raiz quadrada do número de instrumentos.
    """
    try:
        nota_norm = normalize_note_string(nota)
        if nota_norm in spectral_data:
            max_value = max(spectral_data[nota_norm].values())
            return max_value * np.sqrt(num)
        match = re.match(r'([A-Ga-g][#b]?)(\d+)', nota_norm)
        if match:
            nota_base, oitava = match.groups()
            oitava = int(oitava)
            candidatos = []
            for n_existente in spectral_data:
                m = re.match(r'([A-Ga-g][#b]?)(\d+)', n_existente)
                if m:
                    base_existente, oitava_existente = m.groups()
                    oitava_existente = int(oitava_existente)
                    if base_existente.lower() == nota_base.lower():
                        distancia = abs(oitava - oitava_existente)
                        candidatos.append((n_existente, distancia))
            if candidatos:
                candidatos.sort(key=lambda x: x[1])
                nota_proxima = candidatos[0][0]
                max_value = max(spectral_data[nota_proxima].values())
                return max_value * np.sqrt(num)
        valores_maximos = [max(vals.values()) for vals in spectral_data.values()]
        valor_medio = sum(valores_maximos) / len(valores_maximos)
        return valor_medio * np.sqrt(num)
    except Exception as e:
        logger.warning(f"Erro ao obter densidade máxima para {nota}: {e}")
        return 50.0 * np.sqrt(num)

def calculate_max_possible_density(notas, dinamicas, numeros_instrumentos):
    """
    Calcula a densidade máxima possível para um conjunto de notas.
    """
    total = 0.0
    if not notas or not numeros_instrumentos:
        return 100.0
    n = min(len(notas), len(numeros_instrumentos))
    for i in range(n):
        nota = notas[i]
        num = numeros_instrumentos[i]
        nota_density = get_max_note_density(nota, num)
        total += nota_density
    return max(1.0, total)
