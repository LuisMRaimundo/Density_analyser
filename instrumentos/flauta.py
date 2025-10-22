# instrumentos/flauta.py

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern
import re
import logging
from microtonal import QUARTO_TOM_ACIMA, QUARTO_TOM_ABAIXO

# Configurar logging
logger = logging.getLogger('flauta')

# Constantes para símbolos microtonais
QUARTO_TOM_ACIMA = '?'  # Unicode para flecha para cima
QUARTO_TOM_ABAIXO = '?'  # Unicode para flecha para baixo

def converter_notacao(nota):
    """Converte a notação personalizada para a notação padrão.

    Args:
        nota (str): A nota na notação personalizada.

    Returns:
        str: A nota na notação padrão.
    """
    if 'b' in nota:  # Verifica se a nota contém um bemol
        nota_sustenido = nota.replace('b', '#')  # Substitui 'b' por '#'
        if '-' in nota:
            nota_sustenido = nota_sustenido.replace('#-', '#+')  # Substitui '#-' por '#+'
        elif '+' in nota:
            nota_sustenido = nota_sustenido.replace('#+', '#-')  # Substitui '#+' por '#-'
        return nota_sustenido
    return nota  # Retorna a nota original se não precisar converter

# Define the spectral data for each note and dynamic level
spectral_data = {
    'C4': {'pp': 4.723, 'mf': 11.721, 'ff': 18.528},
    'C#-4': {'pp': 4.723, 'mf': 11.721, 'ff': 18.528},
    'C#4': {'pp': 4.518, 'mf': 10.958, 'ff': 18.414},
    'C#+4': {'pp': 4.576, 'mf': 10.527, 'ff': 18.497},
    'D4': {'pp': 4.872, 'mf': 10.282, 'ff': 18.883},
    'D#-4': {'pp': 4.9, 'mf': 10.106, 'ff': 19.127},
    'D#4': {'pp': 4.733, 'mf': 9.465, 'ff': 19.465},
    'D#+4': {'pp': 4.665, 'mf': 9.0, 'ff': 19.005},
    'E4': {'pp': 4.894, 'mf': 8.278, 'ff': 17.054},
    'E#-4': {'pp': 5.206, 'mf': 8.35, 'ff': 16.193},
    'F4': {'pp': 6.193, 'mf': 9.398, 'ff': 14.68},
    'F#-4': {'pp': 6.269, 'mf': 9.965, 'ff': 13.853},
    'F#4': {'pp': 5.321, 'mf': 10.424, 'ff': 13.26},
    'F#+4': {'pp': 5.515, 'mf': 10.893, 'ff': 13.037},
    'G4': {'pp': 7.408, 'mf': 11.903, 'ff': 12.966},
    'G#-4': {'pp': 7.503, 'mf': 12.065, 'ff': 12.688},
    'G#4': {'pp': 6.547, 'mf': 11.894, 'ff': 12.149},
    'G#+4': {'pp': 5.864, 'mf': 11.253, 'ff': 11.99},
    'A4': {'pp': 4.861, 'mf': 9.739, 'ff': 12.007},
    'A#-4': {'pp': 4.403, 'mf': 9.191, 'ff': 11.654},
    'A#4': {'pp': 3.865, 'mf': 8.868, 'ff': 10.374},
    'A#+4': {'pp': 3.595, 'mf': 8.837, 'ff': 10.217},
    'B4': {'pp': 3.044, 'mf': 8.733, 'ff': 10.573},
    'B#-4': {'pp': 2.834, 'mf': 8.625, 'ff': 10.559},
    'C5': {'pp': 2.761, 'mf': 8.203, 'ff': 9.801},
    # ... [resto dos dados espectrais] ...
}

# Mapeamento de notação microtonal para valores fracionários de semitons
MICROTONAL_MAP = {
    # Quartos de tom com notação de símbolos
    f'C{QUARTO_TOM_ACIMA}': 0.5,
    f'D{QUARTO_TOM_ACIMA}': 2.5,
    f'E{QUARTO_TOM_ACIMA}': 4.5,
    f'F{QUARTO_TOM_ACIMA}': 5.5,
    f'G{QUARTO_TOM_ACIMA}': 7.5,
    f'A{QUARTO_TOM_ACIMA}': 9.5,
    f'B{QUARTO_TOM_ACIMA}': 11.5,
    
    # Quartos de tom com notação de adição/subtração
    'C+': 0.5,
    'C#-': 0.5,
    'C#+': 1.5,
    'D-': 1.5,
    'D+': 2.5,
    'D#-': 2.5,
    'D#+': 3.5,
    'E-': 3.5,
    'E+': 4.5,
    'F-': 4.5,
    'F+': 5.5,
    'F#-': 5.5,
    'F#+': 6.5,
    'G-': 6.5,
    'G+': 7.5,
    'G#-': 7.5,
    'G#+': 8.5,
    'A-': 8.5,
    'A+': 9.5,
    'A#-': 9.5,
    'A#+': 10.5,
    'B-': 10.5,
    'B+': 11.5,
    'C-': 11.5,
}

# Function to convert pitch notation to a numerical value (including microtones)
def nota_para_int(nota):
    """
    Converte a notação de altura para um valor numérico, suportando microtons.
    
    Args:
        nota (str): Nota musical (ex: 'C4', 'C?4', 'D#-5')
        
    Returns:
        float: Valor numérico da nota (com quartos de tom como valores fracionários)
        
    Raises:
        ValueError: Se a nota for inválida
    """
    try:
        # Verificar se a nota tem um símbolo microtonal
        if QUARTO_TOM_ACIMA in nota or QUARTO_TOM_ABAIXO in nota:
            # Extrair a nota base e a oitava
            match = re.match(r'([A-Ga-g][#b]?[??])(\d+)', nota)
            if not match:
                raise ValueError(f"Formato de nota microtonal inválido: {nota}")
                
            nota_base, oitava = match.groups()
            oitava = int(oitava)
            
            # Obter o valor microtonal
            if nota_base not in MICROTONAL_MAP:
                raise ValueError(f"Nota microtonal não reconhecida: {nota_base}")
                
            valor_base = MICROTONAL_MAP[nota_base]
            return valor_base + (oitava * 12)
            
        # Verificar notação de quarto de tom com +/-
        if '+' in nota or '-' in nota:
            # Extrair a nota base e a oitava
            match = re.match(r'([A-Ga-g][#b]?[+-])(\d+)', nota)
            if match:
                nota_base, oitava = match.groups()
                oitava = int(oitava)
                
                # Obter o valor microtonal
                if nota_base in MICROTONAL_MAP:
                    valor_base = MICROTONAL_MAP[nota_base]
                    return valor_base + (oitava * 12)
        
        # Processamento para notas cromáticas padrão
        nota_base, oitava = nota[:-1], int(nota[-1])
        
        # Escala cromática padrão
        escala = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }
        
        if nota_base not in escala:
            raise ValueError(f"Nota base desconhecida: {nota_base}")
            
        return escala[nota_base] + (oitava * 12)
        
    except Exception as e:
        logger.error(f"Erro ao converter nota '{nota}' para inteiro: {e}")
        raise ValueError(f"Nota inválida: {nota}")
def preprocess_nota(nota):
    """
    Preprocessa uma nota musical para garantir compatibilidade com diferentes notações microtonais.
    Converte símbolos Unicode (↑/↓) para notação +/-.
    
    Args:
        nota (str): Nota musical em qualquer formato
        
    Returns:
        str: Nota processada
    """
    if not nota:
        return nota
        
    # Converter símbolos Unicode para notação +/-
    if QUARTO_TOM_ACIMA in nota:
        nota = nota.replace(QUARTO_TOM_ACIMA, '+')
    if QUARTO_TOM_ABAIXO in nota:
        nota = nota.replace(QUARTO_TOM_ABAIXO, '-')
        
    return nota


def calcular_densidade(nota, dinamica):
    """
    Calcula a densidade com base nos dados espectrais, com fallback para notas não encontradas.
    
    Args:
        nota (str): Nota musical
        dinamica (str): Dinâmica (pp, mf, ff, etc.)
        
    Returns:
        float: Valor de densidade
    """
    try:
        # Pré-processar a nota para lidar com símbolos Unicode
        nota = preprocess_nota(nota)
        
        # Verificar se a nota existe no dicionário
        if nota in spectral_data and dinamica in spectral_data[nota]:
            return spectral_data[nota][dinamica]
            
            
        # Se não encontrar a nota ou a dinâmica, tentar encontrar a nota mais próxima
        nota_normalizada = nota
        
        # Tratar notas com símbolos microtonais
        if QUARTO_TOM_ACIMA in nota:
            # Converter para notação com + 
            nota_normalizada = nota.replace(QUARTO_TOM_ACIMA, '+')
        elif QUARTO_TOM_ABAIXO in nota:
            # Converter para notação com -
            nota_normalizada = nota.replace(QUARTO_TOM_ABAIXO, '-')
            
        # Tentar extrair nota base e oitava
        match = re.match(r'([A-Ga-g][#b]?[-+]?)(\d+)', nota_normalizada)
        if not match:
            logger.warning(f"Formato de nota inválido: {nota}, usando C4 como fallback")
            return spectral_data.get("C4", {}).get(dinamica, 5.0)
            
        nota_base, oitava = match.groups()
        oitava = int(oitava)
        
        # Procurar notas com a mesma base em oitavas próximas
        candidatos = []
        for nota_existente in spectral_data:
            match_existente = re.match(r'([A-Ga-g][#b]?[-+]?)(\d+)', nota_existente)
            if match_existente:
                base_existente, oitava_existente = match_existente.groups()
                oitava_existente = int(oitava_existente)
                if base_existente == nota_base:
                    # Calcular a "distância" em oitavas
                    distancia = abs(oitava - oitava_existente)
                    candidatos.append((nota_existente, distancia))
        
        # Se encontrou candidatos, usar o mais próximo
        if candidatos:
            # Ordenar por distância (menor primeiro)
            candidatos.sort(key=lambda x: x[1])
            nota_proxima = candidatos[0][0]
            
            # Log para diagnóstico
            logger.info(f"Nota {nota} não encontrada, usando {nota_proxima} como aproximação")
            
            # Verificar se a dinâmica existe
            if dinamica in spectral_data[nota_proxima]:
                return spectral_data[nota_proxima][dinamica]
            
            # Se a dinâmica não existir, tentar usar uma dinâmica padrão
            dinamicas_padrao = ['pp', 'mf', 'ff']
            if dinamica not in dinamicas_padrao:
                if dinamica in ['pppp', 'ppp', 'p']:
                    return spectral_data[nota_proxima]['pp']
                elif dinamica in ['f', 'fff', 'ffff']:
                    return spectral_data[nota_proxima]['ff']
                else:
                    return spectral_data[nota_proxima]['mf']
            
            # Fallback final - usar a média dos valores disponíveis
            return sum(spectral_data[nota_proxima].values()) / len(spectral_data[nota_proxima])
        
        # Fallback para C4 se nenhum candidato for encontrado
        return spectral_data.get("C4", {}).get(dinamica, 5.0)
    except Exception as e:
        # Em caso de erro, retornar um valor padrão
        logger.error(f"Erro ao calcular densidade para {nota}/{dinamica}: {e}")
        return 5.0  # Valor médio arbitrário como fallback final


# Existing function to predict intermediate dynamics with updated methods
def predict_intermediate_dynamics(pitches, pp_values, mf_values, ff_values):
    """Prevê dinâmicas intermediárias usando Gaussian Process Regression."""
    dynamic_levels = {"pppp": 1, "ppp": 2, "pp": 3, "p": 4, "mf": 5, "f": 6, "ff": 7, "fff": 8, "ffff": 9}
    all_dynamics = list(dynamic_levels.keys())
    predictions = {dynamic: [] for dynamic in all_dynamics}

    # Otimização com NumPy:
    existing_levels = np.array([dynamic_levels[d] for d in ["pp", "mf", "ff"]]).reshape(-1, 1)
    all_levels = np.array([dynamic_levels[d] for d in all_dynamics]).reshape(-1, 1)
    y_train = np.array([pp_values, mf_values, ff_values]).T

    matern_kernel = C(1.0) * Matern(length_scale=1.0, nu=1.5)
    gpr = GaussianProcessRegressor(kernel=matern_kernel, n_restarts_optimizer=10, alpha=1e-1)

    for y in y_train:
        gpr.fit(existing_levels, y)
        y_pred = gpr.predict(all_levels)
        for j, dynamic in enumerate(all_dynamics):
            predictions[dynamic].append(y_pred[j])

    return {k: np.array(v) for k, v in predictions.items()}

def get_max_note_density(nota, num):
    """Retorna a densidade máxima da nota."""
    return max(spectral_data.get(nota, {}).values()) * np.sqrt(num) if nota in spectral_data else 0

def calculate_max_possible_density(notas, dinamicas, numeros_instrumentos):
    """Calcula a densidade máxima possível."""
    return sum(get_max_note_density(nota, num) for nota, num in zip(notas, numeros_instrumentos))
