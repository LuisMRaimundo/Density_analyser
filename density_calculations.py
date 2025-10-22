# density_calculations_improved.py
import numpy as np
from typing import List, Tuple, Dict
from utils.notes import normalize_note_string, is_valid_note

def get_register_weight(midi_note: float) -> float:
    """
    Retorna peso perceptual baseado no registro.
    Valores baseados em sensibilidade auditiva por registro.
    """
    if midi_note < 36:  # < C2
        return 0.5
    elif midi_note < 48:  # C2-C3
        return 0.65
    elif midi_note < 60:  # C3-C4
        return 0.8
    elif midi_note < 72:  # C4-C5
        return 1.0
    elif midi_note < 84:  # C5-C6
        return 0.95
    elif midi_note < 96:  # C6-C7
        return 0.85
    else:  # >= C7
        return 0.7



def calcular_densidade_intervalar(notas: List[float],
                                 usar_ponderacao_perceptual: bool = False) -> float:
    """
    Calcula a densidade de intervalo com ponderação perceptual opcional.

    Args:
        notas: Lista de valores MIDI
        usar_ponderacao_perceptual: Se True, aplica ponderação por registro

    Returns:
        float: Densidade intervalar (possivelmente ponderada)
    """
    if len(notas) < 2:
        return 0.0

    notas_ordenadas = sorted(notas)

    # Cálculo existente
    intervalos_adjacentes = [
        notas_ordenadas[i+1] - notas_ordenadas[i]
        for i in range(len(notas_ordenadas)-1)
    ]

    intervalo_total = notas_ordenadas[-1] - notas_ordenadas[0]

    if usar_ponderacao_perceptual:
        # Aplicar ponderação aos intervalos baseada nas notas envolvidas
        intervalos_ponderados = []
        for i in range(len(notas_ordenadas)-1):
            nota1 = notas_ordenadas[i]
            nota2 = notas_ordenadas[i+1]

            # Peso médio das duas notas do intervalo
            peso = (get_register_weight(nota1) + get_register_weight(nota2)) / 2

            # Intervalo ponderado
            intervalo = (nota2 - nota1) / peso  # Dividir pelo peso inverte o efeito
            intervalos_ponderados.append(intervalo)

        # Média ponderada
        if intervalos_ponderados:
            media_adjacentes = np.mean(intervalos_ponderados)
        else:
            media_adjacentes = 0
    else:
        # Cálculo original
        media_adjacentes = np.mean(intervalos_adjacentes) if intervalos_adjacentes else 0

    # Continuar com o cálculo existente
    densidade_intervalar = 0.7 * media_adjacentes + 0.3 * intervalo_total

    return densidade_intervalar

# Atualizar a função analisar_densidade_completa
def analisar_densidade_completa(notas: List[float],
                               pesos: List[float] = None,
                               usar_ponderacao_perceptual: bool = False) -> Dict[str, float]:
    """
    Análise completa de densidade com múltiplas métricas.

    Args:
        notas: Lista de valores MIDI
        pesos: Lista opcional de pesos
        usar_ponderacao_perceptual: Se True, aplica ponderação perceptual aos intervalos

    Returns:
        dict: Dicionário com todas as métricas de densidade
    """
    resultado = {
        'densidade_basica': calcular_densidade(notas),
        'densidade_intervalar': calcular_densidade_intervalar(notas, usar_ponderacao_perceptual),
        'densidade_ponderada': calcular_densidade_ponderada(notas, pesos),
        'distribuicao_espacial': calcular_distribuicao_espacial(notas),
        'massa': calcular_massa(notas),
        'volume': calcular_volume(notas)
    }

    # Adicionar densidades por registro
    densidades_registro = calcular_densidade_por_registro(notas, usar_ponderacao_perceptual)
    resultado.update({
        f'densidade_{k}': v for k, v in densidades_registro.items()
    })

    return resultado

def calcular_massa(notas: List[float]) -> float:
    """
    Calcula a massa de uma banda sonora com base no número de notas.

    Args:
        notas: Lista de valores MIDI (podem ser floats para microtons)

    Returns:
        float: Número de notas (massa)
    """
    return float(len(notas))


def calcular_volume(notas: List[float]) -> float:
    """
    Calcula o volume (intervalo de alturas) em semitons.

    Args:
        notas: Lista de valores MIDI

    Returns:
        float: Intervalo em semitons, mínimo de 1.0 para evitar divisão por zero
    """
    if not notas:
        return 1.0  # Evita divisão por zero

    notas_ordenadas = sorted(notas)
    volume = notas_ordenadas[-1] - notas_ordenadas[0]

    # Garantir volume mínimo de 1 semitom para evitar densidades infinitas
    divisoes_por_tom = 2          # 2 = quartos-de-tom; altera se usares outra resolução
    min_volume = 1.0 / divisoes_por_tom
    return max(volume, min_volume)


def calcular_densidade(notas: List[float],
                      usar_ponderacao_perceptual: bool = False) -> float:
    """
    Calcula a densidade geral para um conjunto de notas.

    Args:
        notas: Lista de valores MIDI
        usar_ponderacao_perceptual: Se True, aplica ponderação por registro

    Returns:
        float: Densidade (massa/volume)
    """
    if not notas:
        return 0.0

    # As notas já são valores MIDI, não precisam de normalização de string.
    notas_validas = notas

    if usar_ponderacao_perceptual:
        massa_ponderada = sum(get_register_weight(nota) for nota in notas_validas)
        massa = massa_ponderada
    else:
        massa = calcular_massa(notas_validas)

    volume = calcular_volume(notas_validas)

    if volume == 0:
        return 0.0

    return massa / volume


def calcular_densidade_ponderada(notas: List[float],
                                pesos: List[float] = None,
                                usar_ponderacao_perceptual: bool = False) -> float:
    """
    Calcula densidade com ponderação opcional por nota.

    Args:
        notas: Lista de valores MIDI
        pesos: Lista de pesos (ex: amplitudes, dinâmicas)
        usar_ponderacao_perceptual: Se True, aplica ponderação por registro

    Returns:
        float: Densidade ponderada
    """
    if not notas:
        return 0.0

    if pesos is None:
        pesos = [1.0] * len(notas)

    # Normalizar pesos
    pesos_norm = np.array(pesos) / np.sum(pesos)

    if usar_ponderacao_perceptual:
        # Aplicar ponderação perceptual adicional
        pesos_registro = [get_register_weight(nota) for nota in notas]
        pesos_combinados = pesos_norm * pesos_registro
        pesos_norm = pesos_combinados / np.sum(pesos_combinados)

    # Massa ponderada
    massa_ponderada = np.sum(pesos_norm * len(notas))

    # Volume permanece o mesmo
    volume = calcular_volume(notas)

    return massa_ponderada / volume


def calcular_densidade_por_registro(notas: List[float],
                                   usar_ponderacao_perceptual: bool = False) -> Dict[str, float]:
    """
    Calcula densidade separadamente por registro.

    Args:
        notas: Lista de valores MIDI
        usar_ponderacao_perceptual: Se True, aplica ponderação por registro

    Returns:
        dict: Densidades por registro e total
    """
    if not notas:
        return {
            'grave': 0.0,
            'medio': 0.0,
            'agudo': 0.0,
            'total': 0.0
        }

    # Separar por registros (baseado em oitavas MIDI)
    registros = {
        'grave': [n for n in notas if n < 48],     # < C3
        'medio': [n for n in notas if 48 <= n < 72], # C3-B5
        'agudo': [n for n in notas if n >= 72]     # >= C6
    }

    # Calcular densidade para cada registro
    densidades = {}
    for nome, notas_registro in registros.items():
        if notas_registro:
            densidades[nome] = calcular_densidade(notas_registro, usar_ponderacao_perceptual)
        else:
            densidades[nome] = 0.0

    # Densidade total
    densidades['total'] = calcular_densidade(notas, usar_ponderacao_perceptual)

    return densidades


def calcular_distribuicao_espacial(notas: List[float]) -> float:
    """
    Calcula quão uniformemente as notas estão distribuídas no espaço de alturas.
    Retorna valor entre 0 (todas agrupadas) e 1 (perfeitamente distribuídas).

    Args:
        notas: Lista de valores MIDI

    Returns:
        float: Índice de distribuição espacial (0-1)
    """
    if len(notas) < 2:
        return 0.0

    notas_ordenadas = sorted(notas)
    n = len(notas_ordenadas)

    # Calcular intervalos entre notas adjacentes
    intervalos = [
        notas_ordenadas[i+1] - notas_ordenadas[i]
        for i in range(n-1)
    ]

    if not intervalos:
        return 0.0

    # Distribuição perfeita teria todos os intervalos iguais
    intervalo_ideal = (notas_ordenadas[-1] - notas_ordenadas[0]) / (n - 1)

    # Calcular desvio da distribuição ideal
    if intervalo_ideal > 0:
        desvios = [abs(i - intervalo_ideal) / intervalo_ideal for i in intervalos]
        uniformidade = 1.0 - np.mean(desvios)
        return max(0.0, min(1.0, uniformidade))  # Limitar entre 0 e 1
    else:
        return 0.0

# Add to density_calculations.py (at the end, before the test section)

def calculate_final_density(notas: List[float],
                          dinamicas: List[str] = None,
                          instrumentos: List[str] = None,
                          interval_weight: float = 0.5,
                          usar_ponderacao_perceptual: bool = False) -> Dict[str, float]:
    """
    Calcula densidade final combinando densidade intervalar e instrumental.

    Args:
        notas: Lista de valores MIDI
        dinamicas: Lista de dinâmicas para cada nota (pp, mf, ff, etc.)
        instrumentos: Lista de instrumentos para cada nota
        interval_weight: Peso para densidade intervalar (0-1)
                        0 = 100% instrumental, 1 = 100% intervalar
        usar_ponderacao_perceptual: Aplicar ponderação perceptual aos intervalos

    Returns:
        dict: Dicionário com densidades e componentes
    """
    # Importar densidade intervalar do módulo específico
    from densidade_intervalar import calcular_densidade_intervalar as calc_intervalar_ext

    # 1. Calcular densidade intervalar
    # Usar a versão do módulo densidade_intervalar que tem calibração lambda
    densidade_intervalar = calc_intervalar_ext(
        [f"C{int(n//12)}" for n in notas],  # Converter MIDI para notas
        lamb=None  # Usar lambda calibrado
    )

    # Alternativamente, usar a versão local com ponderação perceptual opcional
    densidade_intervalar_local = calcular_densidade_intervalar(notas, usar_ponderacao_perceptual)

    # 2. Calcular densidade instrumental
    densidade_instrumental = 0.0

    if instrumentos and dinamicas:
        # Importar funções específicas dos instrumentos
        try:
            # Para cada nota, obter densidade do instrumento específico
            for i, (nota, dinamica, instrumento) in enumerate(zip(notas, dinamicas, instrumentos)):
                if instrumento.lower() == 'clarinete':
                    from instrumentos.clarinete import calcular_densidade as calc_clar
                    densidade_nota = calc_clar(midi_to_note(nota), dinamica)
                # Adicionar outros instrumentos conforme implementados
                # elif instrumento.lower() == 'violino':
                #     from instrumentos.violino import calcular_densidade as calc_viol
                #     densidade_nota = calc_viol(midi_to_note(nota), dinamica)
                else:
                    # Valor padrão se instrumento não implementado
                    densidade_nota = 10.0

                densidade_instrumental += densidade_nota

        except ImportError as e:
            logger.warning(f"Erro ao importar módulo de instrumento: {e}")
            # Usar densidade básica como fallback
            densidade_instrumental = len(notas) * 10.0
    else:
        # Se não há informação de instrumentos, usar densidade básica
        densidade_instrumental = len(notas) * 10.0

    # 3. Combinar densidades com peso do usuário
    densidade_total = (interval_weight * densidade_intervalar +
                      (1 - interval_weight) * densidade_instrumental)

    # 4. Retornar resultados detalhados
    return {
        'densidade_total': densidade_total,
        'densidade_intervalar': densidade_intervalar,
        'densidade_instrumental': densidade_instrumental,
        'peso_intervalar': interval_weight,
        'peso_instrumental': 1 - interval_weight,
        'ponderacao_perceptual': usar_ponderacao_perceptual,
        'componentes': {
            'notas': notas,
            'dinamicas': dinamicas,
            'instrumentos': instrumentos
        }
    }


def midi_to_note(midi_value: float) -> str:
    """
    Converte valor MIDI para notação de nota.

    Args:
        midi_value: Valor MIDI (pode ser float para microtons)

    Returns:
        str: Nota no formato "C4", "D#5", etc.
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Extrair oitava e nota
    octave = int(midi_value // 12) - 1
    note_index = int(midi_value % 12)

    # Verificar se há microtons
    microtone = midi_value - int(midi_value)

    base_note = f"{note_names[note_index]}{octave}"

    # Adicionar indicação de microtons se necessário
    if microtone > 0.45:
        base_note += "↑"  # Quarto de tom acima
    elif microtone > 0.05:
        base_note += f"+{int(microtone * 100)}c"  # Cents

    return base_note

# Testes
if __name__ == "__main__":
    # Teste 1: Acorde de C maior
    c_major = [60, 64, 67]  # C4, E4, G4
    print("C Major:", analisar_densidade_completa(c_major))

    # Teste 2: Cluster cromático
    cluster = list(range(60, 65))  # C4 até E4
    print("\nCluster:", analisar_densidade_completa(cluster))

    # Teste 3: Distribuído
    distributed = [36, 48, 60, 72, 84]  # C2, C3, C4, C5, C6
    print("\nDistribuído:", analisar_densidade_completa(distributed))

    # Teste 4: Com pesos (simulando dinâmicas)
    notes = [60, 64, 67, 72]
    weights = [0.5, 1.0, 0.8, 0.3]  # pp, f, mf, pp
    print("\nCom pesos:", analisar_densidade_completa(notes, weights))
