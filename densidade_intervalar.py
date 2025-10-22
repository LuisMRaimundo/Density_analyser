# densidade_intervalar.py - versão com parâmetros ajustáveis
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import os
from scipy.optimize import minimize
from typing import Optional
from config import USE_LOG_COMPRESSION


# Configurar logging
logger = logging.getLogger('densidade_intervalar')

# Importar funções do módulo centralizado (como estava no seu código original)
from microtonal import (
    nota_para_posicao, escala_microtonal,
    note_to_midi, QUARTO_TOM_ACIMA, QUARTO_TOM_ABAIXO,
    ESCALA_MICROTONAL
)

from psychoacoustic_corrections import (
    critical_band_masking,
    calculate_roughness,
    apply_loudness_correction           # wrapper que devolve ganho médio
)




# ------------------------------------------------------------------------------
# PARÂMETROS GLOBAIS
# ------------------------------------------------------------------------------
TAMANHO_OITAVA_MICROTONAL = 24

# Caminho para o arquivo de configuração de parâmetros calibrados
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'density_params.json')

# Valores de referência para avaliações empíricas de consonância de díades
# Baseados em dados de Hutchinson & Knopoff, Malmberg, Kameoka & Kuriyagawa
CONSONANCE_RATINGS = {
    0: 1.0,    # unisono
    2: -0.582, # M2/m7
    3: 0.594,  # m3/M6
    4: 0.386,  # M3/m6
    5: 1.240,  # P4/P5
    6: -0.453, # TT
}

# Valor padrão para lambda caso não haja calibração disponível
DEFAULT_LAMBDA = 0.05

# Lista de notas para uso em funções internas
llista_notas = list(escala_microtonal.keys())

# Lista simplificada com apenas notas cromáticas padrão (12 por oitava)
lista_notas_cromaticas = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# ------------------------------------------------------------------------------
# Gerenciamento de parâmetros calibrados
# ------------------------------------------------------------------------------

def carregar_parametros_calibrados():
    """
    Carrega parâmetros calibrados de um arquivo JSON.
    Retorna o valor de lambda baseado em dados experimentais.
    """
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                params = json.load(f)
                logger.info(f"Parâmetros carregados: {params}")
                return params.get('lambda', DEFAULT_LAMBDA)
        else:
            logger.warning(f"Arquivo de configuração não encontrado: {CONFIG_PATH}")
            return DEFAULT_LAMBDA
    except Exception as e:
        logger.error(f"Erro ao carregar parâmetros: {e}")
        return DEFAULT_LAMBDA

def salvar_parametros_calibrados(params):
    """
    Salva parâmetros calibrados em um arquivo JSON.
    """
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, 'w') as f:
            json.dump(params, f, indent=4)
            logger.info(f"Parâmetros salvos: {params}")
        return True
    except Exception as e:
        logger.error(f"Erro ao salvar parâmetros: {e}")
        return False

# ------------------------------------------------------------------------------
# Calibração de parâmetros com base em dados experimentais
# ------------------------------------------------------------------------------

def calibrar_lambda(dados_experimentais=None):
    """
    Calibra o valor de lambda usando dados experimentais.
    Se não forem fornecidos dados, usa valores de referência da literatura.
    
    dados_experimentais: dicionário na forma {intervalo: valor_consonancia}
    
    Retorna o valor de lambda otimizado.
    """
    # Se não há dados experimentais fornecidos, usa valores de referência
    if dados_experimentais is None:
        dados_experimentais = CONSONANCE_RATINGS
        
    logger.info(f"Iniciando calibração com dados: {dados_experimentais}")
    
    # Intervalo para busca de lambda (0.01 a 1.0)
    bounds = [(0.01, 1.0)]
    
    # Função objetivo: minimizar o erro quadrático entre predições e dados experimentais
    def objetivo(lambda_val):
        lambda_val = lambda_val[0]  # Desempacotar valor (scipy.optimize requer array)
        error_sum = 0
        
        # Para cada intervalo nos dados experimentais
        for intervalo, valor_exp in dados_experimentais.items():
            # Criar acorde com duas notas separadas pelo intervalo
            if intervalo == 0:  # uníssono, tratado especialmente
                notas = ["C4", "C4"]
            else:
                notas = ["C4", f"{'CDEFGAB'[intervalo % 7]}{4 + (intervalo // 7)}"]
                
            # Calcular densidade para este intervalo usando o lambda atual
            densidade = calcular_densidade_intervalar(notas, lamb=lambda_val)
            
            # Normalizar densidade para o intervalo [-1, 1] para comparar com dados experimentais
            densidade_norm = 2 * (densidade / max(dados_experimentais.values())) - 1
            
            # Adicionar erro quadrático
            error_sum += (densidade_norm - valor_exp) ** 2
            
        logger.debug(f"Lambda: {lambda_val}, Erro: {error_sum}")
        return error_sum
    
    # Executar otimização
    result = minimize(
        objetivo, 
        [DEFAULT_LAMBDA],  # Valor inicial
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    # Extrair lambda otimizado
    lambda_otimizado = result.x[0]
    logger.info(f"Calibração concluída. Lambda otimizado: {lambda_otimizado}")
    
    # Salvar valor calibrado
    salvar_parametros_calibrados({'lambda': lambda_otimizado})
    
    return lambda_otimizado

# ------------------------------------------------------------------------------
# DEBUG: Função para imprimir informações detalhadas sobre cálculo de intervalo
# ------------------------------------------------------------------------------

def debug_intervalo(nota1, nota2, delta):
    """
    Função de debug para imprimir informações detalhadas sobre o cálculo
    do intervalo entre duas notas.
    """
    # [código original mantido]
    # Converter para MIDI para diagnóstico
    try:
        midi1 = note_to_midi(nota1)
        midi2 = note_to_midi(nota2)
        midi_delta = abs(midi1 - midi2)
        
        # Tentar converter para posições microtonais para diagnóstico
        try:
            # Converter símbolos Unicode para notação +/- antes de chamar nota_para_posicao
            nota1_processada = nota1
            nota2_processada = nota2
            
            if QUARTO_TOM_ACIMA in nota1:
                nota1_processada = nota1.replace(QUARTO_TOM_ACIMA, '+')
            elif QUARTO_TOM_ABAIXO in nota1:
                nota1_processada = nota1.replace(QUARTO_TOM_ABAIXO, '-')
                
            if QUARTO_TOM_ACIMA in nota2:
                nota2_processada = nota2.replace(QUARTO_TOM_ACIMA, '+')
            elif QUARTO_TOM_ABAIXO in nota2:
                nota2_processada = nota2.replace(QUARTO_TOM_ABAIXO, '-')
                
            pos1 = nota_para_posicao(nota1_processada)
            pos2 = nota_para_posicao(nota2_processada)
            pos_delta = abs(pos1 - pos2)
            
            logger.debug(f"DEBUG INTERVALO: {nota1} <-> {nota2}")
            logger.debug(f"  MIDI: {midi1:.2f} <-> {midi2:.2f} = {midi_delta:.2f}")
            logger.debug(f"  POS: {pos1:.2f} <-> {pos2:.2f} = {pos_delta:.2f}")
            logger.debug(f"  DELTA FINAL: {delta:.2f}")
            
        except ValueError as e:
            logger.error(f"Erro no debug de intervalo: {e}")
            
    except Exception as e:
        logger.error(f"Erro no debug de intervalo: {e}")

# ------------------------------------------------------------------------------
# 2) TRADUZIR INTERVALO (EM PASSOS) -> STRING
# ------------------------------------------------------------------------------
def traduzir_para_intervalo_tradicional(passos_microtonais):
    """
    Dado um número de passos microtonais (0..), retorna um nome de intervalo 
    (ex.: 'm3', 'P5', etc.), levando em conta oitavas acima de 24 microtons.
    """
    nomes_intervalos = {
        0: 'unisono', 1: 'unisono+', 2: 'm2', 3: 'm2+', 4: 'M2', 5: 'M2+',
        6: 'm3', 7: 'm3+', 8: 'M3', 9: 'M3+', 10: 'P4', 11: 'P4+', 12: 'aug4',
        13: 'aug4+', 14: 'P5', 15: 'P5+', 16: 'm6', 17: 'm6+', 18: 'M6',
        19: 'M6+', 20: 'm7', 21: 'm7+', 22: 'M7', 23: 'M7+', 24: 'oitava'
    }
    oitavas = passos_microtonais // 24
    resto = passos_microtonais % 24
    nome = nomes_intervalos.get(resto, f"?({resto})")
    if oitavas > 0:
        nome += f" + {oitavas} oitava(s)"
    return nome

# ------------------------------------------------------------------------------
# 3) FUNÇÃO MONOTÔNICA DECRESCENTE (EXPO) COM UNÍSSONO=0
# ------------------------------------------------------------------------------

def decaimento_exponencial_modificado(delta, lamb=None):
    """
    Função de decaimento exponencial modificada para densidade intervalar.
    
    CORREÇÃO: Uníssono (delta=0) agora retorna valor máximo (1.0) em vez de 0.
    Isso reflete a realidade acústica onde uníssonos criam máxima densidade.
    
    Args:
        delta (float): Distância intervalar em microtons
        lamb (float, optional): Parâmetro de decaimento. Se None, usa valor calibrado.
        
    Returns:
        float: Contribuição de densidade (1.0 para uníssono, decaindo com a distância)
    """
    if lamb is None:
        lamb = carregar_parametros_calibrados()
    
    # CORREÇÃO CRÍTICA: Uníssono tem densidade máxima, não mínima!
    if delta == 0:
        return 1.0  # Máxima contribuição para uníssono
    else:
        # Decaimento exponencial para intervalos maiores
        return math.exp(-lamb * delta)

# ------------------------------------------------------------------------------
# 4) CALCULAR DENSIDADE INTERVALAR (SOMA PAR-A-PAR)
# ------------------------------------------------------------------------------
def obter_intervalos(notas):
    """
    Gera lista de intervalos no formato:
       ["m2 (intervalo 2)", "M3 (intervalo 8)", ...]
    para cada par (i < j).
    """
    posicoes = [nota_para_posicao(n) for n in notas]
    intervalos_str = []
    for i in range(len(posicoes)):
        for j in range(i+1, len(posicoes)):
            delta = abs(posicoes[i] - posicoes[j])
            nome_trad = traduzir_para_intervalo_tradicional(delta)
            intervalos_str.append(f"{nome_trad} (intervalo {delta})")
    return intervalos_str


def intervalo_para_numero(intervalo_string):
    """
    Extrai o número (int) do texto "X (intervalo N)" -> N.
    Ex.: 'm3 (intervalo 6)' -> 6.
    """
    match = re.search(r'\(intervalo (\d+)\)', intervalo_string)
    return int(match.group(1)) if match else None


def calcular_densidade_intervalar(notas, lamb=None, usar_ponderacao_perceptual=False):
    """
    Faz a soma par-a-par da função decaimento_exponencial_modificado:
      densidade = S_{i<j} e^(-lamb*delta) 
      (onde delta=0 => 0).
    Versão atualizada que também suporta notação de cents e ponderação perceptual.

    - notas: lista de strings, ex.: ["C4","D4","E4"] ou ["C4+50c", "D4-25c"]
    - lamb: define quão rápido cai com a distância. Se None, usa valor calibrado.
    - usar_ponderacao_perceptual: Se True, aplica ponderação por registro
    
    Retorna float (densidade total).
    """
    # Se lamb não for fornecido, usar valor calibrado
    if lamb is None:
        lamb = carregar_parametros_calibrados()
        
    # Use MIDI values for more precision, especially with cents
    pitches = [note_to_midi(nota) for nota in notas if nota]
    
    # Logging para debug
    logger.debug(f"Notas: {notas}")
    logger.debug(f"Pitches MIDI: {pitches}")
    logger.debug(f"Usando lambda: {lamb}")
    logger.debug(f"Ponderação perceptual: {usar_ponderacao_perceptual}")
    
    densidade_total = 0.0
    n = len(pitches)
    
    for i in range(n):
        for j in range(i+1, n):
            delta_semitons = abs(pitches[i] - pitches[j])
            
            # Log cada intervalo para debug
            logger.debug(f"Intervalo entre {notas[i]} e {notas[j]}: {delta_semitons:.2f} semitons")
            
            # Se o intervalo for muito pequeno (menos de 0.01 semitom), 
            # pode ser erro de precisão numérica, então verificamos explicitamente
            if delta_semitons < 0.01:
                # Verificar se as notas são realmente as mesmas ou se têm diferenças microtonais
                if notas[i] != notas[j]:
                    # Forçar um valor mínimo para garantir que o intervalo seja contabilizado
                    logger.debug(f"Intervalo muito pequeno entre notas diferentes: {notas[i]} e {notas[j]}")
                    delta_semitons = max(delta_semitons, 0.25)  # Forçar pelo menos um quarto de tom
            
            # Converter para a escala microtonal
            delta = delta_semitons * 2  # Fator 2 para manter a proporção com a escala original
            
            # Calcular a densidade deste intervalo
            densidade_base = decaimento_exponencial_modificado(delta, lamb)
            
            # APLICAR PONDERAÇÃO PERCEPTUAL SE SOLICITADA
            if usar_ponderacao_perceptual:
                # Calcular peso perceptual para este par de notas
                peso_perceptual = calcular_peso_perceptual_microtonal(pitches[i], pitches[j], delta_semitons)
                densidade_intervalo = densidade_base * peso_perceptual
                logger.debug(f"  Peso perceptual aplicado: {peso_perceptual:.3f}")
            else:
                densidade_intervalo = densidade_base
            
            densidade_total += densidade_intervalo
            
            # Log para debug
            logger.debug(f"  delta = {delta:.2f}, densidade = {densidade_intervalo:.6f}")
            
            # Debug detalhado
            debug_intervalo(notas[i], notas[j], delta)
    
    logger.debug(f"Densidade total: {densidade_total:.6f}")
    return densidade_total


def calcular_densidade_intervalar_psicoaustica(
        notas,
        use_psychoacoustic: bool = True,
        use_perceptual_weighting: bool = False,
        lamb: Optional[float] = None
    ):
    """
    Retorna a densidade intervalar (escala 0-1) e,
    opcionalmente, com correcções psicoacústicas.

    Se USE_LOG_COMPRESSION=True (config.py) devolve o valor
    já comprimido em log10(1+x) – adequado a clusters extremos.
    """

    # --------------------------------------------------
    # 1) densidade intervalar “física” ou perceptual
    #    devolvida como vector de pesos
    # --------------------------------------------------
    pesos = calcular_densidade_intervalar(
        notas,
        lamb=lamb,
        usar_ponderacao_perceptual=use_perceptual_weighting
    )
    n = len(notas)

    if n > 1:
        dens_scalar = float( 2.0 * np.sum(pesos) / (n * (n - 1)) )  # MÉDIA
    else:
        dens_scalar = 0.0

    # compressão log opcional
    if USE_LOG_COMPRESSION:
        dens_scalar = np.log10(1.0 + dens_scalar)

    if not use_psychoacoustic:
        return dens_scalar

    # --------------------------------------------------
    # 2) correcções psicoacústicas
    # --------------------------------------------------
    from microtonal import note_to_midi
    from psychoacoustic_corrections import (
        critical_band_masking,
        calculate_roughness,
        apply_loudness_correction
    )

    pitches = [note_to_midi(n) for n in notas]
    amps    = np.ones(len(pitches))

    # 2.1 Mascaramento de banda crítica
    amps_masked  = critical_band_masking(pitches, amps)
    masking_gain = float(np.mean(amps_masked))          # 0-1

    # 2.2 Roughness
    rough_vec    = calculate_roughness(pitches, amps_masked)
    rough_gain   = 1.0 + float(np.sum(rough_vec)) / np.sqrt(n)  # divide √n

    # 2.3 Loudness
    loud_gain    = float(np.mean(apply_loudness_correction(pitches,
                                                           amps_masked)))

    dens_final = dens_scalar * masking_gain * rough_gain * loud_gain

    # compressão log final (mantém-se monotónico)
    if USE_LOG_COMPRESSION:
        dens_final = np.log10(1.0 + dens_final)

    return dens_final



# ADD this helper function to densidade_intervalar.py:
def calcular_peso_perceptual_microtonal(midi1, midi2, delta_semitons):
    """
    Calcula peso perceptual para o sistema microtonal.
    
    Args:
        midi1, midi2: Valores MIDI das duas notas
        delta_semitons: Diferença em semitons
        
    Returns:
        float: Fator de peso perceptual
    """
    peso = 1.0
    
    # Peso baseado no registro médio
    registro_medio = (midi1 + midi2) / 2
    
    if registro_medio > 84:  # High register (C6+)
        peso *= 1.3
    elif registro_medio > 72:  # Medium-high register (C5-C6)
        peso *= 1.1
    elif registro_medio < 48:  # Low register (below C3)
        peso *= 0.8
    
    # Peso baseado no tamanho do intervalo
    if delta_semitons <= 1:  # Semitom ou menor
        peso *= 1.5
    elif delta_semitons <= 2:  # Tom
        peso *= 1.3
    elif delta_semitons <= 4:  # Até terça maior
        peso *= 1.1
    elif delta_semitons >= 12:  # Oitava ou maior
        peso *= 0.9
    
    return peso

# ------------------------------------------------------------------------------
# Funções adicionais de análise e visualização
# ------------------------------------------------------------------------------

def analisar_consonancia_vs_lambda(intervalos_teste=None, range_lambda=(0.01, 1.0, 0.05)):
    """
    Analisa como diferentes valores de lambda afetam a consonância calculada.
    
    intervalos_teste: lista de tuplas (nome, [notas]) para testar
    range_lambda: tupla (min, max, step) para valores de lambda a testar
    
    Retorna um DataFrame com os resultados e plota um gráfico.
    """
    if intervalos_teste is None:
        # Intervalos padrão para teste
        intervalos_teste = [
            ("Unísono", ["C4", "C4"]),
            ("Segunda menor", ["C4", "Db4"]),
            ("Segunda maior", ["C4", "D4"]),
            ("Terça menor", ["C4", "Eb4"]),
            ("Terça maior", ["C4", "E4"]),
            ("Quarta justa", ["C4", "F4"]),
            ("Trítono", ["C4", "F#4"]),
            ("Quinta justa", ["C4", "G4"]),
            ("Sexta menor", ["C4", "Ab4"]),
            ("Sexta maior", ["C4", "A4"]),
            ("Sétima menor", ["C4", "Bb4"]),
            ("Sétima maior", ["C4", "B4"]),
            ("Oitava", ["C4", "C5"]),
        ]
    
    # Criar lista de valores lambda para testar
    lambda_min, lambda_max, lambda_step = range_lambda
    lambdas = np.arange(lambda_min, lambda_max + lambda_step/2, lambda_step)
    
    # Armazenar resultados
    resultados = []
    
    # Calcular densidade para cada intervalo e cada lambda
    for nome, notas in intervalos_teste:
        for lamb in lambdas:
            densidade = calcular_densidade_intervalar(notas, lamb=lamb)
            resultados.append({
                "Intervalo": nome,
                "Lambda": lamb,
                "Densidade": densidade
            })
    
    # Converter para DataFrame
    import pandas as pd
    df = pd.DataFrame(resultados)
    
    # Plotar resultados
    plt.figure(figsize=(12, 8))
    for nome in set(df["Intervalo"]):
        subset = df[df["Intervalo"] == nome]
        plt.plot(subset["Lambda"], subset["Densidade"], label=nome)
    
    plt.title("Densidade Intervalar vs. Lambda para Diferentes Intervalos")
    plt.xlabel("Lambda")
    plt.ylabel("Densidade Intervalar")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return df

def testar_modelo_calibrado():
    """
    Testa o modelo com o lambda calibrado em diferentes intervalos e compara
    com as avaliações experimentais de consonância.
    """
    lambda_calibrado = carregar_parametros_calibrados()
    
    # Criar um DataFrame com intervalos e seus valores de densidade calculados
    resultados = []
    
    # Testar nos intervalos de referência
    for intervalo, valor_exp in CONSONANCE_RATINGS.items():
        # Mapear intervalo para notas
        if intervalo == 0:  # uníssono
            notas = ["C4", "C4"]
        else:
            # Aproximação simples para demonstração
            notas = ["C4", f"{'CDEFGAB'[intervalo % 7]}{4 + (intervalo // 7)}"]
        
        # Calcular densidade com lambda calibrado
        densidade = calcular_densidade_intervalar(notas, lamb=lambda_calibrado)
        
        # Normalizar para comparação com valores experimentais
        max_valor = max(CONSONANCE_RATINGS.values())
        densidade_norm = 2 * (densidade / max_valor) - 1
        
        resultados.append({
            "Intervalo": traduzir_para_intervalo_tradicional(intervalo * 2),  # * 2 para microtons
            "Valor Experimental": valor_exp,
            "Densidade Calculada": densidade,
            "Densidade Normalizada": densidade_norm,
            "Erro": abs(densidade_norm - valor_exp)
        })
    
    # Converter para DataFrame
    import pandas as pd
    df = pd.DataFrame(resultados)
    
    # Plotar comparação
    plt.figure(figsize=(12, 6))
    
    intervalos = df["Intervalo"].tolist()
    
    plt.subplot(1, 2, 1)
    plt.bar(intervalos, df["Valor Experimental"], alpha=0.7, label="Experimental")
    plt.bar(intervalos, df["Densidade Normalizada"], alpha=0.7, label="Modelo")
    plt.title(f"Comparação: Lambda={lambda_calibrado:.4f}")
    plt.xlabel("Intervalo")
    plt.ylabel("Consonância Normalizada")
    plt.xticks(rotation=45)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.bar(intervalos, df["Erro"])
    plt.title("Erro Absoluto")
    plt.xlabel("Intervalo")
    plt.ylabel("Erro")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return df

# ------------------------------------------------------------------------------
# Funções para validação experimental
# ------------------------------------------------------------------------------

def coletar_dados_experimentais():
    """
    Interface para coleta de dados experimentais de avaliação de consonância.
    Retorna um dicionário na forma {intervalo: valor_consonancia}.
    """
    dados = {}
    
    intervalos = [
        (0, "Unísono (C4-C4)"),
        (2, "Segunda menor (C4-Db4)"),
        (4, "Segunda maior (C4-D4)"), 
        (6, "Terça menor (C4-Eb4)"),
        (8, "Terça maior (C4-E4)"),
        (10, "Quarta justa (C4-F4)"),
        (12, "Trítono (C4-Gb4)"),
        (14, "Quinta justa (C4-G4)")
    ]
    
    print("Avaliação experimental de consonância de intervalos")
    print("Para cada intervalo, avalie a consonância em uma escala de -1 a 1:")
    print("  -1 = muito dissonante")
    print("   0 = neutro")
    print("   1 = muito consonante")
    
    for intervalo, descricao in intervalos:
        while True:
            try:
                valor = float(input(f"{descricao}: "))
                if -1 <= valor <= 1:
                    dados[intervalo] = valor
                    break
                else:
                    print("Valor deve estar entre -1 e 1.")
            except ValueError:
                print("Digite um número válido.")
    
    return dados

# ------------------------------------------------------------------------------
# Funções de demonstração
# ------------------------------------------------------------------------------

def demonstrar_calibracao():
    """
    Demonstra o processo de calibração com coleta de dados e visualização.
    """
    # Opção 1: Usar dados de referência da literatura
    print("Calibrando com dados de referência da literatura...")
    lambda_otimizado = calibrar_lambda()
    
    # Visualizar resultados
    testar_modelo_calibrado()
    
    # Opção 2: Coletar dados experimentais do usuário
    print("\nDeseja realizar calibração personalizada com seus próprios dados? (s/n)")
    opcao = input()
    if opcao.lower() in ('s', 'sim', 'y', 'yes'):
        dados = coletar_dados_experimentais()
        lambda_otimizado = calibrar_lambda(dados)
        
        # Visualizar resultados
        testar_modelo_calibrado()
    
    # Demonstrar o efeito de diferentes valores de lambda
    analisar_consonancia_vs_lambda()


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
__all__ = [
    "calcular_densidade_intervalar",
    "calcular_densidade_intervalar_psicoacustica",
    "calibrar_lambda",
    "debug_intervalo",
    # …adicione aqui outras funções que queira expor…
]


# Funções mantidas do código original

def visualizar_funcao_exponencial(lamb=None, max_delta=48):
    """
    Plota e^(-lamb*delta) mas com delta=0 => 0 (uníssono=0).
    Observaremos de 0..48 microtons (~ 2 oitavas).
    
    Se lamb for None, usa o valor calibrado.
    """
    if lamb is None:
        lamb = carregar_parametros_calibrados()
        
    steps = np.linspace(0, max_delta, 100)
    valores = [decaimento_exponencial_modificado(s, lamb) for s in steps]
    
    plt.figure(figsize=(8,5))
    plt.plot(steps, valores, label=f"uníssono=0, expo = e^(-{lamb}*delta)")
    
    plt.title(f"Decaimento Exponencial (uníssono=0) - lamb={lamb}")
    plt.xlabel("Distância (microtons)")
    plt.ylabel("Peso")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

