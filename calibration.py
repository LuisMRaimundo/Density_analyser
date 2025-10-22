# calibration.py
"""
Módulo para calibração do parâmetro lambda na análise de densidade intervalar.
Este módulo centraliza funções de calibração, visualização e acesso aos 
parâmetros calibrados.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import logging

# Configurar logging
logger = logging.getLogger('calibration')

# Caminho para o arquivo de configuração
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'density_params.json')
DEFAULT_LAMBDA = 0.05  # Valor padrão para lambda

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

def carregar_parametros_calibrados(config_path=None):
    """
    Carrega parâmetros calibrados de um arquivo JSON.
    Retorna o valor de lambda baseado em dados experimentais.
    
    Args:
        config_path (str, optional): Caminho para o arquivo de configuração
        
    Returns:
        float: Valor calibrado de lambda ou valor padrão caso não exista
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
        
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                params = json.load(f)
                logger.info(f"Parâmetros carregados: {params}")
                return params.get('lambda', DEFAULT_LAMBDA)
        else:
            logger.warning(f"Arquivo de configuração não encontrado: {config_path}")
            return DEFAULT_LAMBDA
    except Exception as e:
        logger.error(f"Erro ao carregar parâmetros: {e}")
        return DEFAULT_LAMBDA

def salvar_parametros_calibrados(params, config_path=None):
    """
    Salva parâmetros calibrados em um arquivo JSON.
    
    Args:
        params (dict): Dicionário com parâmetros
        config_path (str, optional): Caminho para o arquivo de configuração
        
    Returns:
        bool: True se salvo com sucesso, False caso contrário
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
        
    try:
        # Garantir que o diretório existe
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(params, f, indent=4)
            logger.info(f"Parâmetros salvos: {params}")
        return True
    except Exception as e:
        logger.error(f"Erro ao salvar parâmetros: {e}")
        return False

def decaimento_exponencial_modificado(delta, lamb=None):
    """
    Se delta=0 (uníssono), retorna 0.
    Caso contrário, e^(-lamb*delta).

    É uma função monotonicamente decrescente no intervalo (0..infinito).
    Parâmetro lamb controla quão rápido decai e pode ser ajustado com base em validação perceptual.
    
    Args:
        delta (float): Valor do intervalo
        lamb (float, optional): Parâmetro lambda. Se None, usa o valor calibrado
        
    Returns:
        float: Valor de decaimento calculado
    """
    if lamb is None:
        lamb = carregar_parametros_calibrados()
        
    if delta == 0:
        return 0.0
    else:
        return np.exp(-lamb * delta)

def calcular_densidade_intervalar(notas, lamb=None):
    """
    Calcula a densidade intervalar para um conjunto de notas.
    
    Args:
        notas (list): Lista de strings representando notas musicais
        lamb (float, optional): Parâmetro lambda. Se None, usa o valor calibrado
        
    Returns:
        float: Densidade intervalar total
    """
    # Implementação simplificada
    #from microtonal import note_to_midi
    from utils.notes import note_to_midi, normalize_note_string

    if lamb is None:
        lamb = carregar_parametros_calibrados()
    
    # Verificar entrada vazia
    if len(notas) < 2:
        return 0.0
    
    # Converter notas para valores MIDI
    pitches = [note_to_midi(nota) for nota in notas if nota]
    
    # Calcular densidade par a par
    densidade_total = 0.0
    for i in range(len(pitches)):
        for j in range(i+1, len(pitches)):
            # Calcular delta em semitons
            delta_semitons = abs(pitches[i] - pitches[j])
            
            # Converter para a escala microtonal (2 passos por semitom)
            delta = delta_semitons * 2
            
            # Calcular densidade parcial
            densidade_parcial = decaimento_exponencial_modificado(delta, lamb)
            densidade_total += densidade_parcial
    
    return densidade_total

def calibrar_lambda(dados_experimentais=None):
    """
    Calibra o valor de lambda usando dados experimentais.
    Se não forem fornecidos dados, usa valores de referência da literatura.
    
    Args:
        dados_experimentais (dict, optional): Dicionário na forma {intervalo: valor_consonancia}
        
    Returns:
        float: Valor de lambda otimizado
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

def visualizar_funcao_exponencial(lamb=None, max_delta=48):
    """
    Plota e^(-lamb*delta) mas com delta=0 => 0 (uníssono=0).
    Observaremos de 0..48 microtons (~ 2 oitavas).
    
    Args:
        lamb (float, optional): Parâmetro lambda. Se None, usa o valor calibrado
        max_delta (int): Valor máximo de delta para visualização
    """
    if lamb is None:
        lamb = carregar_parametros_calibrados()
        
    steps = np.linspace(0, max_delta, 100)
    valores = [decaimento_exponencial_modificado(s, lamb) for s in steps]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, valores, label=f"uníssono=0, expo = e^(-{lamb:.4f}*delta)")
    
    plt.title(f"Decaimento Exponencial (uníssono=0) - lambda={lamb:.4f}")
    plt.xlabel("Distância (microtons)")
    plt.ylabel("Peso")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def testar_modelo_calibrado():
    """
    Testa o modelo com o lambda calibrado em diferentes intervalos e compara
    com as avaliações experimentais de consonância.
    
    Returns:
        plt.Figure: Figura matplotlib com a comparação
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
            "Intervalo": f"{intervalo}",
            "Valor Experimental": valor_exp,
            "Densidade Calculada": densidade,
            "Densidade Normalizada": densidade_norm,
            "Erro": abs(densidade_norm - valor_exp)
        })
    
    # Extrair dados para visualização
    intervalos = [r["Intervalo"] for r in resultados]
    valores_exp = [r["Valor Experimental"] for r in resultados]
    valores_modelo = [r["Densidade Normalizada"] for r in resultados]
    erros = [r["Erro"] for r in resultados]
    
    # Criar figura para visualização
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico de comparação
    x = np.arange(len(intervalos))
    width = 0.35
    
    ax1.bar(x - width/2, valores_exp, width, label='Experimental', alpha=0.7)
    ax1.bar(x + width/2, valores_modelo, width, label='Modelo', alpha=0.7)
    
    ax1.set_title(f"Comparação: Lambda={lambda_calibrado:.4f}")
    ax1.set_xlabel("Intervalo (semitons)")
    ax1.set_ylabel("Consonância Normalizada")
    ax1.set_xticks(x)
    ax1.set_xticklabels(intervalos)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de erro
    ax2.bar(x, erros, alpha=0.7)
    ax2.set_title("Erro Absoluto")
    ax2.set_xlabel("Intervalo (semitons)")
    ax2.set_ylabel("Erro")
    ax2.set_xticks(x)
    ax2.set_xticklabels(intervalos)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def analisar_consonancia_vs_lambda(intervalos_teste=None, range_lambda=(0.01, 1.0, 0.05)):
    """
    Analisa como diferentes valores de lambda afetam a consonância calculada.
    
    Args:
        intervalos_teste (list, optional): Lista de tuplas (nome, [notas]) para testar
        range_lambda (tuple): (min, max, step) para valores de lambda a testar
        
    Returns:
        plt.Figure: Figura matplotlib com os resultados
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
    
    # Criar figura para visualização
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plotar resultados agrupados por intervalo
    intervalos_nomes = [nome for nome, _ in intervalos_teste]
    for nome in intervalos_nomes:
        # Filtrar dados para este intervalo
        dados_intervalo = [(r["Lambda"], r["Densidade"]) for r in resultados if r["Intervalo"] == nome]
        lambdas_intervalo, densidades_intervalo = zip(*dados_intervalo)
        
        # Plotar linha para este intervalo
        ax.plot(lambdas_intervalo, densidades_intervalo, label=nome)
    
    ax.set_title("Densidade Intervalar vs. Lambda para Diferentes Intervalos")
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Densidade Intervalar")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lambda_min, lambda_max)
    plt.tight_layout()
    plt.show()
    
    return fig

def obter_lambda_atual():
    """
    Retorna o valor atual calibrado de lambda.
    
    Returns:
        float: Valor atual de lambda
    """
    return carregar_parametros_calibrados()

def realizar_calibracao(dados_experimentais=None):
    """
    Realiza a calibração do parâmetro lambda com base em dados experimentais.
    Exibe visualizações relevantes e retorna o lambda otimizado.
    
    Args:
        dados_experimentais (dict, optional): Dados para calibração
        
    Returns:
        float: Lambda otimizado
    """
    # Mostrar o valor atual de lambda
    lambda_atual = carregar_parametros_calibrados()
    print(f"Valor atual de lambda: {lambda_atual}")
    
    # Calibrar com os dados fornecidos ou padrões
    lambda_otimizado = calibrar_lambda(dados_experimentais)
    print(f"Lambda otimizado: {lambda_otimizado}")
    
    # Visualizar resultados
    testar_modelo_calibrado()
    
    # Mostrar função exponencial com o novo lambda
    visualizar_funcao_exponencial(lambda_otimizado)
    
    return lambda_otimizado

# Para uso e teste direto do módulo
if __name__ == "__main__":
    realizar_calibracao()
