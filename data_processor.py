# data_processor.py
# Contém funções para processamento de dados e cálculo de métricas

import numpy as np
import pandas as pd
import json
import importlib
import logging
from tkinter import messagebox, filedialog

# handlers e utilitários
from error_handler import handle_exceptions, InputError, CalculationError
from instrumentos import get_instrument_module
from microtonal import (
    note_to_midi, midi_to_hz, is_valid_note,
    converter_para_sustenido, extract_cents
)
from config import MAX_DENS_GLOBAL, USE_LOG_COMPRESSION
from utils.serialize_utils import serialize_for_json

from spectral_analysis import calculate_extended_spectral_moments as calculate_spectral_complexity
from psychoacoustic_corrections import (
    critical_band_masking,
    calculate_roughness,
    apply_loudness_correction
)


# Configurar logging
logger = logging.getLogger('data_processor')

# No arquivo ata_processor.py, adicione esta função após os imports

# Funções de conversão e utilidade
def load_instrument_module(instrument_name):
    """
    Carrega o módulo do instrumento especificado.
    
    Args:
        instrument_name (str): Nome do módulo de instrumento
        
    Returns:
        module: Módulo carregado
        
    Raises:
        ImportError: Se o módulo não for encontrado
    """
    try:
        # Usando a nova função do pacote instrumentos
        return get_instrument_module(instrument_name)
    except ImportError as e:
        logger.error(f"Módulo para {instrument_name} não encontrado: {e}")
        raise ImportError(f"Module for {instrument_name} not found: {str(e)}")

def format_duration_display(duration_value):
    """
    Formata um valor de duração para exibição no relatório.
    
    Args:
        duration_value (float): Valor numérico da duração
    
    Returns:
        str: String formatada "Nome (Símbolo)"
    """
    return DuracaoMusical.format_duration_display(duration_value)

def salvar_resultados(resultados, nome_arquivo=None):
    """
    Salva os resultados da análise em um arquivo JSON.
    
    Args:
        resultados (dict): Dicionário com os resultados da análise.
        nome_arquivo (str): Nome do arquivo para salvar os resultados.
        
    Returns:
        str: Caminho do arquivo salvo ou None se cancelado
    """
    if nome_arquivo is None:
        nome_arquivo = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Salvar resultados da análise"
        )
        if not nome_arquivo:  # Usuário cancelou
            return None
    
    try:
        # Converter valores numpy para Python nativos usando a função centralizada
        resultados_convertidos = serialize_for_json(resultados)
        
        # Salvar em arquivo
        with open(nome_arquivo, 'w', encoding='utf-8') as f:
            json.dump(resultados_convertidos, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Resultados salvos em: {nome_arquivo}")
        return nome_arquivo
    except Exception as e:
        logger.error(f"Erro ao salvar resultados: {e}")
        messagebox.showerror("Erro", f"Erro ao salvar resultados: {e}")
        return None

def calcular_densidade_ponderada_normalizada(DI, DV, metodo="min-max", w=0.5, 
                                      DI_max=100, DV_max=10, 
                                      alpha=0.7, beta=0.4, use_stevens=False):
    """
    Calcula a densidade ponderada normalizada com opção de normalização Min-Max ou Z-score.
    Agora com suporte para a Lei de Stevens com expoentes configuráveis.

    Args:
        DI (float): Densidade do instrumento
        DV (float): Densidade intervalar
        metodo (str): "min-max" (default) ou "z-score"
        w (float): Peso para balancear DI e DV (default 0.5)
        DI_max (float): Densidade máxima teórica do instrumento (default 100)
        DV_max (float): Densidade máxima teórica intervalar (default 10)
        alpha (float): Expoente de Stevens para a densidade do instrumento (default 0.7)
        beta (float): Expoente de Stevens para a densidade intervalar (default 0.4)
        use_stevens (bool): Se True, aplica a Lei de Stevens; se False, usa método linear original

    Returns:
        float: Densidade ponderada normalizada
    """
    try:
        if metodo == "min-max":
            # Normalização Min-Max baseada em limites teóricos
            DI_norm = DI / DI_max
            DV_norm = DV / DV_max
        elif metodo == "z-score":
            # Normalização Z-score com média e desvio padrão fixos
            DI_mean, DI_std = 50, 25  # Valores exemplo; podem ser dinâmicos
            DV_mean, DV_std = 5, 2.5  # Valores exemplo; podem ser dinâmicos
            DI_norm = (DI - DI_mean) / DI_std if DI_std > 0 else 0
            DV_norm = (DV - DV_mean) / DV_std if DV_std > 0 else 0
        else:
            raise ValueError(f"Método inválido: '{metodo}'. Escolha 'min-max' ou 'z-score'.")
        
        # Aplicação da Lei de Stevens se solicitado
        if use_stevens:
            # Aplicar expoentes de Stevens aos valores normalizados
            if DI_norm > 0:
                DI_norm = DI_norm ** alpha
            if DV_norm > 0:
                DV_norm = DV_norm ** beta
        
        # Aplicação da ponderação e escala final
        densidade_ponderada = 10 * (w * DI_norm + (1 - w) * DV_norm)
        return densidade_ponderada

    except Exception as e:
        import logging
        logging.error(f"Erro ao calcular densidade ponderada: {e}")
        return None  # Retorna None para indicar erro sem quebrar o código

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
            # Usar a função atualizada note_to_midi do utils
            midi_value = note_to_midi(nota)
            
            # Garantir que temos um valor válido
            if midi_value is not None and midi_value != 60.0:  # Se não é o valor fallback padrão
                valid_pitches.append(midi_value)
                valid_notas.append(nota)
            else:
                # Se midi_value é o fallback 60.0, vamos verificar se a nota é realmente C4
                if nota.upper().startswith('C4'):
                    valid_pitches.append(midi_value)
                    valid_notas.append(nota)
                else:
                    logger.warning(f"Nota ignorada (convertida para fallback): {nota}")
        except Exception as e:
            logger.error(f"Erro ao converter nota para MIDI: {e}")
    
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


def calcular_densidade_intervalar_psicoaustica(notas, lamb=0.05, use_psychoacoustic=True, use_perceptual_weighting=False):
    """
    Versão melhorada da função de densidade intervalar com correções psicoacústicas.
    
    Args:
        notas (list): Lista de strings de notas
        lamb (float): Parâmetro lambda para o decaimento exponencial
        use_psychoacoustic (bool): Se True, aplica correções psicoacústicas
        use_perceptual_weighting (bool): Se True, aplica ponderação perceptual adicional
        
    Returns:
        float: Densidade total calculada com correções psicoacústicas
    """
    from densidade_intervalar import decaimento_exponencial_modificado
    
    # Validação da entrada
    if not notas or len(notas) < 2:
        logger.info("Menos de duas notas para calcular densidade intervalar")
        return 0.0
    
    # Converter notas para valores MIDI
    valid_pitches = []
    valid_notas = []
    
    for nota in notas:
        if not nota:
            continue
        
        try:
            midi_value = note_to_midi(nota)
            if midi_value is not None:
                valid_pitches.append(midi_value)
                valid_notas.append(nota)
        except Exception as e:
            logger.error(f"Erro ao converter nota para MIDI: {e}")
    
    if len(valid_pitches) < 2:
        logger.warning(f"Menos de duas notas válidas para densidade intervalar")
        return 0.0
    
    # Criar amplitudes iniciais (todas iguais por enquanto)
    base_amplitudes = np.ones(len(valid_pitches))
    
    # Inicializar variáveis para evitar erro de escopo
    corrected_amplitudes = base_amplitudes
    roughness = 0.0
    
    # Aplicar correções psicoacústicas se solicitado
    if use_psychoacoustic:
        # 1. Mascaramento de banda crítica
        masked_amplitudes = critical_band_masking(valid_pitches, base_amplitudes)
        
        # 2. Correção de loudness
        corrected_amplitudes = apply_loudness_correction(valid_pitches, masked_amplitudes)
        
        # 3. Calcular roughness
        roughness = calculate_roughness(valid_pitches, corrected_amplitudes)
    
    # Calcular densidade intervalar básica
    densidade_total = 0.0
    n = len(valid_pitches)
    
    for i in range(n):
        for j in range(i+1, n):
            # Calcular a diferença em semitons
            delta_semitons = abs(valid_pitches[i] - valid_pitches[j])
            
            # Transformar para escala microtonal
            delta = delta_semitons * 2
            
            # Calcular contribuição de densidade
            densidade_parcial = decaimento_exponencial_modificado(delta, lamb)
            
            # Apply perceptual weighting if requested
            if use_perceptual_weighting:
                from densidade_intervalar import calcular_peso_perceptual_microtonal
                peso_perceptual = calcular_peso_perceptual_microtonal(
                    valid_pitches[i], valid_pitches[j], delta_semitons
                )
                densidade_parcial *= peso_perceptual
            
            # Ponderar pela média das amplitudes corrigidas (if psychoacoustic)
            if use_psychoacoustic:
                weight = (corrected_amplitudes[i] + corrected_amplitudes[j]) / 2
                densidade_parcial *= weight
            
            densidade_total += densidade_parcial
    
    # Adicionar contribuição de roughness (se ativado)
    if use_psychoacoustic and roughness > 0:
        # Escalar roughness apropriadamente
        roughness_contribution = roughness * 0.8  # Increased from 0.3
        densidade_total += roughness_contribution
        
        logger.debug(f"Densidade base: {densidade_total - roughness_contribution:.4f}, "
                    f"Roughness: {roughness_contribution:.4f}, "
                    f"Total: {densidade_total:.4f}")
    
    return densidade_total


def calcular_massa_sonora(notas, dinamicas, numeros_instrumentos, densidades_instrumento, duracoes=None):
    """
    Calcula a massa sonora total - uma medida da quantidade absoluta de material sonoro.
    Versão modificada para ignorar durações.
    
    Args:
        notas (list): Lista de notas
        dinamicas (list): Lista de dinâmicas
        numeros_instrumentos (list): Quantidade de cada instrumento
        densidades_instrumento (list): Densidades por nota
        duracoes (list, optional): Mantido por compatibilidade, mas não utilizado
        
    Returns:
        float: Valor da massa sonora total
    """
    # Fator de escala para diferentes dinâmicas
    fatores_dinamica = {
        'pppp': 0.2, 'ppp': 0.3, 'pp': 0.4, 'p': 0.6,
        'mf': 1.0, 
        'f': 1.5, 'ff': 2.0, 'fff': 2.5, 'ffff': 3.0
    }
    
    massa_total = 0.0
    
    for i in range(len(notas)):
        # Obter fator de dinâmica (default 1.0 se não encontrado)
        fator = fatores_dinamica.get(dinamicas[i], 1.0)
        
        # Calcular componente de massa para esta nota (sem multiplicar pela duração)
        massa_nota = densidades_instrumento[i] * fator * numeros_instrumentos[i]
            
        massa_total += massa_nota
        
    return massa_total

def calcular_densidade_fundida(DI, DV, alpha=0.5, DI_max=100, DV_max=10, DI_mean=50, DI_std=25, DV_mean=5, DV_std=2.5):
    """
    Calcula a densidade ponderada usando uma fusão de Min-Max e Z-Score.

    Args:
        DI (float): Densidade do instrumento
        DV (float): Densidade intervalar
        alpha (float): Peso para balancear Min-Max e Z-Score (0.5 = igual peso)
        DI_max (float): Densidade máxima teórica do instrumento
        DV_max (float): Densidade máxima teórica intervalar
        DI_mean (float): Média esperada da densidade do instrumento
        DI_std (float): Desvio padrão da densidade do instrumento
        DV_mean (float): Média esperada da densidade intervalar
        DV_std (float): Desvio padrão da densidade intervalar

    Returns:
        float: Densidade final combinada.
    """
    # Min-Max Normalization
    DI_minmax = DI / DI_max
    DV_minmax = DV / DV_max

    # Z-Score Normalization
    DI_zscore = (DI - DI_mean) / DI_std if DI_std > 0 else 0
    DV_zscore = (DV - DV_mean) / DV_std if DV_std > 0 else 0

    # Combinação dos dois métodos
    DI_fundida = alpha * DI_minmax + (1 - alpha) * DI_zscore
    DV_fundida = alpha * DV_minmax + (1 - alpha) * DV_zscore

    # Aplicação da ponderação
    densidade_fundida = 10 * (DI_fundida + DV_fundida) / 2  # Normaliza o resultado

    return densidade_fundida



@handle_exceptions(show_dialog=False, rethrow=True)
def calcular_metricas(input_data):
    """
    Calcula todas as métricas (versão sem durações).

    Params
    ------
    input_data : dict
        notes, dynamics, instruments, num_instruments, weight_factor, etc.

    Returns
    -------
    tuple
        (resultados_completos, densidades_instrumento, pitches)
    """
    try:
        # ------------------------------------------------------------
        # 1. Extracção e validação básica
        # ------------------------------------------------------------
        notas               = input_data.get('notes', [])
        dinamicas           = input_data.get('dynamics', [])
        instrumentos        = input_data.get('instruments', [])
        numeros_instr       = input_data.get('num_instruments', [])
        weight_factor       = input_data.get('weight_factor', 0.5)

        if not (notas and dinamicas and instrumentos and numeros_instr):
            raise InputError("Notas, dinâmicas, instrumentos e quantidades são obrigatórios.")
        if not (len(notas) == len(dinamicas) == len(instrumentos) == len(numeros_instr)):
            raise InputError("Listas de entrada devem ter o mesmo comprimento.")

        # ------------------------------------------------------------
        # 2. Converter notas para formato sustenido mantendo cents
        # ------------------------------------------------------------
        notas = [
            f"{converter_para_sustenido(base)}{('+' if cents > 0 else '')}{cents}c" if cents else
            converter_para_sustenido(base)
            for base, cents in (extract_cents(n) for n in notas)
        ]

        # ------------------------------------------------------------
        # 3. Densidade intervalar (três modos possíveis)
        # ------------------------------------------------------------
        from densidade_intervalar import calcular_densidade_intervalar
        from densidade_intervalar import calcular_densidade_intervalar_psicoaustica

        densidade_intervalar_val = calcular_densidade_intervalar_psicoaustica(
            notas,
            use_psychoacoustic       = input_data.get('use_psychoacoustic', True),
            use_perceptual_weighting = input_data.get('use_perceptual_weighting', False)
        )

        # ------------------------------------------------------------
        # 4. Densidade de instrumento
        # ------------------------------------------------------------
        instrument_module = load_instrument_module(instrumentos[0])
        densidades_instr  = []

        for n, dyn, num in zip(notas, dinamicas, numeros_instr):
            if dyn in ('pp', 'mf', 'ff'):
                d = instrument_module.calcular_densidade(n, dyn)
            else:
                pp = instrument_module.calcular_densidade(n, 'pp')
                mf = instrument_module.calcular_densidade(n, 'mf')
                ff = instrument_module.calcular_densidade(n, 'ff')
                d  = instrument_module.predict_intermediate_dynamics([n], [pp], [mf], [ff])[dyn][0]
            densidades_instr.append(d * np.sqrt(num))

        densidade_instrumento_val = sum(densidades_instr)

        # ------------------------------------------------------------
        # 5. Densidade ponderada (Lei de Stevens + min-max)
        # ------------------------------------------------------------
        densidade_ponderada_val = calcular_densidade_ponderada_normalizada(
            densidade_instrumento_val,
            densidade_intervalar_val,
            metodo       = "min-max",
            w            = weight_factor,
            alpha        = input_data.get('alpha', 0.7),
            beta         = input_data.get('beta', 0.4),
            use_stevens  = input_data.get('use_stevens', True)
        )

        # ------------------------------------------------------------
        # 6. Conversão para MIDI (com cents) e refinamento
        # ------------------------------------------------------------
        pitches = [note_to_midi(n) for n in notas]
        amplitude_st = max(pitches) - min(pitches) if len(pitches) > 1 else 0
        spectral_spread_st = amplitude_st           # <-- ESTA LINHA
        densidade_refinada_val = densidade_ponderada_val / amplitude_st if amplitude_st else densidade_ponderada_val
 


        # ------------------------------------------------------------
        # 7. Métricas espectrais e auxiliares
        # ------------------------------------------------------------
        from spectral_analysis import (
            calculate_extended_spectral_moments, calculate_chroma_vector,
            calculate_harmonic_ratio
        )
        from advanced_density_analysis import calculate_spectral_complexity
        from timbre_texture_analysis import (
            calculate_texture_density, calculate_timbre_blend,
            calculate_orchestration_balance
        )

        ext_mom   = calculate_extended_spectral_moments(pitches, densidades_instr)
        chroma    = calculate_chroma_vector(pitches, densidades_instr)
        harm_rat  = calculate_harmonic_ratio(pitches, densidades_instr)
        texture   = calculate_texture_density(pitches, None, numeros_instr)
        timbre    = calculate_timbre_blend(instrumentos, densidades_instr)
        orch      = calculate_orchestration_balance(pitches, densidades_instr, instrumentos)
        comp_dict = calculate_spectral_complexity(pitches, densidades_instr)
        complexity_factor = 1 + np.log1p(comp_dict.get("spectral_entropy", 0))

        # ------------------------------------------------------------
        # 8. FACTOR DE COESÃO – agora em semitons
        # ------------------------------------------------------------
        spectral_spread_st = amplitude_st
        coesao_factor = 10 / (1 + spectral_spread_st)

        # ------------------------------------------------------------
        # 9. Massa sonora e ganho dinâmico opcional
        # ------------------------------------------------------------
        massa_sonora_val = calcular_massa_sonora(
            notas, dinamicas, numeros_instr, densidades_instr, None
        )
        dynamic_boost = np.sqrt(massa_sonora_val)      # pode ser 1 se quiser desligar

        # ------------------------------------------------------------
        # 10. Densidade TOTAL
        # ------------------------------------------------------------
        densidade_total_val = (densidade_refinada_val *
                               coesao_factor *
                               complexity_factor *
                               (1 - harm_rat * 0.15) *
                               dynamic_boost)

        densidade_total_val /= MAX_DENS_GLOBAL      # normalização fixa
        # compressão log opcional (clusters extremos não explodem a escala)
        if USE_LOG_COMPRESSION:
            densidade_total_val = np.log10(1.0 + densidade_total_val)


        # ------------------------------------------------------------
        # 11. Densidade absoluta (referência simples)
        # ------------------------------------------------------------
        densidade_absoluta_val = densidade_ponderada_val * np.log1p(len(notas))

        # ------------------------------------------------------------
        # 12. Agregação dos resultados
        # ------------------------------------------------------------
        resultados = {
            "densidade": {
                "intervalar"   : densidade_intervalar_val,
                "instrumento"  : densidade_instrumento_val,
                "ponderada"    : densidade_ponderada_val,
                "refinada"     : densidade_refinada_val,
                "total"        : densidade_total_val,
                "massa_sonora" : massa_sonora_val,
                "absoluta"     : densidade_absoluta_val
            },
            "momentos_espectrais" : ext_mom,
            "metricas_adicionais" : {
                "complexity"    : comp_dict.get("spectral_entropy", 0),
                "harmonic_ratio": harm_rat,
                "chroma_vector" : chroma.tolist() if isinstance(chroma, np.ndarray) else chroma,
            },
            "textura"            : texture,
            "timbre"             : timbre,
            "orquestracao"       : orch,
            "dados_entrada"      : {
                "notas"             : notas,
                "dinamicas"         : dinamicas,
                "instrumentos"      : instrumentos,
                "numeros_instrumentos": numeros_instr
            }
        }

        return resultados, densidades_instr, pitches

    except Exception as e:
        logger.error(f"Erro ao calcular métricas: {e}", exc_info=True)
        raise


def format_output_string(resultados):
    """
    Formata os resultados para exibição no campo de texto.
    
    Args:
        resultados (dict): Resultados completos da análise
        
    Returns:
        str: String formatada para exibição
    """
    try:
        # Extracting values
        densidade_intervalar_val = resultados["densidade"]["intervalar"]
        densidade_instrumento_val = resultados["densidade"]["instrumento"]
        densidade_ponderada_val = resultados["densidade"]["ponderada"]
        densidade_refinada_val = resultados["densidade"]["refinada"]
        densidade_total_val = resultados["densidade"]["total"]
        massa_sonora_val = resultados["densidade"].get("massa_sonora", 0)
        densidade_absoluta_val = resultados["densidade"].get("absoluta", 0)
        
        moments = resultados["momentos_espectrais"]
        spectral_centroid_freq = moments["Centróide"]["frequency"]
        spectral_centroid_note = moments["Centróide"]["note"]
        spectral_spread_deviation = moments["Dispersão"]["deviation"]
        spectral_skewness = moments.get("spectral_skewness" , 0)
        spectral_kurtosis = moments.get("spectral_kurtosis", 0)
        spectral_flatness = moments.get("spectral_flatness", 0)
        spectral_entropy = moments.get("spectral_entropy", 0)
        
        add_metrics = resultados["metricas_adicionais"]
        complexity = add_metrics.get("complexity", 0)
        harmonic_ratio = add_metrics.get("harmonic_ratio", 0)
        
        texture = resultados["textura"]
        timbre = resultados["timbre"]
        orchestration = resultados["orquestracao"]
        
        # Formatting output
        output_string = (
            f"==================== DENSIDADE ====================\n"
            f"Densidade Intervalar: {densidade_intervalar_val:.4f}\n"
            f"Densidade do Instrumento: {densidade_instrumento_val:.4f}\n"
            f"Densidade Ponderada: {densidade_ponderada_val:.4f}\n"
            f"Densidade Refinada: {densidade_refinada_val:.4f}\n"
            f"Densidade Total: {densidade_total_val:.4f}\n"
            f"Massa Sonora Total: {massa_sonora_val:.4f}\n"
            f"Densidade Absoluta: {densidade_absoluta_val:.4f}\n\n"
            
            f"================ MOMENTOS ESPECTRAIS ===============\n"
            f"Centróide:{spectral_centroid_freq:.2f} Hz, Note: {spectral_centroid_note}\n"
            f"Dispersão: ±{spectral_spread_deviation:.2f} Hz\n"
            f"Skewness: {spectral_skewness:.4f}\n"
            f"Kurtosis: {spectral_kurtosis:.4f}\n"
            f"Flatness: {spectral_flatness:.4f}\n"
            f"Entropy: {spectral_entropy:.4f}\n\n"
            
            f"=============== MÉTRICAS AVANÇADAS ===============\n"
            f"Complexidade Espectral: {complexity:.4f}\n"
            f"Razão Harmônica: {harmonic_ratio:.4f}\n\n"
            
            f"================== TEXTURA ======================\n"
        )
        
        # Add texture metrics
        for k, v in texture.items():
            if isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v):
                output_string += f"{k.capitalize()}: {v:.4f}\n"
        
        output_string += "\n================== TIMBRE =======================\n"
        
        # Add timbre metrics
        for k, v in timbre.items():
            if k != "family_contributions" and isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v):
                output_string += f"{k.capitalize()}: {v:.4f}\n"
        
        output_string += "\n================ ORQUESTRAÇÃO ===================\n"
        
        # Add orchestration metrics
        for k, v in orchestration.items():
            if k != "register_distribution" and isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v):
                output_string += f"{k.capitalize()}: {v:.4f}\n"
        
        return output_string
    
    except Exception as e:
        logger.error(f"Erro ao formatar resultados: {e}")
        return f"Erro ao formatar resultados: {e}"

def generate_validation_text(resultados_validacao, num_historico):
    """
    Gera o texto de validação estatística.
    
    Args:
        resultados_validacao (dict): Resultados da validação estatística
        num_historico (int): Número de amostras no histórico
        
    Returns:
        str: Texto formatado para exibição
    """
    try:
        texto = "=== ESTATÍSTICAS DESCRITIVAS ===\n"
        desc_stats = resultados_validacao['descriptive_stats']
        texto += f"Número de amostras: {num_historico}\n\n"
        
        for col in desc_stats.columns:
            texto += f"{col}:\n"
            texto += f"  Média: {desc_stats[col]['mean']:.4f}\n"
            texto += f"  Desvio Padrão: {desc_stats[col]['std']:.4f}\n"
            texto += f"  Mínimo: {desc_stats[col]['min']:.4f}\n"
            texto += f"  Máximo: {desc_stats[col]['max']:.4f}\n"
            texto += f"  Coef. Variação: {resultados_validacao['coefficient_of_variation'][col]:.4f}\n\n"
        
        texto += "=== CORRELAÇÕES SIGNIFICATIVAS ===\n"
        if resultados_validacao['high_correlations']:
            for (m1, m2), corr in resultados_validacao['high_correlations'].items():
                texto += f"{m1} ? {m2}: {corr:.4f}\n"
        else:
            texto += "Nenhuma correlação significativa encontrada.\n"
        
        if 'pca' in resultados_validacao:
            texto += "\n=== ANÁLISE DE COMPONENTES PRINCIPAIS ===\n"
            texto += f"Número de componentes para 95% da variância: {resultados_validacao['pca']['n_components_95']}\n"
        
        return texto
    
    except Exception as e:
        logger.error(f"Erro ao gerar texto de validação: {e}")
        return f"Erro ao gerar texto de validação: {e}"
