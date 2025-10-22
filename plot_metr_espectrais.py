# plot_metr_espectrais.py

import matplotlib.pyplot as plt
import numpy as np
from math import log2
from spectral_analysis import (
    calculate_spectral_moments,
    calculate_extended_spectral_moments,  # Importante: use isso para obter as métricas adicionais
    calculate_chroma_vector,
    robust_gaussian_kde,
)
from microtonal import midi_to_note_name, note_to_midi, frequency_to_note_name
from utils.serialize_utils import safe_operation, safe_show_figure


def plot_metricas_espectrais_completo(metrics, title="Métricas Espectrais", filename=None):
    """
    Gráfico completo com TODAS as métricas espectrais.
    """
    # Configurar a figura
    fig, ax = plt.subplots(figsize=(14, 6))

    # Extrair todos os valores com verificação de segurança
    metrics_values = {
        'Centróide': {'valor': metrics.get('centroid', 0), 'formato': '{:.2f} Hz', 'cor': '#4285F4'},
        'Dispersão': {'valor': metrics.get('spread', 0), 'formato': '{:.2f} Hz', 'cor': '#EA4335'},
        'Assimetria': {'valor': metrics.get('skewness', 0), 'formato': '{:.4f}', 'cor': '#34A853'},
        'Curtose': {'valor': metrics.get('kurtosis', 0), 'formato': '{:.4f}', 'cor': '#FBBC05'},
        'Planura': {'valor': metrics.get('flatness', 0), 'formato': '{:.4f}', 'cor': '#FF6D01'},
        'Entropia': {'valor': metrics.get('entropy', 0), 'formato': '{:.4f}', 'cor': '#46BDC6'}
    }

    # Sanitizar valores para evitar erros de plotagem
    for key in metrics_values:
        if np.isnan(metrics_values[key]['valor']) or np.isinf(metrics_values[key]['valor']):
            metrics_values[key]['valor'] = 0

    # Extrair dados para o gráfico
    labels = list(metrics_values.keys())
    valores = [metrics_values[k]['valor'] for k in labels]
    cores = [metrics_values[k]['cor'] for k in labels]

    # Ajustar valores para melhor visualização
    # Centróide como 0 para mostrar apenas a nota
    valores_display = valores.copy()
    valores_display[0] = 0

    # Criar gráfico de barras
    barras = ax.bar(labels, valores_display, color=cores, width=0.6)

    # Adicionar rótulos e anotações
    for i, (label, barra) in enumerate(zip(labels, barras)):
        valor_original = valores[i]
        formato = metrics_values[label]['formato']

        if label == 'Centróide':
            # Usar a nota E4 directamente (corrigindo a discrepância)
            nota = metrics.get('centroid_note', frequency_to_note_name(valor_original))
            ax.text(i, 0.5, f"{nota}\n({valor_original:.2f} Hz)",
                   ha='center', va='center', fontsize=12, fontweight='bold',
                   color='white', bbox=dict(facecolor=cores[i], boxstyle='round', alpha=0.9))
        else:
            # Mostrar valor formatado acima da barra
            altura = barra.get_height()
            y_pos = altura + 0.05 if altura >= 0 else altura - 2.0
            valor_texto = formato.format(valor_original)
            ax.text(i, y_pos, valor_texto, ha='center', va='center' if altura < 0 else 'bottom',
                   fontweight='bold', bbox=dict(facecolor='white', edgecolor=cores[i],
                                              boxstyle='round,pad=0.2', alpha=0.9))

    # Configurar eixos e título
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Valor', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adicionar linha horizontal no zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Ajustar limites do eixo Y para melhor visualização
    ax.set_ylim(min(min(valores) * 1.2, -5), max(max(valores) * 1.2, 10))

    # Ajustar layout
    plt.tight_layout()

    # Salvar se necessário
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    return fig, ax


def extract_and_plot_metrics(notas, duracoes, instrumentos, numeros_instrumentos, densidades_instrumento, note_to_midi_func=None):
    """
    Versão actualizada que passa a nota do centróide explicitamente
    """
    try:
        # Converter notas para MIDI
        if note_to_midi_func is None:
            note_to_midi_func = note_to_midi

        pitches = [note_to_midi_func(nota) for nota in notas]
        amplitudes = densidades_instrumento

        # 1. Calcular os momentos espectrais básicos
        spectral_results = calculate_spectral_moments(pitches, amplitudes)

        # 2. Calcular métricas estendidas explicitamente
        extended_metrics = calculate_extended_spectral_moments(pitches, amplitudes)

        # 3. Combinar os resultados
        centroid_freq = spectral_results.get("Centróide", {}).get("frequency", 0)
        centroid_note = spectral_results.get("Centróide", {}).get("note", "N/A")
        spread_hz = spectral_results.get("Dispersão", {}).get("deviation", 0)

        # Obter métricas adicionais (mesma lógica anterior)
        skewness = spectral_results.get("spectral_skewness",
                   extended_metrics.get("skewness",
                   extended_metrics.get("spectral_skewness", 0)))

        kurtosis = spectral_results.get("kurtosis",
                  spectral_results.get("spectral_kurtosis",
                  extended_metrics.get("kurtosis",
                  extended_metrics.get("spectral_kurtosis", 0))))

        flatness = spectral_results.get("flatness",
                  spectral_results.get("spectral_flatness",
                  extended_metrics.get("flatness",
                  extended_metrics.get("spectral_flatness", 0))))

        entropy = spectral_results.get("entropy",
                 spectral_results.get("spectral_entropy",
                 extended_metrics.get("entropy",
                 extended_metrics.get("spectral_entropy", 0))))

        # Imprimir todas as métricas disponíveis
        print(f"Centróide: {centroid_freq:.2f} Hz, Nota: {centroid_note}")
        print(f"Dispersão: ±{spread_hz:.2f} Hz")
        print(f"Assimetria: {skewness:.4f}")
        print(f"Curtose: {kurtosis:.4f}")
        print(f"Planura: {flatness:.4f}")
        print(f"Entropia: {entropy:.4f}")

        # Construir dicionário completo
        metrics = {
            'centroid': centroid_freq,
            'centroid_note': centroid_note,  # Passar a nota explicitamente
            'spread': spread_hz,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'flatness': flatness,
            'entropy': entropy
        }

        # Fechar figuras existentes
        plt.close('all')

        # Gerar o gráfico
        fig, ax = plot_metricas_espectrais_completo(metrics, title="Métricas Espectrais")

        # Mostrar o gráfico
        safe_show_figure(fig)

        return metrics

    except Exception as e:
        print(f"Erro ao processar métricas espectrais: {e}")
        import traceback
        traceback.print_exc()
        return {}


# Adicione este código ao final do ficheiro para calcular explicitamente as métricas estendidas
def calculate_all_spectral_metrics(pitches, amplitudes):
    """
    Função que calcula TODAS as métricas espectrais possíveis.
    Útil para chamar directamente quando necessário.
    """
    # Calcular métricas básicas
    basic_metrics = calculate_spectral_moments(pitches, amplitudes)

    # Calcular métricas estendidas
    extended_metrics = calculate_extended_spectral_moments(pitches, amplitudes)

    # Combinar os resultados
    all_metrics = {**basic_metrics}

    # Extrair métricas estendidas e adicionar ao dicionário
    if isinstance(extended_metrics, dict):
        for key, value in extended_metrics.items():
            all_metrics[key] = value

    return all_metrics
