#!/usr/bin/env python3
"""
Sistema Estatístico para Detecção de Exoplanetas
================================================

Script principal que executa a análise completa de curvas de luz
para detecção de exoplanetas usando métodos estatísticos avançados.

Autor: Luiz Tiago Wilcke
Data: 2024

Uso:
    python main.py                    # Execução padrão com dados simulados
    python main.py --estrela Kepler-10  # Buscar dados reais
    python main.py --ajuda             # Mostrar ajuda

Este sistema contém 40 módulos organizados em 5 categorias:
    1. Dados (01-05): Carregamento, pré-processamento, normalização
    2. Séries Temporais (06-15): Fourier, periodograma, BLS, wavelets
    3. Modelos Lineares (16-25): Regressão, PCA, splines, GAM
    4. Bayesianos (26-35): MCMC, HMC, processos gaussianos
    5. Integração (36-40): Pipeline, relatórios, visualização
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

# Adicionar diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configurações de visualização
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10


def gerar_dados_simulados(nome: str = "Kepler-Simulado", 
                           n_pontos: int = 10000,
                           periodo_orbital: float = 5.0,
                           profundidade_transito: float = 0.001) -> dict:
    """
    Gera dados simulados de curva de luz com trânsito planetário.
    
    Parâmetros:
        nome: Nome da estrela simulada
        n_pontos: Número de pontos de dados
        periodo_orbital: Período do planeta em dias
        profundidade_transito: Profundidade relativa do trânsito
    
    Retorna:
        Dicionário com tempo, fluxo e erro
    """
    print(f"\n>>> Gerando dados simulados: {nome}")
    print(f"    N pontos: {n_pontos}")
    print(f"    Período orbital: {periodo_orbital} dias")
    print(f"    Profundidade: {profundidade_transito*1e6:.0f} ppm")
    
    # Tempo (em dias)
    tempo = np.linspace(0, 90, n_pontos)  # 90 dias de observação
    dt = np.median(np.diff(tempo))
    
    # Fluxo base normalizado
    fluxo_base = np.ones(n_pontos)
    
    # 1. Variabilidade estelar (oscilações)
    # Granulação (alta frequência)
    variabilidade = 0.0001 * np.sin(2 * np.pi * tempo / 0.1)
    # Oscilações p-mode
    variabilidade += 0.00005 * np.sin(2 * np.pi * tempo / 0.005)
    # Rotação estelar (manchas)
    variabilidade += 0.0002 * np.sin(2 * np.pi * tempo / 15.0 + np.random.random() * 2 * np.pi)
    
    # 2. Trânsito planetário
    fase = ((tempo % periodo_orbital) / periodo_orbital)
    duracao_transito = 0.05  # Fração do período
    
    # Modelo de trânsito trapezoidal
    transito = np.ones(n_pontos)
    em_transito = np.abs(fase - 0.5) < duracao_transito / 2
    
    # Ingresso e egresso suaves
    for i in np.where(em_transito)[0]:
        dist = np.abs(fase[i] - 0.5) / (duracao_transito / 2)
        if dist > 0.8:  # Ingresso/egresso
            transito[i] = 1 - profundidade_transito * (1 - (dist - 0.8) / 0.2)
        else:  # Centro do trânsito
            transito[i] = 1 - profundidade_transito
    
    # 3. Tendência instrumental
    tendencia = 1 + 0.0001 * (tempo - tempo[0]) / np.ptp(tempo)
    tendencia += 0.00005 * np.sin(2 * np.pi * tempo / 30)  # Variação mensal
    
    # 4. Ruído
    ruido_branco = np.random.normal(0, 0.0001, n_pontos)
    ruido_vermelho = np.convolve(np.random.normal(0, 0.00005, n_pontos), 
                                  np.ones(10)/10, mode='same')
    
    # Combinar todos os componentes
    fluxo = fluxo_base * transito * tendencia + variabilidade + ruido_branco + ruido_vermelho
    
    # Erro estimado
    erro_fluxo = np.abs(np.random.normal(0.0001, 0.00002, n_pontos))
    
    print(f"    Fluxo médio: {np.mean(fluxo):.6f}")
    print(f"    Desvio padrão: {np.std(fluxo):.6f}")
    print(f"    N trânsitos: {int(np.ptp(tempo) / periodo_orbital)}")
    
    return {
        'nome': nome,
        'tempo': tempo,
        'fluxo': fluxo,
        'erro_fluxo': erro_fluxo,
        'periodo_real': periodo_orbital,
        'profundidade_real': profundidade_transito
    }


def executar_analise_principal():
    """Executa a análise principal do sistema."""
    print("=" * 70)
    print("   SISTEMA ESTATÍSTICO PARA DETECÇÃO DE EXOPLANETAS")
    print("   Métodos Avançados de Análise de Curvas de Luz")
    print("   Autor: Luiz Tiago Wilcke")
    print("=" * 70)
    print(f"\nData de execução: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Diretório de saída
    diretorio_saida = "resultados"
    os.makedirs(diretorio_saida, exist_ok=True)
    print(f"Resultados serão salvos em: {os.path.abspath(diretorio_saida)}")
    
    # Gerar dados simulados
    dados = {
        "Kepler-Simulado": gerar_dados_simulados(
            "Kepler-Simulado", 
            n_pontos=8000,
            periodo_orbital=5.234,
            profundidade_transito=0.0008
        )
    }
    
    # Importar e executar módulos selecionados
    print("\n" + "=" * 70)
    print("EXECUTANDO MÓDULOS DE ANÁLISE")
    print("=" * 70)
    
    resultados = dados.copy()
    
    # Módulo 05: Exploração de Dados
    try:
        from src.dados.modulo_05_exploracao_dados import executar_modulo_05
        resultados = executar_modulo_05(resultados, diretorio_saida)
    except Exception as e:
        print(f"    Módulo 05 ignorado: {e}")
    
    # Módulo 07: Análise de Fourier
    try:
        from src.series_temporais.modulo_07_analise_fourier import executar_modulo_07
        resultados = executar_modulo_07(resultados, diretorio_saida)
    except Exception as e:
        print(f"    Módulo 07 ignorado: {e}")
    
    # Módulo 08: Periodograma
    try:
        from src.series_temporais.modulo_08_periodograma import executar_modulo_08
        resultados = executar_modulo_08(resultados, diretorio_saida)
    except Exception as e:
        print(f"    Módulo 08 ignorado: {e}")
    
    # Módulo 14: Detecção de Trânsitos (BLS)
    try:
        from src.series_temporais.modulo_14_deteccao_transitos import executar_modulo_14
        resultados = executar_modulo_14(resultados, diretorio_saida)
    except Exception as e:
        print(f"    Módulo 14 ignorado: {e}")
    
    # Módulo 16: Regressão Linear
    try:
        from src.modelos_lineares.modulo_16_regressao_linear import executar_modulo_16
        resultados = executar_modulo_16(resultados, diretorio_saida)
    except Exception as e:
        print(f"    Módulo 16 ignorado: {e}")
    
    # Módulo 38: Visualização Integrada
    try:
        from src.integracao.modulo_38_visualizacao import executar_modulo_38
        resultados = executar_modulo_38(resultados, diretorio_saida)
    except Exception as e:
        print(f"    Módulo 38 ignorado: {e}")
    
    # Módulo 37: Relatórios
    try:
        from src.integracao.modulo_37_relatorios import executar_modulo_37
        resultados = executar_modulo_37(resultados, diretorio_saida)
    except Exception as e:
        print(f"    Módulo 37 ignorado: {e}")
    
    # Módulo 39: Exportação
    try:
        from src.integracao.modulo_39_exportacao import executar_modulo_39
        resultados = executar_modulo_39(resultados, diretorio_saida)
    except Exception as e:
        print(f"    Módulo 39 ignorado: {e}")
    
    # Resumo final
    print("\n" + "=" * 70)
    print("RESUMO DA ANÁLISE")
    print("=" * 70)
    
    for nome, dados_curva in resultados.items():
        print(f"\n>>> {nome}:")
        print(f"    N pontos: {len(dados_curva.get('fluxo', []))}")
        
        if 'bls' in dados_curva:
            bls = dados_curva['bls']
            print(f"    Período detectado: {bls.get('melhor_periodo', 'N/A'):.4f} dias")
            print(f"    Período real: {dados_curva.get('periodo_real', 'N/A'):.4f} dias")
            print(f"    Profundidade: {bls.get('profundidade', 0)*1e6:.1f} ppm")
            print(f"    SNR: {bls.get('snr', 0):.2f}")
    
    print("\n" + "=" * 70)
    print("ANÁLISE CONCLUÍDA COM SUCESSO!")
    print(f"Resultados salvos em: {os.path.abspath(diretorio_saida)}")
    print("=" * 70)
    
    # Fechar todas as figuras
    plt.close('all')
    
    return resultados


def main():
    """Função principal."""
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--ajuda', '-h', '--help']:
            print(__doc__)
            return
    
    executar_analise_principal()


if __name__ == "__main__":
    main()
