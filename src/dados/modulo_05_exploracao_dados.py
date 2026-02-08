"""
Módulo 05: Exploração de Dados
Autor: Luiz Tiago Wilcke
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Optional
import os


class ExploradorDados:
    """Análise exploratória de curvas de luz."""
    
    def calcular_estatisticas_descritivas(self, tempo: np.ndarray, fluxo: np.ndarray, 
                                           erro: np.ndarray) -> Dict:
        """Calcula estatísticas descritivas completas."""
        return {
            'n': len(fluxo),
            'media': np.mean(fluxo),
            'mediana': np.median(fluxo),
            'desvio_padrao': np.std(fluxo),
            'variancia': np.var(fluxo),
            'minimo': np.min(fluxo),
            'maximo': np.max(fluxo),
            'amplitude': np.max(fluxo) - np.min(fluxo),
            'q1': np.percentile(fluxo, 25),
            'q3': np.percentile(fluxo, 75),
            'iqr': np.percentile(fluxo, 75) - np.percentile(fluxo, 25),
            'curtose': stats.kurtosis(fluxo),
            'assimetria': stats.skew(fluxo),
            'coef_variacao': np.std(fluxo) / np.mean(fluxo),
            'duracao_dias': tempo[-1] - tempo[0],
            'erro_medio': np.mean(erro),
            'snr_medio': np.mean(fluxo / erro)
        }
    
    def detectar_variacoes_significativas(self, tempo: np.ndarray, fluxo: np.ndarray,
                                           threshold_sigma: float = 3.0) -> Dict:
        """Detecta variações significativas no fluxo."""
        mediana = np.median(fluxo)
        mad = 1.4826 * np.median(np.abs(fluxo - mediana))
        
        # Pontos abaixo (possíveis trânsitos)
        limite_inf = mediana - threshold_sigma * mad
        pontos_baixos = np.where(fluxo < limite_inf)[0]
        
        # Pontos acima (possíveis flares)
        limite_sup = mediana + threshold_sigma * mad
        pontos_altos = np.where(fluxo > limite_sup)[0]
        
        return {
            'n_transitos_candidatos': len(pontos_baixos),
            'n_flares_candidatos': len(pontos_altos),
            'indices_transitos': pontos_baixos,
            'indices_flares': pontos_altos,
            'profundidade_max': mediana - np.min(fluxo[pontos_baixos]) if len(pontos_baixos) > 0 else 0,
            'altura_max_flare': np.max(fluxo[pontos_altos]) - mediana if len(pontos_altos) > 0 else 0
        }
    
    def analisar_periodicidade_simples(self, fluxo: np.ndarray) -> Dict:
        """Análise simples de periodicidade via autocorrelação."""
        n = len(fluxo)
        max_lag = min(n // 2, 1000)
        
        acf = np.correlate(fluxo - np.mean(fluxo), fluxo - np.mean(fluxo), 'full')
        acf = acf[n-1:n+max_lag] / acf[n-1]
        
        # Encontrar picos na ACF (possíveis períodos)
        picos = []
        for i in range(2, len(acf) - 1):
            if acf[i] > acf[i-1] and acf[i] > acf[i+1] and acf[i] > 0.1:
                picos.append((i, acf[i]))
        
        picos_ordenados = sorted(picos, key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'acf': acf,
            'picos_lag': [p[0] for p in picos_ordenados],
            'picos_valor': [p[1] for p in picos_ordenados]
        }
    
    def plotar_exploracao_completa(self, tempo, fluxo, erro, nome_estrela: str,
                                    salvar: Optional[str] = None):
        """Gera visualização exploratória completa."""
        fig = plt.figure(figsize=(16, 12))
        
        # Layout
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
        ax2 = plt.subplot2grid((3, 3), (1, 0))
        ax3 = plt.subplot2grid((3, 3), (1, 1))
        ax4 = plt.subplot2grid((3, 3), (1, 2))
        ax5 = plt.subplot2grid((3, 3), (2, 0))
        ax6 = plt.subplot2grid((3, 3), (2, 1))
        ax7 = plt.subplot2grid((3, 3), (2, 2))
        
        # 1. Curva de luz completa
        ax1.scatter(tempo, fluxo, s=1, alpha=0.5, c='#1f77b4')
        ax1.axhline(np.median(fluxo), color='r', ls='--', alpha=0.7, label='Mediana')
        ax1.fill_between(tempo, np.median(fluxo) - 3*np.std(fluxo), 
                        np.median(fluxo) + 3*np.std(fluxo), alpha=0.2, color='r')
        ax1.set_xlabel('Tempo (BJD)'); ax1.set_ylabel('Fluxo')
        ax1.set_title(f'Curva de Luz - {nome_estrela}'); ax1.legend()
        
        # 2. Histograma
        ax2.hist(fluxo, bins=100, density=True, alpha=0.7, color='#2ca02c')
        ax2.axvline(np.median(fluxo), color='r', ls='--', lw=2)
        ax2.set_xlabel('Fluxo'); ax2.set_ylabel('Densidade'); ax2.set_title('Distribuição')
        
        # 3. Box plot
        ax3.boxplot(fluxo, vert=True)
        ax3.set_ylabel('Fluxo'); ax3.set_title('Box Plot')
        
        # 4. Fluxo vs Erro
        ax4.scatter(erro, fluxo, s=1, alpha=0.3)
        ax4.set_xlabel('Erro'); ax4.set_ylabel('Fluxo'); ax4.set_title('Fluxo vs Erro')
        
        # 5. Variação temporal do erro
        ax5.scatter(tempo, erro, s=1, alpha=0.3, c='orange')
        ax5.set_xlabel('Tempo'); ax5.set_ylabel('Erro'); ax5.set_title('Erro ao Longo do Tempo')
        
        # 6. ACF
        autocorr = self.analisar_periodicidade_simples(fluxo)
        ax6.plot(autocorr['acf'][:200], 'b-', lw=0.5)
        ax6.axhline(0, color='k', ls='-'); ax6.axhline(0.1, color='r', ls='--', alpha=0.7)
        ax6.set_xlabel('Lag'); ax6.set_ylabel('ACF'); ax6.set_title('Autocorrelação')
        
        # 7. Fase dobrada (estimativa simples)
        if len(autocorr['picos_lag']) > 0:
            periodo_est = autocorr['picos_lag'][0]
            fase = (tempo % (periodo_est * np.median(np.diff(tempo)))) / (periodo_est * np.median(np.diff(tempo)))
            ax7.scatter(fase, fluxo, s=1, alpha=0.3)
            ax7.set_xlabel('Fase'); ax7.set_ylabel('Fluxo')
            ax7.set_title(f'Curva de Fase (P~{periodo_est} pts)')
        else:
            ax7.text(0.5, 0.5, 'Sem período detectado', ha='center', va='center')
        
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_05(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 05: EXPLORAÇÃO DE DADOS\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    explorador = ExploradorDados()
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        t, f, e = dados['tempo'], dados['fluxo'], dados['erro_fluxo']
        
        # Estatísticas descritivas
        estatisticas = explorador.calcular_estatisticas_descritivas(t, f, e)
        variacoes = explorador.detectar_variacoes_significativas(t, f)
        periodicidade = explorador.analisar_periodicidade_simples(f)
        
        print(f"    Média: {estatisticas['media']:.6f}")
        print(f"    Desvio padrão: {estatisticas['desvio_padrao']:.6f}")
        print(f"    Trânsitos candidatos: {variacoes['n_transitos_candidatos']}")
        print(f"    Flares candidatos: {variacoes['n_flares_candidatos']}")
        
        arq = os.path.join(diretorio_saida, f"exploracao_{nome.replace(' ', '_').lower()}.png")
        explorador.plotar_exploracao_completa(t, f, e, nome, arq)
        
        resultados[nome] = {
            **dados,
            'estatisticas_descritivas': estatisticas,
            'variacoes': variacoes,
            'periodicidade_acf': periodicidade
        }
    
    plt.close('all')
    print("\nMÓDULO 05 CONCLUÍDO")
    return resultados

__all__ = ['ExploradorDados', 'executar_modulo_05']
