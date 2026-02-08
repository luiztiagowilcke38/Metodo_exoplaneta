"""
Módulo 04: Validação de Dados
Autor: Luiz Tiago Wilcke
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Tuple, Optional
import os


class ValidadorDados:
    """Validação estatística de qualidade dos dados."""
    
    def __init__(self, nivel_significancia: float = 0.05):
        self.alpha = nivel_significancia
    
    def teste_normalidade_shapiro(self, dados: np.ndarray) -> Dict:
        """Teste de Shapiro-Wilk para normalidade."""
        amostra = dados[:5000] if len(dados) > 5000 else dados
        estatistica, p_valor = stats.shapiro(amostra)
        return {'estatistica': estatistica, 'p_valor': p_valor, 'normal': p_valor > self.alpha}
    
    def teste_normalidade_ks(self, dados: np.ndarray) -> Dict:
        """Teste de Kolmogorov-Smirnov."""
        dados_padronizados = (dados - np.mean(dados)) / np.std(dados)
        estatistica, p_valor = stats.kstest(dados_padronizados, 'norm')
        return {'estatistica': estatistica, 'p_valor': p_valor, 'normal': p_valor > self.alpha}
    
    def teste_estacionariedade_adf(self, dados: np.ndarray) -> Dict:
        """Teste Augmented Dickey-Fuller simplificado."""
        # Versão simplificada sem statsmodels
        diff = np.diff(dados)
        t_stat = np.mean(diff) / (np.std(diff) / np.sqrt(len(diff)))
        p_valor = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        return {'estatistica': t_stat, 'p_valor': p_valor, 'estacionario': p_valor < self.alpha}
    
    def teste_autocorrelacao_ljungbox(self, dados: np.ndarray, lags: int = 20) -> Dict:
        """Teste Ljung-Box para autocorrelação."""
        n = len(dados)
        acf = np.correlate(dados - np.mean(dados), dados - np.mean(dados), 'full')
        acf = acf[n-1:] / acf[n-1]
        
        Q = n * (n + 2) * np.sum(acf[1:lags+1]**2 / (n - np.arange(1, lags+1)))
        p_valor = 1 - stats.chi2.cdf(Q, lags)
        return {'estatistica': Q, 'p_valor': p_valor, 'autocorrelacionado': p_valor < self.alpha}
    
    def calcular_metricas_qualidade(self, tempo: np.ndarray, fluxo: np.ndarray, 
                                      erro: np.ndarray) -> Dict:
        """Calcula métricas de qualidade dos dados."""
        dt = np.diff(tempo)
        return {
            'n_pontos': len(fluxo),
            'cobertura_temporal': tempo[-1] - tempo[0],
            'cadencia_mediana': np.median(dt) * 24 * 60,  # minutos
            'gaps_longos': np.sum(dt > 3 * np.median(dt)),
            'snr': np.median(fluxo) / np.median(erro),
            'rms': np.std(fluxo),
            'mad': 1.4826 * np.median(np.abs(fluxo - np.median(fluxo))),
            'curtose': stats.kurtosis(fluxo),
            'assimetria': stats.skew(fluxo)
        }
    
    def validar_dados(self, tempo: np.ndarray, fluxo: np.ndarray, erro: np.ndarray) -> Dict:
        """Validação completa dos dados."""
        metricas = self.calcular_metricas_qualidade(tempo, fluxo, erro)
        
        # Testes estatísticos
        teste_norm_sw = self.teste_normalidade_shapiro(fluxo)
        teste_norm_ks = self.teste_normalidade_ks(fluxo)
        teste_estac = self.teste_estacionariedade_adf(fluxo)
        teste_acf = self.teste_autocorrelacao_ljungbox(fluxo)
        
        qualidade = 'BOA' if metricas['snr'] > 100 else 'MÉDIA' if metricas['snr'] > 50 else 'BAIXA'
        
        return {
            'metricas': metricas,
            'testes': {'shapiro': teste_norm_sw, 'ks': teste_norm_ks, 
                       'adf': teste_estac, 'ljungbox': teste_acf},
            'qualidade_geral': qualidade
        }
    
    def plotar_diagnosticos(self, tempo, fluxo, erro, titulo: str, salvar: Optional[str] = None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Série temporal
        axes[0, 0].errorbar(tempo, fluxo, yerr=erro, fmt='.', ms=1, alpha=0.3)
        axes[0, 0].set_title('Série Temporal'); axes[0, 0].set_xlabel('Tempo'); axes[0, 0].set_ylabel('Fluxo')
        
        # Histograma
        axes[0, 1].hist(fluxo, bins=100, density=True, alpha=0.7)
        x = np.linspace(np.min(fluxo), np.max(fluxo), 100)
        axes[0, 1].plot(x, stats.norm.pdf(x, np.mean(fluxo), np.std(fluxo)), 'r-', lw=2)
        axes[0, 1].set_title('Distribuição vs Normal')
        
        # Q-Q plot
        stats.probplot(fluxo, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        # ACF simplificada
        n = min(100, len(fluxo) // 10)
        acf = np.correlate(fluxo - np.mean(fluxo), fluxo - np.mean(fluxo), 'full')
        acf = acf[len(fluxo)-1:len(fluxo)+n] / acf[len(fluxo)-1]
        axes[1, 1].bar(range(len(acf)), acf, alpha=0.7)
        axes[1, 1].axhline(1.96/np.sqrt(len(fluxo)), color='r', ls='--')
        axes[1, 1].axhline(-1.96/np.sqrt(len(fluxo)), color='r', ls='--')
        axes[1, 1].set_title('Autocorrelação')
        
        for ax in axes.flat: ax.grid(True, alpha=0.3)
        plt.suptitle(titulo, fontweight='bold'); plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_04(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 04: VALIDAÇÃO DE DADOS\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    validador = ValidadorDados()
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        t, f, e = dados['tempo'], dados['fluxo'], dados['erro_fluxo']
        
        validacao = validador.validar_dados(t, f, e)
        m = validacao['metricas']
        
        print(f"    SNR: {m['snr']:.1f}")
        print(f"    RMS: {m['rms']:.6f}")
        print(f"    Curtose: {m['curtose']:.2f}")
        print(f"    Qualidade: {validacao['qualidade_geral']}")
        
        arq = os.path.join(diretorio_saida, f"validacao_{nome.replace(' ', '_').lower()}.png")
        validador.plotar_diagnosticos(t, f, e, f"Diagnósticos - {nome}", arq)
        
        resultados[nome] = {**dados, 'validacao': validacao}
    
    plt.close('all')
    print("\nMÓDULO 04 CONCLUÍDO")
    return resultados

__all__ = ['ValidadorDados', 'executar_modulo_04']
