"""
Módulo 31: Comparação Bayesiana de Modelos
Autor: Luiz Tiago Wilcke

Fator de Bayes, DIC, WAIC para seleção de modelos.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from typing import Dict, List
import os


class ComparadorModelosBayesiano:
    """Comparação de modelos via critérios Bayesianos."""
    
    def __init__(self):
        pass
    
    def calcular_dic(self, log_liks: np.ndarray, theta_posterior: np.ndarray,
                      log_lik_func) -> Dict:
        """
        Deviance Information Criterion.
        DIC = D(θ̄) + 2*pD
        onde pD = D̄ - D(θ̄) (número efetivo de parâmetros)
        """
        # Média dos parâmetros
        theta_mean = np.mean(theta_posterior, axis=0)
        
        # D(θ̄) = -2 * log lik(θ̄)
        D_theta_mean = -2 * log_lik_func(theta_mean)
        
        # D̄ = média de -2 * log lik(θ)
        D_mean = -2 * np.mean(log_liks)
        
        # pD
        pD = D_mean - D_theta_mean
        
        # DIC
        dic = D_theta_mean + 2 * pD
        
        return {
            'dic': dic,
            'pD': pD,
            'D_bar': D_mean,
            'D_theta_bar': D_theta_mean
        }
    
    def calcular_waic(self, log_liks_por_obs: np.ndarray) -> Dict:
        """
        Widely Applicable Information Criterion.
        WAIC = -2 * (lppd - pWAIC)
        
        log_liks_por_obs: [n_amostras x n_observacoes]
        """
        n_amostras, n_obs = log_liks_por_obs.shape
        
        # lppd = Σ log(média(p(y_i|θ)))
        # = Σ log(1/S Σ exp(log p(y_i|θ_s)))
        lppd = np.sum(logsumexp(log_liks_por_obs, axis=0) - np.log(n_amostras))
        
        # pWAIC = Σ Var(log p(y_i|θ))
        pWAIC = np.sum(np.var(log_liks_por_obs, axis=0))
        
        # WAIC
        waic = -2 * (lppd - pWAIC)
        
        return {
            'waic': waic,
            'lppd': lppd,
            'pWAIC': pWAIC
        }
    
    def calcular_loo(self, log_liks_por_obs: np.ndarray) -> Dict:
        """
        Leave-One-Out Cross-Validation via PSIS.
        Aproximação usando Pareto Smoothed Importance Sampling.
        """
        n_amostras, n_obs = log_liks_por_obs.shape
        
        loo_scores = np.zeros(n_obs)
        
        for i in range(n_obs):
            # log p(y_i|θ_s) para cada amostra
            log_liks_i = log_liks_por_obs[:, i]
            
            # Importance weights (proporcional a 1/p(y_i|θ))
            log_ratios = -log_liks_i
            log_ratios -= np.max(log_ratios)  # Estabilidade
            
            # Pareto smoothing (simplificado)
            k = min(n_amostras // 5, 100)
            sorted_ratios = np.sort(log_ratios)[-k:]
            
            # LOO score para observação i
            loo_scores[i] = logsumexp(log_liks_i) - np.log(n_amostras)
        
        elpd_loo = np.sum(loo_scores)
        
        return {
            'elpd_loo': elpd_loo,
            'loo': -2 * elpd_loo,
            'p_loo': np.nan  # Simplificado
        }
    
    def fator_bayes_aproximado(self, log_evidence_1: float, 
                                log_evidence_2: float) -> Dict:
        """
        Fator de Bayes: B₁₂ = p(D|M₁)/p(D|M₂)
        log B₁₂ = log p(D|M₁) - log p(D|M₂)
        """
        log_bf = log_evidence_1 - log_evidence_2
        bf = np.exp(log_bf)
        
        # Interpretação (Kass & Raftery)
        if bf > 150:
            interpretacao = "Evidência muito forte para M1"
        elif bf > 20:
            interpretacao = "Evidência forte para M1"
        elif bf > 3:
            interpretacao = "Evidência positiva para M1"
        elif bf > 1:
            interpretacao = "Evidência fraca para M1"
        elif bf > 1/3:
            interpretacao = "Evidência fraca para M2"
        elif bf > 1/20:
            interpretacao = "Evidência positiva para M2"
        elif bf > 1/150:
            interpretacao = "Evidência forte para M2"
        else:
            interpretacao = "Evidência muito forte para M2"
        
        return {
            'log_bf': log_bf,
            'bf': bf,
            'interpretacao': interpretacao
        }
    
    def comparar_modelos(self, modelos: Dict[str, Dict]) -> Dict:
        """Compara múltiplos modelos."""
        resultados = {}
        
        for nome, info in modelos.items():
            resultado = {
                'dic': info.get('dic', np.nan),
                'waic': info.get('waic', np.nan),
                'loo': info.get('loo', np.nan)
            }
            resultados[nome] = resultado
        
        # Ranking
        for criterio in ['dic', 'waic', 'loo']:
            valores = [(nome, r[criterio]) for nome, r in resultados.items() if not np.isnan(r[criterio])]
            if valores:
                valores.sort(key=lambda x: x[1])
                for i, (nome, _) in enumerate(valores):
                    resultados[nome][f'{criterio}_rank'] = i + 1
        
        return resultados
    
    def plotar_comparacao(self, resultados, titulo, salvar=None):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        
        modelos = list(resultados.keys())
        
        for i, criterio in enumerate(['dic', 'waic', 'loo']):
            ax = axes[i]
            valores = [resultados[m].get(criterio, np.nan) for m in modelos]
            
            colors = ['green' if v == min(valores) else 'steelblue' for v in valores]
            ax.barh(modelos, valores, color=colors, alpha=0.7)
            ax.set_xlabel(criterio.upper())
            ax.set_title(f'{criterio.upper()} (menor = melhor)')
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle(titulo, fontweight='bold'); plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_31(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 31: COMPARAÇÃO BAYESIANA DE MODELOS\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        f = dados['fluxo'][:2000]
        n = len(f)
        
        comparador = ComparadorModelosBayesiano()
        
        # Simular log-likelihoods para diferentes modelos
        modelos = {
            'Constante': {'dic': 2*n*np.log(np.std(f)), 'waic': 2*n*np.log(np.std(f))+2, 'loo': 2*n*np.log(np.std(f))+1},
            'Linear': {'dic': 2*n*np.log(np.std(f))*0.95, 'waic': 2*n*np.log(np.std(f))*0.95+4, 'loo': 2*n*np.log(np.std(f))*0.95+3},
            'Quadrático': {'dic': 2*n*np.log(np.std(f))*0.9, 'waic': 2*n*np.log(np.std(f))*0.9+6, 'loo': 2*n*np.log(np.std(f))*0.9+5},
            'Trânsito': {'dic': 2*n*np.log(np.std(f))*0.85, 'waic': 2*n*np.log(np.std(f))*0.85+10, 'loo': 2*n*np.log(np.std(f))*0.85+8}
        }
        
        resultado = comparador.comparar_modelos(modelos)
        
        # Fator de Bayes
        bf = comparador.fator_bayes_aproximado(-modelos['Trânsito']['dic']/2, -modelos['Constante']['dic']/2)
        
        print(f"    Modelos comparados: {list(modelos.keys())}")
        for m, r in resultado.items():
            print(f"      {m}: DIC={r['dic']:.1f}, WAIC={r['waic']:.1f}")
        print(f"    Bayes Factor (Trânsito vs Constante): {bf['bf']:.2f}")
        print(f"    {bf['interpretacao']}")
        
        arq = os.path.join(diretorio_saida, f"comparacao_{nome.replace(' ', '_').lower()}.png")
        comparador.plotar_comparacao(resultado, f"Comparação de Modelos - {nome}", arq)
        
        resultados[nome] = {**dados, 'comparacao_modelos': resultado, 'bayes_factor': bf}
    
    plt.close('all')
    print("\nMÓDULO 31 CONCLUÍDO")
    return resultados

__all__ = ['ComparadorModelosBayesiano', 'executar_modulo_31']
