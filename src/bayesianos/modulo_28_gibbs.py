"""
Módulo 28: Amostrador de Gibbs
Autor: Luiz Tiago Wilcke

Amostragem de Gibbs: amostrar cada parâmetro de sua distribuição condicional.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Optional
import os


class AmostradorGibbs:
    """
    Gibbs Sampling para modelos hierárquicos.
    θ_j^(t+1) ~ p(θ_j | θ_{-j}^(t), dados)
    """
    
    def __init__(self):
        self.cadeia = None
    
    def amostrar_normal_hierarquico(self, dados: np.ndarray, n_amostras: int = 5000,
                                      burnin: int = 1000) -> Dict:
        """
        Modelo hierárquico Normal:
        y_ij ~ N(μ_j, σ²)  (dados do grupo j)
        μ_j ~ N(μ, τ²)     (médias dos grupos)
        μ ~ N(0, 100)       (hiperprior)
        σ² ~ IG(α, β)       (variância dentro)
        τ² ~ IG(α, β)       (variância entre)
        """
        n = len(dados)
        
        # Inicialização
        mu = np.mean(dados)
        sigma2 = np.var(dados)
        
        n_total = burnin + n_amostras
        cadeia = np.zeros((n_total, 3))  # [mu, sigma2, tau2]
        
        # Hiperparâmetros
        alpha0, beta0 = 1, 1
        
        for t in range(n_total):
            # Amostrar μ | dados, σ²
            # Posterior: N(μ_n, σ²_n)
            precision_prior = 1/100
            precision_dados = n / sigma2
            precision_n = precision_prior + precision_dados
            mu_n = precision_dados * np.mean(dados) / precision_n
            sigma2_n = 1 / precision_n
            mu = np.random.normal(mu_n, np.sqrt(sigma2_n))
            
            # Amostrar σ² | dados, μ
            # Posterior: IG(α_n, β_n)
            alpha_n = alpha0 + n/2
            beta_n = beta0 + 0.5 * np.sum((dados - mu)**2)
            sigma2 = 1 / np.random.gamma(alpha_n, 1/beta_n)
            
            # τ² seria amostrado se tivéssemos estrutura hierárquica real
            tau2 = sigma2 * 0.1
            
            cadeia[t] = [mu, sigma2, tau2]
        
        cadeia_final = cadeia[burnin:]
        self.cadeia = cadeia_final
        
        return {
            'cadeia': cadeia_final,
            'mu_posterior': np.mean(cadeia_final[:, 0]),
            'mu_std': np.std(cadeia_final[:, 0]),
            'sigma2_posterior': np.mean(cadeia_final[:, 1]),
            'mu_ic': np.percentile(cadeia_final[:, 0], [2.5, 97.5]),
            'sigma2_ic': np.percentile(cadeia_final[:, 1], [2.5, 97.5])
        }
    
    def modelo_regressao_bayesiana(self, X: np.ndarray, y: np.ndarray,
                                     n_amostras: int = 5000, burnin: int = 1000) -> Dict:
        """
        Regressão Bayesiana via Gibbs.
        y|β,σ² ~ N(Xβ, σ²I)
        β|σ² ~ N(0, σ²/τ I)
        σ² ~ IG(α, β)
        """
        n, p = X.shape
        X_aug = np.column_stack([np.ones(n), X])
        p_aug = p + 1
        
        # Hiperparâmetros
        tau = 0.01
        alpha0, beta0 = 1, 1
        
        # Inicialização
        beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        sigma2 = np.var(y - X_aug @ beta)
        
        n_total = burnin + n_amostras
        cadeia_beta = np.zeros((n_total, p_aug))
        cadeia_sigma2 = np.zeros(n_total)
        
        for t in range(n_total):
            # Amostrar β | y, σ²
            # Posterior: N(m_n, S_n)
            prior_precision = tau * np.eye(p_aug)
            S_n_inv = X_aug.T @ X_aug / sigma2 + prior_precision
            S_n = np.linalg.inv(S_n_inv)
            m_n = S_n @ X_aug.T @ y / sigma2
            
            beta = np.random.multivariate_normal(m_n, S_n)
            
            # Amostrar σ² | y, β
            residuos = y - X_aug @ beta
            alpha_n = alpha0 + n/2
            beta_n = beta0 + 0.5 * np.sum(residuos**2)
            sigma2 = 1 / np.random.gamma(alpha_n, 1/beta_n)
            
            cadeia_beta[t] = beta
            cadeia_sigma2[t] = sigma2
        
        cadeia_beta = cadeia_beta[burnin:]
        cadeia_sigma2 = cadeia_sigma2[burnin:]
        
        # Predição
        y_pred_mean = X_aug @ np.mean(cadeia_beta, axis=0)
        
        return {
            'cadeia_beta': cadeia_beta,
            'cadeia_sigma2': cadeia_sigma2,
            'beta_mean': np.mean(cadeia_beta, axis=0),
            'beta_std': np.std(cadeia_beta, axis=0),
            'sigma2_mean': np.mean(cadeia_sigma2),
            'y_pred': y_pred_mean,
            'beta_ic': np.percentile(cadeia_beta, [2.5, 97.5], axis=0)
        }
    
    def plotar_resultados(self, resultado_normal, resultado_reg, titulo, salvar=None):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Modelo normal - trace plots
        ax = axes[0, 0]
        ax.plot(resultado_normal['cadeia'][:, 0], lw=0.3)
        ax.set_ylabel('μ'); ax.set_title('Trace μ')
        
        ax = axes[0, 1]
        ax.plot(resultado_normal['cadeia'][:, 1], lw=0.3)
        ax.set_ylabel('σ²'); ax.set_title('Trace σ²')
        
        # Posteriors
        ax = axes[0, 2]
        ax.hist(resultado_normal['cadeia'][:, 0], bins=50, density=True, alpha=0.7)
        ax.axvline(resultado_normal['mu_posterior'], color='r', ls='--')
        ax.set_xlabel('μ'); ax.set_title('Posterior μ')
        
        # Regressão - coeficientes
        ax = axes[1, 0]
        n_coefs = resultado_reg['cadeia_beta'].shape[1]
        bp = ax.boxplot([resultado_reg['cadeia_beta'][:, i] for i in range(n_coefs)])
        ax.axhline(0, color='r', ls='--')
        ax.set_xlabel('Coeficiente'); ax.set_ylabel('Valor')
        ax.set_title('Posterior β')
        
        # σ²
        ax = axes[1, 1]
        ax.hist(resultado_reg['cadeia_sigma2'], bins=50, density=True, alpha=0.7)
        ax.set_xlabel('σ²'); ax.set_title('Posterior σ²')
        
        # Correlação entre parâmetros
        ax = axes[1, 2]
        if resultado_reg['cadeia_beta'].shape[1] >= 2:
            ax.scatter(resultado_reg['cadeia_beta'][:, 0], 
                      resultado_reg['cadeia_beta'][:, 1], s=1, alpha=0.3)
            ax.set_xlabel('β₀'); ax.set_ylabel('β₁')
        ax.set_title('Correlação β₀-β₁')
        
        for ax in axes.flat: ax.grid(True, alpha=0.3)
        plt.suptitle(titulo, fontweight='bold'); plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_28(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 28: AMOSTRADOR DE GIBBS\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        t, f = dados['tempo'][:2000], dados['fluxo'][:2000]
        
        gibbs = AmostradorGibbs()
        res_normal = gibbs.amostrar_normal_hierarquico(f, n_amostras=3000)
        
        X = np.column_stack([t - t[0]])
        res_reg = gibbs.modelo_regressao_bayesiana(X, f, n_amostras=3000)
        
        print(f"    μ = {res_normal['mu_posterior']:.6f} IC: {res_normal['mu_ic']}")
        print(f"    σ² = {res_normal['sigma2_posterior']:.8f}")
        print(f"    β = {res_reg['beta_mean']}")
        
        arq = os.path.join(diretorio_saida, f"gibbs_{nome.replace(' ', '_').lower()}.png")
        gibbs.plotar_resultados(res_normal, res_reg, f"Gibbs - {nome}", arq)
        
        resultados[nome] = {**dados, 'gibbs_normal': res_normal, 'gibbs_reg': res_reg}
    
    plt.close('all')
    print("\nMÓDULO 28 CONCLUÍDO")
    return resultados

__all__ = ['AmostradorGibbs', 'executar_modulo_28']
