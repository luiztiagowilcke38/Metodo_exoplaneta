"""
Módulo 26: Inferência Bayesiana
Autor: Luiz Tiago Wilcke

Fundamentos de inferência Bayesiana: P(θ|D) ∝ P(D|θ)P(θ)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gammaln, betaln
from typing import Dict, Optional, Callable
import os


class InferenciaBayesiana:
    """
    Inferência Bayesiana com priors conjugados e não-conjugados.
    Posterior ∝ Likelihood × Prior
    """
    
    def __init__(self):
        pass
    
    def normal_conjugada(self, dados: np.ndarray, mu0: float = 0, 
                          tau0: float = 0.01, alpha0: float = 1,
                          beta0: float = 1) -> Dict:
        """
        Prior conjugado Normal-Gamma para dados normais.
        Dados: x ~ N(μ, σ²)
        Prior: μ|σ² ~ N(μ₀, σ²/τ₀), σ² ~ IG(α₀, β₀)
        Posterior: Normal-Gamma
        """
        n = len(dados)
        x_bar = np.mean(dados)
        s2 = np.var(dados, ddof=1) if n > 1 else 1.0
        
        # Parâmetros posteriori
        tau_n = tau0 + n
        mu_n = (tau0 * mu0 + n * x_bar) / tau_n
        alpha_n = alpha0 + n / 2
        
        ss = np.sum((dados - x_bar) ** 2)
        beta_n = beta0 + 0.5 * ss + 0.5 * tau0 * n * (x_bar - mu0)**2 / tau_n
        
        # Marginal posterior para μ (t de Student)
        df = 2 * alpha_n
        scale = np.sqrt(beta_n / (alpha_n * tau_n))
        
        # Evidência (log-verossimilhança marginal)
        log_evidence = (
            gammaln(alpha_n) - gammaln(alpha0) +
            alpha0 * np.log(beta0) - alpha_n * np.log(beta_n) +
            0.5 * np.log(tau0 / tau_n) -
            n/2 * np.log(2 * np.pi)
        )
        
        return {
            'mu_posterior': mu_n,
            'tau_posterior': tau_n,
            'alpha_posterior': alpha_n,
            'beta_posterior': beta_n,
            'df': df,
            'scale': scale,
            'log_evidence': log_evidence,
            'ic_mu_95': (mu_n - 1.96*scale, mu_n + 1.96*scale),
            'sigma2_mean': beta_n / (alpha_n - 1) if alpha_n > 1 else np.nan
        }
    
    def beta_binomial(self, sucessos: int, n: int, 
                       alpha0: float = 1, beta0: float = 1) -> Dict:
        """
        Prior Beta-Binomial para proporções.
        Dados: k ~ Binomial(n, p)
        Prior: p ~ Beta(α₀, β₀)
        Posterior: p|k ~ Beta(α₀ + k, β₀ + n - k)
        """
        alpha_n = alpha0 + sucessos
        beta_n = beta0 + n - sucessos
        
        # Média e variância posteriori
        media = alpha_n / (alpha_n + beta_n)
        variancia = (alpha_n * beta_n) / ((alpha_n + beta_n)**2 * (alpha_n + beta_n + 1))
        
        # Intervalo de credibilidade
        ic_lower = stats.beta.ppf(0.025, alpha_n, beta_n)
        ic_upper = stats.beta.ppf(0.975, alpha_n, beta_n)
        
        # Log-evidência
        log_evidence = (
            betaln(alpha_n, beta_n) - betaln(alpha0, beta0) +
            gammaln(n + 1) - gammaln(sucessos + 1) - gammaln(n - sucessos + 1)
        )
        
        return {
            'alpha_posterior': alpha_n,
            'beta_posterior': beta_n,
            'media': media,
            'variancia': variancia,
            'ic_95': (ic_lower, ic_upper),
            'log_evidence': log_evidence
        }
    
    def regressao_bayesiana(self, X: np.ndarray, y: np.ndarray,
                             tau_prior: float = 0.01) -> Dict:
        """
        Regressão linear Bayesiana com prior N(0, τ⁻¹I).
        Posterior: β|y,X,σ² ~ N(m_n, S_n)
        """
        n, p = X.shape if X.ndim > 1 else (len(X), 1)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Adicionar intercepto
        X_aug = np.column_stack([np.ones(n), X])
        p_aug = p + 1
        
        # Estimativa de σ² via MLE
        beta_ols = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        residuos = y - X_aug @ beta_ols
        sigma2 = np.sum(residuos ** 2) / (n - p_aug)
        
        # Prior: β ~ N(0, σ²/τ I)
        # Posterior: β ~ N(m_n, S_n)
        prior_precision = tau_prior * np.eye(p_aug)
        
        S_n_inv = X_aug.T @ X_aug / sigma2 + prior_precision
        S_n = np.linalg.inv(S_n_inv)
        m_n = S_n @ X_aug.T @ y / sigma2
        
        # Predição
        y_pred = X_aug @ m_n
        
        # Incerteza nas predições
        pred_var = sigma2 + np.sum((X_aug @ S_n) * X_aug, axis=1)
        pred_std = np.sqrt(pred_var)
        
        # Log-evidência (marginal likelihood)
        log_evidence = (
            -n/2 * np.log(2 * np.pi * sigma2) -
            0.5 * np.sum(residuos ** 2) / sigma2 +
            0.5 * np.log(np.linalg.det(S_n)) +
            0.5 * p_aug * np.log(tau_prior) -
            0.5 * m_n @ prior_precision @ m_n
        )
        
        return {
            'beta_mean': m_n,
            'beta_cov': S_n,
            'sigma2': sigma2,
            'y_pred': y_pred,
            'pred_std': pred_std,
            'log_evidence': log_evidence,
            'ic_beta': [(m_n[i] - 1.96*np.sqrt(S_n[i,i]), m_n[i] + 1.96*np.sqrt(S_n[i,i])) 
                       for i in range(p_aug)]
        }
    
    def plotar_resultados(self, dados, resultado_normal, resultado_reg, titulo, salvar=None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Posterior de μ
        ax = axes[0, 0]
        x = np.linspace(resultado_normal['mu_posterior'] - 4*resultado_normal['scale'],
                       resultado_normal['mu_posterior'] + 4*resultado_normal['scale'], 200)
        pdf = stats.t.pdf(x, resultado_normal['df'], 
                         resultado_normal['mu_posterior'], resultado_normal['scale'])
        ax.plot(x, pdf, 'b-', lw=2)
        ax.axvline(resultado_normal['mu_posterior'], color='r', ls='--', label='MAP')
        ax.axvline(np.mean(dados), color='g', ls=':', label='MLE')
        ax.fill_between(x, pdf, where=(x >= resultado_normal['ic_mu_95'][0]) & 
                       (x <= resultado_normal['ic_mu_95'][1]), alpha=0.3)
        ax.set_xlabel('μ'); ax.set_ylabel('Densidade')
        ax.set_title('Posterior de μ'); ax.legend()
        
        # Histograma dos dados
        ax = axes[0, 1]
        ax.hist(dados, bins=50, density=True, alpha=0.7, label='Dados')
        x = np.linspace(dados.min(), dados.max(), 100)
        pdf = stats.norm.pdf(x, resultado_normal['mu_posterior'], 
                            np.sqrt(resultado_normal['sigma2_mean']))
        ax.plot(x, pdf, 'r-', lw=2, label='Posterior Preditiva')
        ax.set_xlabel('Valor'); ax.set_ylabel('Densidade')
        ax.set_title('Dados vs Posterior Preditiva'); ax.legend()
        
        # Regressão Bayesiana
        ax = axes[1, 0]
        t = np.arange(len(resultado_reg['y_pred']))
        ax.scatter(t, dados, s=1, alpha=0.3, c='gray', label='Dados')
        ax.plot(t, resultado_reg['y_pred'], 'b-', lw=1, label='Média Posterior')
        ax.fill_between(t, 
                       resultado_reg['y_pred'] - 2*resultado_reg['pred_std'],
                       resultado_reg['y_pred'] + 2*resultado_reg['pred_std'],
                       alpha=0.2, label='IC 95%')
        ax.set_xlabel('t'); ax.set_ylabel('y')
        ax.set_title('Regressão Bayesiana'); ax.legend()
        
        # Posterior dos coeficientes
        ax = axes[1, 1]
        beta = resultado_reg['beta_mean']
        std = np.sqrt(np.diag(resultado_reg['beta_cov']))
        x = np.arange(len(beta))
        ax.bar(x, beta, yerr=1.96*std, capsize=5, alpha=0.7)
        ax.axhline(0, color='r', ls='--')
        ax.set_xlabel('Coeficiente'); ax.set_ylabel('Valor')
        ax.set_title('Posterior dos β (± 95% IC)')
        
        for ax in axes.flat: ax.grid(True, alpha=0.3)
        plt.suptitle(titulo, fontweight='bold'); plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_26(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 26: INFERÊNCIA BAYESIANA\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        f = dados['fluxo']
        t = dados['tempo']
        
        bayes = InferenciaBayesiana()
        res_normal = bayes.normal_conjugada(f)
        
        X = np.column_stack([t - t[0], (t - t[0])**2])
        res_reg = bayes.regressao_bayesiana(X, f)
        
        print(f"    μ posterior: {res_normal['mu_posterior']:.6f}")
        print(f"    IC 95%: [{res_normal['ic_mu_95'][0]:.6f}, {res_normal['ic_mu_95'][1]:.6f}]")
        print(f"    Log-evidência: {res_normal['log_evidence']:.2f}")
        
        arq = os.path.join(diretorio_saida, f"bayes_{nome.replace(' ', '_').lower()}.png")
        bayes.plotar_resultados(f, res_normal, res_reg, f"Inferência Bayesiana - {nome}", arq)
        
        resultados[nome] = {**dados, 'bayes_normal': res_normal, 'bayes_reg': res_reg}
    
    plt.close('all')
    print("\nMÓDULO 26 CONCLUÍDO")
    return resultados

__all__ = ['InferenciaBayesiana', 'executar_modulo_26']
