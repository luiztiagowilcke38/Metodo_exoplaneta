"""
Módulo 27: MCMC Metropolis-Hastings
Autor: Luiz Tiago Wilcke

Algoritmo Metropolis-Hastings para amostragem de distribuições posteriori.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Callable, Optional
import os


class MetropolisHastings:
    """
    MCMC via Metropolis-Hastings.
    Aceita proposta θ* com probabilidade min(1, π(θ*)/π(θ_t) × q(θ_t|θ*)/q(θ*|θ_t))
    """
    
    def __init__(self, log_posterior: Callable, n_params: int):
        self.log_posterior = log_posterior
        self.n_params = n_params
        self.cadeia = None
        self.log_probs = None
    
    def proposta_normal(self, theta_atual: np.ndarray, 
                         escalas: np.ndarray) -> np.ndarray:
        """Proposta simétrica Normal: q(θ*|θ) = N(θ, Σ)"""
        return theta_atual + escalas * np.random.randn(self.n_params)
    
    def executar(self, theta_inicial: np.ndarray, n_amostras: int = 10000,
                  escalas: np.ndarray = None, burnin: int = 1000,
                  thin: int = 1) -> Dict:
        """
        Executa o algoritmo Metropolis-Hastings.
        """
        if escalas is None:
            escalas = np.ones(self.n_params) * 0.1
        
        n_total = burnin + n_amostras * thin
        cadeia = np.zeros((n_total, self.n_params))
        log_probs = np.zeros(n_total)
        
        theta = theta_inicial.copy()
        log_prob = self.log_posterior(theta)
        aceitos = 0
        
        for i in range(n_total):
            # Proposta
            theta_proposta = self.proposta_normal(theta, escalas)
            log_prob_proposta = self.log_posterior(theta_proposta)
            
            # Razão de aceitação (log)
            log_alpha = log_prob_proposta - log_prob
            
            # Aceitar/rejeitar
            if np.log(np.random.rand()) < log_alpha:
                theta = theta_proposta
                log_prob = log_prob_proposta
                aceitos += 1
            
            cadeia[i] = theta
            log_probs[i] = log_prob
        
        # Remover burn-in e aplicar thinning
        cadeia_final = cadeia[burnin::thin]
        log_probs_final = log_probs[burnin::thin]
        
        self.cadeia = cadeia_final
        self.log_probs = log_probs_final
        
        taxa_aceitacao = aceitos / n_total
        
        return {
            'cadeia': cadeia_final,
            'log_probs': log_probs_final,
            'taxa_aceitacao': taxa_aceitacao,
            'medias': np.mean(cadeia_final, axis=0),
            'stds': np.std(cadeia_final, axis=0),
            'n_amostras': len(cadeia_final)
        }
    
    def diagnosticos(self) -> Dict:
        """Diagnósticos de convergência."""
        n, p = self.cadeia.shape
        
        # Effective Sample Size (ESS) via autocorrelação
        ess = np.zeros(p)
        for j in range(p):
            x = self.cadeia[:, j]
            x_cent = x - np.mean(x)
            
            # Autocorrelação
            acf = np.correlate(x_cent, x_cent, mode='full')
            acf = acf[len(acf)//2:]
            acf = acf / acf[0]
            
            # ESS = n / (1 + 2*Σρ_k)
            # Truncar quando autocorrelação fica negativa
            idx_neg = np.where(acf < 0)[0]
            if len(idx_neg) > 0:
                acf = acf[:idx_neg[0]]
            
            tau = 1 + 2 * np.sum(acf[1:])
            ess[j] = n / tau
        
        # Gelman-Rubin (R-hat) - requer múltiplas cadeias, aqui aproximamos
        # dividindo a cadeia em 2
        meio = n // 2
        cadeia1 = self.cadeia[:meio]
        cadeia2 = self.cadeia[meio:]
        
        r_hat = np.zeros(p)
        for j in range(p):
            B = (meio / 2) * (np.mean(cadeia1[:, j]) - np.mean(cadeia2[:, j]))**2
            W = 0.5 * (np.var(cadeia1[:, j]) + np.var(cadeia2[:, j]))
            var_hat = (1 - 1/meio) * W + (1/meio) * B
            r_hat[j] = np.sqrt(var_hat / W) if W > 0 else np.nan
        
        return {
            'ess': ess,
            'r_hat': r_hat,
            'convergiu': np.all(r_hat < 1.1) if not np.any(np.isnan(r_hat)) else True
        }
    
    def plotar_diagnosticos(self, nomes_params: list = None, titulo: str = "",
                             salvar: Optional[str] = None) -> plt.Figure:
        if nomes_params is None:
            nomes_params = [f'θ{i}' for i in range(self.n_params)]
        
        n_params = min(self.n_params, 4)
        fig, axes = plt.subplots(n_params, 3, figsize=(14, 3*n_params))
        if n_params == 1:
            axes = axes.reshape(1, -1)
        
        diag = self.diagnosticos()
        
        for i in range(n_params):
            # Trace plot
            ax = axes[i, 0]
            ax.plot(self.cadeia[:, i], lw=0.3)
            ax.set_ylabel(nomes_params[i])
            if i == 0: ax.set_title('Trace Plot')
            if i == n_params - 1: ax.set_xlabel('Iteração')
            
            # Histograma
            ax = axes[i, 1]
            ax.hist(self.cadeia[:, i], bins=50, density=True, alpha=0.7)
            ax.axvline(np.mean(self.cadeia[:, i]), color='r', ls='--')
            if i == 0: ax.set_title('Posterior')
            
            # Autocorrelação
            ax = axes[i, 2]
            x = self.cadeia[:, i] - np.mean(self.cadeia[:, i])
            acf = np.correlate(x, x, mode='full')[len(x)-1:len(x)+50]
            acf = acf / acf[0]
            ax.bar(range(len(acf)), acf, alpha=0.7)
            ax.axhline(1.96/np.sqrt(len(self.cadeia)), color='r', ls='--')
            ax.axhline(-1.96/np.sqrt(len(self.cadeia)), color='r', ls='--')
            if i == 0: ax.set_title(f'ACF (ESS={diag["ess"][i]:.0f})')
        
        for ax in axes.flat: ax.grid(True, alpha=0.3)
        plt.suptitle(titulo, fontweight='bold'); plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_27(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 27: MCMC METROPOLIS-HASTINGS\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        f = dados['fluxo'][:2000]
        
        # Posterior: N(μ|dados) × IG(σ²|dados)
        def log_posterior(theta):
            mu, log_sigma = theta
            sigma = np.exp(log_sigma)
            
            # Prior
            log_prior = -0.5 * mu**2 / 100 - log_sigma  # N(0,100), Jeffreys para σ
            
            # Likelihood
            n = len(f)
            log_lik = -n*log_sigma - 0.5*np.sum((f - mu)**2) / sigma**2
            
            return log_prior + log_lik
        
        mh = MetropolisHastings(log_posterior, n_params=2)
        resultado = mh.executar(np.array([np.mean(f), np.log(np.std(f))]), 
                               n_amostras=5000, escalas=np.array([0.001, 0.01]))
        
        diag = mh.diagnosticos()
        
        print(f"    Taxa de aceitação: {resultado['taxa_aceitacao']:.2%}")
        print(f"    μ = {resultado['medias'][0]:.6f} ± {resultado['stds'][0]:.6f}")
        print(f"    σ = {np.exp(resultado['medias'][1]):.6f}")
        print(f"    ESS: {diag['ess']}")
        print(f"    R-hat: {diag['r_hat']}")
        
        arq = os.path.join(diretorio_saida, f"mcmc_{nome.replace(' ', '_').lower()}.png")
        mh.plotar_diagnosticos(['μ', 'log(σ)'], f"MCMC - {nome}", arq)
        
        resultados[nome] = {**dados, 'mcmc': resultado, 'mcmc_diag': diag}
    
    plt.close('all')
    print("\nMÓDULO 27 CONCLUÍDO")
    return resultados

__all__ = ['MetropolisHastings', 'executar_modulo_27']
