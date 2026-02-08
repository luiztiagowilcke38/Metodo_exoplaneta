"""
Módulo 29: Hamiltonian Monte Carlo (HMC)
Autor: Luiz Tiago Wilcke

HMC usa dinâmica Hamiltoniana para propostas eficientes em alta dimensão.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Callable, Optional
import os


class HamiltonianMonteCarlo:
    """
    HMC: usa gradiente para explorar o espaço de parâmetros.
    H(θ, p) = U(θ) + K(p)
    onde U(θ) = -log π(θ) e K(p) = p'M⁻¹p/2
    """
    
    def __init__(self, log_posterior: Callable, grad_log_posterior: Callable,
                 n_params: int):
        self.log_posterior = log_posterior
        self.grad_log_posterior = grad_log_posterior
        self.n_params = n_params
    
    def leapfrog(self, theta: np.ndarray, momentum: np.ndarray,
                  epsilon: float, L: int) -> tuple:
        """
        Integrador Leapfrog para dinâmica Hamiltoniana.
        """
        theta = theta.copy()
        momentum = momentum.copy()
        
        # Meio passo para momentum
        grad = self.grad_log_posterior(theta)
        momentum = momentum + 0.5 * epsilon * grad
        
        # L passos completos
        for _ in range(L - 1):
            theta = theta + epsilon * momentum
            grad = self.grad_log_posterior(theta)
            momentum = momentum + epsilon * grad
        
        # Último passo completo para theta
        theta = theta + epsilon * momentum
        
        # Meio passo final para momentum
        grad = self.grad_log_posterior(theta)
        momentum = momentum + 0.5 * epsilon * grad
        
        # Negar momentum (para reversibilidade)
        momentum = -momentum
        
        return theta, momentum
    
    def executar(self, theta_inicial: np.ndarray, n_amostras: int = 5000,
                  epsilon: float = 0.01, L: int = 20,
                  burnin: int = 1000) -> Dict:
        """
        Executa HMC.
        """
        n_total = burnin + n_amostras
        cadeia = np.zeros((n_total, self.n_params))
        log_probs = np.zeros(n_total)
        
        theta = theta_inicial.copy()
        aceitos = 0
        
        for i in range(n_total):
            # Amostrar momentum
            momentum = np.random.randn(self.n_params)
            
            # Hamiltoniano atual
            U = -self.log_posterior(theta)
            K = 0.5 * np.sum(momentum**2)
            H_atual = U + K
            
            # Leapfrog
            theta_proposta, momentum_proposta = self.leapfrog(theta, momentum, epsilon, L)
            
            # Hamiltoniano proposto
            U_proposta = -self.log_posterior(theta_proposta)
            K_proposta = 0.5 * np.sum(momentum_proposta**2)
            H_proposta = U_proposta + K_proposta
            
            # Aceitar/rejeitar via Metropolis
            delta_H = H_proposta - H_atual
            if np.log(np.random.rand()) < -delta_H:
                theta = theta_proposta
                aceitos += 1
            
            cadeia[i] = theta
            log_probs[i] = self.log_posterior(theta)
        
        cadeia_final = cadeia[burnin:]
        self.cadeia = cadeia_final
        
        return {
            'cadeia': cadeia_final,
            'log_probs': log_probs[burnin:],
            'taxa_aceitacao': aceitos / n_total,
            'medias': np.mean(cadeia_final, axis=0),
            'stds': np.std(cadeia_final, axis=0)
        }
    
    def diagnosticos(self) -> Dict:
        """Diagnósticos de convergência."""
        n, p = self.cadeia.shape
        
        ess = np.zeros(p)
        for j in range(p):
            x = self.cadeia[:, j] - np.mean(self.cadeia[:, j])
            acf = np.correlate(x, x, mode='full')[len(x)-1:len(x)+50]
            acf = acf / acf[0]
            idx_neg = np.where(acf < 0)[0]
            if len(idx_neg) > 0:
                acf = acf[:idx_neg[0]]
            tau = 1 + 2 * np.sum(acf[1:])
            ess[j] = n / tau
        
        return {'ess': ess}
    
    def plotar_resultados(self, nomes_params=None, titulo="", salvar=None):
        if nomes_params is None:
            nomes_params = [f'θ{i}' for i in range(self.n_params)]
        
        n_params = min(self.n_params, 4)
        fig, axes = plt.subplots(n_params, 2, figsize=(12, 3*n_params))
        if n_params == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_params):
            ax = axes[i, 0]
            ax.plot(self.cadeia[:, i], lw=0.3)
            ax.set_ylabel(nomes_params[i])
            if i == 0: ax.set_title('Trace Plot')
            
            ax = axes[i, 1]
            ax.hist(self.cadeia[:, i], bins=50, density=True, alpha=0.7)
            ax.axvline(np.mean(self.cadeia[:, i]), color='r', ls='--')
            if i == 0: ax.set_title('Posterior')
        
        for ax in axes.flat: ax.grid(True, alpha=0.3)
        plt.suptitle(titulo, fontweight='bold'); plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_29(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 29: HAMILTONIAN MONTE CARLO\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        f = dados['fluxo'][:1000]
        n = len(f)
        s_n = np.sum(f)
        ss_n = np.sum(f**2)
        
        def log_posterior(theta):
            mu, log_sigma = theta
            sigma = np.exp(log_sigma)
            log_prior = -0.5 * mu**2 / 100 - log_sigma
            log_lik = -n*log_sigma - 0.5*(ss_n - 2*mu*s_n + n*mu**2) / sigma**2
            return log_prior + log_lik
        
        def grad_log_posterior(theta):
            mu, log_sigma = theta
            sigma = np.exp(log_sigma)
            sigma2 = sigma**2
            
            grad_mu = -mu/100 + (s_n - n*mu) / sigma2
            grad_log_sigma = -1 + (ss_n - 2*mu*s_n + n*mu**2) / sigma2 - n
            
            return np.array([grad_mu, grad_log_sigma])
        
        hmc = HamiltonianMonteCarlo(log_posterior, grad_log_posterior, n_params=2)
        resultado = hmc.executar(np.array([np.mean(f), np.log(np.std(f))]),
                                n_amostras=2000, epsilon=0.001, L=20)
        
        diag = hmc.diagnosticos()
        
        print(f"    Taxa de aceitação: {resultado['taxa_aceitacao']:.2%}")
        print(f"    μ = {resultado['medias'][0]:.6f}")
        print(f"    σ = {np.exp(resultado['medias'][1]):.6f}")
        print(f"    ESS: {diag['ess']}")
        
        arq = os.path.join(diretorio_saida, f"hmc_{nome.replace(' ', '_').lower()}.png")
        hmc.plotar_resultados(['μ', 'log(σ)'], f"HMC - {nome}", arq)
        
        resultados[nome] = {**dados, 'hmc': resultado}
    
    plt.close('all')
    print("\nMÓDULO 29 CONCLUÍDO")
    return resultados

__all__ = ['HamiltonianMonteCarlo', 'executar_modulo_29']
