"""
Módulo 32: Nested Sampling
Autor: Luiz Tiago Wilcke

Nested Sampling para cálculo de evidência e posterior.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Callable
import os


class NestedSampling:
    """
    Nested Sampling (Skilling 2006).
    Calcula evidência Z = ∫ L(θ) π(θ) dθ via integração em níveis de likelihood.
    """
    
    def __init__(self, log_likelihood: Callable, prior_sampler: Callable,
                 n_params: int, n_live: int = 100):
        self.log_likelihood = log_likelihood
        self.prior_sampler = prior_sampler
        self.n_params = n_params
        self.n_live = n_live
    
    def executar(self, n_iteracoes: int = 1000, tol: float = 0.01) -> Dict:
        """
        Executa Nested Sampling.
        """
        # Pontos vivos iniciais do prior
        pontos_vivos = np.array([self.prior_sampler() for _ in range(self.n_live)])
        log_L_vivos = np.array([self.log_likelihood(p) for p in pontos_vivos])
        
        # Armazenar pontos mortos
        pontos_mortos = []
        log_L_mortos = []
        log_X = 0  # log(prior mass restante)
        
        # log(evidência)
        log_Z = -np.inf
        H = 0  # Informação
        
        for i in range(n_iteracoes):
            # Encontrar pior ponto vivo
            idx_pior = np.argmin(log_L_vivos)
            L_min = log_L_vivos[idx_pior]
            
            # Salvar como morto
            pontos_mortos.append(pontos_vivos[idx_pior].copy())
            log_L_mortos.append(L_min)
            
            # Atualizar evidência
            # log dX ≈ log(exp(-i/n_live) - exp(-(i+1)/n_live))
            log_dX = -i / self.n_live + np.log(1 - np.exp(-1/self.n_live))
            log_Z_new = np.logaddexp(log_Z, L_min + log_dX)
            
            # Informação
            if log_Z_new > -np.inf:
                H = np.exp(L_min + log_dX - log_Z_new) * L_min + \
                    np.exp(log_Z - log_Z_new) * (H + log_Z) - log_Z_new
            
            log_Z = log_Z_new
            
            # Substituir pior ponto
            # Amostrar do prior acima de L_min
            aceito = False
            for _ in range(1000):
                novo_ponto = self.prior_sampler()
                novo_log_L = self.log_likelihood(novo_ponto)
                if novo_log_L > L_min:
                    pontos_vivos[idx_pior] = novo_ponto
                    log_L_vivos[idx_pior] = novo_log_L
                    aceito = True
                    break
            
            if not aceito:
                # Perturbar ponto existente
                idx_ref = np.random.randint(self.n_live)
                pontos_vivos[idx_pior] = pontos_vivos[idx_ref] + 0.01 * np.random.randn(self.n_params)
                log_L_vivos[idx_pior] = self.log_likelihood(pontos_vivos[idx_pior])
            
            # Critério de parada
            log_Z_restante = np.max(log_L_vivos) - (i+1)/self.n_live
            if log_Z_restante < log_Z + np.log(tol):
                break
        
        # Adicionar pontos vivos finais
        for p, ll in zip(pontos_vivos, log_L_vivos):
            pontos_mortos.append(p)
            log_L_mortos.append(ll)
        
        # Calcular pesos posteriori
        pontos_mortos = np.array(pontos_mortos)
        log_L_mortos = np.array(log_L_mortos)
        n_mortos = len(log_L_mortos)
        
        log_pesos = log_L_mortos + np.linspace(0, -n_mortos/self.n_live, n_mortos) - log_Z
        pesos = np.exp(log_pesos - np.max(log_pesos))
        pesos /= np.sum(pesos)
        
        # Posterior
        posterior_mean = np.average(pontos_mortos, weights=pesos, axis=0)
        posterior_std = np.sqrt(np.average((pontos_mortos - posterior_mean)**2, weights=pesos, axis=0))
        
        return {
            'log_Z': log_Z,
            'Z': np.exp(log_Z),
            'H': H,
            'n_iteracoes': i + 1,
            'pontos': pontos_mortos,
            'log_L': log_L_mortos,
            'pesos': pesos,
            'posterior_mean': posterior_mean,
            'posterior_std': posterior_std
        }
    
    def plotar_resultados(self, resultado, nomes_params=None, titulo="", salvar=None):
        if nomes_params is None:
            nomes_params = [f'θ{i}' for i in range(self.n_params)]
        
        n_params = min(self.n_params, 4)
        fig, axes = plt.subplots(2, n_params, figsize=(4*n_params, 8))
        if n_params == 1:
            axes = axes.reshape(2, 1)
        
        # Posteriors
        for i in range(n_params):
            ax = axes[0, i]
            ax.hist(resultado['pontos'][:, i], weights=resultado['pesos'], bins=50, density=True, alpha=0.7)
            ax.axvline(resultado['posterior_mean'][i], color='r', ls='--')
            ax.set_xlabel(nomes_params[i])
            ax.set_title(f"{nomes_params[i]} = {resultado['posterior_mean'][i]:.4f}")
        
        # Likelihood vs iteração
        ax = axes[1, 0]
        ax.plot(resultado['log_L'], lw=0.5)
        ax.set_xlabel('Iteração'); ax.set_ylabel('log L')
        ax.set_title('Evolução da Likelihood')
        
        # Pesos
        ax = axes[1, 1] if n_params > 1 else axes[1, 0]
        ax.plot(resultado['pesos'], lw=0.5)
        ax.set_xlabel('Ponto'); ax.set_ylabel('Peso')
        ax.set_title('Pesos Posteriori')
        
        # Info
        if n_params > 2:
            ax = axes[1, 2]
            ax.axis('off')
            info = f"""Nested Sampling
            
log Z = {resultado['log_Z']:.2f}
Z = {resultado['Z']:.2e}
H = {resultado['H']:.2f} nats
n_iter = {resultado['n_iteracoes']}"""
            ax.text(0.1, 0.5, info, fontsize=12, family='monospace', va='center')
        
        for ax in axes.flat:
            if ax.has_data(): ax.grid(True, alpha=0.3)
        plt.suptitle(titulo, fontweight='bold'); plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_32(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 32: NESTED SAMPLING\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        f = dados['fluxo'][:500]
        n = len(f)
        
        def log_lik(theta):
            mu, log_sigma = theta
            sigma = np.exp(log_sigma)
            return -n*np.log(sigma) - 0.5*np.sum((f - mu)**2)/sigma**2
        
        def prior_sampler():
            mu = np.random.uniform(f.min(), f.max())
            log_sigma = np.random.uniform(-10, 0)
            return np.array([mu, log_sigma])
        
        ns = NestedSampling(log_lik, prior_sampler, n_params=2, n_live=50)
        resultado = ns.executar(n_iteracoes=500)
        
        print(f"    log Z = {resultado['log_Z']:.2f}")
        print(f"    H = {resultado['H']:.2f} nats")
        print(f"    μ = {resultado['posterior_mean'][0]:.6f} ± {resultado['posterior_std'][0]:.6f}")
        print(f"    σ = {np.exp(resultado['posterior_mean'][1]):.6f}")
        
        arq = os.path.join(diretorio_saida, f"nested_{nome.replace(' ', '_').lower()}.png")
        ns.plotar_resultados(resultado, ['μ', 'log(σ)'], f"Nested Sampling - {nome}", arq)
        
        resultados[nome] = {**dados, 'nested_sampling': resultado}
    
    plt.close('all')
    print("\nMÓDULO 32 CONCLUÍDO")
    return resultados

__all__ = ['NestedSampling', 'executar_modulo_32']
