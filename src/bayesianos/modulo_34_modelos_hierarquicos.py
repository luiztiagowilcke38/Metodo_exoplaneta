"""
Módulo 34: Modelos Hierárquicos Bayesianos
Sistema Estatístico para Detecção de Exoplanetas
Autor: Luiz Tiago Wilcke

Modelos hierárquicos para análise de múltiplas curvas de luz.
Permite combinar informação de diferentes estrelas.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os


class ModeloHierarquico:
    """
    Modelo Hierárquico Bayesiano para exoplanetas.
    
    Estrutura:
        Hiperpriors: μ_global, σ_global
        Priors por grupo: θ_j ~ N(μ_global, σ_global²)
        Likelihood: y_ij ~ N(f(x_ij, θ_j), σ²)
    
    Permite estimar parâmetros comuns a todos os sistemas
    enquanto modela variação individual.
    """
    
    def __init__(self, n_grupos: int):
        """
        Inicializa modelo hierárquico.
        
        Parâmetros:
            n_grupos: Número de grupos (estrelas/curvas de luz)
        """
        self.n_grupos = n_grupos
        self.hiperparametros = {}
        self.parametros_grupo = {}
    
    def gibbs_hierarquico(self, dados_grupos: List[np.ndarray], 
                           n_amostras: int = 3000,
                           burnin: int = 1000) -> Dict:
        """
        Amostragem de Gibbs para modelo hierárquico Normal.
        
        Modelo:
            y_ij | μ_j, σ² ~ N(μ_j, σ²)
            μ_j | μ, τ² ~ N(μ, τ²)
            μ ~ N(0, 100)
            σ² ~ IG(1, 1)
            τ² ~ IG(1, 1)
        """
        J = len(dados_grupos)
        n_j = [len(d) for d in dados_grupos]
        y_bar_j = [np.mean(d) for d in dados_grupos]
        
        # Inicialização
        mu_j = np.array(y_bar_j)
        mu = np.mean(mu_j)
        sigma2 = np.mean([np.var(d) for d in dados_grupos])
        tau2 = np.var(mu_j)
        
        n_total = burnin + n_amostras
        cadeia_mu = np.zeros(n_total)
        cadeia_tau2 = np.zeros(n_total)
        cadeia_sigma2 = np.zeros(n_total)
        cadeia_mu_j = np.zeros((n_total, J))
        
        for t in range(n_total):
            # Amostrar μ_j | resto (para cada grupo)
            for j in range(J):
                # Posterior: N(m_j, v_j)
                precisao_dados = n_j[j] / sigma2
                precisao_prior = 1 / tau2
                v_j = 1 / (precisao_dados + precisao_prior)
                m_j = v_j * (precisao_dados * y_bar_j[j] + precisao_prior * mu)
                mu_j[j] = np.random.normal(m_j, np.sqrt(v_j))
            
            # Amostrar μ | μ_j, τ²
            precisao_prior = 1 / 100
            precisao_dados = J / tau2
            v_mu = 1 / (precisao_prior + precisao_dados)
            m_mu = v_mu * precisao_dados * np.mean(mu_j)
            mu = np.random.normal(m_mu, np.sqrt(v_mu))
            
            # Amostrar τ² | μ, μ_j (Inverse Gamma)
            alpha_tau = 1 + J/2
            beta_tau = 1 + 0.5 * np.sum((mu_j - mu)**2)
            tau2 = 1 / np.random.gamma(alpha_tau, 1/beta_tau)
            
            # Amostrar σ² | resto (Inverse Gamma)
            ss_total = sum([np.sum((d - mu_j[j])**2) for j, d in enumerate(dados_grupos)])
            n_total_obs = sum(n_j)
            alpha_sigma = 1 + n_total_obs/2
            beta_sigma = 1 + 0.5 * ss_total
            sigma2 = 1 / np.random.gamma(alpha_sigma, 1/beta_sigma)
            
            cadeia_mu[t] = mu
            cadeia_tau2[t] = tau2
            cadeia_sigma2[t] = sigma2
            cadeia_mu_j[t] = mu_j.copy()
        
        # Remover burn-in
        cadeia_mu = cadeia_mu[burnin:]
        cadeia_tau2 = cadeia_tau2[burnin:]
        cadeia_sigma2 = cadeia_sigma2[burnin:]
        cadeia_mu_j = cadeia_mu_j[burnin:]
        
        # Calcular shrinkage (contração para a média global)
        shrinkage = np.zeros(J)
        for j in range(J):
            var_posterior = np.var(cadeia_mu_j[:, j])
            var_dados = sigma2 / n_j[j]
            shrinkage[j] = 1 - var_posterior / var_dados if var_dados > 0 else 0
        
        return {
            'mu_global': np.mean(cadeia_mu),
            'mu_global_std': np.std(cadeia_mu),
            'tau2': np.mean(cadeia_tau2),
            'sigma2': np.mean(cadeia_sigma2),
            'mu_j': np.mean(cadeia_mu_j, axis=0),
            'mu_j_std': np.std(cadeia_mu_j, axis=0),
            'shrinkage': shrinkage,
            'icc': np.mean(cadeia_tau2) / (np.mean(cadeia_tau2) + np.mean(cadeia_sigma2)),
            'cadeia_mu': cadeia_mu,
            'cadeia_mu_j': cadeia_mu_j
        }
    
    def plotar_resultados(self, dados_grupos: List[np.ndarray], resultado: Dict,
                          titulo: str, salvar: Optional[str] = None) -> plt.Figure:
        """Visualiza resultados do modelo hierárquico."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        J = len(dados_grupos)
        
        # Médias por grupo (shrinkage)
        ax = axes[0, 0]
        y_bar = [np.mean(d) for d in dados_grupos]
        mu_j = resultado['mu_j']
        mu_global = resultado['mu_global']
        
        ax.scatter(range(J), y_bar, s=50, c='blue', label='Média amostral', zorder=3)
        ax.scatter(range(J), mu_j, s=50, c='red', marker='x', label='Média posterior', zorder=3)
        ax.axhline(mu_global, color='green', ls='--', lw=2, label=f'μ global = {mu_global:.4f}')
        
        # Conectar com linhas (shrinkage visível)
        for j in range(J):
            ax.plot([j, j], [y_bar[j], mu_j[j]], 'k-', alpha=0.3)
        
        ax.set_xlabel('Grupo'); ax.set_ylabel('Média')
        ax.set_title('Shrinkage para Média Global'); ax.legend()
        
        # Posterior de μ global
        ax = axes[0, 1]
        ax.hist(resultado['cadeia_mu'], bins=50, density=True, alpha=0.7)
        ax.axvline(resultado['mu_global'], color='r', ls='--', lw=2)
        ax.set_xlabel('μ global'); ax.set_ylabel('Densidade')
        ax.set_title('Distribuição Posterior de μ')
        
        # Shrinkage por grupo
        ax = axes[1, 0]
        ax.bar(range(J), resultado['shrinkage'], alpha=0.7)
        ax.set_xlabel('Grupo'); ax.set_ylabel('Shrinkage')
        ax.set_title('Coeficiente de Shrinkage por Grupo')
        ax.axhline(0.5, color='r', ls='--')
        
        # Informações
        ax = axes[1, 1]
        ax.axis('off')
        info = f"""Modelo Hierárquico Bayesiano

Hiperparâmetros:
  μ global = {resultado['mu_global']:.6f} ± {resultado['mu_global_std']:.6f}
  τ² (entre grupos) = {resultado['tau2']:.6f}
  σ² (dentro grupos) = {resultado['sigma2']:.6f}

Métricas:
  ICC = {resultado['icc']:.4f}
  Shrinkage médio = {np.mean(resultado['shrinkage']):.4f}
  Número de grupos = {J}

Interpretação:
  ICC alto → grupos muito diferentes
  Shrinkage alto → dados puxados para média global"""
        ax.text(0.1, 0.5, info, fontsize=11, family='monospace', va='center')
        
        for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]: ax.grid(True, alpha=0.3)
        plt.suptitle(titulo, fontweight='bold'); plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_34(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    """Executa análise hierárquica Bayesiana."""
    print("=" * 60)
    print("MÓDULO 34: MODELOS HIERÁRQUICOS BAYESIANOS")
    print("Sistema de Detecção de Exoplanetas")
    print("Autor: Luiz Tiago Wilcke")
    print("=" * 60)
    
    os.makedirs(diretorio_saida, exist_ok=True)
    resultados = {}
    
    # Combinar todas as curvas de luz
    todas_curvas = list(dados_entrada.values())
    n_grupos = len(todas_curvas)
    
    if n_grupos > 1:
        dados_grupos = [d['fluxo'][:500] for d in todas_curvas]
    else:
        # Dividir uma curva em grupos artificiais
        f = todas_curvas[0]['fluxo']
        n = len(f)
        n_grupos = 10
        dados_grupos = [f[i*n//n_grupos:(i+1)*n//n_grupos] for i in range(n_grupos)]
    
    print(f"\n>>> Analisando {n_grupos} grupos")
    
    modelo = ModeloHierarquico(n_grupos)
    resultado = modelo.gibbs_hierarquico(dados_grupos, n_amostras=2000)
    
    print(f"    μ global = {resultado['mu_global']:.6f} ± {resultado['mu_global_std']:.6f}")
    print(f"    τ² = {resultado['tau2']:.8f}")
    print(f"    σ² = {resultado['sigma2']:.8f}")
    print(f"    ICC = {resultado['icc']:.4f}")
    
    arq = os.path.join(diretorio_saida, "hierarquico.png")
    modelo.plotar_resultados(dados_grupos, resultado, "Modelo Hierárquico", arq)
    print(f"    Gráfico salvo: {arq}")
    
    for nome in dados_entrada:
        resultados[nome] = {**dados_entrada[nome], 'hierarquico': resultado}
    
    plt.close('all')
    print("\n" + "=" * 60)
    print("MÓDULO 34 CONCLUÍDO")
    print("=" * 60)
    
    return resultados


__all__ = ['ModeloHierarquico', 'executar_modulo_34']
