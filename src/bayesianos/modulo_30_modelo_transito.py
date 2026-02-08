"""
Módulo 30: Modelo de Trânsito Bayesiano
Autor: Luiz Tiago Wilcke

Modelo completo de trânsito planetário com inferência Bayesiana.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Dict, Optional
import os


class ModeloTransitoBayesiano:
    """Modelo de trânsito com priors físicos e MCMC."""
    
    def __init__(self):
        pass
    
    def modelo_transito(self, tempo: np.ndarray, t0: float, periodo: float,
                         rp_rs: float, a_rs: float, inc: float,
                         u1: float = 0.4, u2: float = 0.2) -> np.ndarray:
        """
        Modelo de trânsito com limb darkening quadrático.
        
        Parâmetros:
            t0: Época central
            periodo: Período orbital
            rp_rs: Raio do planeta / Raio da estrela
            a_rs: Semi-eixo maior / Raio da estrela
            inc: Inclinação (graus)
            u1, u2: Limb darkening quadrático
        """
        fase = 2 * np.pi * (tempo - t0) / periodo
        
        # Posição projetada
        inc_rad = np.radians(inc)
        z = a_rs * np.sqrt(np.sin(fase)**2 + (np.cos(inc_rad) * np.cos(fase))**2)
        
        # Fluxo normalizado
        fluxo = np.ones_like(tempo)
        
        # Taxa de trânsito (integral do limb darkening)
        em_transito = z < 1 + rp_rs
        
        for i in np.where(em_transito)[0]:
            zi = z[i]
            if zi >= 1 + rp_rs:
                delta = 0
            elif zi <= 1 - rp_rs:
                # Totalmente dentro
                delta = rp_rs**2
            else:
                # Parcialmente
                k0 = np.arccos((zi**2 + rp_rs**2 - 1) / (2*zi*rp_rs)) if zi > 0 else 0
                k1 = np.arccos((zi**2 - rp_rs**2 + 1) / (2*zi)) if zi > 0 else 0
                delta = (rp_rs**2 * k0 + k1 - 
                        0.5*np.sqrt(max(0, (1+rp_rs-zi)*(zi+rp_rs-1)*
                                       (zi-rp_rs+1)*(zi+rp_rs+1)))) / np.pi
            
            # Limb darkening
            c1 = 1 - u1 - 2*u2
            c2 = u1 + 2*u2
            c3 = -u2
            fator_ld = (c1 + c2*2/3 + c3/2) / (c1 + c2 + c3)
            
            fluxo[i] = 1 - delta * fator_ld
        
        return fluxo
    
    def log_prior(self, theta: np.ndarray) -> float:
        """Priors físicos para parâmetros de trânsito."""
        t0, log_periodo, rp_rs, log_a_rs, inc = theta
        
        periodo = np.exp(log_periodo)
        a_rs = np.exp(log_a_rs)
        
        # Verificar limites físicos
        if rp_rs < 0 or rp_rs > 0.5:
            return -np.inf
        if periodo < 0.1 or periodo > 1000:
            return -np.inf
        if a_rs < 1 or a_rs > 1000:
            return -np.inf
        if inc < 70 or inc > 90:
            return -np.inf
        
        # Priors
        log_prior = 0
        log_prior += -0.5 * (rp_rs - 0.1)**2 / 0.05**2  # Prior Rp/Rs ~ N(0.1, 0.05)
        log_prior += -0.5 * (inc - 87)**2 / 3**2  # Prior inc ~ N(87, 3)
        
        return log_prior
    
    def log_likelihood(self, theta: np.ndarray, tempo: np.ndarray, 
                        fluxo: np.ndarray, erro: np.ndarray) -> float:
        """Verossimilhança Gaussiana."""
        t0, log_periodo, rp_rs, log_a_rs, inc = theta
        periodo = np.exp(log_periodo)
        a_rs = np.exp(log_a_rs)
        
        try:
            modelo = self.modelo_transito(tempo, t0, periodo, rp_rs, a_rs, inc)
        except:
            return -np.inf
        
        chi2 = np.sum(((fluxo - modelo) / erro)**2)
        return -0.5 * chi2
    
    def log_posterior(self, theta: np.ndarray, tempo: np.ndarray,
                       fluxo: np.ndarray, erro: np.ndarray) -> float:
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, tempo, fluxo, erro)
    
    def ajustar_mcmc(self, tempo: np.ndarray, fluxo: np.ndarray, erro: np.ndarray,
                      theta_inicial: np.ndarray = None, n_amostras: int = 5000,
                      burnin: int = 1000) -> Dict:
        """MCMC para o modelo de trânsito."""
        n_params = 5
        
        if theta_inicial is None:
            # Estimativas iniciais
            t0 = tempo[np.argmin(fluxo)]
            theta_inicial = np.array([t0, np.log(3.0), 0.1, np.log(10.0), 87.0])
        
        escalas = np.array([0.001, 0.001, 0.001, 0.01, 0.1])
        
        n_total = burnin + n_amostras
        cadeia = np.zeros((n_total, n_params))
        log_probs = np.zeros(n_total)
        
        theta = theta_inicial.copy()
        log_p = self.log_posterior(theta, tempo, fluxo, erro)
        aceitos = 0
        
        for i in range(n_total):
            theta_proposta = theta + escalas * np.random.randn(n_params)
            log_p_proposta = self.log_posterior(theta_proposta, tempo, fluxo, erro)
            
            if np.log(np.random.rand()) < log_p_proposta - log_p:
                theta = theta_proposta
                log_p = log_p_proposta
                aceitos += 1
            
            cadeia[i] = theta
            log_probs[i] = log_p
        
        cadeia_final = cadeia[burnin:]
        
        # Converter para parâmetros físicos
        t0_post = cadeia_final[:, 0]
        periodo_post = np.exp(cadeia_final[:, 1])
        rp_rs_post = cadeia_final[:, 2]
        a_rs_post = np.exp(cadeia_final[:, 3])
        inc_post = cadeia_final[:, 4]
        
        return {
            'cadeia': cadeia_final,
            't0': (np.mean(t0_post), np.std(t0_post)),
            'periodo': (np.mean(periodo_post), np.std(periodo_post)),
            'rp_rs': (np.mean(rp_rs_post), np.std(rp_rs_post)),
            'a_rs': (np.mean(a_rs_post), np.std(a_rs_post)),
            'inc': (np.mean(inc_post), np.std(inc_post)),
            'taxa_aceitacao': aceitos / n_total
        }
    
    def plotar_resultados(self, tempo, fluxo, resultado, titulo, salvar=None):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Modelo ajustado
        ax = axes[0, 0]
        t0, periodo = resultado['t0'][0], resultado['periodo'][0]
        rp_rs, a_rs = resultado['rp_rs'][0], resultado['a_rs'][0]
        inc = resultado['inc'][0]
        
        fase = ((tempo - t0) % periodo) / periodo
        fase[fase > 0.5] -= 1
        ordem = np.argsort(fase)
        
        modelo = self.modelo_transito(tempo, t0, periodo, rp_rs, a_rs, inc)
        
        ax.scatter(fase, fluxo, s=1, alpha=0.3, c='gray')
        ax.plot(fase[ordem], modelo[ordem], 'r-', lw=2)
        ax.set_xlabel('Fase'); ax.set_ylabel('Fluxo')
        ax.set_title('Modelo Ajustado')
        
        # Distribuições posteriori
        nomes = ['T₀', 'P (dias)', 'Rp/Rs', 'a/Rs', 'inc (°)']
        for i, nome in enumerate(nomes):
            ax = axes[(i+1)//3, (i+1)%3]
            dados = resultado['cadeia'][:, i]
            if i in [1, 3]:
                dados = np.exp(dados)
            ax.hist(dados, bins=50, density=True, alpha=0.7)
            ax.axvline(np.mean(dados), color='r', ls='--')
            ax.set_xlabel(nome)
            ax.set_title(f'{nome} = {np.mean(dados):.4f} ± {np.std(dados):.4f}')
        
        for ax in axes.flat: ax.grid(True, alpha=0.3)
        plt.suptitle(titulo, fontweight='bold'); plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_30(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 30: MODELO DE TRÂNSITO BAYESIANO\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        t = dados['tempo']
        f = dados['fluxo']
        e = dados.get('erro_fluxo', np.ones_like(f) * 0.0001)
        
        modelo = ModeloTransitoBayesiano()
        resultado = modelo.ajustar_mcmc(t, f, e, n_amostras=2000)
        
        print(f"    T₀ = {resultado['t0'][0]:.4f} ± {resultado['t0'][1]:.4f}")
        print(f"    P = {resultado['periodo'][0]:.4f} ± {resultado['periodo'][1]:.4f}")
        print(f"    Rp/Rs = {resultado['rp_rs'][0]:.4f} ± {resultado['rp_rs'][1]:.4f}")
        print(f"    Taxa: {resultado['taxa_aceitacao']:.2%}")
        
        arq = os.path.join(diretorio_saida, f"transito_bayes_{nome.replace(' ', '_').lower()}.png")
        modelo.plotar_resultados(t, f, resultado, f"Trânsito Bayesiano - {nome}", arq)
        
        resultados[nome] = {**dados, 'modelo_transito_bayes': resultado}
    
    plt.close('all')
    print("\nMÓDULO 30 CONCLUÍDO")
    return resultados

__all__ = ['ModeloTransitoBayesiano', 'executar_modulo_30']
