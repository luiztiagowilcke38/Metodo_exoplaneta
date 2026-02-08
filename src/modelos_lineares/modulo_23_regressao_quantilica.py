"""
Módulo 23: Regressão Quantílica
Autor: Luiz Tiago Wilcke

Regressão nos quantis: min Σ ρ_τ(y_i - x_i'β)
onde ρ_τ(u) = u(τ - I(u<0))
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize
from typing import Dict, List, Optional
import os


class RegressaoQuantilica:
    """Regressão Quantílica via programação linear."""
    
    def __init__(self, quantis: List[float] = None):
        if quantis is None:
            self.quantis = [0.1, 0.25, 0.5, 0.75, 0.9]
        else:
            self.quantis = quantis
    
    def check_function(self, u: np.ndarray, tau: float) -> np.ndarray:
        """Função check: ρ_τ(u) = u(τ - I(u<0))."""
        return u * (tau - (u < 0))
    
    def ajustar_quantil(self, X: np.ndarray, y: np.ndarray, tau: float) -> Dict:
        """
        Ajusta regressão para quantil τ via IRLS.
        """
        n, p = X.shape
        X_aug = np.column_stack([np.ones(n), X])
        p_aug = p + 1
        
        # Inicializar com OLS
        beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        
        # IRLS para minimizar check function
        for iteration in range(100):
            beta_old = beta.copy()
            
            residuos = y - X_aug @ beta
            
            # Pesos para check function
            pesos = np.where(residuos >= 0, tau, 1 - tau)
            pesos = pesos / (np.abs(residuos) + 1e-6)
            
            # WLS
            W = np.diag(pesos)
            try:
                beta = np.linalg.solve(X_aug.T @ W @ X_aug, X_aug.T @ W @ y)
            except:
                beta = np.linalg.lstsq(X_aug.T @ W @ X_aug, X_aug.T @ W @ y, rcond=None)[0]
            
            if np.max(np.abs(beta - beta_old)) < 1e-6:
                break
        
        y_pred = X_aug @ beta
        residuos = y - y_pred
        
        # Pseudo R² de Koenker-Machado
        check_model = np.sum(self.check_function(residuos, tau))
        check_null = np.sum(self.check_function(y - np.quantile(y, tau), tau))
        pseudo_r2 = 1 - check_model / check_null
        
        return {
            'tau': tau,
            'intercepto': beta[0],
            'beta': beta[1:],
            'y_pred': y_pred,
            'residuos': residuos,
            'pseudo_r2': pseudo_r2,
            'check_value': check_model
        }
    
    def ajustar_todos_quantis(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Ajusta regressão para todos os quantis especificados."""
        resultados = {}
        for tau in self.quantis:
            resultados[tau] = self.ajustar_quantil(X, y, tau)
        return resultados
    
    def plotar_resultados(self, X, y, resultados, titulo, salvar=None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Curvas de regressão
        ax = axes[0, 0]
        ax.scatter(X[:, 0] if X.ndim > 1 else X, y, s=1, alpha=0.2, c='gray')
        
        cores = plt.cm.viridis(np.linspace(0, 1, len(self.quantis)))
        for (tau, res), cor in zip(resultados.items(), cores):
            ordem = np.argsort(X[:, 0] if X.ndim > 1 else X)
            ax.plot((X[ordem, 0] if X.ndim > 1 else X[ordem]), 
                   res['y_pred'][ordem], color=cor, lw=2, label=f'τ={tau}')
        
        ax.legend(loc='upper right')
        ax.set_xlabel('X'); ax.set_ylabel('y')
        ax.set_title('Regressões Quantílicas')
        
        # Coeficientes por quantil
        ax = axes[0, 1]
        taus = list(resultados.keys())
        betas = np.array([resultados[t]['beta'] for t in taus])
        
        for i in range(betas.shape[1]):
            ax.plot(taus, betas[:, i], 'o-', label=f'β{i+1}')
        
        ax.axhline(0, color='k', ls='--', alpha=0.3)
        ax.set_xlabel('Quantil (τ)'); ax.set_ylabel('Coeficiente')
        ax.set_title('Coeficientes por Quantil'); ax.legend()
        
        # Pseudo R²
        ax = axes[1, 0]
        r2s = [resultados[t]['pseudo_r2'] for t in taus]
        ax.plot(taus, r2s, 'bo-', markersize=8)
        ax.set_xlabel('Quantil (τ)'); ax.set_ylabel('Pseudo R²')
        ax.set_title('Pseudo R² de Koenker-Machado')
        
        # Quantis observados vs preditos
        ax = axes[1, 1]
        for tau in [0.1, 0.5, 0.9]:
            if tau in resultados:
                res = resultados[tau]
                prop_abaixo = np.mean(y < res['y_pred'])
                ax.bar(tau, prop_abaixo, width=0.08, alpha=0.7, label=f'τ={tau}: {prop_abaixo:.2f}')
        
        ax.plot([0, 1], [0, 1], 'r--', label='45°')
        ax.set_xlabel('Quantil Nominal'); ax.set_ylabel('Proporção Observada')
        ax.set_title('Calibração dos Quantis'); ax.legend()
        
        for ax in axes.flat: ax.grid(True, alpha=0.3)
        plt.suptitle(titulo, fontweight='bold'); plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_23(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 23: REGRESSÃO QUANTÍLICA\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        t, f = dados['tempo'], dados['fluxo']
        X = np.column_stack([t - t[0]])
        
        qreg = RegressaoQuantilica()
        resultado = qreg.ajustar_todos_quantis(X, f)
        
        for tau in [0.25, 0.5, 0.75]:
            print(f"    τ={tau}: β={resultado[tau]['beta'][0]:.6f}, R²={resultado[tau]['pseudo_r2']:.4f}")
        
        arq = os.path.join(diretorio_saida, f"quantilica_{nome.replace(' ', '_').lower()}.png")
        qreg.plotar_resultados(X, f, resultado, f"Regressão Quantílica - {nome}", arq)
        
        resultados[nome] = {**dados, 'quantilica': resultado}
    
    plt.close('all')
    print("\nMÓDULO 23 CONCLUÍDO")
    return resultados

__all__ = ['RegressaoQuantilica', 'executar_modulo_23']
