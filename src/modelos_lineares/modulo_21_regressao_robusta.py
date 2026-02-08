"""
Módulo 21: Regressão Robusta
Autor: Luiz Tiago Wilcke

Regressão resistente a outliers usando Huber loss e M-estimators.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Dict, Optional
import os


class RegressaoRobusta:
    """
    Regressão robusta via M-estimadores.
    Minimiza: Σ ρ((y_i - x_i'β)/σ)
    """
    
    def __init__(self, metodo: str = 'huber', c: float = 1.345):
        self.metodo = metodo
        self.c = c  # Constante de tuning
    
    def rho_huber(self, r: np.ndarray) -> np.ndarray:
        """Função ρ de Huber."""
        abs_r = np.abs(r)
        return np.where(abs_r <= self.c, 0.5 * r**2, self.c * abs_r - 0.5 * self.c**2)
    
    def psi_huber(self, r: np.ndarray) -> np.ndarray:
        """Função ψ de Huber (derivada de ρ)."""
        return np.clip(r, -self.c, self.c)
    
    def peso_huber(self, r: np.ndarray) -> np.ndarray:
        """Pesos w = ψ(r)/r."""
        eps = 1e-10
        return np.where(np.abs(r) <= self.c, 1.0, self.c / (np.abs(r) + eps))
    
    def rho_tukey(self, r: np.ndarray) -> np.ndarray:
        """Função ρ biweight de Tukey."""
        c = 4.685
        return np.where(np.abs(r) <= c, (c**2/6) * (1 - (1 - (r/c)**2)**3), c**2/6)
    
    def peso_tukey(self, r: np.ndarray) -> np.ndarray:
        """Pesos biweight de Tukey."""
        c = 4.685
        return np.where(np.abs(r) <= c, (1 - (r/c)**2)**2, 0)
    
    def escala_mad(self, r: np.ndarray) -> float:
        """Estimador robusto de escala: MAD."""
        return 1.4826 * np.median(np.abs(r - np.median(r)))
    
    def ajustar_irls(self, X: np.ndarray, y: np.ndarray, 
                      max_iter: int = 50, tol: float = 1e-6) -> Dict:
        """
        Iteratively Reweighted Least Squares (IRLS).
        1. Ajustar OLS inicial
        2. Calcular resíduos padronizados
        3. Calcular pesos w_i = ψ(r_i)/r_i
        4. Ajustar WLS com pesos
        5. Repetir até convergência
        """
        n, p = X.shape
        X_aug = np.column_stack([np.ones(n), X])
        
        # OLS inicial
        beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        
        for iteration in range(max_iter):
            beta_old = beta.copy()
            
            # Resíduos
            residuos = y - X_aug @ beta
            sigma = self.escala_mad(residuos)
            r_padronizado = residuos / (sigma + 1e-10)
            
            # Pesos
            if self.metodo == 'huber':
                pesos = self.peso_huber(r_padronizado)
            else:
                pesos = self.peso_tukey(r_padronizado)
            
            # WLS: β = (X'WX)^{-1} X'Wy
            W = np.diag(pesos)
            XtWX = X_aug.T @ W @ X_aug
            XtWy = X_aug.T @ W @ y
            
            try:
                beta = np.linalg.solve(XtWX, XtWy)
            except:
                beta = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
            
            # Convergência
            if np.max(np.abs(beta - beta_old)) < tol:
                break
        
        intercepto = beta[0]
        coef = beta[1:]
        
        y_pred = X_aug @ beta
        residuos_final = y - y_pred
        
        # R² robusto
        tss = np.sum((y - np.median(y))**2)
        rss = np.sum(pesos * residuos_final**2)
        r2 = 1 - rss / np.sum(pesos * (y - np.median(y))**2)
        
        # Outliers detectados
        outliers = np.where(pesos < 0.5)[0]
        
        return {
            'beta': coef,
            'intercepto': intercepto,
            'sigma': sigma,
            'pesos': pesos,
            'r2': r2,
            'iteracoes': iteration + 1,
            'outliers': outliers,
            'n_outliers': len(outliers),
            'y_pred': y_pred,
            'residuos': residuos_final
        }
    
    def plotar_resultados(self, X, y, resultado, titulo, salvar=None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Dados com outliers marcados
        ax = axes[0, 0]
        cores = np.where(resultado['pesos'] < 0.5, 'red', 'blue')
        ax.scatter(range(len(y)), y, c=cores, s=5, alpha=0.5)
        ax.plot(resultado['y_pred'], 'g-', lw=1, label='Ajuste robusto')
        ax.set_title(f"Outliers detectados: {resultado['n_outliers']}")
        ax.legend()
        
        # Pesos
        ax = axes[0, 1]
        ax.scatter(range(len(resultado['pesos'])), resultado['pesos'], s=3, alpha=0.5)
        ax.axhline(0.5, color='r', ls='--')
        ax.set_xlabel('Observação'); ax.set_ylabel('Peso')
        ax.set_title('Pesos do M-estimador')
        
        # Resíduos padronizados
        ax = axes[1, 0]
        r_pad = resultado['residuos'] / resultado['sigma']
        ax.scatter(resultado['y_pred'], r_pad, c=cores, s=5, alpha=0.5)
        ax.axhline(0, color='k')
        ax.axhline(self.c, color='r', ls='--')
        ax.axhline(-self.c, color='r', ls='--')
        ax.set_xlabel('Predito'); ax.set_ylabel('Resíduo Padronizado')
        
        # Distribuição dos pesos
        ax = axes[1, 1]
        ax.hist(resultado['pesos'], bins=50, alpha=0.7)
        ax.set_xlabel('Peso'); ax.set_ylabel('Frequência')
        ax.set_title('Distribuição dos Pesos')
        
        for ax in axes.flat: ax.grid(True, alpha=0.3)
        plt.suptitle(f"{titulo} ({self.metodo.capitalize()})", fontweight='bold')
        plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_21(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 21: REGRESSÃO ROBUSTA\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        t, f = dados['tempo'], dados['fluxo']
        X = np.column_stack([t - t[0], (t - t[0])**2])
        
        robusta = RegressaoRobusta(metodo='huber')
        resultado = robusta.ajustar_irls(X, f)
        
        print(f"    R² robusto: {resultado['r2']:.4f}")
        print(f"    Outliers: {resultado['n_outliers']} ({100*resultado['n_outliers']/len(f):.1f}%)")
        print(f"    σ (MAD): {resultado['sigma']:.6f}")
        
        arq = os.path.join(diretorio_saida, f"robusta_{nome.replace(' ', '_').lower()}.png")
        robusta.plotar_resultados(X, f, resultado, f"Regressão Robusta - {nome}", arq)
        
        resultados[nome] = {**dados, 'regressao_robusta': resultado}
    
    plt.close('all')
    print("\nMÓDULO 21 CONCLUÍDO")
    return resultados

__all__ = ['RegressaoRobusta', 'executar_modulo_21']
