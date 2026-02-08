"""
Módulo 18: Regressão LASSO
Autor: Luiz Tiago Wilcke

Regularização L1: min (1/2n)||y - Xβ||² + λ||β||₁
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import os


class RegressaoLASSO:
    """LASSO via Coordinate Descent."""
    
    def __init__(self, lambda_valores: np.ndarray = None):
        if lambda_valores is None:
            self.lambda_valores = np.logspace(-4, 1, 100)
        else:
            self.lambda_valores = lambda_valores
    
    def soft_threshold(self, x: float, lambda_: float) -> float:
        """Operador soft thresholding."""
        if x > lambda_:
            return x - lambda_
        elif x < -lambda_:
            return x + lambda_
        return 0.0
    
    def ajustar(self, X: np.ndarray, y: np.ndarray, lambda_: float,
                max_iter: int = 1000, tol: float = 1e-6) -> Dict:
        """
        LASSO via Coordinate Descent.
        Para cada j: β_j = S(r_j + β_j, λ) / ||X_j||²
        """
        n, p = X.shape
        
        # Centralizar
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1
        X_scaled = (X - X_mean) / X_std
        y_mean = np.mean(y)
        y_cent = y - y_mean
        
        # Inicializar
        beta = np.zeros(p)
        X_norm_sq = np.sum(X_scaled ** 2, axis=0)
        
        for iteration in range(max_iter):
            beta_old = beta.copy()
            
            for j in range(p):
                # Residual parcial
                r_j = y_cent - X_scaled @ beta + X_scaled[:, j] * beta[j]
                
                # Atualização
                rho_j = X_scaled[:, j] @ r_j
                beta[j] = self.soft_threshold(rho_j, lambda_ * n) / X_norm_sq[j]
            
            # Convergência
            if np.max(np.abs(beta - beta_old)) < tol:
                break
        
        # Desescalar
        beta_orig = beta / X_std
        intercepto = y_mean - np.sum(beta_orig * X_mean)
        
        y_pred = X @ beta_orig + intercepto
        residuos = y - y_pred
        
        rss = np.sum(residuos ** 2)
        tss = np.sum((y - y_mean) ** 2)
        r2 = 1 - rss / tss
        
        n_nao_zero = np.sum(np.abs(beta_orig) > 1e-10)
        
        return {
            'beta': beta_orig,
            'intercepto': intercepto,
            'r2': r2,
            'n_nao_zero': n_nao_zero,
            'lambda': lambda_,
            'iteracoes': iteration + 1,
            'y_pred': y_pred,
            'residuos': residuos
        }
    
    def validacao_cruzada(self, X: np.ndarray, y: np.ndarray, k: int = 5) -> Dict:
        """K-fold CV."""
        n = len(y)
        indices = np.arange(n)
        np.random.shuffle(indices)
        folds = np.array_split(indices, k)
        
        cv_scores = np.zeros((len(self.lambda_valores), k))
        
        for i, lambda_ in enumerate(self.lambda_valores):
            for j, fold_teste in enumerate(folds):
                mascara_treino = ~np.isin(np.arange(n), fold_teste)
                
                X_treino, y_treino = X[mascara_treino], y[mascara_treino]
                X_teste, y_teste = X[fold_teste], y[fold_teste]
                
                resultado = self.ajustar(X_treino, y_treino, lambda_)
                y_pred = X_teste @ resultado['beta'] + resultado['intercepto']
                cv_scores[i, j] = np.mean((y_teste - y_pred) ** 2)
        
        cv_mean = np.mean(cv_scores, axis=1)
        idx_melhor = np.argmin(cv_mean)
        
        return {
            'lambda_otimo': self.lambda_valores[idx_melhor],
            'cv_mean': cv_mean,
            'lambdas': self.lambda_valores
        }
    
    def plotar_caminho(self, X, y, cv_resultado, titulo, salvar=None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Caminho dos coeficientes
        betas = []
        n_nao_zero = []
        for lambda_ in self.lambda_valores:
            res = self.ajustar(X, y, lambda_, max_iter=500)
            betas.append(res['beta'])
            n_nao_zero.append(res['n_nao_zero'])
        betas = np.array(betas)
        
        ax = axes[0, 0]
        for i in range(betas.shape[1]):
            ax.semilogx(self.lambda_valores, betas[:, i], lw=1)
        ax.axvline(cv_resultado['lambda_otimo'], color='r', ls='--')
        ax.set_xlabel('λ'); ax.set_ylabel('β')
        ax.set_title('Caminho LASSO')
        
        # CV
        ax = axes[0, 1]
        ax.semilogx(cv_resultado['lambdas'], cv_resultado['cv_mean'])
        ax.axvline(cv_resultado['lambda_otimo'], color='r', ls='--')
        ax.set_xlabel('λ'); ax.set_ylabel('MSE')
        ax.set_title('Validação Cruzada')
        
        # Sparsity
        ax = axes[1, 0]
        ax.semilogx(self.lambda_valores, n_nao_zero)
        ax.axvline(cv_resultado['lambda_otimo'], color='r', ls='--')
        ax.set_xlabel('λ'); ax.set_ylabel('# coeficientes ≠ 0')
        ax.set_title('Esparsidade')
        
        # Coeficientes finais
        resultado = self.ajustar(X, y, cv_resultado['lambda_otimo'])
        ax = axes[1, 1]
        cores = ['green' if b != 0 else 'gray' for b in resultado['beta']]
        ax.bar(range(len(resultado['beta'])), resultado['beta'], color=cores, alpha=0.7)
        ax.set_xlabel('Feature'); ax.set_ylabel('β')
        ax.set_title(f"Coeficientes (λ = {cv_resultado['lambda_otimo']:.4f})")
        
        for ax in axes.flat: ax.grid(True, alpha=0.3)
        plt.suptitle(titulo, fontweight='bold'); plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_18(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 18: REGRESSÃO LASSO\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        t, f = dados['tempo'], dados['fluxo']
        X = np.column_stack([t - t[0], (t - t[0])**2, np.sin(2*np.pi*t/10), np.cos(2*np.pi*t/10)])
        
        lasso = RegressaoLASSO()
        cv = lasso.validacao_cruzada(X, f, k=5)
        resultado = lasso.ajustar(X, f, cv['lambda_otimo'])
        
        print(f"    λ ótimo: {cv['lambda_otimo']:.4f}")
        print(f"    Coeficientes não-zero: {resultado['n_nao_zero']}")
        print(f"    R²: {resultado['r2']:.4f}")
        
        arq = os.path.join(diretorio_saida, f"lasso_{nome.replace(' ', '_').lower()}.png")
        lasso.plotar_caminho(X, f, cv, f"LASSO - {nome}", arq)
        
        resultados[nome] = {**dados, 'lasso': resultado, 'lasso_cv': cv}
    
    plt.close('all')
    print("\nMÓDULO 18 CONCLUÍDO")
    return resultados

__all__ = ['RegressaoLASSO', 'executar_modulo_18']
