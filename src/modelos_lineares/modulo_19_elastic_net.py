"""
Módulo 19: Elastic Net
Autor: Luiz Tiago Wilcke

Combinação L1 + L2: min (1/2n)||y - Xβ||² + λ[α||β||₁ + (1-α)||β||²/2]
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import os


class ElasticNet:
    """Elastic Net via Coordinate Descent."""
    
    def __init__(self, lambda_valores: np.ndarray = None, alpha_valores: np.ndarray = None):
        if lambda_valores is None:
            self.lambda_valores = np.logspace(-4, 1, 50)
        else:
            self.lambda_valores = lambda_valores
        
        if alpha_valores is None:
            self.alpha_valores = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
        else:
            self.alpha_valores = alpha_valores
    
    def soft_threshold(self, x: float, lambda_: float) -> float:
        if x > lambda_:
            return x - lambda_
        elif x < -lambda_:
            return x + lambda_
        return 0.0
    
    def ajustar(self, X: np.ndarray, y: np.ndarray, lambda_: float, alpha: float,
                max_iter: int = 1000, tol: float = 1e-6) -> Dict:
        """
        Elastic Net: β_j = S(r_j, λα) / (||X_j||² + λ(1-α))
        """
        n, p = X.shape
        
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1
        X_scaled = (X - X_mean) / X_std
        y_mean = np.mean(y)
        y_cent = y - y_mean
        
        beta = np.zeros(p)
        X_norm_sq = np.sum(X_scaled ** 2, axis=0)
        
        for iteration in range(max_iter):
            beta_old = beta.copy()
            
            for j in range(p):
                r_j = y_cent - X_scaled @ beta + X_scaled[:, j] * beta[j]
                rho_j = X_scaled[:, j] @ r_j
                
                # Elastic Net update
                beta[j] = self.soft_threshold(rho_j, lambda_ * alpha * n) / (X_norm_sq[j] + lambda_ * (1 - alpha) * n)
            
            if np.max(np.abs(beta - beta_old)) < tol:
                break
        
        beta_orig = beta / X_std
        intercepto = y_mean - np.sum(beta_orig * X_mean)
        
        y_pred = X @ beta_orig + intercepto
        residuos = y - y_pred
        
        rss = np.sum(residuos ** 2)
        tss = np.sum((y - y_mean) ** 2)
        r2 = 1 - rss / tss
        
        return {
            'beta': beta_orig,
            'intercepto': intercepto,
            'r2': r2,
            'n_nao_zero': np.sum(np.abs(beta_orig) > 1e-10),
            'lambda': lambda_,
            'alpha': alpha,
            'y_pred': y_pred,
            'residuos': residuos
        }
    
    def validacao_cruzada_2d(self, X: np.ndarray, y: np.ndarray, k: int = 5) -> Dict:
        """Grid search 2D para lambda e alpha."""
        n = len(y)
        indices = np.arange(n)
        np.random.shuffle(indices)
        folds = np.array_split(indices, k)
        
        cv_scores = np.zeros((len(self.lambda_valores), len(self.alpha_valores)))
        
        for i, lambda_ in enumerate(self.lambda_valores):
            for j, alpha in enumerate(self.alpha_valores):
                scores = []
                for fold_teste in folds:
                    mascara = ~np.isin(np.arange(n), fold_teste)
                    resultado = self.ajustar(X[mascara], y[mascara], lambda_, alpha)
                    y_pred = X[fold_teste] @ resultado['beta'] + resultado['intercepto']
                    scores.append(np.mean((y[fold_teste] - y_pred) ** 2))
                cv_scores[i, j] = np.mean(scores)
        
        idx_min = np.unravel_index(np.argmin(cv_scores), cv_scores.shape)
        
        return {
            'lambda_otimo': self.lambda_valores[idx_min[0]],
            'alpha_otimo': self.alpha_valores[idx_min[1]],
            'cv_scores': cv_scores,
            'lambdas': self.lambda_valores,
            'alphas': self.alpha_valores
        }
    
    def plotar_resultados(self, X, y, cv, titulo, salvar=None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Heatmap CV
        ax = axes[0, 0]
        im = ax.imshow(np.log10(cv['cv_scores']).T, aspect='auto', origin='lower',
                       extent=[np.log10(cv['lambdas'][0]), np.log10(cv['lambdas'][-1]),
                              cv['alphas'][0], cv['alphas'][-1]])
        ax.set_xlabel('log₁₀(λ)'); ax.set_ylabel('α')
        ax.set_title('log₁₀(MSE) - Validação Cruzada')
        plt.colorbar(im, ax=ax)
        
        # Caminho para alpha ótimo
        betas = []
        for lambda_ in self.lambda_valores:
            res = self.ajustar(X, y, lambda_, cv['alpha_otimo'], max_iter=300)
            betas.append(res['beta'])
        betas = np.array(betas)
        
        ax = axes[0, 1]
        for i in range(min(betas.shape[1], 10)):
            ax.semilogx(self.lambda_valores, betas[:, i], lw=1)
        ax.axvline(cv['lambda_otimo'], color='r', ls='--')
        ax.set_xlabel('λ'); ax.set_ylabel('β')
        ax.set_title(f"Caminho (α = {cv['alpha_otimo']:.2f})")
        
        # Ajuste final
        resultado = self.ajustar(X, y, cv['lambda_otimo'], cv['alpha_otimo'])
        ax = axes[1, 0]
        ax.scatter(resultado['y_pred'], y, s=5, alpha=0.5)
        lim = [min(resultado['y_pred'].min(), y.min()), max(resultado['y_pred'].max(), y.max())]
        ax.plot(lim, lim, 'r--')
        ax.set_xlabel('Predito'); ax.set_ylabel('Observado')
        ax.set_title(f"R² = {resultado['r2']:.4f}")
        
        # Coeficientes
        ax = axes[1, 1]
        cores = ['green' if abs(b) > 1e-10 else 'gray' for b in resultado['beta']]
        ax.bar(range(len(resultado['beta'])), resultado['beta'], color=cores, alpha=0.7)
        ax.set_title(f"λ={cv['lambda_otimo']:.4f}, α={cv['alpha_otimo']:.2f}")
        
        for ax in axes.flat: ax.grid(True, alpha=0.3)
        plt.suptitle(titulo, fontweight='bold'); plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_19(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 19: ELASTIC NET\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        t, f = dados['tempo'], dados['fluxo']
        X = np.column_stack([t - t[0], (t - t[0])**2, np.sin(2*np.pi*t/10), np.cos(2*np.pi*t/10)])
        
        enet = ElasticNet()
        cv = enet.validacao_cruzada_2d(X, f, k=5)
        resultado = enet.ajustar(X, f, cv['lambda_otimo'], cv['alpha_otimo'])
        
        print(f"    λ ótimo: {cv['lambda_otimo']:.4f}")
        print(f"    α ótimo: {cv['alpha_otimo']:.2f}")
        print(f"    R²: {resultado['r2']:.4f}")
        
        arq = os.path.join(diretorio_saida, f"elasticnet_{nome.replace(' ', '_').lower()}.png")
        enet.plotar_resultados(X, f, cv, f"Elastic Net - {nome}", arq)
        
        resultados[nome] = {**dados, 'elastic_net': resultado, 'enet_cv': cv}
    
    plt.close('all')
    print("\nMÓDULO 19 CONCLUÍDO")
    return resultados

__all__ = ['ElasticNet', 'executar_modulo_19']
