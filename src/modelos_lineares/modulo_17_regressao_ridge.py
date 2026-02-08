"""
Módulo 17: Regressão Ridge
Autor: Luiz Tiago Wilcke

Regularização L2: β̂ = (X'X + λI)⁻¹X'y
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
from typing import Dict, Optional
import os


class RegressaoRidge:
    """Regressão Ridge com validação cruzada para seleção de λ."""
    
    def __init__(self, lambda_valores: np.ndarray = None):
        if lambda_valores is None:
            self.lambda_valores = np.logspace(-4, 4, 100)
        else:
            self.lambda_valores = lambda_valores
    
    def ajustar(self, X: np.ndarray, y: np.ndarray, lambda_: float) -> Dict:
        """
        Ajusta regressão Ridge.
        β̂_ridge = (X'X + λI)⁻¹X'y
        """
        n, p = X.shape
        
        # Centralizar e escalar
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1
        X_scaled = (X - X_mean) / X_std
        
        y_mean = np.mean(y)
        y_cent = y - y_mean
        
        # Ridge
        I = np.eye(p)
        XtX = X_scaled.T @ X_scaled
        beta_scaled = inv(XtX + lambda_ * I) @ X_scaled.T @ y_cent
        
        # Desescalar
        beta = beta_scaled / X_std
        intercepto = y_mean - np.sum(beta * X_mean)
        
        # Predições
        y_pred = X @ beta + intercepto
        residuos = y - y_pred
        
        # Estatísticas
        rss = np.sum(residuos ** 2)
        tss = np.sum((y - y_mean) ** 2)
        r2 = 1 - rss / tss
        
        # Graus de liberdade efetivos
        eigenvalues = np.linalg.eigvalsh(XtX)
        df = np.sum(eigenvalues / (eigenvalues + lambda_))
        
        # GCV (Generalized Cross-Validation)
        gcv = n * rss / (n - df) ** 2
        
        return {
            'beta': beta,
            'intercepto': intercepto,
            'r2': r2,
            'gcv': gcv,
            'df_efetivo': df,
            'lambda': lambda_,
            'y_pred': y_pred,
            'residuos': residuos
        }
    
    def validacao_cruzada(self, X: np.ndarray, y: np.ndarray, k: int = 5) -> Dict:
        """K-fold CV para seleção de λ."""
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
        cv_std = np.std(cv_scores, axis=1)
        
        idx_melhor = np.argmin(cv_mean)
        lambda_otimo = self.lambda_valores[idx_melhor]
        
        # Regra 1-SE: lambda mais restritivo dentro de 1 desvio padrão
        threshold = cv_mean[idx_melhor] + cv_std[idx_melhor]
        idx_1se = np.where(cv_mean <= threshold)[0][-1]
        lambda_1se = self.lambda_valores[idx_1se]
        
        return {
            'lambda_otimo': lambda_otimo,
            'lambda_1se': lambda_1se,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'lambdas': self.lambda_valores
        }
    
    def plotar_caminho_regularizacao(self, X, y, cv_resultado, titulo, salvar=None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Caminho dos coeficientes
        betas = []
        for lambda_ in self.lambda_valores:
            res = self.ajustar(X, y, lambda_)
            betas.append(res['beta'])
        betas = np.array(betas)
        
        ax = axes[0, 0]
        for i in range(betas.shape[1]):
            ax.semilogx(self.lambda_valores, betas[:, i], lw=1)
        ax.axvline(cv_resultado['lambda_otimo'], color='r', ls='--', label='λ ótimo')
        ax.set_xlabel('λ'); ax.set_ylabel('β')
        ax.set_title('Caminho de Regularização'); ax.legend()
        
        # CV score
        ax = axes[0, 1]
        ax.semilogx(cv_resultado['lambdas'], cv_resultado['cv_mean'], 'b-')
        ax.fill_between(cv_resultado['lambdas'],
                       cv_resultado['cv_mean'] - cv_resultado['cv_std'],
                       cv_resultado['cv_mean'] + cv_resultado['cv_std'], alpha=0.2)
        ax.axvline(cv_resultado['lambda_otimo'], color='r', ls='--')
        ax.axvline(cv_resultado['lambda_1se'], color='g', ls='--')
        ax.set_xlabel('λ'); ax.set_ylabel('MSE (CV)')
        ax.set_title('Validação Cruzada')
        
        # Ajuste com λ ótimo
        resultado = self.ajustar(X, y, cv_resultado['lambda_otimo'])
        ax = axes[1, 0]
        ax.scatter(resultado['y_pred'], y, s=5, alpha=0.5)
        lim = [min(resultado['y_pred'].min(), y.min()), max(resultado['y_pred'].max(), y.max())]
        ax.plot(lim, lim, 'r--')
        ax.set_xlabel('Predito'); ax.set_ylabel('Observado')
        ax.set_title(f"R² = {resultado['r2']:.4f}")
        
        # Coeficientes
        ax = axes[1, 1]
        ax.bar(range(len(resultado['beta'])), resultado['beta'], alpha=0.7)
        ax.set_xlabel('Feature'); ax.set_ylabel('β')
        ax.set_title(f"Coeficientes (λ = {cv_resultado['lambda_otimo']:.4f})")
        
        for ax in axes.flat: ax.grid(True, alpha=0.3)
        plt.suptitle(titulo, fontweight='bold'); plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_17(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 17: REGRESSÃO RIDGE\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        t, f = dados['tempo'], dados['fluxo']
        X = np.column_stack([t - t[0], (t - t[0])**2, np.sin(2*np.pi*t/10), np.cos(2*np.pi*t/10)])
        
        ridge = RegressaoRidge()
        cv = ridge.validacao_cruzada(X, f, k=5)
        resultado = ridge.ajustar(X, f, cv['lambda_otimo'])
        
        print(f"    λ ótimo: {cv['lambda_otimo']:.4f}")
        print(f"    λ 1-SE: {cv['lambda_1se']:.4f}")
        print(f"    R²: {resultado['r2']:.4f}")
        print(f"    df efetivo: {resultado['df_efetivo']:.2f}")
        
        arq = os.path.join(diretorio_saida, f"ridge_{nome.replace(' ', '_').lower()}.png")
        ridge.plotar_caminho_regularizacao(X, f, cv, f"Ridge - {nome}", arq)
        
        resultados[nome] = {**dados, 'ridge': resultado, 'ridge_cv': cv}
    
    plt.close('all')
    print("\nMÓDULO 17 CONCLUÍDO")
    return resultados

__all__ = ['RegressaoRidge', 'executar_modulo_17']
