"""
Módulo 25: Modelos Aditivos Generalizados (GAM)
Autor: Luiz Tiago Wilcke

GAM: g(E[Y]) = β₀ + f₁(x₁) + f₂(x₂) + ... + fₚ(xₚ)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from typing import Dict, List, Optional
import os


class ModeloGAM:
    """
    Modelo Aditivo Generalizado com backfitting.
    Cada f_j é uma função suave (spline penalizado).
    """
    
    def __init__(self, n_nos: int = 10, grau: int = 3, lambdas: List[float] = None):
        self.n_nos = n_nos
        self.grau = grau
        self.lambdas = lambdas
        self.funcoes = {}
    
    def criar_base_spline(self, x: np.ndarray) -> np.ndarray:
        """Cria base B-spline."""
        nos = np.linspace(x.min(), x.max(), self.n_nos)
        nos_ext = np.concatenate([
            [x.min()] * self.grau, nos, [x.max()] * self.grau
        ])
        
        n_base = len(nos_ext) - self.grau - 1
        base = np.zeros((len(x), n_base))
        
        for i in range(n_base):
            coefs = np.zeros(n_base)
            coefs[i] = 1
            spl = BSpline(nos_ext, coefs, self.grau)
            base[:, i] = spl(x)
        
        return base, nos_ext
    
    def ajustar_smooth(self, x: np.ndarray, y: np.ndarray, lambda_: float) -> Dict:
        """Ajusta função suave com P-spline."""
        base, nos = self.criar_base_spline(x)
        n_base = base.shape[1]
        
        # Matriz de diferenças de segunda ordem
        D = np.diff(np.eye(n_base), n=2, axis=0)
        
        # Solução penalizada
        BtB = base.T @ base
        DtD = D.T @ D
        Bty = base.T @ y
        
        coefs = np.linalg.solve(BtB + lambda_ * DtD, Bty)
        
        return {
            'base': base,
            'coefs': coefs,
            'nos': nos,
            'fitted': base @ coefs
        }
    
    def backfitting(self, X: np.ndarray, y: np.ndarray, 
                     max_iter: int = 100, tol: float = 1e-6) -> Dict:
        """
        Algoritmo de backfitting para ajustar GAM.
        
        Para cada j:
            R_j = y - α - Σ_{k≠j} f_k(x_k)
            Ajustar f_j(x_j) aos resíduos parciais R_j
        """
        n, p = X.shape
        
        if self.lambdas is None:
            self.lambdas = [1.0] * p
        
        # Inicializar
        alpha = np.mean(y)
        f_values = np.zeros((n, p))
        
        for iteration in range(max_iter):
            f_old = f_values.copy()
            
            for j in range(p):
                # Resíduos parciais
                r_j = y - alpha - np.sum(f_values[:, np.arange(p) != j], axis=1)
                
                # Ajustar smooth
                smooth = self.ajustar_smooth(X[:, j], r_j, self.lambdas[j])
                f_values[:, j] = smooth['fitted']
                
                # Centralizar
                f_values[:, j] -= np.mean(f_values[:, j])
                
                self.funcoes[j] = smooth
            
            # Atualizar intercepto
            alpha = np.mean(y - np.sum(f_values, axis=1))
            
            # Convergência
            if np.max(np.abs(f_values - f_old)) < tol:
                break
        
        y_pred = alpha + np.sum(f_values, axis=1)
        residuos = y - y_pred
        
        rss = np.sum(residuos ** 2)
        tss = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - rss / tss
        
        # Graus de liberdade aproximados
        df_total = 1  # intercepto
        for j in range(p):
            df_total += self.funcoes[j]['base'].shape[1]
        
        aic = n * np.log(rss / n) + 2 * df_total
        
        return {
            'alpha': alpha,
            'f_values': f_values,
            'y_pred': y_pred,
            'residuos': residuos,
            'r2': r2,
            'aic': aic,
            'df': df_total,
            'iteracoes': iteration + 1
        }
    
    def prever(self, X_novo: np.ndarray) -> np.ndarray:
        """Predição para novos dados."""
        n = X_novo.shape[0]
        p = len(self.funcoes)
        
        y_pred = np.ones(n) * self.alpha
        
        for j in range(p):
            base, _ = self.criar_base_spline(X_novo[:, j])
            y_pred += base @ self.funcoes[j]['coefs']
        
        return y_pred
    
    def plotar_efeitos_parciais(self, X, resultado, titulo, salvar=None):
        n_vars = X.shape[1]
        n_cols = min(3, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols + 1
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = np.atleast_2d(axes)
        
        # Efeitos parciais
        for j in range(n_vars):
            ax = axes[j // n_cols, j % n_cols]
            ordem = np.argsort(X[:, j])
            
            # Resíduos parciais
            r_parcial = resultado['residuos'] + resultado['f_values'][:, j]
            ax.scatter(X[:, j], r_parcial, s=1, alpha=0.3, c='gray')
            ax.plot(X[ordem, j], resultado['f_values'][ordem, j], 'b-', lw=2)
            
            ax.axhline(0, color='r', ls='--', alpha=0.5)
            ax.set_xlabel(f'x{j+1}'); ax.set_ylabel(f'f{j+1}(x{j+1})')
            ax.set_title(f'Efeito Parcial de x{j+1}')
        
        # Esconder eixos vazios
        for i in range(n_vars, n_rows * n_cols - 2):
            axes[i // n_cols, i % n_cols].axis('off')
        
        # Ajuste geral
        ax_ajuste = axes[-1, 0]
        ax_ajuste.scatter(resultado['y_pred'], resultado['y_pred'] + resultado['residuos'], 
                         s=2, alpha=0.3)
        lim = [resultado['y_pred'].min(), resultado['y_pred'].max()]
        ax_ajuste.plot(lim, lim, 'r--')
        ax_ajuste.set_xlabel('Predito'); ax_ajuste.set_ylabel('Observado')
        ax_ajuste.set_title(f"R² = {resultado['r2']:.4f}")
        
        # Resíduos
        ax_res = axes[-1, 1] if n_cols > 1 else axes[-1, 0]
        ax_res.hist(resultado['residuos'], bins=50, density=True, alpha=0.7)
        ax_res.set_xlabel('Resíduo'); ax_res.set_ylabel('Densidade')
        ax_res.set_title('Distribuição dos Resíduos')
        
        for ax in axes.flat:
            if ax.has_data(): ax.grid(True, alpha=0.3)
        
        plt.suptitle(titulo, fontweight='bold'); plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_25(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 25: MODELOS ADITIVOS GENERALIZADOS\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        t, f = dados['tempo'], dados['fluxo']
        
        # Features não-lineares
        X = np.column_stack([
            t - t[0],
            np.sin(2 * np.pi * t / 10),
            np.cos(2 * np.pi * t / 10)
        ])
        
        gam = ModeloGAM(n_nos=10, lambdas=[1.0, 10.0, 10.0])
        resultado = gam.backfitting(X, f)
        
        print(f"    R²: {resultado['r2']:.4f}")
        print(f"    AIC: {resultado['aic']:.2f}")
        print(f"    Iterações: {resultado['iteracoes']}")
        
        arq = os.path.join(diretorio_saida, f"gam_{nome.replace(' ', '_').lower()}.png")
        gam.plotar_efeitos_parciais(X, resultado, f"GAM - {nome}", arq)
        
        resultados[nome] = {**dados, 'gam': resultado}
    
    plt.close('all')
    print("\nMÓDULO 25 CONCLUÍDO")
    return resultados

__all__ = ['ModeloGAM', 'executar_modulo_25']
