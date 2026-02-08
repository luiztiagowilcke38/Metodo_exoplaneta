"""
Módulo 24: Regressão com Splines
Autor: Luiz Tiago Wilcke

B-splines e smoothing splines para modelagem não-linear.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, splrep, splev
from scipy.linalg import solve_banded
from typing import Dict, Optional
import os


class RegressaoSplines:
    """
    Regressão com splines:
    - B-splines: base de funções polinomiais por partes
    - Smoothing splines: minimiza RSS + λ∫f''(x)²dx
    - Penalized splines (P-splines)
    """
    
    def __init__(self, n_nos: int = 10, grau: int = 3):
        self.n_nos = n_nos
        self.grau = grau
    
    def criar_base_bspline(self, x: np.ndarray, nos: np.ndarray = None) -> np.ndarray:
        """
        Cria matriz de base B-spline.
        B_{i,k}(x) definido recursivamente (Cox-de Boor).
        """
        if nos is None:
            nos = np.linspace(x.min(), x.max(), self.n_nos)
        
        # Adicionar nós extras nas extremidades
        nos_ext = np.concatenate([
            [x.min()] * self.grau,
            nos,
            [x.max()] * self.grau
        ])
        
        n_base = len(nos_ext) - self.grau - 1
        base = np.zeros((len(x), n_base))
        
        for i in range(n_base):
            coefs = np.zeros(n_base)
            coefs[i] = 1
            spl = BSpline(nos_ext, coefs, self.grau)
            base[:, i] = spl(x)
        
        return base, nos_ext
    
    def ajustar_bspline(self, x: np.ndarray, y: np.ndarray, 
                         nos: np.ndarray = None) -> Dict:
        """Ajusta regressão B-spline via OLS."""
        base, nos_ext = self.criar_base_bspline(x, nos)
        
        # OLS: β = (B'B)^{-1}B'y
        coefs = np.linalg.lstsq(base, y, rcond=None)[0]
        
        y_pred = base @ coefs
        residuos = y - y_pred
        
        rss = np.sum(residuos ** 2)
        tss = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - rss / tss
        
        # df efetivos = número de colunas da base
        df = base.shape[1]
        gcv = len(y) * rss / (len(y) - df) ** 2
        
        return {
            'coefs': coefs,
            'base': base,
            'nos': nos_ext,
            'y_pred': y_pred,
            'residuos': residuos,
            'r2': r2,
            'df': df,
            'gcv': gcv
        }
    
    def ajustar_pspline(self, x: np.ndarray, y: np.ndarray,
                         lambda_: float = 1.0, ordem_penalidade: int = 2) -> Dict:
        """
        P-spline: B-spline + penalidade nas diferenças dos coeficientes.
        min ||y - Bβ||² + λ||D_d β||²
        onde D_d é a matriz de diferenças de ordem d.
        """
        base, nos_ext = self.criar_base_bspline(x)
        n_base = base.shape[1]
        
        # Matriz de diferenças de ordem d
        D = np.eye(n_base)
        for _ in range(ordem_penalidade):
            D = np.diff(D, axis=0)
        
        # Solução: (B'B + λD'D)β = B'y
        BtB = base.T @ base
        DtD = D.T @ D
        Bty = base.T @ y
        
        coefs = np.linalg.solve(BtB + lambda_ * DtD, Bty)
        
        y_pred = base @ coefs
        residuos = y - y_pred
        
        rss = np.sum(residuos ** 2)
        tss = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - rss / tss
        
        # Graus de liberdade efetivos via matriz hat
        try:
            hat_diag = np.diag(base @ np.linalg.inv(BtB + lambda_ * DtD) @ base.T)
            df = np.sum(hat_diag)
        except:
            df = n_base
        
        gcv = len(y) * rss / (len(y) - df) ** 2
        
        return {
            'coefs': coefs,
            'lambda': lambda_,
            'y_pred': y_pred,
            'residuos': residuos,
            'r2': r2,
            'df': df,
            'gcv': gcv
        }
    
    def selecionar_lambda_cv(self, x: np.ndarray, y: np.ndarray,
                              lambdas: np.ndarray = None, k: int = 5) -> Dict:
        """Seleção de λ via validação cruzada."""
        if lambdas is None:
            lambdas = np.logspace(-4, 4, 50)
        
        n = len(y)
        indices = np.arange(n)
        np.random.shuffle(indices)
        folds = np.array_split(indices, k)
        
        cv_scores = np.zeros(len(lambdas))
        
        for i, lambda_ in enumerate(lambdas):
            scores = []
            for fold in folds:
                mascara = ~np.isin(np.arange(n), fold)
                try:
                    res = self.ajustar_pspline(x[mascara], y[mascara], lambda_)
                    base_teste, _ = self.criar_base_bspline(x[fold])
                    y_pred = base_teste @ res['coefs']
                    scores.append(np.mean((y[fold] - y_pred) ** 2))
                except:
                    scores.append(np.inf)
            cv_scores[i] = np.mean(scores)
        
        idx_melhor = np.argmin(cv_scores)
        
        return {
            'lambda_otimo': lambdas[idx_melhor],
            'cv_scores': cv_scores,
            'lambdas': lambdas
        }
    
    def plotar_resultados(self, x, y, resultado_bspline, resultado_pspline, titulo, salvar=None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ordem = np.argsort(x)
        
        # B-spline
        ax = axes[0, 0]
        ax.scatter(x, y, s=1, alpha=0.3, c='gray')
        ax.plot(x[ordem], resultado_bspline['y_pred'][ordem], 'b-', lw=2, label='B-spline')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(f"B-spline (R² = {resultado_bspline['r2']:.4f})")
        ax.legend()
        
        # P-spline
        ax = axes[0, 1]
        ax.scatter(x, y, s=1, alpha=0.3, c='gray')
        ax.plot(x[ordem], resultado_pspline['y_pred'][ordem], 'r-', lw=2, label='P-spline')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(f"P-spline (λ={resultado_pspline['lambda']:.2f}, R²={resultado_pspline['r2']:.4f})")
        ax.legend()
        
        # Base B-spline
        ax = axes[1, 0]
        base = resultado_bspline['base']
        for i in range(min(base.shape[1], 15)):
            ax.plot(x[ordem], base[ordem, i], lw=1, alpha=0.7)
        ax.set_xlabel('x'); ax.set_ylabel('Valor da base')
        ax.set_title(f'Funções de Base B-spline ({base.shape[1]} funções)')
        
        # Resíduos
        ax = axes[1, 1]
        ax.scatter(resultado_pspline['y_pred'], resultado_pspline['residuos'], s=2, alpha=0.5)
        ax.axhline(0, color='r', ls='--')
        ax.set_xlabel('Predito'); ax.set_ylabel('Resíduo')
        ax.set_title('Resíduos P-spline')
        
        for ax in axes.flat: ax.grid(True, alpha=0.3)
        plt.suptitle(titulo, fontweight='bold'); plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_24(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 24: REGRESSÃO COM SPLINES\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        t, f = dados['tempo'], dados['fluxo']
        
        splines = RegressaoSplines(n_nos=15, grau=3)
        res_bspline = splines.ajustar_bspline(t, f)
        
        cv = splines.selecionar_lambda_cv(t, f)
        res_pspline = splines.ajustar_pspline(t, f, cv['lambda_otimo'])
        
        print(f"    B-spline R²: {res_bspline['r2']:.4f}, df: {res_bspline['df']}")
        print(f"    P-spline R²: {res_pspline['r2']:.4f}, λ: {cv['lambda_otimo']:.4f}")
        
        arq = os.path.join(diretorio_saida, f"splines_{nome.replace(' ', '_').lower()}.png")
        splines.plotar_resultados(t, f, res_bspline, res_pspline, f"Splines - {nome}", arq)
        
        resultados[nome] = {**dados, 'bspline': res_bspline, 'pspline': res_pspline}
    
    plt.close('all')
    print("\nMÓDULO 24 CONCLUÍDO")
    return resultados

__all__ = ['RegressaoSplines', 'executar_modulo_24']
