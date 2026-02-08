"""
Módulo 16: Regressão Linear Múltipla
Sistema Estatístico para Detecção de Exoplanetas
Autor: Luiz Tiago Wilcke

Implementação completa de regressão linear com diagnósticos estatísticos.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import inv, lstsq
from typing import Dict, Optional, Tuple
import os


class RegressaoLinear:
    """
    Regressão linear múltipla via OLS com diagnósticos completos.
    
    Modelo: Y = Xβ + ε, onde ε ~ N(0, σ²I)
    Estimador OLS: β̂ = (X'X)⁻¹X'Y
    """
    
    def __init__(self):
        self.beta = None
        self.residuos = None
        self.sigma2 = None
        self.cov_beta = None
        self.r2 = None
        self.r2_ajustado = None
    
    def ajustar(self, X: np.ndarray, y: np.ndarray, 
                intercepto: bool = True) -> Dict:
        """
        Ajusta modelo de regressão linear.
        
        Parâmetros:
            X: Matriz de preditores [n x p]
            y: Vetor resposta [n]
            intercepto: Se True, adiciona coluna de 1s
        """
        n = len(y)
        
        if intercepto:
            X = np.column_stack([np.ones(n), X])
        
        p = X.shape[1]
        
        # Estimador OLS: β̂ = (X'X)⁻¹X'y
        XtX = X.T @ X
        Xty = X.T @ y
        
        try:
            XtX_inv = inv(XtX)
            self.beta = XtX_inv @ Xty
        except:
            self.beta, _, _, _ = lstsq(X, y)
            XtX_inv = np.linalg.pinv(XtX)
        
        # Valores ajustados e resíduos
        y_pred = X @ self.beta
        self.residuos = y - y_pred
        
        # Variância residual: σ² = RSS / (n - p)
        rss = np.sum(self.residuos ** 2)
        self.sigma2 = rss / (n - p)
        
        # Covariância dos estimadores: Var(β̂) = σ²(X'X)⁻¹
        self.cov_beta = self.sigma2 * XtX_inv
        
        # Erros padrão
        se_beta = np.sqrt(np.diag(self.cov_beta))
        
        # Estatísticas t e p-valores
        t_stats = self.beta / se_beta
        p_valores = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p))
        
        # R² e R² ajustado
        tss = np.sum((y - np.mean(y)) ** 2)
        self.r2 = 1 - rss / tss
        self.r2_ajustado = 1 - (1 - self.r2) * (n - 1) / (n - p)
        
        # Estatística F
        if p > 1:
            f_stat = (tss - rss) / (p - 1) / (rss / (n - p))
            f_pvalor = 1 - stats.f.cdf(f_stat, p - 1, n - p)
        else:
            f_stat = np.nan
            f_pvalor = np.nan
        
        # Log-verossimilhança
        log_lik = -n/2 * np.log(2 * np.pi * self.sigma2) - rss / (2 * self.sigma2)
        
        # Critérios de informação
        aic = -2 * log_lik + 2 * p
        bic = -2 * log_lik + p * np.log(n)
        
        return {
            'beta': self.beta,
            'se_beta': se_beta,
            't_stats': t_stats,
            'p_valores': p_valores,
            'sigma2': self.sigma2,
            'r2': self.r2,
            'r2_ajustado': self.r2_ajustado,
            'f_stat': f_stat,
            'f_pvalor': f_pvalor,
            'aic': aic,
            'bic': bic,
            'log_verossimilhanca': log_lik,
            'residuos': self.residuos,
            'y_pred': y_pred
        }
    
    def diagnosticos(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Testes de diagnóstico da regressão."""
        n = len(y)
        
        # Teste de Durbin-Watson (autocorrelação)
        dw = np.sum(np.diff(self.residuos) ** 2) / np.sum(self.residuos ** 2)
        
        # Teste de Jarque-Bera (normalidade)
        jb_stat, jb_pvalor = stats.jarque_bera(self.residuos)
        
        # Teste de Breusch-Pagan (heterocedasticidade)
        res2 = self.residuos ** 2
        X_bp = np.column_stack([np.ones(n), X])
        beta_bp = lstsq(X_bp, res2)[0]
        res2_pred = X_bp @ beta_bp
        ss_bp = np.sum((res2 - res2_pred) ** 2)
        ss_tot = np.sum((res2 - np.mean(res2)) ** 2)
        r2_bp = 1 - ss_bp / ss_tot
        bp_stat = n * r2_bp
        bp_pvalor = 1 - stats.chi2.cdf(bp_stat, X.shape[1] if X.ndim > 1 else 1)
        
        # Alavancagem (hat values)
        X_full = np.column_stack([np.ones(n), X]) if X.ndim > 1 else np.column_stack([np.ones(n), X])
        H = X_full @ inv(X_full.T @ X_full) @ X_full.T
        alavancagem = np.diag(H)
        
        # Resíduos studentizados
        mse = np.sum(self.residuos ** 2) / (n - X_full.shape[1])
        residuos_stud = self.residuos / np.sqrt(mse * (1 - alavancagem))
        
        # Distância de Cook
        p = X_full.shape[1]
        cook_d = (residuos_stud ** 2 / p) * (alavancagem / (1 - alavancagem))
        
        return {
            'durbin_watson': dw,
            'jarque_bera': (jb_stat, jb_pvalor),
            'breusch_pagan': (bp_stat, bp_pvalor),
            'alavancagem': alavancagem,
            'residuos_studentizados': residuos_stud,
            'cook_d': cook_d,
            'pontos_influentes': np.where(cook_d > 4 / n)[0]
        }
    
    def plotar_diagnosticos(self, X, y, resultado, titulo, salvar=None):
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # Valores ajustados vs observados
        ax = axes[0, 0]
        ax.scatter(resultado['y_pred'], y, s=5, alpha=0.5)
        lim = [min(resultado['y_pred'].min(), y.min()), max(resultado['y_pred'].max(), y.max())]
        ax.plot(lim, lim, 'r--', lw=2)
        ax.set_xlabel('Predito'); ax.set_ylabel('Observado')
        ax.set_title(f"R² = {resultado['r2']:.4f}")
        
        # Resíduos vs ajustados
        ax = axes[0, 1]
        ax.scatter(resultado['y_pred'], resultado['residuos'], s=5, alpha=0.5)
        ax.axhline(0, color='r', ls='--')
        ax.set_xlabel('Predito'); ax.set_ylabel('Resíduo')
        ax.set_title('Resíduos vs Ajustados')
        
        # Q-Q plot
        ax = axes[0, 2]
        stats.probplot(resultado['residuos'], dist="norm", plot=ax)
        ax.set_title('Q-Q Plot dos Resíduos')
        
        # Histograma dos resíduos
        ax = axes[1, 0]
        ax.hist(resultado['residuos'], bins=50, density=True, alpha=0.7)
        x = np.linspace(resultado['residuos'].min(), resultado['residuos'].max(), 100)
        ax.plot(x, stats.norm.pdf(x, 0, np.sqrt(resultado['sigma2'])), 'r-', lw=2)
        ax.set_title('Distribuição dos Resíduos')
        
        # Scale-location
        ax = axes[1, 1]
        res_sqrt = np.sqrt(np.abs(resultado['residuos'] / np.std(resultado['residuos'])))
        ax.scatter(resultado['y_pred'], res_sqrt, s=5, alpha=0.5)
        ax.set_xlabel('Predito'); ax.set_ylabel('√|Resíduo Padronizado|')
        ax.set_title('Scale-Location')
        
        # Cook's distance
        ax = axes[1, 2]
        diag = self.diagnosticos(X, y)
        ax.bar(range(len(diag['cook_d'])), diag['cook_d'], alpha=0.7)
        ax.axhline(4 / len(y), color='r', ls='--')
        ax.set_xlabel('Observação'); ax.set_ylabel("Cook's D")
        ax.set_title('Distância de Cook')
        
        for ax in axes.flat: ax.grid(True, alpha=0.3)
        plt.suptitle(titulo, fontweight='bold'); plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_16(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 16: REGRESSÃO LINEAR MÚLTIPLA\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        t, f = dados['tempo'], dados['fluxo']
        
        # Criar features: tempo, tempo², etc.
        X = np.column_stack([t - t[0], (t - t[0])**2, np.sin(2*np.pi*t/10)])
        
        reg = RegressaoLinear()
        resultado = reg.ajustar(X, f)
        diag = reg.diagnosticos(X, f)
        
        print(f"    R²: {resultado['r2']:.4f}")
        print(f"    R² ajustado: {resultado['r2_ajustado']:.4f}")
        print(f"    AIC: {resultado['aic']:.2f}")
        print(f"    Durbin-Watson: {diag['durbin_watson']:.2f}")
        print(f"    Pontos influentes: {len(diag['pontos_influentes'])}")
        
        arq = os.path.join(diretorio_saida, f"regressao_{nome.replace(' ', '_').lower()}.png")
        reg.plotar_diagnosticos(X, f, resultado, f"Regressão Linear - {nome}", arq)
        
        resultados[nome] = {**dados, 'regressao_linear': resultado, 'diagnosticos': diag}
    
    plt.close('all')
    print("\nMÓDULO 16 CONCLUÍDO")
    return resultados

__all__ = ['RegressaoLinear', 'executar_modulo_16']
