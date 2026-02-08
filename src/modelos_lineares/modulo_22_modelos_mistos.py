"""
Módulo 22: Modelos Lineares Mistos
Autor: Luiz Tiago Wilcke

Modelos com efeitos fixos e aleatórios: y = Xβ + Zb + ε
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv, block_diag
from scipy.optimize import minimize
from typing import Dict, Optional
import os


class ModeloMisto:
    """
    Modelo Linear Misto via REML.
    y_i = X_i β + Z_i b_i + ε_i
    b_i ~ N(0, G), ε_i ~ N(0, R_i)
    """
    
    def __init__(self):
        self.beta = None
        self.var_efeito_aleatorio = None
        self.var_residual = None
    
    def criar_grupos(self, tempo: np.ndarray, n_grupos: int = 10) -> np.ndarray:
        """Divide dados em grupos para efeitos aleatórios."""
        indices = np.floor(np.linspace(0, n_grupos, len(tempo))).astype(int)
        indices = np.minimum(indices, n_grupos - 1)
        return indices
    
    def log_verossimilhanca_reml(self, theta: np.ndarray, y: np.ndarray,
                                   X: np.ndarray, Z: np.ndarray) -> float:
        """
        Log-verossimilhança restrita (REML).
        l_REML = -0.5[log|V| + log|X'V⁻¹X| + y'Py]
        onde P = V⁻¹ - V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹
        """
        sigma2_b = np.exp(theta[0])  # Variância do efeito aleatório
        sigma2_e = np.exp(theta[1])  # Variância residual
        
        n = len(y)
        p = X.shape[1]
        
        # V = ZGZ' + R = σ²_b ZZ' + σ²_e I
        V = sigma2_b * (Z @ Z.T) + sigma2_e * np.eye(n)
        
        try:
            V_inv = inv(V)
            log_det_V = np.log(np.linalg.det(V))
        except:
            return 1e10
        
        XtV_inv = X.T @ V_inv
        XtV_invX = XtV_inv @ X
        
        try:
            XtV_invX_inv = inv(XtV_invX)
            log_det_XtV_invX = np.log(np.linalg.det(XtV_invX))
        except:
            return 1e10
        
        # Matriz P
        P = V_inv - V_inv @ X @ XtV_invX_inv @ XtV_inv
        
        # REML
        reml = 0.5 * (log_det_V + log_det_XtV_invX + y @ P @ y)
        
        return reml
    
    def ajustar(self, X: np.ndarray, y: np.ndarray, grupos: np.ndarray) -> Dict:
        """
        Ajusta modelo misto via REML.
        """
        n = len(y)
        n_grupos = len(np.unique(grupos))
        
        # Matriz de design para efeitos aleatórios
        Z = np.zeros((n, n_grupos))
        for i, g in enumerate(grupos):
            Z[i, g] = 1
        
        # Adicionar intercepto aos efeitos fixos
        X_aug = np.column_stack([np.ones(n), X])
        
        # Otimizar componentes de variância
        theta0 = np.array([0.0, 0.0])  # log(sigma2_b), log(sigma2_e)
        
        resultado = minimize(
            self.log_verossimilhanca_reml,
            theta0,
            args=(y, X_aug, Z),
            method='L-BFGS-B'
        )
        
        sigma2_b = np.exp(resultado.x[0])
        sigma2_e = np.exp(resultado.x[1])
        
        self.var_efeito_aleatorio = sigma2_b
        self.var_residual = sigma2_e
        
        # Estimar efeitos fixos (β) via GLS
        V = sigma2_b * (Z @ Z.T) + sigma2_e * np.eye(n)
        V_inv = inv(V)
        
        XtV_inv = X_aug.T @ V_inv
        self.beta = inv(XtV_inv @ X_aug) @ XtV_inv @ y
        
        # BLUP para efeitos aleatórios (b)
        residuos = y - X_aug @ self.beta
        G = sigma2_b * np.eye(n_grupos)
        b_hat = G @ Z.T @ V_inv @ residuos
        
        # Valores ajustados
        y_pred_fixos = X_aug @ self.beta
        y_pred_total = y_pred_fixos + Z @ b_hat
        
        # ICC (Intraclass Correlation Coefficient)
        icc = sigma2_b / (sigma2_b + sigma2_e)
        
        # R² marginal e condicional
        var_fixos = np.var(y_pred_fixos)
        r2_marginal = var_fixos / (var_fixos + sigma2_b + sigma2_e)
        r2_condicional = (var_fixos + sigma2_b) / (var_fixos + sigma2_b + sigma2_e)
        
        return {
            'beta': self.beta,
            'efeitos_aleatorios': b_hat,
            'var_efeito_aleatorio': sigma2_b,
            'var_residual': sigma2_e,
            'icc': icc,
            'r2_marginal': r2_marginal,
            'r2_condicional': r2_condicional,
            'y_pred_fixos': y_pred_fixos,
            'y_pred_total': y_pred_total,
            'residuos': y - y_pred_total,
            'reml': resultado.fun
        }
    
    def plotar_resultados(self, X, y, grupos, resultado, titulo, salvar=None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Dados por grupo
        ax = axes[0, 0]
        for g in np.unique(grupos):
            mascara = grupos == g
            ax.scatter(X[mascara, 0] if X.ndim > 1 else X[mascara], y[mascara], s=3, alpha=0.3)
        ax.set_xlabel('X'); ax.set_ylabel('y')
        ax.set_title(f'Dados por Grupo ({len(np.unique(grupos))} grupos)')
        
        # Efeitos aleatórios
        ax = axes[0, 1]
        ax.bar(range(len(resultado['efeitos_aleatorios'])), resultado['efeitos_aleatorios'], alpha=0.7)
        ax.axhline(0, color='r', ls='--')
        ax.set_xlabel('Grupo'); ax.set_ylabel('Efeito Aleatório')
        ax.set_title('BLUPs dos Efeitos Aleatórios')
        
        # Ajuste
        ax = axes[1, 0]
        ax.scatter(resultado['y_pred_total'], y, s=2, alpha=0.3)
        lim = [min(y.min(), resultado['y_pred_total'].min()),
               max(y.max(), resultado['y_pred_total'].max())]
        ax.plot(lim, lim, 'r--')
        ax.set_xlabel('Predito'); ax.set_ylabel('Observado')
        ax.set_title(f"R² cond. = {resultado['r2_condicional']:.4f}")
        
        # Componentes de variância
        ax = axes[1, 1]
        ax.axis('off')
        info = f"""Componentes de Variância:
        
σ²_aleatório:  {resultado['var_efeito_aleatorio']:.6f}
σ²_residual:   {resultado['var_residual']:.6f}
ICC:           {resultado['icc']:.4f}

R² marginal:     {resultado['r2_marginal']:.4f}
R² condicional:  {resultado['r2_condicional']:.4f}
REML:            {resultado['reml']:.2f}"""
        ax.text(0.1, 0.5, info, fontsize=12, family='monospace', va='center')
        
        for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]: ax.grid(True, alpha=0.3)
        plt.suptitle(titulo, fontweight='bold'); plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_22(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 22: MODELOS MISTOS\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        t, f = dados['tempo'][:2000], dados['fluxo'][:2000]  # Limitar
        X = np.column_stack([t - t[0]])
        
        misto = ModeloMisto()
        grupos = misto.criar_grupos(t, n_grupos=20)
        resultado = misto.ajustar(X, f, grupos)
        
        print(f"    ICC: {resultado['icc']:.4f}")
        print(f"    R² marginal: {resultado['r2_marginal']:.4f}")
        print(f"    R² condicional: {resultado['r2_condicional']:.4f}")
        
        arq = os.path.join(diretorio_saida, f"misto_{nome.replace(' ', '_').lower()}.png")
        misto.plotar_resultados(X, f, grupos, resultado, f"Modelo Misto - {nome}", arq)
        
        resultados[nome] = {**dados, 'modelo_misto': resultado}
    
    plt.close('all')
    print("\nMÓDULO 22 CONCLUÍDO")
    return resultados

__all__ = ['ModeloMisto', 'executar_modulo_22']
