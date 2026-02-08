"""
Módulo 09: Autocorrelação
Autor: Luiz Tiago Wilcke
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import os


class AnalisadorAutocorrelacao:
    """Análise de autocorrelação e correlação cruzada."""
    
    def calcular_acf(self, dados: np.ndarray, max_lag: int = None) -> np.ndarray:
        """
        Função de Autocorrelação (ACF).
        ρ(k) = Cov(X_t, X_{t+k}) / Var(X_t)
        """
        n = len(dados)
        if max_lag is None:
            max_lag = min(n // 4, 500)
        
        dados_cent = dados - np.mean(dados)
        variancia = np.var(dados)
        
        acf = np.zeros(max_lag + 1)
        for k in range(max_lag + 1):
            acf[k] = np.sum(dados_cent[:n-k] * dados_cent[k:]) / (n * variancia)
        
        return acf
    
    def calcular_pacf(self, dados: np.ndarray, max_lag: int = None) -> np.ndarray:
        """
        Função de Autocorrelação Parcial (PACF).
        Usa algoritmo de Durbin-Levinson.
        """
        acf = self.calcular_acf(dados, max_lag)
        n_lags = len(acf)
        pacf = np.zeros(n_lags)
        pacf[0] = 1.0
        
        if n_lags > 1:
            pacf[1] = acf[1]
        
        phi = np.zeros((n_lags, n_lags))
        phi[1, 1] = acf[1]
        
        for k in range(2, n_lags):
            # Numerador
            num = acf[k] - np.sum(phi[k-1, 1:k] * acf[k-1:0:-1])
            # Denominador
            den = 1 - np.sum(phi[k-1, 1:k] * acf[1:k])
            
            if abs(den) > 1e-10:
                phi[k, k] = num / den
            
            # Atualizar coeficientes
            for j in range(1, k):
                phi[k, j] = phi[k-1, j] - phi[k, k] * phi[k-1, k-j]
            
            pacf[k] = phi[k, k]
        
        return pacf
    
    def correlacao_cruzada(self, x: np.ndarray, y: np.ndarray, max_lag: int = None) -> Dict:
        """Correlação cruzada entre duas séries."""
        n = min(len(x), len(y))
        if max_lag is None:
            max_lag = n // 4
        
        x_cent = x[:n] - np.mean(x[:n])
        y_cent = y[:n] - np.mean(y[:n])
        
        lags = np.arange(-max_lag, max_lag + 1)
        ccf = np.zeros(len(lags))
        
        norm = np.sqrt(np.sum(x_cent**2) * np.sum(y_cent**2))
        
        for i, lag in enumerate(lags):
            if lag >= 0:
                ccf[i] = np.sum(x_cent[:n-lag] * y_cent[lag:]) / norm
            else:
                ccf[i] = np.sum(x_cent[-lag:] * y_cent[:n+lag]) / norm
        
        return {'lags': lags, 'ccf': ccf}
    
    def identificar_ordem_arima(self, acf: np.ndarray, pacf: np.ndarray,
                                 n_dados: int) -> Dict:
        """Sugere ordens p, q para ARIMA baseado em ACF/PACF."""
        limite_ic = 1.96 / np.sqrt(n_dados)
        
        # Ordem q (MA): último lag significativo na ACF
        q_sugerido = 0
        for k in range(1, len(acf)):
            if abs(acf[k]) > limite_ic:
                q_sugerido = k
        
        # Ordem p (AR): último lag significativo na PACF
        p_sugerido = 0
        for k in range(1, len(pacf)):
            if abs(pacf[k]) > limite_ic:
                p_sugerido = k
        
        return {
            'p_sugerido': min(p_sugerido, 5),
            'q_sugerido': min(q_sugerido, 5),
            'limite_ic': limite_ic
        }
    
    def plotar_acf_pacf(self, dados: np.ndarray, titulo: str, salvar: Optional[str] = None):
        acf = self.calcular_acf(dados, 50)
        pacf = self.calcular_pacf(dados, 50)
        n = len(dados)
        limite = 1.96 / np.sqrt(n)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # ACF
        axes[0, 0].bar(range(len(acf)), acf, alpha=0.7, color='#1f77b4')
        axes[0, 0].axhline(limite, color='r', ls='--', alpha=0.7)
        axes[0, 0].axhline(-limite, color='r', ls='--', alpha=0.7)
        axes[0, 0].axhline(0, color='k'); axes[0, 0].set_title('ACF')
        axes[0, 0].set_xlabel('Lag'); axes[0, 0].set_ylabel('Autocorrelação')
        
        # PACF
        axes[0, 1].bar(range(len(pacf)), pacf, alpha=0.7, color='#2ca02c')
        axes[0, 1].axhline(limite, color='r', ls='--', alpha=0.7)
        axes[0, 1].axhline(-limite, color='r', ls='--', alpha=0.7)
        axes[0, 1].axhline(0, color='k'); axes[0, 1].set_title('PACF')
        axes[0, 1].set_xlabel('Lag'); axes[0, 1].set_ylabel('Autocorrelação Parcial')
        
        # Série temporal
        axes[1, 0].plot(dados, 'b-', lw=0.5, alpha=0.7)
        axes[1, 0].set_title('Série Original'); axes[1, 0].set_xlabel('t')
        
        # Identificação
        ordem = self.identificar_ordem_arima(acf, pacf, n)
        axes[1, 1].axis('off')
        texto = f"Identificação de Ordem ARIMA(p,d,q):\n\n"
        texto += f"p sugerido (AR): {ordem['p_sugerido']}\n"
        texto += f"q sugerido (MA): {ordem['q_sugerido']}\n"
        texto += f"Limite IC 95%: ±{ordem['limite_ic']:.4f}"
        axes[1, 1].text(0.1, 0.5, texto, fontsize=14, va='center')
        
        for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]: ax.grid(True, alpha=0.3)
        plt.suptitle(titulo, fontweight='bold'); plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_09(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 09: AUTOCORRELAÇÃO\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    analisador = AnalisadorAutocorrelacao()
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        f = dados['fluxo']
        
        acf = analisador.calcular_acf(f, 50)
        pacf = analisador.calcular_pacf(f, 50)
        ordem = analisador.identificar_ordem_arima(acf, pacf, len(f))
        
        print(f"    p sugerido (AR): {ordem['p_sugerido']}")
        print(f"    q sugerido (MA): {ordem['q_sugerido']}")
        
        arq = os.path.join(diretorio_saida, f"acf_{nome.replace(' ', '_').lower()}.png")
        analisador.plotar_acf_pacf(f, f"ACF/PACF - {nome}", arq)
        
        resultados[nome] = {**dados, 'acf': acf, 'pacf': pacf, 'ordem_arima': ordem}
    
    plt.close('all')
    print("\nMÓDULO 09 CONCLUÍDO")
    return resultados

__all__ = ['AnalisadorAutocorrelacao', 'executar_modulo_09']
