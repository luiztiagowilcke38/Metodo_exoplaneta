"""
Módulo 10: Modelos ARIMA
Autor: Luiz Tiago Wilcke
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional
import os


class ModeloARIMA:
    """
    Modelo ARIMA(p,d,q) para séries temporais.
    X_t = c + Σφ_i X_{t-i} + ε_t + Σθ_j ε_{t-j}
    """
    
    def __init__(self, ordem: Tuple[int, int, int] = (1, 0, 1)):
        self.p, self.d, self.q = ordem
        self.phi = None  # Coeficientes AR
        self.theta = None  # Coeficientes MA
        self.constante = None
        self.sigma2 = None
        self.residuos = None
    
    def diferenciar(self, dados: np.ndarray, d: int = 1) -> np.ndarray:
        """Aplica diferenciação d vezes."""
        resultado = dados.copy()
        for _ in range(d):
            resultado = np.diff(resultado)
        return resultado
    
    def _log_verossimilhanca(self, params: np.ndarray, dados: np.ndarray) -> float:
        """Calcula log-verossimilhança negativa para otimização."""
        c = params[0]
        phi = params[1:self.p+1] if self.p > 0 else np.array([])
        theta = params[self.p+1:self.p+self.q+1] if self.q > 0 else np.array([])
        
        n = len(dados)
        residuos = np.zeros(n)
        
        for t in range(max(self.p, self.q), n):
            pred = c
            for i in range(self.p):
                if t - i - 1 >= 0:
                    pred += phi[i] * dados[t - i - 1]
            for j in range(self.q):
                if t - j - 1 >= 0:
                    pred += theta[j] * residuos[t - j - 1]
            residuos[t] = dados[t] - pred
        
        sigma2 = np.var(residuos[max(self.p, self.q):])
        if sigma2 <= 0:
            return 1e10
        
        ll = -0.5 * n * np.log(2 * np.pi * sigma2) - np.sum(residuos**2) / (2 * sigma2)
        return -ll
    
    def ajustar(self, dados: np.ndarray) -> Dict:
        """Ajusta o modelo ARIMA aos dados."""
        # Diferenciar se necessário
        dados_diff = self.diferenciar(dados, self.d) if self.d > 0 else dados
        
        # Parâmetros iniciais
        n_params = 1 + self.p + self.q
        x0 = np.zeros(n_params)
        x0[0] = np.mean(dados_diff)
        
        # Otimizar
        resultado = minimize(self._log_verossimilhanca, x0, args=(dados_diff,),
                           method='L-BFGS-B', options={'maxiter': 500})
        
        params = resultado.x
        self.constante = params[0]
        self.phi = params[1:self.p+1] if self.p > 0 else np.array([])
        self.theta = params[self.p+1:self.p+self.q+1] if self.q > 0 else np.array([])
        
        # Calcular resíduos finais
        n = len(dados_diff)
        self.residuos = np.zeros(n)
        for t in range(max(self.p, self.q), n):
            pred = self.constante
            for i in range(self.p):
                if t - i - 1 >= 0:
                    pred += self.phi[i] * dados_diff[t - i - 1]
            for j in range(self.q):
                if t - j - 1 >= 0:
                    pred += self.theta[j] * self.residuos[t - j - 1]
            self.residuos[t] = dados_diff[t] - pred
        
        self.sigma2 = np.var(self.residuos[max(self.p, self.q):])
        
        return {
            'constante': self.constante,
            'phi': self.phi,
            'theta': self.theta,
            'sigma2': self.sigma2,
            'aic': 2 * n_params + 2 * resultado.fun,
            'bic': n_params * np.log(n) + 2 * resultado.fun,
            'log_verossimilhanca': -resultado.fun
        }
    
    def prever(self, dados: np.ndarray, n_passos: int = 10) -> np.ndarray:
        """Previsão n passos à frente."""
        dados_ext = np.concatenate([dados, np.zeros(n_passos)])
        res_ext = np.concatenate([self.residuos, np.zeros(n_passos)])
        
        n = len(dados)
        for t in range(n, n + n_passos):
            pred = self.constante
            for i in range(self.p):
                pred += self.phi[i] * dados_ext[t - i - 1]
            for j in range(self.q):
                pred += self.theta[j] * res_ext[t - j - 1]
            dados_ext[t] = pred
        
        return dados_ext[n:]
    
    def plotar_diagnostico(self, dados: np.ndarray, titulo: str, salvar: Optional[str] = None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Dados e ajuste
        n = len(dados)
        ajustado = dados - np.concatenate([np.zeros(len(dados) - len(self.residuos)), self.residuos])
        axes[0, 0].plot(dados, 'b-', lw=0.5, alpha=0.7, label='Observado')
        axes[0, 0].plot(ajustado, 'r-', lw=1, alpha=0.8, label='Ajustado')
        axes[0, 0].legend(); axes[0, 0].set_title('Dados vs Ajuste')
        
        # Resíduos
        axes[0, 1].scatter(range(len(self.residuos)), self.residuos, s=1, alpha=0.5)
        axes[0, 1].axhline(0, color='r', ls='--')
        axes[0, 1].set_title('Resíduos')
        
        # Histograma dos resíduos
        axes[1, 0].hist(self.residuos, bins=50, density=True, alpha=0.7)
        axes[1, 0].set_title('Distribuição dos Resíduos')
        
        # ACF dos resíduos
        acf_res = np.correlate(self.residuos, self.residuos, 'full')
        acf_res = acf_res[len(self.residuos)-1:len(self.residuos)+30] / acf_res[len(self.residuos)-1]
        axes[1, 1].bar(range(len(acf_res)), acf_res, alpha=0.7)
        axes[1, 1].axhline(1.96/np.sqrt(n), color='r', ls='--')
        axes[1, 1].axhline(-1.96/np.sqrt(n), color='r', ls='--')
        axes[1, 1].set_title('ACF dos Resíduos')
        
        for ax in axes.flat: ax.grid(True, alpha=0.3)
        plt.suptitle(f"{titulo} - ARIMA({self.p},{self.d},{self.q})", fontweight='bold')
        plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_10(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 10: MODELOS ARIMA\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        f = dados['fluxo'][:2000]  # Limitar para eficiência
        
        # Testar algumas ordens
        melhor_aic = np.inf
        melhor_modelo = None
        melhor_ordem = (1, 0, 1)
        
        for p in [1, 2]:
            for q in [0, 1]:
                modelo = ModeloARIMA((p, 0, q))
                try:
                    resultado = modelo.ajustar(f)
                    if resultado['aic'] < melhor_aic:
                        melhor_aic = resultado['aic']
                        melhor_modelo = modelo
                        melhor_ordem = (p, 0, q)
                except:
                    continue
        
        if melhor_modelo:
            resultado = melhor_modelo.ajustar(f)
            print(f"    Melhor ordem: ARIMA{melhor_ordem}")
            print(f"    AIC: {resultado['aic']:.2f}")
            print(f"    σ²: {resultado['sigma2']:.6f}")
            
            arq = os.path.join(diretorio_saida, f"arima_{nome.replace(' ', '_').lower()}.png")
            melhor_modelo.plotar_diagnostico(f, nome, arq)
            
            resultados[nome] = {**dados, 'arima': resultado, 'ordem_arima': melhor_ordem}
    
    plt.close('all')
    print("\nMÓDULO 10 CONCLUÍDO")
    return resultados

__all__ = ['ModeloARIMA', 'executar_modulo_10']
