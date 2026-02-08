"""
Módulo 11: Modelos SARIMA
Autor: Luiz Tiago Wilcke
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional
import os


class ModeloSARIMA:
    """
    Modelo SARIMA(p,d,q)(P,D,Q)_s para séries com sazonalidade.
    Combina componentes ARIMA regulares com sazonais.
    """
    
    def __init__(self, ordem: Tuple = (1, 0, 1), ordem_sazonal: Tuple = (1, 0, 1), periodo: int = 100):
        self.p, self.d, self.q = ordem
        self.P, self.D, self.Q = ordem_sazonal
        self.s = periodo
        self.params = None
        self.residuos = None
        self.sigma2 = None
    
    def diferenciar_sazonal(self, dados: np.ndarray) -> np.ndarray:
        """Aplica diferenciação sazonal: (1 - B^s)^D."""
        resultado = dados.copy()
        for _ in range(self.D):
            resultado = resultado[self.s:] - resultado[:-self.s]
        return resultado
    
    def _calcular_residuos(self, params: np.ndarray, dados: np.ndarray) -> np.ndarray:
        """Calcula resíduos do modelo."""
        c = params[0]
        phi = params[1:self.p+1]
        theta = params[self.p+1:self.p+self.q+1]
        Phi = params[self.p+self.q+1:self.p+self.q+self.P+1]
        Theta = params[self.p+self.q+self.P+1:]
        
        n = len(dados)
        inicio = max(self.p, self.q, self.P * self.s, self.Q * self.s)
        residuos = np.zeros(n)
        
        for t in range(inicio, n):
            pred = c
            # Termos AR regulares
            for i in range(self.p):
                pred += phi[i] * dados[t - i - 1]
            # Termos MA regulares
            for j in range(self.q):
                pred += theta[j] * residuos[t - j - 1]
            # Termos AR sazonais
            for i in range(self.P):
                idx = t - (i + 1) * self.s
                if idx >= 0:
                    pred += Phi[i] * dados[idx]
            # Termos MA sazonais
            for j in range(self.Q):
                idx = t - (j + 1) * self.s
                if idx >= 0:
                    pred += Theta[j] * residuos[idx]
            
            residuos[t] = dados[t] - pred
        
        return residuos
    
    def _log_verossimilhanca(self, params: np.ndarray, dados: np.ndarray) -> float:
        """Log-verossimilhança negativa."""
        try:
            residuos = self._calcular_residuos(params, dados)
            n_validos = len(dados) - max(self.p, self.q, self.P * self.s, self.Q * self.s)
            sigma2 = np.var(residuos[-n_validos:])
            if sigma2 <= 0:
                return 1e10
            ll = -0.5 * n_validos * np.log(2 * np.pi * sigma2) - np.sum(residuos[-n_validos:]**2) / (2 * sigma2)
            return -ll
        except:
            return 1e10
    
    def ajustar(self, dados: np.ndarray) -> Dict:
        """Ajusta o modelo SARIMA."""
        # Diferenciação
        dados_proc = dados.copy()
        if self.d > 0:
            for _ in range(self.d):
                dados_proc = np.diff(dados_proc)
        if self.D > 0:
            dados_proc = self.diferenciar_sazonal(dados_proc)
        
        # Parâmetros iniciais
        n_params = 1 + self.p + self.q + self.P + self.Q
        x0 = np.zeros(n_params)
        x0[0] = np.mean(dados_proc)
        
        # Otimizar
        resultado = minimize(self._log_verossimilhanca, x0, args=(dados_proc,),
                           method='L-BFGS-B', options={'maxiter': 300})
        
        self.params = resultado.x
        self.residuos = self._calcular_residuos(self.params, dados_proc)
        n_validos = len(dados_proc) - max(self.p, self.q, self.P * self.s, self.Q * self.s)
        self.sigma2 = np.var(self.residuos[-n_validos:])
        
        return {
            'constante': self.params[0],
            'phi': self.params[1:self.p+1],
            'theta': self.params[self.p+1:self.p+self.q+1],
            'Phi': self.params[self.p+self.q+1:self.p+self.q+self.P+1],
            'Theta': self.params[self.p+self.q+self.P+1:],
            'sigma2': self.sigma2,
            'aic': 2 * n_params + 2 * resultado.fun,
            'bic': n_params * np.log(len(dados_proc)) + 2 * resultado.fun
        }
    
    def plotar_resultado(self, dados: np.ndarray, titulo: str, salvar: Optional[str] = None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].plot(dados, 'b-', lw=0.5, alpha=0.7)
        axes[0, 0].set_title('Série Original'); axes[0, 0].set_xlabel('t')
        
        if self.residuos is not None:
            axes[0, 1].scatter(range(len(self.residuos)), self.residuos, s=1, alpha=0.5)
            axes[0, 1].axhline(0, color='r', ls='--')
            axes[0, 1].set_title('Resíduos')
            
            axes[1, 0].hist(self.residuos, bins=50, density=True, alpha=0.7)
            axes[1, 0].set_title('Distribuição dos Resíduos')
            
            # ACF sazonal
            acf = np.correlate(self.residuos, self.residuos, 'full')
            acf = acf[len(self.residuos)-1:] / acf[len(self.residuos)-1]
            lags = [0, 1, 2, self.s-1, self.s, self.s+1, 2*self.s]
            lags = [l for l in lags if l < len(acf)]
            axes[1, 1].bar(range(len(lags)), [acf[l] for l in lags], alpha=0.7)
            axes[1, 1].set_xticks(range(len(lags)))
            axes[1, 1].set_xticklabels([str(l) for l in lags])
            axes[1, 1].set_title('ACF em Lags Sazonais')
        
        for ax in axes.flat: ax.grid(True, alpha=0.3)
        ordem_str = f"SARIMA({self.p},{self.d},{self.q})({self.P},{self.D},{self.Q})_{self.s}"
        plt.suptitle(f"{titulo}\n{ordem_str}", fontweight='bold')
        plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_11(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 11: MODELOS SARIMA\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        f = dados['fluxo'][:2000]
        
        # Estimar período
        periodo = dados.get('periodos_detectados', {}).get('periodos', [100])[0] if 'periodos_detectados' in dados else 100
        periodo = int(min(max(periodo, 10), 200))
        
        modelo = ModeloSARIMA((1, 0, 1), (1, 0, 0), periodo)
        resultado = modelo.ajustar(f)
        
        print(f"    Período sazonal: {periodo}")
        print(f"    AIC: {resultado['aic']:.2f}")
        print(f"    σ²: {resultado['sigma2']:.6f}")
        
        arq = os.path.join(diretorio_saida, f"sarima_{nome.replace(' ', '_').lower()}.png")
        modelo.plotar_resultado(f, nome, arq)
        
        resultados[nome] = {**dados, 'sarima': resultado, 'periodo_sazonal': periodo}
    
    plt.close('all')
    print("\nMÓDULO 11 CONCLUÍDO")
    return resultados

__all__ = ['ModeloSARIMA', 'executar_modulo_11']
