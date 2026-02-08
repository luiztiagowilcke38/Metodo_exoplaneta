"""
Módulo 35: Análise de Posteriors
Sistema Estatístico para Detecção de Exoplanetas
Autor: Luiz Tiago Wilcke

Ferramentas para análise e diagnóstico de distribuições posteriores.
Inclui sumários, intervalos de credibilidade e visualizações.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Optional, Tuple
import os


class AnalisadorPosterior:
    """
    Análise completa de distribuições posteriores MCMC.
    
    Funcionalidades:
    - Estatísticas sumárias (média, mediana, moda, IC)
    - Diagnósticos de convergência (R-hat, ESS, ACF)
    - Análise de correlação entre parâmetros
    - Visualizações (trace, corner, forest plots)
    """
    
    def __init__(self, amostras: np.ndarray, nomes_parametros: List[str] = None):
        """
        Inicializa analisador.
        
        Parâmetros:
            amostras: Array [n_amostras x n_parametros]
            nomes_parametros: Lista de nomes dos parâmetros
        """
        self.amostras = np.atleast_2d(amostras)
        if self.amostras.shape[0] < self.amostras.shape[1]:
            self.amostras = self.amostras.T
        
        self.n_amostras, self.n_params = self.amostras.shape
        
        if nomes_parametros is None:
            self.nomes = [f'θ{i}' for i in range(self.n_params)]
        else:
            self.nomes = nomes_parametros
    
    def sumario(self) -> Dict:
        """Calcula estatísticas sumárias para cada parâmetro."""
        sumarios = {}
        
        for i, nome in enumerate(self.nomes):
            x = self.amostras[:, i]
            
            # Estatísticas básicas
            media = np.mean(x)
            mediana = np.median(x)
            desvio = np.std(x)
            
            # Moda (via KDE)
            try:
                kde = stats.gaussian_kde(x)
                x_grid = np.linspace(x.min(), x.max(), 1000)
                moda = x_grid[np.argmax(kde(x_grid))]
            except:
                moda = mediana
            
            # Intervalos de credibilidade
            hdi_95 = self._hdi(x, 0.95)
            hdi_89 = self._hdi(x, 0.89)
            
            # Quantis
            q = np.percentile(x, [2.5, 25, 50, 75, 97.5])
            
            sumarios[nome] = {
                'media': media,
                'mediana': mediana,
                'moda': moda,
                'desvio': desvio,
                'hdi_95': hdi_95,
                'hdi_89': hdi_89,
                'q2.5': q[0],
                'q25': q[1],
                'q50': q[2],
                'q75': q[3],
                'q97.5': q[4]
            }
        
        return sumarios
    
    def _hdi(self, x: np.ndarray, prob: float = 0.95) -> Tuple[float, float]:
        """
        Calcula Highest Density Interval (HDI).
        Menor intervalo que contém prob% da distribuição.
        """
        x_sorted = np.sort(x)
        n = len(x)
        credMass = int(np.floor(prob * n))
        
        intervalWidth = x_sorted[credMass:] - x_sorted[:(n - credMass)]
        minIdx = np.argmin(intervalWidth)
        
        return (x_sorted[minIdx], x_sorted[minIdx + credMass])
    
    def diagnosticos_convergencia(self) -> Dict:
        """Calcula diagnósticos de convergência MCMC."""
        diagnosticos = {}
        
        for i, nome in enumerate(self.nomes):
            x = self.amostras[:, i]
            
            # ESS (Effective Sample Size)
            acf = self._autocorrelacao(x)
            tau = 1 + 2 * np.sum(acf[1:])
            ess = self.n_amostras / max(tau, 1)
            
            # R-hat (dividir cadeia em 2)
            meio = self.n_amostras // 2
            x1, x2 = x[:meio], x[meio:]
            B = 0.5 * ((np.mean(x1) - np.mean(x2))**2) * meio
            W = 0.5 * (np.var(x1) + np.var(x2))
            var_hat = (1 - 1/meio) * W + B/meio
            r_hat = np.sqrt(var_hat / W) if W > 0 else np.nan
            
            # MCSE (Monte Carlo Standard Error)
            mcse = np.std(x) / np.sqrt(ess)
            
            diagnosticos[nome] = {
                'ess': ess,
                'ess_por_segundo': np.nan,  # Seria calculado com tempo de execução
                'r_hat': r_hat,
                'mcse': mcse,
                'convergiu': r_hat < 1.1 if not np.isnan(r_hat) else True,
                'acf': acf[:50]
            }
        
        return diagnosticos
    
    def _autocorrelacao(self, x: np.ndarray, max_lag: int = 100) -> np.ndarray:
        """Calcula função de autocorrelação."""
        n = len(x)
        x_cent = x - np.mean(x)
        result = np.correlate(x_cent, x_cent, mode='full')[n-1:n+max_lag]
        return result / result[0] if result[0] != 0 else result
    
    def correlacao_parametros(self) -> np.ndarray:
        """Calcula matriz de correlação entre parâmetros."""
        return np.corrcoef(self.amostras.T)
    
    def probabilidade_direcional(self, idx: int, direcao: str = 'positivo') -> float:
        """
        Calcula probabilidade de um parâmetro ser positivo/negativo.
        Útil para testar hipóteses direcionais.
        """
        x = self.amostras[:, idx]
        if direcao == 'positivo':
            return np.mean(x > 0)
        else:
            return np.mean(x < 0)
    
    def plotar_trace(self, salvar: Optional[str] = None) -> plt.Figure:
        """Gera trace plots para todos os parâmetros."""
        n_rows = min(self.n_params, 4)
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        diag = self.diagnosticos_convergencia()
        
        for i in range(n_rows):
            # Trace
            ax = axes[i, 0]
            ax.plot(self.amostras[:, i], lw=0.3, alpha=0.7)
            ax.set_ylabel(self.nomes[i])
            if i == n_rows - 1: ax.set_xlabel('Iteração')
            ax.set_title(f'R-hat = {diag[self.nomes[i]]["r_hat"]:.3f}')
            
            # Posterior
            ax = axes[i, 1]
            ax.hist(self.amostras[:, i], bins=50, density=True, alpha=0.7)
            ax.axvline(np.mean(self.amostras[:, i]), color='r', ls='--', label='Média')
            ax.axvline(np.median(self.amostras[:, i]), color='g', ls=':', label='Mediana')
            ax.set_title(f'ESS = {diag[self.nomes[i]]["ess"]:.0f}')
            ax.legend(fontsize=8)
        
        for ax in axes.flat: ax.grid(True, alpha=0.3)
        plt.suptitle('Diagnóstico MCMC', fontweight='bold')
        plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig
    
    def plotar_corner(self, salvar: Optional[str] = None) -> plt.Figure:
        """Gera corner plot (matriz de dispersão + marginais)."""
        n = min(self.n_params, 5)
        fig, axes = plt.subplots(n, n, figsize=(3*n, 3*n))
        
        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal: histograma marginal
                    ax.hist(self.amostras[:, i], bins=30, density=True, alpha=0.7)
                    ax.axvline(np.mean(self.amostras[:, i]), color='r', ls='--')
                elif i > j:
                    # Abaixo: scatter plot
                    ax.scatter(self.amostras[:, j], self.amostras[:, i], s=1, alpha=0.1)
                else:
                    # Acima: correlação
                    corr = np.corrcoef(self.amostras[:, i], self.amostras[:, j])[0, 1]
                    ax.text(0.5, 0.5, f'{corr:.2f}', fontsize=16, ha='center', va='center',
                           transform=ax.transAxes, color='red' if abs(corr) > 0.5 else 'black')
                    ax.axis('off')
                
                if i == n - 1: ax.set_xlabel(self.nomes[j])
                if j == 0 and i > 0: ax.set_ylabel(self.nomes[i])
                if i < n - 1: ax.set_xticklabels([])
                if j > 0 and i != j: ax.set_yticklabels([])
        
        plt.suptitle('Corner Plot', fontweight='bold')
        plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_35(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    """Executa análise de posteriors."""
    print("=" * 60)
    print("MÓDULO 35: ANÁLISE DE POSTERIORS")
    print("Sistema de Detecção de Exoplanetas")
    print("Autor: Luiz Tiago Wilcke")
    print("=" * 60)
    
    os.makedirs(diretorio_saida, exist_ok=True)
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> Processando: {nome}")
        
        # Verificar se há amostras MCMC disponíveis
        if 'mcmc' in dados:
            amostras = dados['mcmc'].get('cadeia', None)
        else:
            # Gerar amostras sintéticas para demonstração
            f = dados['fluxo'][:1000]
            mu = np.random.normal(np.mean(f), np.std(f)/10, 2000)
            sigma = np.abs(np.random.normal(np.std(f), np.std(f)/10, 2000))
            amostras = np.column_stack([mu, sigma])
        
        analisador = AnalisadorPosterior(amostras, ['μ', 'σ'])
        sumario = analisador.sumario()
        diagnosticos = analisador.diagnosticos_convergencia()
        
        for param, s in sumario.items():
            print(f"    {param}:")
            print(f"      Média: {s['media']:.6f} ± {s['desvio']:.6f}")
            print(f"      HDI 95%: [{s['hdi_95'][0]:.6f}, {s['hdi_95'][1]:.6f}]")
        
        for param, d in diagnosticos.items():
            print(f"    {param}: ESS={d['ess']:.0f}, R-hat={d['r_hat']:.3f}")
        
        # Plotar
        arq_trace = os.path.join(diretorio_saida, f"trace_{nome.replace(' ', '_').lower()}.png")
        analisador.plotar_trace(arq_trace)
        
        arq_corner = os.path.join(diretorio_saida, f"corner_{nome.replace(' ', '_').lower()}.png")
        analisador.plotar_corner(arq_corner)
        
        resultados[nome] = {**dados, 'sumario_posterior': sumario, 'diagnosticos': diagnosticos}
    
    plt.close('all')
    print("\n" + "=" * 60)
    print("MÓDULO 35 CONCLUÍDO")
    print("=" * 60)
    
    return resultados


__all__ = ['AnalisadorPosterior', 'executar_modulo_35']
