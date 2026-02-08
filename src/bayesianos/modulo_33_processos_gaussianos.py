"""
Módulo 33: Processos Gaussianos
Sistema Estatístico para Detecção de Exoplanetas
Autor: Luiz Tiago Wilcke

Modelagem de curvas de luz com Processos Gaussianos (GP).
Útil para modelar variabilidade estelar e ruído correlacionado.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize
from typing import Dict, Callable, Optional
import os


class ProcessoGaussiano:
    """
    Processo Gaussiano para regressão não-paramétrica.
    
    f(x) ~ GP(m(x), k(x,x'))
    
    onde m(x) é a função média e k(x,x') é o kernel de covariância.
    """
    
    def __init__(self, kernel: str = 'se', escala: float = 1.0, 
                 comprimento: float = 1.0, amplitude: float = 1.0):
        """
        Inicializa o Processo Gaussiano.
        
        Parâmetros:
            kernel: Tipo de kernel ('se', 'matern32', 'periodic')
            escala: Escala do ruído
            comprimento: Escala de comprimento do kernel
            amplitude: Amplitude do kernel
        """
        self.kernel_nome = kernel
        self.escala = escala
        self.comprimento = comprimento
        self.amplitude = amplitude
        
        # Dados de treino
        self.X_treino = None
        self.y_treino = None
        self.K_inv = None
        self.alpha = None
    
    def kernel_se(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Kernel Squared Exponential (RBF).
        k(x,x') = σ² exp(-||x-x'||² / (2l²))
        """
        dist_sq = np.subtract.outer(x1, x2) ** 2
        return self.amplitude**2 * np.exp(-dist_sq / (2 * self.comprimento**2))
    
    def kernel_matern32(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Kernel Matérn ν=3/2.
        k(x,x') = σ²(1 + √3|x-x'|/l) exp(-√3|x-x'|/l)
        """
        dist = np.abs(np.subtract.outer(x1, x2))
        sqrt3 = np.sqrt(3)
        return self.amplitude**2 * (1 + sqrt3*dist/self.comprimento) * \
               np.exp(-sqrt3*dist/self.comprimento)
    
    def kernel_periodic(self, x1: np.ndarray, x2: np.ndarray, 
                        periodo: float = 1.0) -> np.ndarray:
        """
        Kernel Periódico.
        k(x,x') = σ² exp(-2 sin²(π|x-x'|/p) / l²)
        """
        dist = np.abs(np.subtract.outer(x1, x2))
        return self.amplitude**2 * np.exp(-2 * np.sin(np.pi * dist / periodo)**2 / 
                                          self.comprimento**2)
    
    def kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Calcula matriz de covariância usando o kernel escolhido."""
        if self.kernel_nome == 'se':
            return self.kernel_se(x1, x2)
        elif self.kernel_nome == 'matern32':
            return self.kernel_matern32(x1, x2)
        elif self.kernel_nome == 'periodic':
            return self.kernel_periodic(x1, x2)
        else:
            raise ValueError(f"Kernel desconhecido: {self.kernel_nome}")
    
    def ajustar(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Ajusta o GP aos dados de treino.
        
        Calcula K⁻¹ e α = K⁻¹y para predição eficiente.
        """
        self.X_treino = X
        self.y_treino = y
        n = len(X)
        
        # Matriz de covariância com ruído
        K = self.kernel(X, X) + self.escala**2 * np.eye(n)
        
        # Decomposição de Cholesky para estabilidade
        try:
            L = cholesky(K, lower=True)
            self.L = L
            
            # α = K⁻¹y = L⁻ᵀL⁻¹y
            self.alpha = solve_triangular(L.T, solve_triangular(L, y, lower=True))
        except:
            # Fallback para inversão direta
            self.K_inv = np.linalg.inv(K + 1e-6 * np.eye(n))
            self.alpha = self.K_inv @ y
    
    def prever(self, X_new: np.ndarray) -> tuple:
        """
        Predição para novos pontos.
        
        μ* = K* α
        σ²* = K** - K* K⁻¹ K*ᵀ
        
        Retorna:
            Tupla (média, desvio padrão)
        """
        # Covariância entre treino e teste
        K_star = self.kernel(self.X_treino, X_new)
        
        # Média posterior
        mu = K_star.T @ self.alpha
        
        # Variância posterior
        K_ss = self.kernel(X_new, X_new) + 1e-6 * np.eye(len(X_new))
        
        if hasattr(self, 'L'):
            v = solve_triangular(self.L, K_star, lower=True)
            var = np.diag(K_ss) - np.sum(v**2, axis=0)
        else:
            var = np.diag(K_ss) - np.diag(K_star.T @ self.K_inv @ K_star)
        
        var = np.maximum(var, 1e-10)  # Garantir positividade
        
        return mu, np.sqrt(var)
    
    def log_marginal_likelihood(self, theta: np.ndarray = None) -> float:
        """
        Log-verossimilhança marginal.
        log p(y|X,θ) = -0.5 yᵀK⁻¹y - 0.5 log|K| - n/2 log(2π)
        """
        if theta is not None:
            self.amplitude, self.comprimento, self.escala = np.exp(theta)
        
        n = len(self.X_treino)
        K = self.kernel(self.X_treino, self.X_treino) + self.escala**2 * np.eye(n)
        
        try:
            L = cholesky(K, lower=True)
            alpha = solve_triangular(L.T, solve_triangular(L, self.y_treino, lower=True))
            
            lml = -0.5 * self.y_treino @ alpha
            lml -= np.sum(np.log(np.diag(L)))
            lml -= n/2 * np.log(2 * np.pi)
        except:
            return -np.inf
        
        return lml
    
    def otimizar_hiperparametros(self) -> Dict:
        """Otimiza hiperparâmetros maximizando log-verossimilhança marginal."""
        def objetivo(theta):
            return -self.log_marginal_likelihood(theta)
        
        theta0 = np.log([self.amplitude, self.comprimento, self.escala])
        
        resultado = minimize(objetivo, theta0, method='L-BFGS-B',
                           bounds=[(-5, 5), (-5, 5), (-10, 2)])
        
        self.amplitude, self.comprimento, self.escala = np.exp(resultado.x)
        
        # Reajustar com parâmetros otimizados
        self.ajustar(self.X_treino, self.y_treino)
        
        return {
            'amplitude': self.amplitude,
            'comprimento': self.comprimento,
            'escala_ruido': self.escala,
            'lml': -resultado.fun
        }
    
    def plotar_resultados(self, X_teste: np.ndarray = None, titulo: str = "",
                          salvar: Optional[str] = None) -> plt.Figure:
        """Visualiza ajuste do GP."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        if X_teste is None:
            X_teste = np.linspace(self.X_treino.min(), self.X_treino.max(), 500)
        
        mu, std = self.prever(X_teste)
        
        # Ajuste
        ax = axes[0, 0]
        ax.scatter(self.X_treino, self.y_treino, s=5, alpha=0.3, c='gray', label='Dados')
        ax.plot(X_teste, mu, 'b-', lw=2, label='Média GP')
        ax.fill_between(X_teste, mu - 2*std, mu + 2*std, alpha=0.2, color='blue', label='IC 95%')
        ax.set_xlabel('Tempo'); ax.set_ylabel('Fluxo')
        ax.set_title('Ajuste do Processo Gaussiano'); ax.legend()
        
        # Resíduos
        ax = axes[0, 1]
        mu_treino, _ = self.prever(self.X_treino)
        residuos = self.y_treino - mu_treino
        ax.scatter(self.X_treino, residuos, s=2, alpha=0.5)
        ax.axhline(0, color='r', ls='--')
        ax.set_xlabel('Tempo'); ax.set_ylabel('Resíduo')
        ax.set_title('Resíduos')
        
        # Kernel
        ax = axes[1, 0]
        tau = np.linspace(0, self.comprimento * 5, 100)
        k_tau = self.amplitude**2 * np.exp(-tau**2 / (2*self.comprimento**2))
        ax.plot(tau, k_tau, 'b-', lw=2)
        ax.axhline(0, color='k', ls='--')
        ax.axvline(self.comprimento, color='r', ls='--', label=f'l = {self.comprimento:.3f}')
        ax.set_xlabel('|τ|'); ax.set_ylabel('k(τ)')
        ax.set_title('Função de Covariância'); ax.legend()
        
        # Amostras da posterior
        ax = axes[1, 1]
        n_amostras = 5
        K = self.kernel(X_teste, X_teste) + 1e-6 * np.eye(len(X_teste))
        try:
            L = cholesky(K, lower=True)
            for _ in range(n_amostras):
                amostra = mu + L @ np.random.randn(len(X_teste))
                ax.plot(X_teste, amostra, lw=0.5, alpha=0.7)
            ax.set_xlabel('Tempo'); ax.set_ylabel('Fluxo')
            ax.set_title('Amostras da Distribuição Posterior')
        except:
            ax.text(0.5, 0.5, 'Erro na amostragem', ha='center', transform=ax.transAxes)
        
        for ax in axes.flat: ax.grid(True, alpha=0.3)
        plt.suptitle(titulo, fontweight='bold'); plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_33(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    """Executa análise com Processos Gaussianos."""
    print("=" * 60)
    print("MÓDULO 33: PROCESSOS GAUSSIANOS")
    print("Sistema de Detecção de Exoplanetas")
    print("Autor: Luiz Tiago Wilcke")
    print("=" * 60)
    
    os.makedirs(diretorio_saida, exist_ok=True)
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> Processando: {nome}")
        
        # Usar subconjunto para eficiência
        t = dados['tempo'][:1000]
        f = dados['fluxo'][:1000]
        
        # Criar e ajustar GP
        gp = ProcessoGaussiano(kernel='se', escala=np.std(f)*0.1, 
                               comprimento=np.ptp(t)*0.1, amplitude=np.std(f))
        gp.ajustar(t, f)
        
        # Otimizar hiperparâmetros
        params = gp.otimizar_hiperparametros()
        
        print(f"    Amplitude: {params['amplitude']:.6f}")
        print(f"    Comprimento: {params['comprimento']:.4f}")
        print(f"    Ruído: {params['escala_ruido']:.6f}")
        print(f"    Log-ML: {params['lml']:.2f}")
        
        # Plotar
        arq = os.path.join(diretorio_saida, f"gp_{nome.replace(' ', '_').lower()}.png")
        gp.plotar_resultados(titulo=f"Processo Gaussiano - {nome}", salvar=arq)
        print(f"    Gráfico salvo: {arq}")
        
        resultados[nome] = {**dados, 'gp': params}
    
    plt.close('all')
    print("\n" + "=" * 60)
    print("MÓDULO 33 CONCLUÍDO")
    print("=" * 60)
    
    return resultados


__all__ = ['ProcessoGaussiano', 'executar_modulo_33']
