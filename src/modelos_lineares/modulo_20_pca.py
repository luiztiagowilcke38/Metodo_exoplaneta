"""
Módulo 20: Análise de Componentes Principais (PCA)
Autor: Luiz Tiago Wilcke

Decomposição em componentes ortogonais que maximizam variância.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import os


class AnalisePCA:
    """
    PCA via SVD: X = UΣV'
    Componentes principais: Z = XV = UΣ
    """
    
    def __init__(self, n_componentes: int = None):
        self.n_componentes = n_componentes
        self.componentes = None
        self.variancia_explicada = None
        self.media = None
        self.std = None
    
    def ajustar(self, X: np.ndarray) -> Dict:
        """
        Ajusta PCA via decomposição SVD.
        """
        n, p = X.shape
        
        # Centralizar e escalar
        self.media = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1
        X_scaled = (X - self.media) / self.std
        
        # SVD: X = UΣV'
        U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
        
        # Variância explicada
        variancia = S ** 2 / (n - 1)
        variancia_total = np.sum(variancia)
        self.variancia_explicada = variancia / variancia_total
        variancia_acumulada = np.cumsum(self.variancia_explicada)
        
        # Número de componentes
        if self.n_componentes is None:
            # Critério de Kaiser: autovalores > 1 (após escalar)
            self.n_componentes = min(p, max(1, np.sum(variancia > 1)))
        
        self.componentes = Vt[:self.n_componentes].T
        
        # Scores (projeções)
        scores = X_scaled @ self.componentes
        
        # Loadings (correlações)
        loadings = self.componentes * np.sqrt(variancia[:self.n_componentes])
        
        # Comunalidades
        comunalidades = np.sum(loadings ** 2, axis=1)
        
        return {
            'scores': scores,
            'loadings': loadings,
            'componentes': self.componentes,
            'variancia_explicada': self.variancia_explicada[:self.n_componentes],
            'variancia_acumulada': variancia_acumulada[:self.n_componentes],
            'comunalidades': comunalidades,
            'autovalores': variancia[:self.n_componentes],
            'n_componentes': self.n_componentes
        }
    
    def transformar(self, X: np.ndarray) -> np.ndarray:
        """Projeta novos dados nos componentes principais."""
        X_scaled = (X - self.media) / self.std
        return X_scaled @ self.componentes
    
    def reconstruir(self, scores: np.ndarray) -> np.ndarray:
        """Reconstrói dados a partir dos scores."""
        X_scaled = scores @ self.componentes.T
        return X_scaled * self.std + self.media
    
    def plotar_resultados(self, resultado, X, titulo, salvar=None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Scree plot
        ax = axes[0, 0]
        x = np.arange(1, len(resultado['variancia_explicada']) + 1)
        ax.bar(x, resultado['variancia_explicada'], alpha=0.7, label='Individual')
        ax.plot(x, resultado['variancia_acumulada'], 'ro-', label='Acumulada')
        ax.axhline(0.9, color='g', ls='--', alpha=0.7, label='90%')
        ax.set_xlabel('Componente'); ax.set_ylabel('Proporção da Variância')
        ax.set_title('Scree Plot'); ax.legend()
        
        # Biplot (PC1 vs PC2)
        ax = axes[0, 1]
        scores = resultado['scores']
        if scores.shape[1] >= 2:
            ax.scatter(scores[:, 0], scores[:, 1], s=1, alpha=0.3)
            ax.set_xlabel(f"PC1 ({resultado['variancia_explicada'][0]:.1%})")
            ax.set_ylabel(f"PC2 ({resultado['variancia_explicada'][1]:.1%})")
        ax.set_title('Scores PC1 vs PC2')
        
        # Loadings
        ax = axes[1, 0]
        loadings = resultado['loadings']
        n_vars = min(loadings.shape[0], 20)
        n_pcs = min(loadings.shape[1], 5)
        im = ax.imshow(loadings[:n_vars, :n_pcs].T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xlabel('Variável'); ax.set_ylabel('PC')
        ax.set_title('Loadings'); plt.colorbar(im, ax=ax)
        
        # Comunalidades
        ax = axes[1, 1]
        ax.bar(range(len(resultado['comunalidades'])), resultado['comunalidades'], alpha=0.7)
        ax.axhline(0.5, color='r', ls='--')
        ax.set_xlabel('Variável'); ax.set_ylabel('Comunalidade')
        ax.set_title('Comunalidades')
        
        for ax in axes.flat: ax.grid(True, alpha=0.3)
        plt.suptitle(titulo, fontweight='bold'); plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_20(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 20: PCA\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        t, f = dados['tempo'], dados['fluxo']
        
        # Criar matriz de features expandida
        X = np.column_stack([
            f, np.gradient(f), np.gradient(np.gradient(f)),
            np.sin(2*np.pi*t/5), np.sin(2*np.pi*t/10), np.sin(2*np.pi*t/20),
            np.cos(2*np.pi*t/5), np.cos(2*np.pi*t/10), np.cos(2*np.pi*t/20)
        ])
        
        pca = AnalisePCA(n_componentes=5)
        resultado = pca.ajustar(X)
        
        print(f"    Componentes: {resultado['n_componentes']}")
        print(f"    Variância explicada: {resultado['variancia_acumulada'][-1]:.1%}")
        
        arq = os.path.join(diretorio_saida, f"pca_{nome.replace(' ', '_').lower()}.png")
        pca.plotar_resultados(resultado, X, f"PCA - {nome}", arq)
        
        resultados[nome] = {**dados, 'pca': resultado}
    
    plt.close('all')
    print("\nMÓDULO 20 CONCLUÍDO")
    return resultados

__all__ = ['AnalisePCA', 'executar_modulo_20']
