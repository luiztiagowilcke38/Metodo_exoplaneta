"""
Módulo 06: Decomposição Temporal
Autor: Luiz Tiago Wilcke
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from typing import Dict, Tuple, Optional
import os


class DecompositorTemporal:
    """Decomposição de séries temporais em componentes."""
    
    def decomposicao_stl_simplificada(self, tempo: np.ndarray, fluxo: np.ndarray,
                                       periodo: int = 100) -> Dict:
        """
        Decomposição STL simplificada: Tendência + Sazonalidade + Resíduo.
        F(t) = T(t) + S(t) + R(t)
        """
        # Tendência via LOWESS simplificado (média móvel ponderada)
        janela = min(periodo * 2 + 1, len(fluxo) // 3)
        if janela % 2 == 0: janela += 1
        tendencia = signal.savgol_filter(fluxo, janela, 2)
        
        # Remover tendência
        detrended = fluxo - tendencia
        
        # Sazonalidade via média por fase
        n_ciclos = max(1, len(fluxo) // periodo)
        sazonalidade = np.zeros(len(fluxo))
        
        for i in range(periodo):
            indices = np.arange(i, len(fluxo), periodo)
            if len(indices) > 0:
                media_sazonal = np.mean(detrended[indices])
                sazonalidade[indices] = media_sazonal
        
        # Resíduo
        residuo = fluxo - tendencia - sazonalidade
        
        # Força das componentes
        var_total = np.var(fluxo)
        forca_tendencia = 1 - np.var(residuo + sazonalidade) / var_total
        forca_sazonalidade = 1 - np.var(residuo + tendencia - np.mean(tendencia)) / var_total
        
        return {
            'tendencia': tendencia,
            'sazonalidade': sazonalidade,
            'residuo': residuo,
            'forca_tendencia': max(0, forca_tendencia),
            'forca_sazonalidade': max(0, forca_sazonalidade),
            'periodo_usado': periodo
        }
    
    def decomposicao_emd_simplificada(self, fluxo: np.ndarray, n_imfs: int = 5) -> Dict:
        """
        Decomposição empírica de modos (EMD) simplificada.
        Extrai IMFs (Intrinsic Mode Functions) iterativamente.
        """
        imfs = []
        residuo = fluxo.copy()
        
        for _ in range(n_imfs):
            h = residuo.copy()
            
            # Iteração de sifting simplificada
            for _ in range(10):
                # Encontrar extremos locais
                maximos = signal.argrelmax(h)[0]
                minimos = signal.argrelmin(h)[0]
                
                if len(maximos) < 2 or len(minimos) < 2:
                    break
                
                # Interpolar envelopes
                x = np.arange(len(h))
                env_max = np.interp(x, maximos, h[maximos])
                env_min = np.interp(x, minimos, h[minimos])
                
                # Média dos envelopes
                media = (env_max + env_min) / 2
                h = h - media
            
            imfs.append(h)
            residuo = residuo - h
        
        return {'imfs': imfs, 'residuo': residuo}
    
    def plotar_decomposicao(self, tempo, fluxo, decomp: Dict, titulo: str,
                             salvar: Optional[str] = None):
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        
        axes[0].plot(tempo, fluxo, 'b-', lw=0.5, alpha=0.7)
        axes[0].set_ylabel('Original'); axes[0].set_title(titulo)
        
        axes[1].plot(tempo, decomp['tendencia'], 'g-', lw=1)
        axes[1].set_ylabel(f"Tendência\n(força={decomp['forca_tendencia']:.2f})")
        
        axes[2].plot(tempo, decomp['sazonalidade'], 'orange', lw=0.5)
        axes[2].set_ylabel(f"Sazonalidade\n(força={decomp['forca_sazonalidade']:.2f})")
        
        axes[3].scatter(tempo, decomp['residuo'], s=1, alpha=0.5, c='red')
        axes[3].set_ylabel('Resíduo'); axes[3].set_xlabel('Tempo (BJD)')
        
        for ax in axes: ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_06(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 06: DECOMPOSIÇÃO TEMPORAL\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    decompositor = DecompositorTemporal()
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        t, f = dados['tempo'], dados['fluxo']
        
        decomp = decompositor.decomposicao_stl_simplificada(t, f, periodo=100)
        print(f"    Força da tendência: {decomp['forca_tendencia']:.3f}")
        print(f"    Força da sazonalidade: {decomp['forca_sazonalidade']:.3f}")
        print(f"    Desvio do resíduo: {np.std(decomp['residuo']):.6f}")
        
        arq = os.path.join(diretorio_saida, f"decomposicao_{nome.replace(' ', '_').lower()}.png")
        decompositor.plotar_decomposicao(t, f, decomp, f"Decomposição STL - {nome}", arq)
        
        resultados[nome] = {**dados, 'decomposicao': decomp}
    
    plt.close('all')
    print("\nMÓDULO 06 CONCLUÍDO")
    return resultados

__all__ = ['DecompositorTemporal', 'executar_modulo_06']
