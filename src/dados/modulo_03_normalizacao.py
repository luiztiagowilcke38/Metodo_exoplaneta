"""
Módulo 03: Normalização de Dados
Autor: Luiz Tiago Wilcke
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from typing import Tuple, Dict, Optional
import os


class NormalizadorCurvaLuz:
    """Normalização e remoção de tendências de curvas de luz."""
    
    def __init__(self, janela_tendencia: int = 201, ordem_polinomio: int = 3):
        self.janela_tendencia = janela_tendencia
        self.ordem_polinomio = ordem_polinomio
    
    def normalizar_mediana(self, fluxo: np.ndarray) -> np.ndarray:
        """Normaliza dividindo pela mediana."""
        return fluxo / np.median(fluxo)
    
    def normalizar_zscore(self, fluxo: np.ndarray) -> np.ndarray:
        """Normalização Z-score: (x - média) / desvio_padrão."""
        return (fluxo - np.mean(fluxo)) / np.std(fluxo)
    
    def normalizar_minmax(self, fluxo: np.ndarray) -> np.ndarray:
        """Normalização Min-Max para intervalo [0, 1]."""
        return (fluxo - np.min(fluxo)) / (np.max(fluxo) - np.min(fluxo))
    
    def remover_tendencia_polinomial(self, tempo: np.ndarray, fluxo: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove tendência ajustando polinômio de grau n."""
        coefs = np.polyfit(tempo, fluxo, self.ordem_polinomio)
        tendencia = np.polyval(coefs, tempo)
        fluxo_detrend = fluxo / tendencia
        return fluxo_detrend, tendencia
    
    def remover_tendencia_savgol(self, fluxo: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove tendência usando filtro Savitzky-Golay."""
        janela = self.janela_tendencia if self.janela_tendencia % 2 == 1 else self.janela_tendencia + 1
        tendencia = signal.savgol_filter(fluxo, janela, 3)
        fluxo_detrend = fluxo / tendencia
        return fluxo_detrend, tendencia
    
    def remover_tendencia_mediana_movel(self, fluxo: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove tendência usando mediana móvel."""
        tendencia = pd.Series(fluxo).rolling(self.janela_tendencia, center=True, min_periods=1).median().values
        fluxo_detrend = fluxo / tendencia
        return fluxo_detrend, tendencia
    
    def normalizar_por_segmentos(self, tempo: np.ndarray, fluxo: np.ndarray, 
                                  n_segmentos: int = 10) -> np.ndarray:
        """Normaliza cada segmento independentemente."""
        indices = np.array_split(np.arange(len(fluxo)), n_segmentos)
        fluxo_norm = np.zeros_like(fluxo)
        for idx in indices:
            fluxo_norm[idx] = fluxo[idx] / np.median(fluxo[idx])
        return fluxo_norm
    
    def normalizar_completo(self, tempo: np.ndarray, fluxo: np.ndarray,
                            metodo_tendencia: str = 'savgol') -> Tuple[np.ndarray, Dict]:
        """Pipeline completo de normalização."""
        # Remover tendência
        if metodo_tendencia == 'polinomial':
            fluxo_dt, tendencia = self.remover_tendencia_polinomial(tempo, fluxo)
        elif metodo_tendencia == 'mediana':
            fluxo_dt, tendencia = self.remover_tendencia_mediana_movel(fluxo)
        else:
            fluxo_dt, tendencia = self.remover_tendencia_savgol(fluxo)
        
        # Normalizar para mediana = 1
        fluxo_norm = self.normalizar_mediana(fluxo_dt)
        
        stats = {
            'media_original': np.mean(fluxo),
            'desvio_original': np.std(fluxo),
            'media_normalizado': np.mean(fluxo_norm),
            'desvio_normalizado': np.std(fluxo_norm),
            'amplitude_tendencia': np.max(tendencia) - np.min(tendencia)
        }
        return fluxo_norm, stats
    
    def plotar_normalizacao(self, tempo, fluxo_orig, fluxo_norm, tendencia,
                            titulo: str, salvar: Optional[str] = None):
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        axes[0].scatter(tempo, fluxo_orig, s=1, alpha=0.5, c='gray')
        axes[0].plot(tempo, tendencia, 'r-', lw=2, label='Tendência')
        axes[0].set_title('Fluxo Original com Tendência'); axes[0].legend()
        
        axes[1].scatter(tempo, fluxo_norm, s=1, alpha=0.5, c='#1f77b4')
        axes[1].axhline(1.0, color='r', ls='--', alpha=0.7)
        axes[1].set_title('Fluxo Normalizado')
        
        axes[2].hist(fluxo_norm, bins=100, density=True, alpha=0.7)
        axes[2].axvline(1.0, color='r', ls='--')
        axes[2].set_title('Distribuição Normalizada')
        
        for ax in axes: ax.set_xlabel('Tempo (BJD)'); ax.grid(True, alpha=0.3)
        plt.suptitle(titulo, fontweight='bold'); plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_03(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 03: NORMALIZAÇÃO\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    normalizador = NormalizadorCurvaLuz()
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        t, f = dados['tempo'], dados['fluxo']
        
        # Normalização com Savitzky-Golay
        f_dt, tendencia = normalizador.remover_tendencia_savgol(f)
        f_norm = normalizador.normalizar_mediana(f_dt)
        
        stats = {'media_norm': np.mean(f_norm), 'std_norm': np.std(f_norm)}
        print(f"    Média normalizada: {stats['media_norm']:.6f}")
        print(f"    Desvio padrão: {stats['std_norm']:.6f}")
        
        arq = os.path.join(diretorio_saida, f"normalizacao_{nome.replace(' ', '_').lower()}.png")
        normalizador.plotar_normalizacao(t, f, f_norm, tendencia, f"Normalização - {nome}", arq)
        
        resultados[nome] = {'tempo': t, 'fluxo': f_norm, 'erro_fluxo': dados['erro_fluxo'],
                            'tendencia': tendencia, 'estatisticas': stats}
    
    plt.close('all')
    print("\nMÓDULO 03 CONCLUÍDO")
    return resultados

__all__ = ['NormalizadorCurvaLuz', 'executar_modulo_03']
