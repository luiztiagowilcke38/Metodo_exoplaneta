"""
Módulo 02: Preprocessamento de Dados
Autor: Luiz Tiago Wilcke
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
from typing import Tuple, Dict, Optional
import os


class PreprocessadorCurvaLuz:
    """Preprocessamento de curvas de luz com sigma-clipping e detecção de outliers."""
    
    def __init__(self, sigma_clip: float = 5.0, iteracoes_clip: int = 3, janela_mediana: int = 25):
        self.sigma_clip = sigma_clip
        self.iteracoes_clip = iteracoes_clip
        self.janela_mediana = janela_mediana
        self.estatisticas_remocao = {}
    
    def remover_valores_invalidos(self, tempo: np.ndarray, fluxo: np.ndarray, 
                                   erro_fluxo: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mascara = np.isfinite(tempo) & np.isfinite(fluxo) & np.isfinite(erro_fluxo) & (erro_fluxo > 0)
        self.estatisticas_remocao['valores_invalidos'] = np.sum(~mascara)
        return tempo[mascara], fluxo[mascara], erro_fluxo[mascara]
    
    def sigma_clipping_iterativo(self, tempo: np.ndarray, fluxo: np.ndarray,
                                  erro_fluxo: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sigma-clipping iterativo usando MAD robusto."""
        mascara = np.ones(len(fluxo), dtype=bool)
        
        for _ in range(self.iteracoes_clip):
            mediana = np.median(fluxo[mascara])
            mad = 1.4826 * np.median(np.abs(fluxo[mascara] - mediana))
            nova_mascara = (fluxo >= mediana - self.sigma_clip * mad) & (fluxo <= mediana + self.sigma_clip * mad)
            if np.sum(mascara & nova_mascara) == np.sum(mascara):
                break
            mascara = mascara & nova_mascara
        
        self.estatisticas_remocao['sigma_clipping'] = np.sum(~mascara)
        return tempo[mascara], fluxo[mascara], erro_fluxo[mascara], ~mascara
    
    def detectar_outliers_locais(self, tempo: np.ndarray, fluxo: np.ndarray,
                                  erro_fluxo: np.ndarray, fator: float = 3.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mediana_local = pd.Series(fluxo).rolling(self.janela_mediana, center=True, min_periods=1).median().values
        residuos = np.abs(fluxo - mediana_local)
        mad_local = pd.Series(residuos).rolling(self.janela_mediana, center=True, min_periods=1).median().values * 1.4826
        mad_local = np.maximum(mad_local, 1e-10)
        mascara = (residuos / mad_local) < fator
        self.estatisticas_remocao['outliers_locais'] = np.sum(~mascara)
        return tempo[mascara], fluxo[mascara], erro_fluxo[mascara]
    
    def preprocessar_completo(self, tempo: np.ndarray, fluxo: np.ndarray,
                               erro_fluxo: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        n_orig = len(tempo)
        tempo, fluxo, erro_fluxo = self.remover_valores_invalidos(tempo, fluxo, erro_fluxo)
        tempo, fluxo, erro_fluxo, _ = self.sigma_clipping_iterativo(tempo, fluxo, erro_fluxo)
        tempo, fluxo, erro_fluxo = self.detectar_outliers_locais(tempo, fluxo, erro_fluxo)
        
        stats = {'pontos_originais': n_orig, 'pontos_finais': len(tempo),
                 'percentual_removido': 100 * (1 - len(tempo) / n_orig), **self.estatisticas_remocao}
        return tempo, fluxo, erro_fluxo, stats
    
    def plotar_comparacao(self, t_orig, f_orig, t_proc, f_proc, titulo: str, salvar: Optional[str] = None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].scatter(t_orig, f_orig, s=1, alpha=0.3, c='gray')
        axes[0, 0].set_title('Original'); axes[0, 0].set_xlabel('Tempo'); axes[0, 0].set_ylabel('Fluxo')
        axes[0, 1].scatter(t_proc, f_proc, s=1, alpha=0.5, c='#1f77b4')
        axes[0, 1].set_title('Preprocessado'); axes[0, 1].set_xlabel('Tempo'); axes[0, 1].set_ylabel('Fluxo')
        axes[1, 0].hist(f_orig, bins=100, alpha=0.7, color='gray', density=True, label='Original')
        axes[1, 0].hist(f_proc, bins=100, alpha=0.7, color='#1f77b4', density=True, label='Preprocessado')
        axes[1, 0].legend(); axes[1, 0].set_title('Distribuição')
        stats.probplot(f_proc, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot')
        plt.suptitle(titulo, fontweight='bold'); plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_02(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 02: PREPROCESSAMENTO\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    preprocessador = PreprocessadorCurvaLuz()
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        t, f, e, st = preprocessador.preprocessar_completo(dados['tempo'], dados['fluxo'], dados['erro_fluxo'])
        print(f"    Original: {st['pontos_originais']} -> Final: {st['pontos_finais']} ({st['percentual_removido']:.1f}% removido)")
        arq = os.path.join(diretorio_saida, f"preproc_{nome.replace(' ', '_').lower()}.png")
        preprocessador.plotar_comparacao(dados['tempo'], dados['fluxo'], t, f, f"Preprocessamento - {nome}", arq)
        resultados[nome] = {'tempo': t, 'fluxo': f, 'erro_fluxo': e, 'estatisticas': st}
    
    plt.close('all')
    print("\nMÓDULO 02 CONCLUÍDO")
    return resultados

__all__ = ['PreprocessadorCurvaLuz', 'executar_modulo_02']
