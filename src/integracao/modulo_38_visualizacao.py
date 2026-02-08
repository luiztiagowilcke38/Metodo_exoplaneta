"""
Módulo 38: Visualização Integrada
Sistema Estatístico para Detecção de Exoplanetas
Autor: Luiz Tiago Wilcke

Sistema de visualização unificado para todos os resultados.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple
import os


class VisualizadorIntegrado:
    """
    Sistema de visualização integrado para análise de exoplanetas.
    
    Gera dashboards e visualizações compostas combinando
    resultados de múltiplos módulos de análise.
    """
    
    def __init__(self, estilo: str = 'seaborn-v0_8-whitegrid'):
        """
        Inicializa o visualizador.
        
        Parâmetros:
            estilo: Estilo matplotlib para os gráficos
        """
        try:
            plt.style.use(estilo)
        except:
            plt.style.use('default')
        
        self.cores = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    def criar_dashboard_curva_luz(self, tempo: np.ndarray, fluxo: np.ndarray,
                                    resultados: Dict, nome: str,
                                    salvar: Optional[str] = None) -> plt.Figure:
        """
        Cria dashboard completo para uma curva de luz.
        
        Parâmetros:
            tempo: Array de tempos
            fluxo: Array de fluxos
            resultados: Dicionário com resultados de análises
            nome: Nome da curva de luz
            salvar: Caminho para salvar a figura
        """
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Curva de luz completa
        ax1 = fig.add_subplot(gs[0, :3])
        ax1.scatter(tempo, fluxo, s=1, alpha=0.3, c=self.cores[0])
        ax1.set_xlabel('Tempo (BJD)')
        ax1.set_ylabel('Fluxo Normalizado')
        ax1.set_title('Curva de Luz Completa')
        ax1.grid(True, alpha=0.3)
        
        # 2. Histograma do fluxo
        ax2 = fig.add_subplot(gs[0, 3])
        ax2.hist(fluxo, bins=50, orientation='horizontal', alpha=0.7, color=self.cores[0])
        ax2.set_xlabel('Contagem')
        ax2.axhline(np.median(fluxo), color='r', ls='--', label='Mediana')
        ax2.legend()
        
        # 3. Periodograma (se disponível)
        ax3 = fig.add_subplot(gs[1, :2])
        if 'bls' in resultados:
            bls = resultados['bls']
            ax3.plot(bls.get('periodos', []), bls.get('potencia', []), 
                    color=self.cores[1], lw=0.5)
            if 'melhor_periodo' in bls:
                ax3.axvline(bls['melhor_periodo'], color='r', ls='--', 
                           label=f"P = {bls['melhor_periodo']:.4f} d")
            ax3.set_xlabel('Período (dias)')
            ax3.set_ylabel('Potência BLS')
            ax3.set_xscale('log')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'BLS não disponível', ha='center', va='center',
                    transform=ax3.transAxes)
        ax3.set_title('Periodograma BLS')
        ax3.grid(True, alpha=0.3)
        
        # 4. Curva dobrada
        ax4 = fig.add_subplot(gs[1, 2:])
        if 'bls' in resultados and 'melhor_periodo' in resultados['bls']:
            periodo = resultados['bls']['melhor_periodo']
            fase = ((tempo - tempo[0]) % periodo) / periodo
            fase[fase > 0.5] -= 1
            ordem = np.argsort(fase)
            ax4.scatter(fase, fluxo, s=1, alpha=0.3, c=self.cores[2])
            ax4.set_xlabel('Fase')
            ax4.set_ylabel('Fluxo')
            ax4.set_title(f'Curva Dobrada (P = {periodo:.4f} d)')
        else:
            ax4.text(0.5, 0.5, 'Período não detectado', ha='center', va='center',
                    transform=ax4.transAxes)
        ax4.grid(True, alpha=0.3)
        
        # 5. Estatísticas
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.axis('off')
        stats_texto = f"""Estatísticas:
N pontos: {len(fluxo):,}
Média: {np.mean(fluxo):.6f}
Mediana: {np.median(fluxo):.6f}
Desvio: {np.std(fluxo):.6f}
Min: {np.min(fluxo):.6f}
Max: {np.max(fluxo):.6f}
Range: {np.ptp(fluxo):.6f}"""
        ax5.text(0.1, 0.9, stats_texto, fontsize=10, family='monospace',
                verticalalignment='top', transform=ax5.transAxes)
        ax5.set_title('Estatísticas Descritivas')
        
        # 6. Autocorrelação
        ax6 = fig.add_subplot(gs[2, 1])
        fluxo_cent = fluxo - np.mean(fluxo)
        acf = np.correlate(fluxo_cent[:min(len(fluxo), 5000)], 
                          fluxo_cent[:min(len(fluxo), 5000)], mode='full')
        acf = acf[len(acf)//2:][:100]
        acf = acf / acf[0]
        ax6.bar(range(len(acf)), acf, alpha=0.7, color=self.cores[3])
        ax6.axhline(1.96/np.sqrt(len(fluxo)), color='r', ls='--')
        ax6.axhline(-1.96/np.sqrt(len(fluxo)), color='r', ls='--')
        ax6.set_xlabel('Lag')
        ax6.set_ylabel('ACF')
        ax6.set_title('Autocorrelação')
        ax6.grid(True, alpha=0.3)
        
        # 7. Espectro de potência
        ax7 = fig.add_subplot(gs[2, 2:])
        n = len(fluxo)
        dt = np.median(np.diff(tempo))
        freqs = np.fft.rfftfreq(n, d=dt)[1:]
        psd = np.abs(np.fft.rfft(fluxo - np.mean(fluxo)))[1:]**2
        ax7.loglog(freqs, psd, color=self.cores[4], lw=0.5)
        ax7.set_xlabel('Frequência (1/dia)')
        ax7.set_ylabel('Potência')
        ax7.set_title('Espectro de Potência (FFT)')
        ax7.grid(True, alpha=0.3)
        
        # 8. Informações da análise
        ax8 = fig.add_subplot(gs[3, :2])
        ax8.axis('off')
        
        info_texto = f"""Resultados da Análise:
        
Curva de Luz: {nome}
Duração: {np.ptp(tempo):.2f} dias
Cadência média: {dt*24*60:.2f} minutos
"""
        if 'bls' in resultados:
            info_texto += f"""
Detecção BLS:
  Período: {resultados['bls'].get('melhor_periodo', 'N/A'):.6f} dias
  SNR: {resultados['bls'].get('snr', 'N/A'):.2f}
  Profundidade: {resultados['bls'].get('profundidade', 0)*1e6:.1f} ppm
"""
        ax8.text(0.05, 0.95, info_texto, fontsize=10, family='monospace',
                verticalalignment='top', transform=ax8.transAxes)
        ax8.set_title('Resumo da Análise')
        
        # 9. Zoom em região de interesse
        ax9 = fig.add_subplot(gs[3, 2:])
        # Encontrar região com menor fluxo (possível trânsito)
        janela = min(100, len(fluxo) // 10)
        fluxo_suavizado = np.convolve(fluxo, np.ones(janela)/janela, mode='valid')
        idx_min = np.argmin(fluxo_suavizado)
        
        inicio = max(0, idx_min - 500)
        fim = min(len(fluxo), idx_min + 500)
        
        ax9.scatter(tempo[inicio:fim], fluxo[inicio:fim], s=3, alpha=0.5, c=self.cores[0])
        ax9.set_xlabel('Tempo (BJD)')
        ax9.set_ylabel('Fluxo')
        ax9.set_title('Zoom em Região de Interesse')
        ax9.grid(True, alpha=0.3)
        
        plt.suptitle(f'Dashboard de Análise: {nome}', fontsize=14, fontweight='bold', y=1.02)
        
        if salvar:
            plt.savefig(salvar, dpi=150, bbox_inches='tight')
        
        return fig
    
    def criar_comparativo(self, dados_multiplos: Dict,
                          salvar: Optional[str] = None) -> plt.Figure:
        """Cria visualização comparativa de múltiplas curvas de luz."""
        n = len(dados_multiplos)
        fig, axes = plt.subplots(n, 3, figsize=(16, 4*n))
        if n == 1:
            axes = axes.reshape(1, -1)
        
        for i, (nome, dados) in enumerate(dados_multiplos.items()):
            t, f = dados['tempo'], dados['fluxo']
            
            # Curva de luz
            axes[i, 0].scatter(t, f, s=1, alpha=0.3)
            axes[i, 0].set_ylabel(nome)
            if i == n-1: axes[i, 0].set_xlabel('Tempo')
            
            # Histograma
            axes[i, 1].hist(f, bins=50, alpha=0.7)
            if i == n-1: axes[i, 1].set_xlabel('Fluxo')
            
            # Box plot
            axes[i, 2].boxplot(f)
            axes[i, 2].set_xticklabels([nome])
        
        axes[0, 0].set_title('Curva de Luz')
        axes[0, 1].set_title('Distribuição do Fluxo')
        axes[0, 2].set_title('Box Plot')
        
        for ax in axes.flat: ax.grid(True, alpha=0.3)
        plt.suptitle('Comparação de Curvas de Luz', fontweight='bold')
        plt.tight_layout()
        
        if salvar:
            plt.savefig(salvar, dpi=150, bbox_inches='tight')
        
        return fig


def executar_modulo_38(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    """Gera visualizações integradas."""
    print("=" * 60)
    print("MÓDULO 38: VISUALIZAÇÃO INTEGRADA")
    print("Sistema de Detecção de Exoplanetas")
    print("Autor: Luiz Tiago Wilcke")
    print("=" * 60)
    
    os.makedirs(diretorio_saida, exist_ok=True)
    
    viz = VisualizadorIntegrado()
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> Gerando dashboard: {nome}")
        
        arq = os.path.join(diretorio_saida, f"dashboard_{nome.replace(' ', '_').lower()}.png")
        viz.criar_dashboard_curva_luz(dados['tempo'], dados['fluxo'], dados, nome, arq)
        print(f"    Salvo: {arq}")
    
    # Comparativo
    arq_comp = os.path.join(diretorio_saida, "comparativo.png")
    viz.criar_comparativo(dados_entrada, arq_comp)
    print(f"\n>>> Comparativo salvo: {arq_comp}")
    
    plt.close('all')
    print("\n" + "=" * 60)
    print("MÓDULO 38 CONCLUÍDO")
    print("=" * 60)
    
    return dados_entrada


__all__ = ['VisualizadorIntegrado', 'executar_modulo_38']
