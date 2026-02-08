"""
Módulo 07: Análise de Fourier
Autor: Luiz Tiago Wilcke
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from typing import Dict, Tuple, Optional
import os


class AnalisadorFourier:
    """Análise de Fourier e espectro de potência para curvas de luz."""
    
    def calcular_fft(self, tempo: np.ndarray, fluxo: np.ndarray) -> Dict:
        """
        Calcula a Transformada de Fourier Discreta.
        X(f) = Σ x(t) * exp(-2πift)
        """
        n = len(fluxo)
        dt = np.median(np.diff(tempo))
        
        # FFT
        fft_valores = fft.fft(fluxo - np.mean(fluxo))
        frequencias = fft.fftfreq(n, dt)
        
        # Apenas frequências positivas
        mascara = frequencias > 0
        freq_pos = frequencias[mascara]
        amplitude = 2 * np.abs(fft_valores[mascara]) / n
        fase = np.angle(fft_valores[mascara])
        
        # Potência
        potencia = amplitude ** 2
        
        return {
            'frequencias': freq_pos,
            'amplitude': amplitude,
            'fase': fase,
            'potencia': potencia,
            'frequencia_nyquist': 1 / (2 * dt),
            'resolucao_frequencia': 1 / (n * dt)
        }
    
    def calcular_espectro_potencia(self, tempo: np.ndarray, fluxo: np.ndarray,
                                    janela: str = 'hann') -> Dict:
        """
        Calcula o espectro de potência usando janelamento.
        P(f) = |X(f)|² / N
        """
        n = len(fluxo)
        dt = np.median(np.diff(tempo))
        
        # Aplicar janela
        if janela == 'hann':
            w = np.hanning(n)
        elif janela == 'hamming':
            w = np.hamming(n)
        else:
            w = np.ones(n)
        
        fluxo_jan = (fluxo - np.mean(fluxo)) * w
        
        # FFT com janela
        fft_valores = fft.fft(fluxo_jan)
        frequencias = fft.fftfreq(n, dt)
        
        mascara = frequencias > 0
        freq_pos = frequencias[mascara]
        potencia = np.abs(fft_valores[mascara]) ** 2 / (n * np.sum(w ** 2))
        
        # Normalização para densidade espectral de potência
        psd = potencia * 2 * n * dt
        
        return {'frequencias': freq_pos, 'psd': psd, 'potencia': potencia}
    
    def encontrar_picos_frequencia(self, frequencias: np.ndarray, potencia: np.ndarray,
                                    n_picos: int = 10, threshold_snr: float = 3.0) -> Dict:
        """Encontra picos significativos no espectro."""
        # Ruído de fundo (mediana)
        ruido = np.median(potencia)
        snr = potencia / ruido
        
        # Encontrar picos locais
        picos = []
        for i in range(1, len(potencia) - 1):
            if snr[i] > threshold_snr and potencia[i] > potencia[i-1] and potencia[i] > potencia[i+1]:
                picos.append((frequencias[i], potencia[i], snr[i]))
        
        # Ordenar por potência
        picos_ordenados = sorted(picos, key=lambda x: x[1], reverse=True)[:n_picos]
        
        return {
            'frequencias_pico': [p[0] for p in picos_ordenados],
            'periodos_pico': [1/p[0] for p in picos_ordenados] if picos_ordenados else [],
            'potencias_pico': [p[1] for p in picos_ordenados],
            'snr_picos': [p[2] for p in picos_ordenados],
            'ruido_base': ruido
        }
    
    def plotar_espectro(self, resultado_fft: Dict, resultado_picos: Dict,
                         titulo: str, salvar: Optional[str] = None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Espectro de amplitude
        axes[0, 0].semilogy(resultado_fft['frequencias'], resultado_fft['amplitude'], 'b-', lw=0.5)
        axes[0, 0].set_xlabel('Frequência (1/dia)'); axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title('Espectro de Amplitude')
        
        # PSD
        axes[0, 1].loglog(resultado_fft['frequencias'], resultado_fft['potencia'], 'g-', lw=0.5)
        for fp in resultado_picos['frequencias_pico'][:5]:
            axes[0, 1].axvline(fp, color='r', ls='--', alpha=0.5)
        axes[0, 1].set_xlabel('Frequência (1/dia)'); axes[0, 1].set_ylabel('Potência')
        axes[0, 1].set_title('Espectro de Potência (log-log)')
        
        # Período vs Potência
        periodos = 1 / resultado_fft['frequencias']
        axes[1, 0].semilogx(periodos, resultado_fft['potencia'], 'b-', lw=0.5)
        for pp in resultado_picos['periodos_pico'][:5]:
            axes[1, 0].axvline(pp, color='r', ls='--', alpha=0.5)
        axes[1, 0].set_xlabel('Período (dias)'); axes[1, 0].set_ylabel('Potência')
        axes[1, 0].set_title('Potência vs Período')
        axes[1, 0].set_xlim(0.1, 100)
        
        # Tabela de picos
        axes[1, 1].axis('off')
        if resultado_picos['frequencias_pico']:
            texto = "Principais Picos:\n\n"
            texto += f"{'Freq (1/d)':<12}{'Período (d)':<12}{'SNR':<8}\n"
            texto += "-" * 32 + "\n"
            for i in range(min(5, len(resultado_picos['frequencias_pico']))):
                texto += f"{resultado_picos['frequencias_pico'][i]:<12.4f}"
                texto += f"{resultado_picos['periodos_pico'][i]:<12.4f}"
                texto += f"{resultado_picos['snr_picos'][i]:<8.1f}\n"
            axes[1, 1].text(0.1, 0.5, texto, family='monospace', fontsize=12, va='center')
        
        for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]: ax.grid(True, alpha=0.3)
        plt.suptitle(titulo, fontweight='bold'); plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_07(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 07: ANÁLISE DE FOURIER\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    analisador = AnalisadorFourier()
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        t, f = dados['tempo'], dados['fluxo']
        
        fft_res = analisador.calcular_fft(t, f)
        picos = analisador.encontrar_picos_frequencia(fft_res['frequencias'], fft_res['potencia'])
        
        print(f"    Frequência Nyquist: {fft_res['frequencia_nyquist']:.4f} 1/dia")
        if picos['periodos_pico']:
            print(f"    Período dominante: {picos['periodos_pico'][0]:.4f} dias")
            print(f"    SNR do pico: {picos['snr_picos'][0]:.1f}")
        
        arq = os.path.join(diretorio_saida, f"fourier_{nome.replace(' ', '_').lower()}.png")
        analisador.plotar_espectro(fft_res, picos, f"Análise de Fourier - {nome}", arq)
        
        resultados[nome] = {**dados, 'fourier': fft_res, 'picos_frequencia': picos}
    
    plt.close('all')
    print("\nMÓDULO 07 CONCLUÍDO")
    return resultados

__all__ = ['AnalisadorFourier', 'executar_modulo_07']
