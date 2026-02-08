"""
Módulo 13: Análise de Wavelets
Sistema Estatístico para Detecção de Exoplanetas
Autor: Luiz Tiago Wilcke

Implementação completa de análise tempo-frequência via wavelets
para detecção de trânsitos e caracterização de variabilidade estelar.

A Transformada Contínua de Wavelet (CWT) é definida como:
    W(a,b) = (1/√a) ∫ x(t) ψ*((t-b)/a) dt

Onde:
    a: Escala (relacionada à frequência)
    b: Posição (tempo)
    ψ: Wavelet mãe
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Dict, Tuple, Optional, Callable
import os


class AnalisadorWavelets:
    """
    Análise de wavelets para curvas de luz estelares.
    
    Suporta:
    - CWT (Continuous Wavelet Transform)
    - DWT (Discrete Wavelet Transform)
    - Múltiplas wavelets mãe
    - Análise de potência escalar
    - Detecção de eventos transientes
    """
    
    def __init__(self):
        self.wavelets_disponiveis = {
            'morlet': self._morlet,
            'mexican_hat': self._mexican_hat,
            'paul': self._paul,
            'dog': self._dog  # Derivative of Gaussian
        }
    
    def _morlet(self, t: np.ndarray, omega0: float = 6.0) -> np.ndarray:
        """
        Wavelet de Morlet.
        ψ(t) = π^(-1/4) * exp(iω₀t) * exp(-t²/2)
        
        Parâmetros:
            t: Array de tempo normalizado
            omega0: Frequência central (padrão: 6 para admissibilidade)
        """
        return (np.pi ** -0.25) * np.exp(1j * omega0 * t) * np.exp(-t**2 / 2)
    
    def _mexican_hat(self, t: np.ndarray) -> np.ndarray:
        """
        Wavelet Mexican Hat (segunda derivada da Gaussiana).
        ψ(t) = (2/√3) * π^(-1/4) * (1 - t²) * exp(-t²/2)
        """
        norm = 2 / (np.sqrt(3) * np.pi**0.25)
        return norm * (1 - t**2) * np.exp(-t**2 / 2)
    
    def _paul(self, t: np.ndarray, m: int = 4) -> np.ndarray:
        """
        Wavelet de Paul de ordem m.
        ψ(t) = (2^m * i^m * m!) / (π * (2m)!) * (1 - it)^(-(m+1))
        """
        from math import factorial
        norm = (2**m * factorial(m)) / np.sqrt(np.pi * factorial(2*m))
        return norm * (1 - 1j * t) ** (-(m + 1))
    
    def _dog(self, t: np.ndarray, m: int = 2) -> np.ndarray:
        """
        Derivative of Gaussian de ordem m.
        ψ(t) = (-1)^(m+1) / √(Γ(m+0.5)) * d^m/dt^m (exp(-t²/2))
        """
        from scipy.special import gamma
        norm = (-1)**(m+1) / np.sqrt(gamma(m + 0.5))
        
        if m == 1:
            return norm * (-t) * np.exp(-t**2 / 2)
        elif m == 2:
            return norm * (t**2 - 1) * np.exp(-t**2 / 2)
        else:
            # Fórmula geral usando polinômios de Hermite
            from numpy.polynomial.hermite import hermval
            coefs = np.zeros(m + 1)
            coefs[m] = 1
            return norm * hermval(t, coefs) * np.exp(-t**2 / 2)
    
    def cwt(self, dados: np.ndarray, dt: float = 1.0,
            escalas: np.ndarray = None, wavelet: str = 'morlet',
            omega0: float = 6.0) -> Dict:
        """
        Transformada Contínua de Wavelet.
        
        W(a,b) = ∑ x_n * ψ*((n-b)dt/a) * √(dt/a)
        
        Parâmetros:
            dados: Série temporal
            dt: Intervalo de amostragem
            escalas: Array de escalas (ou None para automático)
            wavelet: Nome da wavelet ('morlet', 'mexican_hat', 'paul', 'dog')
            omega0: Frequência central (para Morlet)
            
        Retorna:
            Dicionário com coeficientes, escalas, frequências, etc.
        """
        N = len(dados)
        
        # Definir escalas se não fornecidas
        if escalas is None:
            # Escalas logaritmicamente espaçadas
            s0 = 2 * dt  # Menor escala
            J = int(np.log2(N * dt / s0))  # Número de oitavas
            dj = 0.25  # Resolução em escala
            escalas = s0 * 2 ** (np.arange(0, J + 1) * dj)
        
        n_escalas = len(escalas)
        
        # Preparar dados (centralizar)
        dados_cent = dados - np.mean(dados)
        
        # Calcular CWT via convolução
        coeficientes = np.zeros((n_escalas, N), dtype=complex)
        
        # Para cada escala
        for i, escala in enumerate(escalas):
            # Criar wavelet na escala atual
            # Tamanho da wavelet: proporcional à escala
            M = min(10 * int(escala / dt), N)
            if M % 2 == 0:
                M += 1
            
            t_wavelet = np.arange(-M//2, M//2 + 1) * dt / escala
            
            if wavelet == 'morlet':
                psi = self._morlet(t_wavelet, omega0)
            elif wavelet == 'mexican_hat':
                psi = self._mexican_hat(t_wavelet)
            elif wavelet == 'paul':
                psi = self._paul(t_wavelet)
            elif wavelet == 'dog':
                psi = self._dog(t_wavelet)
            else:
                raise ValueError(f"Wavelet desconhecida: {wavelet}")
            
            # Normalização para preservar energia
            psi = psi * np.sqrt(dt / escala)
            
            # Convolução
            coeficientes[i, :] = signal.convolve(dados_cent, np.conj(psi[::-1]), 
                                                  mode='same')
        
        # Potência
        potencia = np.abs(coeficientes) ** 2
        
        # Converter escalas para períodos/frequências
        if wavelet == 'morlet':
            # Para Morlet: período = (4π * escala) / (ω₀ + √(2 + ω₀²))
            periodos = (4 * np.pi * escalas) / (omega0 + np.sqrt(2 + omega0**2))
        else:
            # Aproximação genérica
            periodos = escalas
        
        frequencias = 1 / periodos
        
        # Cone de influência (COI)
        # Região afetada por efeitos de borda
        coi = N / 2 - np.abs(np.arange(N) - N/2)
        coi = coi * dt * np.sqrt(2)
        
        # Espectro global de wavelet (média temporal)
        espectro_global = np.mean(potencia, axis=1)
        
        return {
            'coeficientes': coeficientes,
            'potencia': potencia,
            'escalas': escalas,
            'periodos': periodos,
            'frequencias': frequencias,
            'coi': coi,
            'espectro_global': espectro_global,
            'wavelet': wavelet
        }
    
    def dwt(self, dados: np.ndarray, nivel: int = None,
            wavelet: str = 'db4') -> Dict:
        """
        Transformada Discreta de Wavelet usando banco de filtros.
        
        Implementa decomposição multinível via filtros de quadratura.
        
        Parâmetros:
            dados: Série temporal
            nivel: Número de níveis de decomposição
            wavelet: Tipo de wavelet (db4 = Daubechies 4)
        """
        N = len(dados)
        
        if nivel is None:
            nivel = int(np.log2(N)) - 1
        
        # Filtros de Daubechies D4
        h0 = np.array([
            (1 + np.sqrt(3)) / (4 * np.sqrt(2)),
            (3 + np.sqrt(3)) / (4 * np.sqrt(2)),
            (3 - np.sqrt(3)) / (4 * np.sqrt(2)),
            (1 - np.sqrt(3)) / (4 * np.sqrt(2))
        ])  # Passa-baixa
        
        h1 = np.array([h0[3], -h0[2], h0[1], -h0[0]])  # Passa-alta
        
        aproximacoes = []
        detalhes = []
        
        sinal = dados.copy()
        
        for n in range(nivel):
            # Convolução e decimação
            a = signal.convolve(sinal, h0, mode='same')[::2]
            d = signal.convolve(sinal, h1, mode='same')[::2]
            
            aproximacoes.append(a)
            detalhes.append(d)
            
            sinal = a
        
        # Reconstrução
        reconstruido = aproximacoes[-1].copy()
        for n in range(nivel - 1, -1, -1):
            # Upsampling e convolução
            up = np.zeros(2 * len(reconstruido))
            up[::2] = reconstruido
            
            reconstruido = signal.convolve(up, h0[::-1], mode='same')[:len(detalhes[n])*2]
        
        # Energia por nível
        energia_por_nivel = [np.sum(d**2) for d in detalhes]
        energia_total = sum(energia_por_nivel) + np.sum(aproximacoes[-1]**2)
        energia_relativa = [e / energia_total for e in energia_por_nivel]
        
        return {
            'aproximacoes': aproximacoes,
            'detalhes': detalhes,
            'energia_por_nivel': energia_por_nivel,
            'energia_relativa': energia_relativa,
            'nivel': nivel
        }
    
    def detectar_eventos_transientes(self, cwt_resultado: Dict,
                                      threshold_sigma: float = 3.0) -> Dict:
        """
        Detecta eventos transientes (como trânsitos) usando CWT.
        
        Parâmetros:
            cwt_resultado: Resultado da CWT
            threshold_sigma: Threshold em unidades de desvio padrão
            
        Retorna:
            Dicionário com eventos detectados
        """
        potencia = cwt_resultado['potencia']
        periodos = cwt_resultado['periodos']
        
        # Calcular média e desvio por escala
        media_por_escala = np.mean(potencia, axis=1, keepdims=True)
        std_por_escala = np.std(potencia, axis=1, keepdims=True)
        
        # Z-score
        z_score = (potencia - media_por_escala) / (std_por_escala + 1e-10)
        
        # Detectar picos significativos
        eventos = []
        for i in range(potencia.shape[0]):
            for j in range(potencia.shape[1]):
                if z_score[i, j] > threshold_sigma:
                    eventos.append({
                        'indice_tempo': j,
                        'indice_escala': i,
                        'periodo': periodos[i],
                        'potencia': potencia[i, j],
                        'z_score': z_score[i, j]
                    })
        
        # Ordenar por z-score
        eventos = sorted(eventos, key=lambda x: x['z_score'], reverse=True)
        
        return {
            'eventos': eventos[:100],  # Top 100
            'z_score_map': z_score,
            'n_eventos': len(eventos)
        }
    
    def plotar_cwt(self, tempo: np.ndarray, dados: np.ndarray,
                   cwt_resultado: Dict, titulo: str,
                   salvar: Optional[str] = None) -> plt.Figure:
        """Visualização completa da análise CWT."""
        fig = plt.figure(figsize=(16, 12))
        
        # Layout
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
        ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
        ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
        ax4 = plt.subplot2grid((3, 3), (0, 2))
        
        # Série temporal
        ax1.plot(tempo, dados, 'b-', lw=0.5)
        ax1.set_xlabel('Tempo (BJD)')
        ax1.set_ylabel('Fluxo')
        ax1.set_title('Curva de Luz')
        ax1.set_xlim(tempo[0], tempo[-1])
        
        # Escalograma (potência CWT)
        potencia = cwt_resultado['potencia']
        periodos = cwt_resultado['periodos']
        
        # Log da potência para melhor visualização
        pot_log = np.log10(potencia + 1e-10)
        
        im = ax2.pcolormesh(tempo, periodos, pot_log,
                           shading='auto', cmap='jet')
        
        # Cone de influência
        coi = cwt_resultado['coi']
        ax2.fill_between(tempo, coi, periodos[-1], 
                        alpha=0.3, color='white', hatch='/')
        
        ax2.set_xlabel('Tempo (BJD)')
        ax2.set_ylabel('Período (dias)')
        ax2.set_title(f"Escalograma CWT ({cwt_resultado['wavelet']})")
        ax2.set_yscale('log')
        ax2.set_ylim(periodos[0], periodos[-1])
        ax2.set_xlim(tempo[0], tempo[-1])
        
        plt.colorbar(im, ax=ax2, label='log₁₀(Potência)')
        
        # Espectro global
        espectro = cwt_resultado['espectro_global']
        ax3.plot(espectro, periodos, 'b-', lw=1)
        ax3.set_xlabel('Potência Média')
        ax3.set_ylabel('Período (dias)')
        ax3.set_title('Espectro Global')
        ax3.set_yscale('log')
        ax3.set_ylim(periodos[0], periodos[-1])
        
        # Potência integrada por tempo
        potencia_tempo = np.sum(potencia, axis=0)
        ax4.plot(tempo, potencia_tempo, 'g-', lw=0.5)
        ax4.set_xlabel('Tempo')
        ax4.set_ylabel('Potência Total')
        ax4.set_title('Potência Integrada')
        ax4.set_xlim(tempo[0], tempo[-1])
        
        for ax in [ax1, ax3, ax4]:
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(titulo, fontweight='bold', fontsize=14)
        plt.tight_layout()
        
        if salvar:
            plt.savefig(salvar, dpi=150, bbox_inches='tight')
        
        return fig


def executar_modulo_13(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    """Executa análise de wavelets."""
    print("=" * 60)
    print("MÓDULO 13: ANÁLISE DE WAVELETS")
    print("Sistema de Detecção de Exoplanetas")
    print("Autor: Luiz Tiago Wilcke")
    print("=" * 60)
    
    os.makedirs(diretorio_saida, exist_ok=True)
    analisador = AnalisadorWavelets()
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> Processando: {nome}")
        
        tempo = dados['tempo']
        fluxo = dados['fluxo']
        dt = np.median(np.diff(tempo))
        
        # CWT com Morlet
        cwt_resultado = analisador.cwt(fluxo, dt=dt, wavelet='morlet', omega0=6.0)
        
        # DWT
        dwt_resultado = analisador.dwt(fluxo, nivel=5)
        
        # Detectar eventos
        eventos = analisador.detectar_eventos_transientes(cwt_resultado, threshold_sigma=4.0)
        
        print(f"    Escalas analisadas: {len(cwt_resultado['escalas'])}")
        print(f"    Período mínimo: {cwt_resultado['periodos'][0]:.4f} dias")
        print(f"    Período máximo: {cwt_resultado['periodos'][-1]:.4f} dias")
        print(f"    Eventos transientes: {eventos['n_eventos']}")
        print(f"    Energia DWT por nível: {[f'{e:.2%}' for e in dwt_resultado['energia_relativa'][:5]]}")
        
        # Plotar
        arquivo = os.path.join(diretorio_saida, f"wavelets_{nome.replace(' ', '_').lower()}.png")
        analisador.plotar_cwt(tempo, fluxo, cwt_resultado, f"Wavelets - {nome}", arquivo)
        print(f"    Gráfico salvo em: {arquivo}")
        
        resultados[nome] = {
            **dados,
            'cwt': cwt_resultado,
            'dwt': dwt_resultado,
            'eventos_wavelet': eventos
        }
    
    plt.close('all')
    print("\n" + "=" * 60)
    print("MÓDULO 13 CONCLUÍDO")
    print("=" * 60)
    
    return resultados


__all__ = ['AnalisadorWavelets', 'executar_modulo_13']
