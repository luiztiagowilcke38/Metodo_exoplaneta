"""
Módulo 08: Periodograma Lomb-Scargle
Autor: Luiz Tiago Wilcke
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import os


class PeriodogramaLombScargle:
    """
    Periodograma Lomb-Scargle para dados irregularmente amostrados.
    
    P(ω) = (1/2) * { [Σ(x-x̄)cos(ω(t-τ))]² / Σcos²(ω(t-τ)) + 
                     [Σ(x-x̄)sin(ω(t-τ))]² / Σsin²(ω(t-τ)) }
    """
    
    def calcular_periodograma(self, tempo: np.ndarray, fluxo: np.ndarray,
                               freq_min: float = None, freq_max: float = None,
                               n_frequencias: int = 10000) -> Dict:
        """Calcula o periodograma Lomb-Scargle."""
        n = len(tempo)
        fluxo_cent = fluxo - np.mean(fluxo)
        
        # Definir intervalo de frequências
        if freq_min is None:
            freq_min = 1.0 / (tempo[-1] - tempo[0])
        if freq_max is None:
            freq_max = n / (2 * (tempo[-1] - tempo[0]))
        
        frequencias = np.linspace(freq_min, freq_max, n_frequencias)
        potencia = np.zeros(n_frequencias)
        
        for i, freq in enumerate(frequencias):
            omega = 2 * np.pi * freq
            
            # Calcular τ (fase de referência)
            tau = np.arctan2(np.sum(np.sin(2 * omega * tempo)),
                            np.sum(np.cos(2 * omega * tempo))) / (2 * omega)
            
            # Termos do periodograma
            cos_term = np.cos(omega * (tempo - tau))
            sin_term = np.sin(omega * (tempo - tau))
            
            # Numeradores
            sum_cos = np.sum(fluxo_cent * cos_term) ** 2
            sum_sin = np.sum(fluxo_cent * sin_term) ** 2
            
            # Denominadores
            sum_cos2 = np.sum(cos_term ** 2)
            sum_sin2 = np.sum(sin_term ** 2)
            
            # Potência normalizada
            if sum_cos2 > 0 and sum_sin2 > 0:
                potencia[i] = 0.5 * (sum_cos / sum_cos2 + sum_sin / sum_sin2)
        
        # Normalizar
        potencia = potencia / np.var(fluxo_cent)
        
        # FAP (False Alarm Probability) aproximada
        M = n_frequencias
        fap_levels = [0.1, 0.01, 0.001]
        z_fap = [-np.log(1 - (1 - fap) ** (1/M)) for fap in fap_levels]
        
        return {
            'frequencias': frequencias,
            'periodos': 1 / frequencias,
            'potencia': potencia,
            'fap_levels': dict(zip(fap_levels, z_fap)),
            'n_frequencias': n_frequencias
        }
    
    def encontrar_melhores_periodos(self, resultado: Dict, n_periodos: int = 10) -> Dict:
        """Encontra os períodos mais significativos."""
        freq = resultado['frequencias']
        pot = resultado['potencia']
        
        # Encontrar máximos locais
        picos = []
        for i in range(1, len(pot) - 1):
            if pot[i] > pot[i-1] and pot[i] > pot[i+1]:
                picos.append((freq[i], 1/freq[i], pot[i]))
        
        # Ordenar por potência
        picos_ord = sorted(picos, key=lambda x: x[2], reverse=True)
        
        # Filtrar harmônicos (períodos muito próximos)
        picos_filtrados = []
        for p in picos_ord:
            eh_harmonico = False
            for pf in picos_filtrados:
                razao = p[1] / pf[1]
                if abs(razao - round(razao)) < 0.05:
                    eh_harmonico = True
                    break
            if not eh_harmonico:
                picos_filtrados.append(p)
            if len(picos_filtrados) >= n_periodos:
                break
        
        return {
            'frequencias': [p[0] for p in picos_filtrados],
            'periodos': [p[1] for p in picos_filtrados],
            'potencias': [p[2] for p in picos_filtrados]
        }
    
    def calcular_fap(self, z: float, n_freq: int, n_pontos: int) -> float:
        """Calcula False Alarm Probability para um dado nível de potência."""
        return 1 - (1 - np.exp(-z)) ** n_freq
    
    def plotar_periodograma(self, resultado: Dict, picos: Dict, titulo: str,
                             salvar: Optional[str] = None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Frequência vs Potência
        axes[0, 0].plot(resultado['frequencias'], resultado['potencia'], 'b-', lw=0.5)
        for fap, z in resultado['fap_levels'].items():
            axes[0, 0].axhline(z, color='r', ls='--', alpha=0.5, label=f'FAP={fap}')
        axes[0, 0].set_xlabel('Frequência (1/dia)'); axes[0, 0].set_ylabel('Potência')
        axes[0, 0].set_title('Periodograma Lomb-Scargle'); axes[0, 0].legend()
        
        # Período vs Potência
        axes[0, 1].plot(resultado['periodos'], resultado['potencia'], 'g-', lw=0.5)
        for p in picos['periodos'][:5]:
            axes[0, 1].axvline(p, color='r', ls='--', alpha=0.7)
        axes[0, 1].set_xlabel('Período (dias)'); axes[0, 1].set_ylabel('Potência')
        axes[0, 1].set_title('Potência vs Período'); axes[0, 1].set_xlim(0, 50)
        
        # Período (log) vs Potência
        axes[1, 0].semilogx(resultado['periodos'], resultado['potencia'], 'b-', lw=0.5)
        axes[1, 0].set_xlabel('Período (dias) [log]'); axes[1, 0].set_ylabel('Potência')
        axes[1, 0].set_title('Potência vs Período (escala log)')
        
        # Tabela de períodos
        axes[1, 1].axis('off')
        texto = "Melhores Períodos:\n\n"
        texto += f"{'Período (d)':<15}{'Freq (1/d)':<15}{'Potência':<12}\n"
        texto += "-" * 42 + "\n"
        for i in range(min(8, len(picos['periodos']))):
            texto += f"{picos['periodos'][i]:<15.4f}{picos['frequencias'][i]:<15.4f}{picos['potencias'][i]:<12.4f}\n"
        axes[1, 1].text(0.1, 0.5, texto, family='monospace', fontsize=11, va='center')
        
        for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]: ax.grid(True, alpha=0.3)
        plt.suptitle(titulo, fontweight='bold'); plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_08(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 08: PERIODOGRAMA LOMB-SCARGLE\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    ls = PeriodogramaLombScargle()
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        t, f = dados['tempo'], dados['fluxo']
        
        periodograma = ls.calcular_periodograma(t, f)
        picos = ls.encontrar_melhores_periodos(periodograma)
        
        if picos['periodos']:
            print(f"    Melhor período: {picos['periodos'][0]:.4f} dias")
            print(f"    Potência máxima: {picos['potencias'][0]:.4f}")
        
        arq = os.path.join(diretorio_saida, f"lomb_scargle_{nome.replace(' ', '_').lower()}.png")
        ls.plotar_periodograma(periodograma, picos, f"Lomb-Scargle - {nome}", arq)
        
        resultados[nome] = {**dados, 'lomb_scargle': periodograma, 'periodos_detectados': picos}
    
    plt.close('all')
    print("\nMÓDULO 08 CONCLUÍDO")
    return resultados

__all__ = ['PeriodogramaLombScargle', 'executar_modulo_08']
