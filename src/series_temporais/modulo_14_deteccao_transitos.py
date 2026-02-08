"""
Módulo 14: Detecção de Trânsitos
Sistema Estatístico para Detecção de Exoplanetas
Autor: Luiz Tiago Wilcke

Implementação do algoritmo BLS (Box-fitting Least Squares) para
detecção de trânsitos planetários em curvas de luz.

O modelo de trânsito é uma caixa retangular:
    m(t) = { L = 1 - δ  se t está em trânsito
           { H = 1      caso contrário

A estatística BLS é:
    SR = max { [s(r) - r·s]² / [r·(1-r)] }

Onde:
    s = Σwᵢxᵢ (fluxo médio ponderado)
    r = Σwᵢ para pontos em trânsito
    s(r) = Σwᵢxᵢ para pontos em trânsito
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
from scipy import signal
from scipy.optimize import minimize_scalar
import os


class DetectorTransitos:
    """
    Detector de trânsitos planetários usando múltiplos métodos:
    - BLS (Box-fitting Least Squares)
    - Matched Filter
    - Detecção de dips estatística
    """
    
    def __init__(self, periodo_min: float = 0.5, periodo_max: float = 50.0,
                 n_periodos: int = 10000, duracao_min: float = 0.01,
                 duracao_max: float = 0.15):
        """
        Inicializa o detector.
        
        Parâmetros:
            periodo_min, periodo_max: Intervalo de busca de períodos (dias)
            n_periodos: Número de períodos a testar
            duracao_min, duracao_max: Fração do período (duração/período)
        """
        self.periodo_min = periodo_min
        self.periodo_max = periodo_max
        self.n_periodos = n_periodos
        self.duracao_min = duracao_min
        self.duracao_max = duracao_max
        self.n_bins_fase = 300
        self.n_duracoes = 15
    
    def bls(self, tempo: np.ndarray, fluxo: np.ndarray,
            erro: np.ndarray = None) -> Dict:
        """
        Box-fitting Least Squares (Kovács et al. 2002).
        
        Busca o período, fase e duração que maximizam a estatística SR.
        
        SR = sqrt(N) * [s_in - r*s_total]² / [σ² * r * (1-r)]
        
        Parâmetros:
            tempo: Array de tempos
            fluxo: Array de fluxos
            erro: Array de erros (opcional)
            
        Retorna:
            Dicionário com resultados BLS
        """
        N = len(tempo)
        
        # Pesos
        if erro is None:
            pesos = np.ones(N)
        else:
            pesos = 1 / (erro ** 2)
        pesos = pesos / np.sum(pesos)
        
        # Normalizar fluxo
        fluxo_norm = fluxo - np.average(fluxo, weights=pesos)
        
        # Grid de períodos (distribuição logarítmica mais densa em períodos curtos)
        periodos = np.exp(np.linspace(
            np.log(self.periodo_min),
            np.log(self.periodo_max),
            self.n_periodos
        ))
        
        # Frações de duração a testar
        duracoes_rel = np.linspace(self.duracao_min, self.duracao_max, self.n_duracoes)
        
        # Arrays para resultados
        potencia_bls = np.zeros(self.n_periodos)
        melhores_fases = np.zeros(self.n_periodos)
        melhores_duracoes = np.zeros(self.n_periodos)
        profundidades = np.zeros(self.n_periodos)
        
        # Para cada período
        for i, periodo in enumerate(periodos):
            # Fase dobrada
            fase = (tempo % periodo) / periodo
            
            # Ordenar por fase
            ordem = np.argsort(fase)
            fase_ord = fase[ordem]
            fluxo_ord = fluxo_norm[ordem]
            pesos_ord = pesos[ordem]
            
            # Binning de fase para eficiência
            bins_fase = np.linspace(0, 1, self.n_bins_fase + 1)
            indices_bin = np.digitize(fase_ord, bins_fase) - 1
            
            # Somas cumulativas para cálculo rápido
            soma_w = np.zeros(self.n_bins_fase + 1)
            soma_wf = np.zeros(self.n_bins_fase + 1)
            
            for j in range(N):
                bin_idx = min(indices_bin[j], self.n_bins_fase - 1)
                soma_w[bin_idx + 1] += pesos_ord[j]
                soma_wf[bin_idx + 1] += pesos_ord[j] * fluxo_ord[j]
            
            # Somas cumulativas
            soma_w = np.cumsum(soma_w)
            soma_wf = np.cumsum(soma_wf)
            
            melhor_sr = 0
            melhor_fase_inicio = 0
            melhor_duracao = 0
            melhor_profundidade = 0
            
            # Para cada duração
            for duracao_rel in duracoes_rel:
                n_bins_transito = max(1, int(duracao_rel * self.n_bins_fase))
                
                # Deslizar janela de trânsito
                for j in range(self.n_bins_fase):
                    k = (j + n_bins_transito) % self.n_bins_fase
                    
                    # Somas dentro da janela
                    if k > j:
                        r = soma_w[k] - soma_w[j]
                        s_in = soma_wf[k] - soma_wf[j]
                    else:
                        r = soma_w[self.n_bins_fase] - soma_w[j] + soma_w[k]
                        s_in = soma_wf[self.n_bins_fase] - soma_wf[j] + soma_wf[k]
                    
                    # Evitar divisão por zero
                    if r > 0 and r < 1:
                        # Estatística SR
                        sr = (s_in ** 2) / (r * (1 - r))
                        
                        if sr > melhor_sr:
                            melhor_sr = sr
                            melhor_fase_inicio = j / self.n_bins_fase
                            melhor_duracao = duracao_rel
                            # Profundidade estimada
                            melhor_profundidade = s_in / r
            
            potencia_bls[i] = melhor_sr
            melhores_fases[i] = melhor_fase_inicio
            melhores_duracoes[i] = melhor_duracao
            profundidades[i] = abs(melhor_profundidade)
        
        # Encontrar melhor período
        idx_melhor = np.argmax(potencia_bls)
        
        # Calcular SNR da detecção
        ruido = np.std(potencia_bls)
        snr = (potencia_bls[idx_melhor] - np.median(potencia_bls)) / ruido
        
        return {
            'periodos': periodos,
            'potencia': potencia_bls,
            'melhor_periodo': periodos[idx_melhor],
            'melhor_fase': melhores_fases[idx_melhor],
            'melhor_duracao': melhores_duracoes[idx_melhor],
            'profundidade': profundidades[idx_melhor],
            'snr': snr,
            'fases': melhores_fases,
            'duracoes': melhores_duracoes
        }
    
    def matched_filter(self, tempo: np.ndarray, fluxo: np.ndarray,
                       modelo_transito: np.ndarray) -> Dict:
        """
        Detecção via filtro casado (matched filter).
        
        Correlação cruzada com modelo de trânsito conhecido.
        
        SNR(t) = (x * m)(t) / √(σ² * Σm²)
        """
        N = len(fluxo)
        
        # Normalizar modelo
        modelo_norm = modelo_transito - np.mean(modelo_transito)
        modelo_norm = modelo_norm / np.sqrt(np.sum(modelo_norm ** 2))
        
        # Correlação cruzada
        correlacao = signal.correlate(fluxo - np.mean(fluxo), modelo_norm, mode='same')
        
        # Normalizar para SNR
        sigma = np.std(fluxo)
        snr = correlacao / sigma
        
        # Encontrar picos
        picos, _ = signal.find_peaks(-snr, height=3.0, distance=50)
        
        return {
            'snr': snr,
            'picos_tempo': tempo[picos] if len(picos) > 0 else np.array([]),
            'picos_snr': snr[picos] if len(picos) > 0 else np.array([]),
            'n_deteccoes': len(picos)
        }
    
    def gerar_modelo_transito_mandel_agol(self, tempo_rel: np.ndarray,
                                           periodo: float, duracao: float,
                                           profundidade: float,
                                           u1: float = 0.4, u2: float = 0.2) -> np.ndarray:
        """
        Gera modelo de trânsito com limb darkening quadrático.
        
        Modelo simplificado de Mandel & Agol (2002).
        
        I(μ) = 1 - u1*(1-μ) - u2*(1-μ)²
        
        Parâmetros:
            tempo_rel: Tempo relativo à época central
            periodo: Período orbital (dias)
            duracao: Duração do trânsito (dias)
            profundidade: Profundidade (δ = (Rp/Rs)²)
            u1, u2: Coeficientes de limb darkening
        """
        # Fase
        fase = np.abs((tempo_rel % periodo) / periodo - 0.5) * 2
        
        # Parâmetro de impacto normalizado
        duracao_rel = duracao / periodo
        raio_normalizado = np.sqrt(profundidade)
        
        modelo = np.ones_like(tempo_rel)
        
        # Durante o trânsito
        em_transito = fase < duracao_rel
        
        if np.any(em_transito):
            # Posição no disco (z normalizado)
            z = fase[em_transito] / duracao_rel
            
            # Integral do limb darkening
            # F/F0 = 1 - δ * (1 - u1/3 - u2/6) * f(z)
            fator_ld = 1 - u1/3 - u2/6
            
            # f(z): fração obscurecida (aproximação)
            # Para z < 1 - p: totalmente dentro
            # Para 1 - p < z < 1 + p: cruzando a borda
            p = raio_normalizado
            
            f = np.ones_like(z)
            # Transição suave na ingresso/egresso
            mascara = z > 0.5
            if np.any(mascara):
                z_trans = (z[mascara] - 0.5) / 0.5
                f[mascara] = 1 - z_trans ** 2
            
            modelo[em_transito] = 1 - profundidade * fator_ld * f
        
        return modelo
    
    def dobrar_fase(self, tempo: np.ndarray, fluxo: np.ndarray,
                    periodo: float, epoca: float = None) -> Dict:
        """
        Dobra a curva de luz na fase do período encontrado.
        
        Parâmetros:
            tempo: Array de tempos
            fluxo: Array de fluxos
            periodo: Período para dobrar
            epoca: Época de referência (centro do trânsito)
        """
        if epoca is None:
            epoca = tempo[0]
        
        fase = ((tempo - epoca) % periodo) / periodo
        fase[fase > 0.5] -= 1  # Centrar em 0
        
        # Ordenar por fase
        ordem = np.argsort(fase)
        
        return {
            'fase': fase[ordem],
            'fluxo': fluxo[ordem],
            'periodo': periodo,
            'epoca': epoca
        }
    
    def refinar_periodo(self, tempo: np.ndarray, fluxo: np.ndarray,
                        periodo_inicial: float, tolerancia: float = 0.01) -> Dict:
        """
        Refina o período minimizando a dispersão na curva dobrada.
        """
        def dispersao(p):
            fase = (tempo % p) / p
            n_bins = 50
            bins = np.linspace(0, 1, n_bins + 1)
            dispersao_total = 0
            for i in range(n_bins):
                mascara = (fase >= bins[i]) & (fase < bins[i+1])
                if np.sum(mascara) > 1:
                    dispersao_total += np.std(fluxo[mascara])
            return dispersao_total
        
        resultado = minimize_scalar(
            dispersao,
            bounds=(periodo_inicial * (1 - tolerancia), periodo_inicial * (1 + tolerancia)),
            method='bounded'
        )
        
        return {
            'periodo_refinado': resultado.x,
            'dispersao_minima': resultado.fun
        }
    
    def plotar_resultados_bls(self, tempo: np.ndarray, fluxo: np.ndarray,
                               resultado_bls: Dict, titulo: str,
                               salvar: Optional[str] = None) -> plt.Figure:
        """Visualização dos resultados BLS."""
        fig = plt.figure(figsize=(16, 12))
        
        # Periodograma BLS
        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
        ax1.plot(resultado_bls['periodos'], resultado_bls['potencia'], 'b-', lw=0.5)
        ax1.axvline(resultado_bls['melhor_periodo'], color='r', ls='--', lw=2,
                   label=f"P = {resultado_bls['melhor_periodo']:.4f} d")
        ax1.set_xlabel('Período (dias)')
        ax1.set_ylabel('Potência BLS')
        ax1.set_title('Periodograma BLS')
        ax1.set_xscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Curva de luz original
        ax2 = plt.subplot2grid((3, 2), (1, 0))
        ax2.scatter(tempo, fluxo, s=1, alpha=0.3, c='#1f77b4')
        ax2.set_xlabel('Tempo (BJD)')
        ax2.set_ylabel('Fluxo')
        ax2.set_title('Curva de Luz')
        ax2.grid(True, alpha=0.3)
        
        # Curva dobrada na fase
        ax3 = plt.subplot2grid((3, 2), (1, 1))
        dobrada = self.dobrar_fase(tempo, fluxo, resultado_bls['melhor_periodo'])
        ax3.scatter(dobrada['fase'], dobrada['fluxo'], s=2, alpha=0.5, c='#2ca02c')
        ax3.axvline(0, color='r', ls='--', alpha=0.5)
        ax3.set_xlabel('Fase')
        ax3.set_ylabel('Fluxo')
        ax3.set_title(f"Dobrada no Período (P={resultado_bls['melhor_periodo']:.4f} d)")
        ax3.set_xlim(-0.5, 0.5)
        ax3.grid(True, alpha=0.3)
        
        # Zoom no trânsito
        ax4 = plt.subplot2grid((3, 2), (2, 0))
        duracao = resultado_bls['melhor_duracao']
        mascara_transito = np.abs(dobrada['fase']) < duracao * 2
        ax4.scatter(dobrada['fase'][mascara_transito], 
                   dobrada['fluxo'][mascara_transito], s=5, alpha=0.7)
        ax4.axvspan(-duracao/2, duracao/2, alpha=0.2, color='red')
        ax4.set_xlabel('Fase')
        ax4.set_ylabel('Fluxo')
        ax4.set_title('Zoom no Trânsito')
        ax4.grid(True, alpha=0.3)
        
        # Informações
        ax5 = plt.subplot2grid((3, 2), (2, 1))
        ax5.axis('off')
        info = f"""Resultados da Detecção BLS:
        
Período:        {resultado_bls['melhor_periodo']:.6f} dias
Fase inicial:   {resultado_bls['melhor_fase']:.4f}
Duração:        {resultado_bls['melhor_duracao']:.4f} (fração)
Profundidade:   {resultado_bls['profundidade']*1e6:.1f} ppm
SNR:            {resultado_bls['snr']:.1f}

Raio planetário estimado:
Rp/Rs = {np.sqrt(resultado_bls['profundidade']):.4f}
"""
        ax5.text(0.1, 0.5, info, fontsize=12, family='monospace',
                verticalalignment='center', transform=ax5.transAxes)
        
        plt.suptitle(titulo, fontweight='bold', fontsize=14)
        plt.tight_layout()
        
        if salvar:
            plt.savefig(salvar, dpi=150, bbox_inches='tight')
        
        return fig


def executar_modulo_14(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    """Executa detecção de trânsitos."""
    print("=" * 60)
    print("MÓDULO 14: DETECÇÃO DE TRÂNSITOS (BLS)")
    print("Sistema de Detecção de Exoplanetas")
    print("Autor: Luiz Tiago Wilcke")
    print("=" * 60)
    
    os.makedirs(diretorio_saida, exist_ok=True)
    detector = DetectorTransitos(n_periodos=5000)
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> Processando: {nome}")
        
        tempo = dados['tempo']
        fluxo = dados['fluxo']
        erro = dados.get('erro_fluxo', None)
        
        # BLS
        resultado_bls = detector.bls(tempo, fluxo, erro)
        
        # Refinar período
        refinado = detector.refinar_periodo(tempo, fluxo, resultado_bls['melhor_periodo'])
        
        print(f"    Período detectado: {resultado_bls['melhor_periodo']:.6f} dias")
        print(f"    Período refinado: {refinado['periodo_refinado']:.6f} dias")
        print(f"    Profundidade: {resultado_bls['profundidade']*1e6:.1f} ppm")
        print(f"    Rp/Rs estimado: {np.sqrt(resultado_bls['profundidade']):.4f}")
        print(f"    SNR: {resultado_bls['snr']:.1f}")
        
        # Plotar
        arquivo = os.path.join(diretorio_saida, f"bls_{nome.replace(' ', '_').lower()}.png")
        detector.plotar_resultados_bls(tempo, fluxo, resultado_bls, f"BLS - {nome}", arquivo)
        print(f"    Gráfico salvo em: {arquivo}")
        
        resultados[nome] = {
            **dados,
            'bls': resultado_bls,
            'periodo_refinado': refinado
        }
    
    plt.close('all')
    print("\n" + "=" * 60)
    print("MÓDULO 14 CONCLUÍDO")
    print("=" * 60)
    
    return resultados


__all__ = ['DetectorTransitos', 'executar_modulo_14']
