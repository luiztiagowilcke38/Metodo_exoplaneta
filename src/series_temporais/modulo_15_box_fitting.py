"""
Módulo 15: Box-Fitting Least Squares Otimizado
Autor: Luiz Tiago Wilcke
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from typing import Dict, Optional
import os


class BoxFittingOtimizado:
    """
    BLS otimizado com ajuste não-linear de parâmetros do trânsito.
    Inclui modelagem de limb darkening e ingresso/egresso.
    """
    
    def __init__(self):
        self.parametros_otimos = None
    
    def modelo_transito_completo(self, tempo: np.ndarray, t0: float, periodo: float,
                                   duracao: float, profundidade: float,
                                   u1: float = 0.4, u2: float = 0.2) -> np.ndarray:
        """
        Modelo de trânsito com limb darkening quadrático.
        
        Parâmetros:
            t0: Época central
            periodo: Período orbital
            duracao: Duração total
            profundidade: Profundidade máxima
            u1, u2: Coeficientes de limb darkening
        """
        fase = ((tempo - t0) % periodo) / periodo
        fase[fase > 0.5] -= 1
        
        duracao_rel = duracao / periodo
        modelo = np.ones_like(tempo)
        
        # Regiões do trânsito
        ingresso = np.abs(fase + duracao_rel/2) < duracao_rel * 0.1
        egresso = np.abs(fase - duracao_rel/2) < duracao_rel * 0.1
        centro = (np.abs(fase) < duracao_rel/2) & ~ingresso & ~egresso
        
        # Limb darkening
        fator_ld = 1 - u1/3 - u2/6
        
        # Dentro do trânsito
        z = np.abs(fase) / (duracao_rel/2)
        profundidade_efetiva = profundidade * fator_ld * (1 - 0.2 * z**2)
        
        modelo[centro] = 1 - profundidade_efetiva[centro]
        
        # Ingresso/egresso suaves
        for mascara, sinal in [(ingresso, -1), (egresso, 1)]:
            if np.any(mascara):
                x = (fase[mascara] - sinal * duracao_rel/2) / (duracao_rel * 0.1)
                modelo[mascara] = 1 - profundidade * fator_ld * 0.5 * (1 + np.tanh(-x * 3))
        
        return modelo
    
    def funcao_custo(self, params: np.ndarray, tempo: np.ndarray, 
                     fluxo: np.ndarray, erro: np.ndarray) -> float:
        """Chi-quadrado reduzido."""
        t0, periodo, duracao, profundidade = params
        
        if profundidade < 0 or profundidade > 0.1 or duracao < 0.001:
            return 1e10
        
        modelo = self.modelo_transito_completo(tempo, t0, periodo, duracao, profundidade)
        chi2 = np.sum(((fluxo - modelo) / erro) ** 2)
        return chi2
    
    def ajustar(self, tempo: np.ndarray, fluxo: np.ndarray, erro: np.ndarray,
                periodo_inicial: float, t0_inicial: float = None) -> Dict:
        """Ajuste não-linear dos parâmetros."""
        if t0_inicial is None:
            t0_inicial = tempo[np.argmin(fluxo)]
        
        # Bounds
        bounds = [
            (tempo[0], tempo[-1]),           # t0
            (periodo_inicial * 0.9, periodo_inicial * 1.1),  # período
            (0.001, 0.3),                    # duração (fração)
            (1e-6, 0.1)                      # profundidade
        ]
        
        x0 = [t0_inicial, periodo_inicial, 0.05, 0.001]
        
        resultado = minimize(
            self.funcao_custo, x0, args=(tempo, fluxo, erro),
            method='L-BFGS-B', bounds=bounds
        )
        
        self.parametros_otimos = resultado.x
        t0, periodo, duracao, profundidade = resultado.x
        
        # Modelo final
        modelo_final = self.modelo_transito_completo(tempo, t0, periodo, duracao, profundidade)
        residuos = fluxo - modelo_final
        
        # Estatísticas
        n = len(fluxo)
        k = 4
        chi2_red = resultado.fun / (n - k)
        bic = resultado.fun + k * np.log(n)
        
        return {
            't0': t0,
            'periodo': periodo,
            'duracao': duracao * periodo,
            'profundidade': profundidade,
            'rp_rs': np.sqrt(profundidade),
            'chi2_reduzido': chi2_red,
            'bic': bic,
            'modelo': modelo_final,
            'residuos': residuos
        }
    
    def calcular_incertezas_mcmc(self, tempo: np.ndarray, fluxo: np.ndarray,
                                  erro: np.ndarray, n_samples: int = 5000) -> Dict:
        """Estima incertezas via MCMC simplificado."""
        if self.parametros_otimos is None:
            raise ValueError("Execute ajustar() primeiro")
        
        params = self.parametros_otimos.copy()
        escalas = np.array([0.001, 0.0001, 0.001, 1e-5])
        
        amostras = np.zeros((n_samples, 4))
        log_prob_atual = -self.funcao_custo(params, tempo, fluxo, erro)
        
        aceitos = 0
        for i in range(n_samples):
            proposta = params + escalas * np.random.randn(4)
            log_prob_proposta = -self.funcao_custo(proposta, tempo, fluxo, erro)
            
            if np.log(np.random.random()) < log_prob_proposta - log_prob_atual:
                params = proposta
                log_prob_atual = log_prob_proposta
                aceitos += 1
            
            amostras[i] = params
        
        # Estatísticas
        burnin = n_samples // 4
        amostras = amostras[burnin:]
        
        return {
            't0': (np.mean(amostras[:, 0]), np.std(amostras[:, 0])),
            'periodo': (np.mean(amostras[:, 1]), np.std(amostras[:, 1])),
            'duracao': (np.mean(amostras[:, 2]), np.std(amostras[:, 2])),
            'profundidade': (np.mean(amostras[:, 3]), np.std(amostras[:, 3])),
            'taxa_aceitacao': aceitos / n_samples,
            'amostras': amostras
        }
    
    def plotar_ajuste(self, tempo, fluxo, resultado, titulo, salvar=None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Curva de luz com modelo
        ax = axes[0, 0]
        ax.scatter(tempo, fluxo, s=1, alpha=0.3, c='gray')
        ax.plot(tempo, resultado['modelo'], 'r-', lw=1, label='Modelo')
        ax.set_xlabel('Tempo'); ax.set_ylabel('Fluxo')
        ax.set_title('Ajuste do Trânsito'); ax.legend()
        
        # Dobrada na fase
        ax = axes[0, 1]
        fase = ((tempo - resultado['t0']) % resultado['periodo']) / resultado['periodo']
        fase[fase > 0.5] -= 1
        ordem = np.argsort(fase)
        ax.scatter(fase, fluxo, s=2, alpha=0.5)
        ax.plot(fase[ordem], resultado['modelo'][ordem], 'r-', lw=2)
        ax.set_xlabel('Fase'); ax.set_ylabel('Fluxo')
        ax.set_title('Curva Dobrada')
        
        # Resíduos
        ax = axes[1, 0]
        ax.scatter(tempo, resultado['residuos'], s=1, alpha=0.5)
        ax.axhline(0, color='r', ls='--')
        ax.set_xlabel('Tempo'); ax.set_ylabel('Resíduo')
        ax.set_title(f"Resíduos (χ²ᵣ = {resultado['chi2_reduzido']:.2f})")
        
        # Info
        ax = axes[1, 1]
        ax.axis('off')
        info = f"""Parâmetros Ajustados:
        
T₀:          {resultado['t0']:.4f} BJD
Período:     {resultado['periodo']:.6f} dias
Duração:     {resultado['duracao']:.4f} dias
Profund.:    {resultado['profundidade']*1e6:.1f} ppm
Rp/Rs:       {resultado['rp_rs']:.4f}
BIC:         {resultado['bic']:.1f}"""
        ax.text(0.1, 0.5, info, fontsize=12, family='monospace', va='center')
        
        for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]: ax.grid(True, alpha=0.3)
        plt.suptitle(titulo, fontweight='bold'); plt.tight_layout()
        if salvar: plt.savefig(salvar, dpi=150, bbox_inches='tight')
        return fig


def executar_modulo_15(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    print("=" * 60 + "\nMÓDULO 15: BOX-FITTING OTIMIZADO\nAutor: Luiz Tiago Wilcke\n" + "=" * 60)
    os.makedirs(diretorio_saida, exist_ok=True)
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> {nome}")
        t, f, e = dados['tempo'], dados['fluxo'], dados.get('erro_fluxo', np.ones_like(dados['fluxo'])*0.0001)
        
        periodo_inicial = dados.get('bls', {}).get('melhor_periodo', 5.0)
        
        bfo = BoxFittingOtimizado()
        resultado = bfo.ajustar(t, f, e, periodo_inicial)
        
        print(f"    Período: {resultado['periodo']:.6f} dias")
        print(f"    Profundidade: {resultado['profundidade']*1e6:.1f} ppm")
        print(f"    Rp/Rs: {resultado['rp_rs']:.4f}")
        print(f"    χ²ᵣ: {resultado['chi2_reduzido']:.2f}")
        
        arq = os.path.join(diretorio_saida, f"boxfit_{nome.replace(' ', '_').lower()}.png")
        bfo.plotar_ajuste(t, f, resultado, f"Box-Fitting - {nome}", arq)
        
        resultados[nome] = {**dados, 'box_fitting': resultado}
    
    plt.close('all')
    print("\nMÓDULO 15 CONCLUÍDO")
    return resultados

__all__ = ['BoxFittingOtimizado', 'executar_modulo_15']
