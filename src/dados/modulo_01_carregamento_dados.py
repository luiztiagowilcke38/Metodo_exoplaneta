"""
Módulo 01: Carregamento de Dados
Sistema Estatístico para Detecção de Exoplanetas
Autor: Luiz Tiago Wilcke

Este módulo implementa o carregamento de curvas de luz reais
das missões Kepler e TESS através da biblioteca lightkurve.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Union
import warnings
import os

# Tentativa de importar lightkurve, com fallback para dados simulados
try:
    import lightkurve as lk
    LIGHTKURVE_DISPONIVEL = True
except ImportError:
    LIGHTKURVE_DISPONIVEL = False
    warnings.warn("lightkurve não disponível. Usando dados simulados.")


class CarregadorDadosExoplanetas:
    """
    Classe para carregamento de curvas de luz de estrelas
    com exoplanetas confirmados ou candidatos.
    
    Atributos:
        diretorio_cache (str): Diretório para armazenar dados baixados
        estrelas_alvo (dict): Dicionário com informações das estrelas alvo
    """
    
    def __init__(self, diretorio_cache: str = "dados"):
        """
        Inicializa o carregador de dados.
        
        Parâmetros:
            diretorio_cache: Caminho para salvar dados baixados
        """
        self.diretorio_cache = diretorio_cache
        self.estrelas_alvo = self._definir_estrelas_alvo()
        
        # Criar diretório se não existir
        if not os.path.exists(diretorio_cache):
            os.makedirs(diretorio_cache)
    
    def _definir_estrelas_alvo(self) -> Dict:
        """
        Define as estrelas alvo com exoplanetas conhecidos.
        
        Retorna:
            Dicionário com informações das estrelas
        """
        estrelas = {
            "Kepler-10": {
                "tipo": "Kepler",
                "periodo_orbital": 0.8375,  # dias
                "profundidade_transito": 0.00015,  # fração
                "descricao": "Primeiro exoplaneta rochoso confirmado pelo Kepler"
            },
            "Kepler-22": {
                "tipo": "Kepler", 
                "periodo_orbital": 289.8623,  # dias
                "profundidade_transito": 0.00049,
                "descricao": "Exoplaneta na zona habitável"
            },
            "Kepler-62": {
                "tipo": "Kepler",
                "periodo_orbital": 122.3874,  # dias (Kepler-62e)
                "profundidade_transito": 0.00084,
                "descricao": "Sistema com múltiplos planetas na zona habitável"
            },
            "TIC 141914082": {
                "tipo": "TESS",
                "periodo_orbital": 3.2,
                "profundidade_transito": 0.001,
                "descricao": "Candidato TESS"
            }
        }
        return estrelas
    
    def carregar_curva_luz_kepler(self, 
                                   nome_estrela: str,
                                   trimestre: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Carrega curva de luz do Kepler para uma estrela específica.
        
        Parâmetros:
            nome_estrela: Nome da estrela (ex: "Kepler-10")
            trimestre: Trimestre específico (1-17) ou None para todos
            
        Retorna:
            Tupla com (tempo, fluxo, erro_fluxo)
        """
        if LIGHTKURVE_DISPONIVEL:
            try:
                # Buscar dados via lightkurve
                resultado_busca = lk.search_lightcurve(nome_estrela, mission='Kepler')
                
                if len(resultado_busca) == 0:
                    raise ValueError(f"Nenhum dado encontrado para {nome_estrela}")
                
                # Baixar curva de luz
                if trimestre is not None:
                    colecao = resultado_busca[trimestre].download()
                else:
                    colecao = resultado_busca.download_all()
                    colecao = colecao.stitch()
                
                # Extrair arrays
                tempo = colecao.time.value
                fluxo = colecao.flux.value
                erro_fluxo = colecao.flux_err.value
                
                return tempo, fluxo, erro_fluxo
                
            except Exception as e:
                warnings.warn(f"Erro ao baixar dados: {e}. Usando simulação.")
                return self._gerar_dados_simulados(nome_estrela)
        else:
            return self._gerar_dados_simulados(nome_estrela)
    
    def carregar_curva_luz_tess(self,
                                 tic_id: str,
                                 setor: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Carrega curva de luz do TESS para um TIC ID específico.
        
        Parâmetros:
            tic_id: Identificador TIC (ex: "TIC 141914082")
            setor: Setor específico ou None para todos
            
        Retorna:
            Tupla com (tempo, fluxo, erro_fluxo)
        """
        if LIGHTKURVE_DISPONIVEL:
            try:
                resultado_busca = lk.search_lightcurve(tic_id, mission='TESS')
                
                if len(resultado_busca) == 0:
                    raise ValueError(f"Nenhum dado encontrado para {tic_id}")
                
                if setor is not None:
                    colecao = resultado_busca[setor].download()
                else:
                    colecao = resultado_busca.download_all()
                    colecao = colecao.stitch()
                
                tempo = colecao.time.value
                fluxo = colecao.flux.value
                erro_fluxo = colecao.flux_err.value
                
                return tempo, fluxo, erro_fluxo
                
            except Exception as e:
                warnings.warn(f"Erro ao baixar dados TESS: {e}. Usando simulação.")
                return self._gerar_dados_simulados(tic_id)
        else:
            return self._gerar_dados_simulados(tic_id)
    
    def _gerar_dados_simulados(self, 
                                nome_estrela: str,
                                numero_pontos: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gera dados simulados realísticos de curva de luz com trânsitos.
        
        Modelo físico:
            F(t) = F0 * [1 - δ * T(t)] + ε(t) + V(t) + S(t)
            
        Onde:
            F0: Fluxo base normalizado
            δ: Profundidade do trânsito
            T(t): Função de trânsito (box-shape modificado)
            ε(t): Ruído gaussiano
            V(t): Variabilidade estelar (processo estocástico)
            S(t): Tendência sistemática instrumental
        
        Parâmetros:
            nome_estrela: Nome para buscar parâmetros
            numero_pontos: Quantidade de pontos na série
            
        Retorna:
            Tupla com (tempo, fluxo, erro_fluxo)
        """
        np.random.seed(42)  # Reprodutibilidade
        
        # Obter parâmetros da estrela
        if nome_estrela in self.estrelas_alvo:
            params = self.estrelas_alvo[nome_estrela]
            periodo = params["periodo_orbital"]
            profundidade = params["profundidade_transito"]
        else:
            # Parâmetros padrão
            periodo = 5.0
            profundidade = 0.001
        
        # Gerar tempo (simular ~90 dias de observação como Kepler)
        duracao_observacao = 90.0  # dias
        tempo = np.linspace(0, duracao_observacao, numero_pontos)
        
        # Cadência típica do Kepler: ~30 minutos
        # Adicionar gaps realísticos (10% dos dados)
        mascara_gaps = np.random.random(numero_pontos) > 0.10
        tempo = tempo[mascara_gaps]
        numero_pontos = len(tempo)
        
        # 1. Fluxo base normalizado
        fluxo_base = np.ones(numero_pontos)
        
        # 2. Modelar trânsitos planetários
        duracao_transito = 0.1 * periodo  # ~10% do período
        fase = (tempo % periodo) / periodo
        
        # Função de trânsito com limb darkening quadrático
        # I(μ) = 1 - c1*(1-μ) - c2*(1-μ)²
        # μ = cos(θ), onde θ é o ângulo do centro do disco
        c1, c2 = 0.4, 0.2  # Coeficientes típicos para estrela tipo solar
        
        # Trânsito simplificado com bordas suavizadas
        centro_transito = 0.5
        largura_transito = duracao_transito / periodo
        
        # Calcular distância do centro do trânsito
        distancia = np.abs(fase - centro_transito)
        
        # Aplicar modelo de trânsito
        transito = np.ones(numero_pontos)
        em_transito = distancia < largura_transito / 2
        
        # Modelo com limb darkening
        x = distancia[em_transito] / (largura_transito / 2)
        mu = np.sqrt(1 - x**2)
        fator_limb = 1 - c1 * (1 - mu) - c2 * (1 - mu)**2
        transito[em_transito] = 1 - profundidade * fator_limb
        
        # 3. Variabilidade estelar (processo Ornstein-Uhlenbeck)
        # dX = θ(μ - X)dt + σdW
        theta_ou = 0.5  # Taxa de reversão à média
        mu_ou = 0.0     # Média de longo prazo
        sigma_ou = 0.0002  # Volatilidade
        
        variabilidade = np.zeros(numero_pontos)
        dt = np.diff(tempo, prepend=tempo[0])
        
        for i in range(1, numero_pontos):
            variabilidade[i] = (variabilidade[i-1] + 
                               theta_ou * (mu_ou - variabilidade[i-1]) * dt[i] +
                               sigma_ou * np.sqrt(dt[i]) * np.random.randn())
        
        # 4. Tendência sistemática (polinômio de baixo grau + oscilação)
        tendencia = (0.0001 * (tempo / duracao_observacao) + 
                    0.00005 * np.sin(2 * np.pi * tempo / 30))
        
        # 5. Ruído fotométrico
        # Ruído de Poisson + ruído de leitura + ruído de céu
        erro_base = 0.0001  # ~100 ppm típico do Kepler
        erro_fluxo = erro_base * (1 + 0.1 * np.random.random(numero_pontos))
        ruido = np.random.normal(0, erro_fluxo)
        
        # Combinar componentes
        fluxo = fluxo_base * transito + variabilidade + tendencia + ruido
        
        return tempo, fluxo, erro_fluxo
    
    def salvar_dados(self, 
                     nome_estrela: str,
                     tempo: np.ndarray,
                     fluxo: np.ndarray,
                     erro_fluxo: np.ndarray) -> str:
        """
        Salva os dados em arquivo CSV.
        
        Parâmetros:
            nome_estrela: Nome identificador
            tempo, fluxo, erro_fluxo: Arrays de dados
            
        Retorna:
            Caminho do arquivo salvo
        """
        df = pd.DataFrame({
            'tempo_bjd': tempo,
            'fluxo_normalizado': fluxo,
            'erro_fluxo': erro_fluxo
        })
        
        nome_arquivo = os.path.join(
            self.diretorio_cache, 
            f"{nome_estrela.replace(' ', '_').lower()}_curva_luz.csv"
        )
        df.to_csv(nome_arquivo, index=False)
        
        return nome_arquivo
    
    def carregar_dados_salvos(self, nome_arquivo: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Carrega dados previamente salvos.
        
        Parâmetros:
            nome_arquivo: Caminho do arquivo CSV
            
        Retorna:
            Tupla com (tempo, fluxo, erro_fluxo)
        """
        df = pd.read_csv(nome_arquivo)
        return (df['tempo_bjd'].values, 
                df['fluxo_normalizado'].values, 
                df['erro_fluxo'].values)
    
    def plotar_curva_luz(self,
                         tempo: np.ndarray,
                         fluxo: np.ndarray,
                         erro_fluxo: np.ndarray,
                         titulo: str = "Curva de Luz",
                         salvar_como: Optional[str] = None) -> plt.Figure:
        """
        Plota a curva de luz com barras de erro.
        
        Parâmetros:
            tempo, fluxo, erro_fluxo: Arrays de dados
            titulo: Título do gráfico
            salvar_como: Caminho para salvar (opcional)
            
        Retorna:
            Objeto Figure do matplotlib
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), 
                                  gridspec_kw={'height_ratios': [3, 1]})
        
        # Painel superior: curva de luz completa
        ax1 = axes[0]
        ax1.errorbar(tempo, fluxo, yerr=erro_fluxo, 
                     fmt='.', markersize=1, alpha=0.5, 
                     color='#1f77b4', ecolor='#aec7e8', elinewidth=0.5)
        ax1.set_xlabel('Tempo (BJD)', fontsize=12)
        ax1.set_ylabel('Fluxo Normalizado', fontsize=12)
        ax1.set_title(titulo, fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Adicionar média móvel
        janela = min(100, len(fluxo) // 10)
        if janela > 1:
            media_movel = pd.Series(fluxo).rolling(window=janela, center=True).mean()
            ax1.plot(tempo, media_movel, 'r-', linewidth=1.5, 
                     label=f'Média Móvel (janela={janela})', alpha=0.8)
            ax1.legend(loc='upper right')
        
        # Painel inferior: zoom em região com possível trânsito
        ax2 = axes[1]
        
        # Encontrar região com menor fluxo (possível trânsito)
        indice_minimo = np.argmin(fluxo)
        margem = len(tempo) // 20
        inicio = max(0, indice_minimo - margem)
        fim = min(len(tempo), indice_minimo + margem)
        
        ax2.errorbar(tempo[inicio:fim], fluxo[inicio:fim], 
                     yerr=erro_fluxo[inicio:fim],
                     fmt='o', markersize=3, alpha=0.7,
                     color='#2ca02c', ecolor='#98df8a')
        ax2.set_xlabel('Tempo (BJD)', fontsize=12)
        ax2.set_ylabel('Fluxo', fontsize=12)
        ax2.set_title('Zoom em Região de Interesse', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if salvar_como:
            plt.savefig(salvar_como, dpi=150, bbox_inches='tight')
            
        return fig
    
    def obter_estatisticas_basicas(self,
                                    tempo: np.ndarray,
                                    fluxo: np.ndarray,
                                    erro_fluxo: np.ndarray) -> Dict:
        """
        Calcula estatísticas descritivas básicas dos dados.
        
        Parâmetros:
            tempo, fluxo, erro_fluxo: Arrays de dados
            
        Retorna:
            Dicionário com estatísticas
        """
        estatisticas = {
            'numero_pontos': len(tempo),
            'duracao_dias': tempo[-1] - tempo[0],
            'cadencia_media_min': np.median(np.diff(tempo)) * 24 * 60,
            'fluxo_medio': np.mean(fluxo),
            'fluxo_desvio_padrao': np.std(fluxo),
            'fluxo_mediana': np.median(fluxo),
            'fluxo_minimo': np.min(fluxo),
            'fluxo_maximo': np.max(fluxo),
            'erro_medio': np.mean(erro_fluxo),
            'snr_estimado': np.mean(fluxo) / np.mean(erro_fluxo),
            'percentil_1': np.percentile(fluxo, 1),
            'percentil_99': np.percentile(fluxo, 99)
        }
        
        return estatisticas


def executar_modulo_01(diretorio_saida: str = "resultados") -> Dict:
    """
    Função principal para executar o módulo de carregamento.
    
    Parâmetros:
        diretorio_saida: Diretório para salvar resultados
        
    Retorna:
        Dicionário com resultados do módulo
    """
    print("=" * 60)
    print("MÓDULO 01: CARREGAMENTO DE DADOS")
    print("Sistema de Detecção de Exoplanetas")
    print("Autor: Luiz Tiago Wilcke")
    print("=" * 60)
    
    # Criar diretório de saída
    if not os.path.exists(diretorio_saida):
        os.makedirs(diretorio_saida)
    
    # Inicializar carregador
    carregador = CarregadorDadosExoplanetas()
    
    resultados = {}
    
    # Carregar dados para cada estrela alvo
    for nome_estrela, info in carregador.estrelas_alvo.items():
        print(f"\n>>> Processando: {nome_estrela}")
        print(f"    Descrição: {info['descricao']}")
        print(f"    Período orbital: {info['periodo_orbital']:.4f} dias")
        
        # Carregar dados
        if info['tipo'] == 'Kepler':
            tempo, fluxo, erro = carregador.carregar_curva_luz_kepler(nome_estrela)
        else:
            tempo, fluxo, erro = carregador.carregar_curva_luz_tess(nome_estrela)
        
        # Salvar dados
        arquivo = carregador.salvar_dados(nome_estrela, tempo, fluxo, erro)
        print(f"    Dados salvos em: {arquivo}")
        
        # Calcular estatísticas
        stats = carregador.obter_estatisticas_basicas(tempo, fluxo, erro)
        
        print(f"\n    Estatísticas:")
        print(f"    - Número de pontos: {stats['numero_pontos']}")
        print(f"    - Duração: {stats['duracao_dias']:.2f} dias")
        print(f"    - Cadência média: {stats['cadencia_media_min']:.2f} min")
        print(f"    - SNR estimado: {stats['snr_estimado']:.1f}")
        print(f"    - Desvio padrão do fluxo: {stats['fluxo_desvio_padrao']:.6f}")
        
        # Plotar curva de luz
        nome_grafico = os.path.join(
            diretorio_saida, 
            f"curva_luz_{nome_estrela.replace(' ', '_').lower()}.png"
        )
        carregador.plotar_curva_luz(
            tempo, fluxo, erro,
            titulo=f"Curva de Luz - {nome_estrela}",
            salvar_como=nome_grafico
        )
        print(f"    Gráfico salvo em: {nome_grafico}")
        
        resultados[nome_estrela] = {
            'tempo': tempo,
            'fluxo': fluxo,
            'erro_fluxo': erro,
            'estatisticas': stats,
            'arquivo': arquivo
        }
    
    print("\n" + "=" * 60)
    print("MÓDULO 01 CONCLUÍDO")
    print("=" * 60)
    
    plt.close('all')
    
    return resultados


# Exportar classes e funções
__all__ = ['CarregadorDadosExoplanetas', 'executar_modulo_01']


if __name__ == "__main__":
    resultados = executar_modulo_01()
