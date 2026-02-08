"""
Módulo 40: Execução Completa
Sistema Estatístico para Detecção de Exoplanetas
Autor: Luiz Tiago Wilcke

Módulo integrador que executa a análise completa de curvas de luz
para detecção de exoplanetas usando todos os 40 módulos do sistema.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from datetime import datetime
import os
import sys


class ExecutorCompleto:
    """
    Executor integrado de todos os módulos do sistema.
    
    Orquestra a execução de:
    1. Módulos de dados (01-05)
    2. Módulos de séries temporais (06-15)
    3. Módulos de modelos lineares (16-25)
    4. Módulos Bayesianos (26-35)
    5. Módulos de integração (36-40)
    """
    
    def __init__(self, diretorio_saida: str = "resultados"):
        """
        Inicializa o executor.
        
        Parâmetros:
            diretorio_saida: Diretório para salvar todos os resultados
        """
        self.diretorio_saida = diretorio_saida
        self.resultados = {}
        self.log = []
        
        os.makedirs(diretorio_saida, exist_ok=True)
    
    def _log(self, mensagem: str) -> None:
        """Registra mensagem com timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entrada = f"[{timestamp}] {mensagem}"
        self.log.append(entrada)
        print(entrada)
    
    def executar_analise_completa(self, dados_entrada: Dict,
                                    modulos_selecionados: List[int] = None) -> Dict:
        """
        Executa análise completa usando todos os módulos.
        
        Parâmetros:
            dados_entrada: Dicionário com dados de entrada
            modulos_selecionados: Lista de módulos a executar (None = todos)
            
        Retorna:
            Dicionário com todos os resultados
        """
        self._log("=" * 60)
        self._log("SISTEMA DE DETECÇÃO DE EXOPLANETAS")
        self._log("Análise Estatística Avançada")
        self._log("Autor: Luiz Tiago Wilcke")
        self._log("=" * 60)
        
        inicio = datetime.now()
        dados_atuais = dados_entrada.copy()
        
        # Lista de módulos disponíveis
        modulos = {
            # Dados
            1: ("Carregamento de Dados", "src.dados.modulo_01_carregamento_dados", "executar_modulo_01"),
            2: ("Pré-processamento", "src.dados.modulo_02_preprocessamento", "executar_modulo_02"),
            3: ("Normalização", "src.dados.modulo_03_normalizacao", "executar_modulo_03"),
            4: ("Validação", "src.dados.modulo_04_validacao_dados", "executar_modulo_04"),
            5: ("Exploração", "src.dados.modulo_05_exploracao_dados", "executar_modulo_05"),
            
            # Séries Temporais
            6: ("Decomposição Temporal", "src.series_temporais.modulo_06_decomposicao_temporal", "executar_modulo_06"),
            7: ("Análise de Fourier", "src.series_temporais.modulo_07_analise_fourier", "executar_modulo_07"),
            8: ("Periodograma", "src.series_temporais.modulo_08_periodograma", "executar_modulo_08"),
            9: ("Autocorrelação", "src.series_temporais.modulo_09_autocorrelacao", "executar_modulo_09"),
            10: ("ARIMA", "src.series_temporais.modulo_10_arima", "executar_modulo_10"),
            11: ("SARIMA", "src.series_temporais.modulo_11_sarima", "executar_modulo_11"),
            12: ("Filtro de Kalman", "src.series_temporais.modulo_12_filtros_kalman", "executar_modulo_12"),
            13: ("Wavelets", "src.series_temporais.modulo_13_wavelets", "executar_modulo_13"),
            14: ("Detecção de Trânsitos", "src.series_temporais.modulo_14_deteccao_transitos", "executar_modulo_14"),
            15: ("Box-Fitting", "src.series_temporais.modulo_15_box_fitting", "executar_modulo_15"),
            
            # Modelos Lineares
            16: ("Regressão Linear", "src.modelos_lineares.modulo_16_regressao_linear", "executar_modulo_16"),
            17: ("Regressão Ridge", "src.modelos_lineares.modulo_17_regressao_ridge", "executar_modulo_17"),
            18: ("Regressão LASSO", "src.modelos_lineares.modulo_18_regressao_lasso", "executar_modulo_18"),
            19: ("Elastic Net", "src.modelos_lineares.modulo_19_elastic_net", "executar_modulo_19"),
            20: ("PCA", "src.modelos_lineares.modulo_20_pca", "executar_modulo_20"),
            21: ("Regressão Robusta", "src.modelos_lineares.modulo_21_regressao_robusta", "executar_modulo_21"),
            22: ("Modelos Mistos", "src.modelos_lineares.modulo_22_modelos_mistos", "executar_modulo_22"),
            23: ("Regressão Quantílica", "src.modelos_lineares.modulo_23_regressao_quantilica", "executar_modulo_23"),
            24: ("Splines", "src.modelos_lineares.modulo_24_splines", "executar_modulo_24"),
            25: ("GAM", "src.modelos_lineares.modulo_25_gam", "executar_modulo_25"),
            
            # Bayesianos
            26: ("Inferência Bayesiana", "src.bayesianos.modulo_26_inferencia_bayesiana", "executar_modulo_26"),
            27: ("MCMC Metropolis", "src.bayesianos.modulo_27_mcmc_metropolis", "executar_modulo_27"),
            28: ("Gibbs Sampler", "src.bayesianos.modulo_28_gibbs", "executar_modulo_28"),
            29: ("HMC", "src.bayesianos.modulo_29_hmc", "executar_modulo_29"),
            30: ("Modelo de Trânsito", "src.bayesianos.modulo_30_modelo_transito", "executar_modulo_30"),
            31: ("Comparação de Modelos", "src.bayesianos.modulo_31_comparacao_modelos", "executar_modulo_31"),
            32: ("Nested Sampling", "src.bayesianos.modulo_32_nested_sampling", "executar_modulo_32"),
            33: ("Processos Gaussianos", "src.bayesianos.modulo_33_processos_gaussianos", "executar_modulo_33"),
            34: ("Modelos Hierárquicos", "src.bayesianos.modulo_34_modelos_hierarquicos", "executar_modulo_34"),
            35: ("Análise de Posteriors", "src.bayesianos.modulo_35_analise_posteriors", "executar_modulo_35"),
            
            # Integração
            36: ("Pipeline", "src.integracao.modulo_36_pipeline", "executar_modulo_36"),
            37: ("Relatórios", "src.integracao.modulo_37_relatorios", "executar_modulo_37"),
            38: ("Visualização", "src.integracao.modulo_38_visualizacao", "executar_modulo_38"),
            39: ("Exportação", "src.integracao.modulo_39_exportacao", "executar_modulo_39"),
        }
        
        if modulos_selecionados is None:
            modulos_selecionados = list(modulos.keys())
        
        for num_modulo in modulos_selecionados:
            if num_modulo not in modulos:
                continue
            
            nome, modulo_path, funcao_nome = modulos[num_modulo]
            
            self._log(f"\n>>> MÓDULO {num_modulo:02d}: {nome}")
            
            try:
                # Importar dinamicamente
                modulo = __import__(modulo_path, fromlist=[funcao_nome])
                funcao = getattr(modulo, funcao_nome)
                
                # Executar
                dados_atuais = funcao(dados_atuais, diretorio_saida=self.diretorio_saida)
                
                self._log(f"    ✓ Concluído")
                self.resultados[num_modulo] = {'status': 'sucesso', 'nome': nome}
                
            except Exception as e:
                self._log(f"    ✗ Erro: {str(e)[:50]}")
                self.resultados[num_modulo] = {'status': 'erro', 'nome': nome, 'erro': str(e)}
        
        duracao = (datetime.now() - inicio).total_seconds()
        
        self._log(f"\n{'='*60}")
        self._log(f"ANÁLISE CONCLUÍDA em {duracao:.1f} segundos")
        self._log(f"{'='*60}")
        
        # Salvar log
        log_path = os.path.join(self.diretorio_saida, "execucao_log.txt")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.log))
        
        return dados_atuais
    
    def gerar_sumario(self) -> str:
        """Gera sumário da execução."""
        sucessos = sum(1 for r in self.resultados.values() if r['status'] == 'sucesso')
        erros = sum(1 for r in self.resultados.values() if r['status'] == 'erro')
        
        linhas = []
        linhas.append("=" * 50)
        linhas.append("SUMÁRIO DA EXECUÇÃO")
        linhas.append("=" * 50)
        linhas.append(f"Módulos executados com sucesso: {sucessos}")
        linhas.append(f"Módulos com erro: {erros}")
        linhas.append("")
        
        for num, res in sorted(self.resultados.items()):
            status = "✓" if res['status'] == 'sucesso' else "✗"
            linhas.append(f"  {status} Módulo {num:02d}: {res['nome']}")
        
        return '\n'.join(linhas)


def executar_modulo_40(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    """Executa análise completa do sistema."""
    print("=" * 60)
    print("MÓDULO 40: EXECUÇÃO COMPLETA")
    print("Sistema de Detecção de Exoplanetas")
    print("Autor: Luiz Tiago Wilcke")
    print("=" * 60)
    
    executor = ExecutorCompleto(diretorio_saida)
    
    # Executar apenas módulos essenciais para demonstração rápida
    modulos_demo = [4, 5, 7, 8, 14, 16, 26, 37, 38, 39]
    
    print("\n>>> Executando módulos selecionados para demonstração...")
    resultados = executor.executar_analise_completa(dados_entrada, modulos_demo)
    
    print("\n" + executor.gerar_sumario())
    
    print("\n" + "=" * 60)
    print("MÓDULO 40 CONCLUÍDO")
    print("SISTEMA DE DETECÇÃO DE EXOPLANETAS FINALIZADO")
    print("=" * 60)
    
    return resultados


__all__ = ['ExecutorCompleto', 'executar_modulo_40']
