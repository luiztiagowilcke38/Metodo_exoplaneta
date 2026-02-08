"""
Módulo 36: Pipeline de Processamento
Sistema Estatístico para Detecção de Exoplanetas
Autor: Luiz Tiago Wilcke

Pipeline completo que orquestra todos os módulos de processamento.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Callable
from datetime import datetime
import os
import json


class PipelineProcessamento:
    """
    Pipeline de processamento de curvas de luz para detecção de exoplanetas.
    
    Orquestra a execução sequencial de:
    1. Carregamento de dados
    2. Pré-processamento
    3. Normalização
    4. Validação
    5. Análise exploratória
    """
    
    def __init__(self, diretorio_saida: str = "resultados"):
        """
        Inicializa o pipeline.
        
        Parâmetros:
            diretorio_saida: Diretório para salvar resultados
        """
        self.diretorio_saida = diretorio_saida
        self.etapas = []
        self.resultados = {}
        self.log = []
        
        os.makedirs(diretorio_saida, exist_ok=True)
    
    def adicionar_etapa(self, nome: str, funcao: Callable, **kwargs) -> None:
        """
        Adiciona uma etapa ao pipeline.
        
        Parâmetros:
            nome: Nome descritivo da etapa
            funcao: Função a ser executada
            kwargs: Argumentos adicionais para a função
        """
        self.etapas.append({
            'nome': nome,
            'funcao': funcao,
            'kwargs': kwargs
        })
        self._log(f"Etapa adicionada: {nome}")
    
    def _log(self, mensagem: str) -> None:
        """Registra mensagem no log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entrada = f"[{timestamp}] {mensagem}"
        self.log.append(entrada)
        print(entrada)
    
    def executar(self, dados_iniciais: Dict) -> Dict:
        """
        Executa todas as etapas do pipeline.
        
        Parâmetros:
            dados_iniciais: Dicionário com dados de entrada
            
        Retorna:
            Dicionário com resultados de todas as etapas
        """
        self._log("=" * 60)
        self._log("INICIANDO PIPELINE DE PROCESSAMENTO")
        self._log("=" * 60)
        
        dados_atuais = dados_iniciais.copy()
        inicio_total = datetime.now()
        
        for i, etapa in enumerate(self.etapas):
            self._log(f"\n>>> Etapa {i+1}/{len(self.etapas)}: {etapa['nome']}")
            inicio = datetime.now()
            
            try:
                dados_atuais = etapa['funcao'](
                    dados_atuais, 
                    diretorio_saida=self.diretorio_saida,
                    **etapa['kwargs']
                )
                duracao = (datetime.now() - inicio).total_seconds()
                self._log(f"    Concluída em {duracao:.2f}s")
                self.resultados[etapa['nome']] = {'status': 'sucesso', 'duracao': duracao}
            except Exception as e:
                self._log(f"    ERRO: {str(e)}")
                self.resultados[etapa['nome']] = {'status': 'erro', 'mensagem': str(e)}
        
        duracao_total = (datetime.now() - inicio_total).total_seconds()
        self._log(f"\n{'='*60}")
        self._log(f"PIPELINE CONCLUÍDO em {duracao_total:.2f}s")
        self._log(f"{'='*60}")
        
        # Salvar log
        log_path = os.path.join(self.diretorio_saida, "pipeline_log.txt")
        with open(log_path, 'w') as f:
            f.write('\n'.join(self.log))
        
        return dados_atuais
    
    def gerar_relatorio(self) -> str:
        """Gera relatório resumido do pipeline."""
        relatorio = []
        relatorio.append("=" * 60)
        relatorio.append("RELATÓRIO DO PIPELINE")
        relatorio.append("=" * 60)
        
        for nome, resultado in self.resultados.items():
            status = "✓" if resultado['status'] == 'sucesso' else "✗"
            duracao = resultado.get('duracao', 0)
            relatorio.append(f"  {status} {nome}: {duracao:.2f}s")
        
        return '\n'.join(relatorio)


def executar_modulo_36(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    """Executa pipeline de processamento."""
    print("=" * 60)
    print("MÓDULO 36: PIPELINE DE PROCESSAMENTO")
    print("Sistema de Detecção de Exoplanetas")
    print("Autor: Luiz Tiago Wilcke")
    print("=" * 60)
    
    os.makedirs(diretorio_saida, exist_ok=True)
    
    # Criar pipeline de demonstração
    pipeline = PipelineProcessamento(diretorio_saida)
    
    # Funções de exemplo para etapas
    def etapa_validacao(dados, diretorio_saida):
        for nome in dados:
            dados[nome]['validado'] = True
        return dados
    
    def etapa_estatisticas(dados, diretorio_saida):
        for nome, d in dados.items():
            d['estatisticas'] = {
                'n_pontos': len(d['fluxo']),
                'media': np.mean(d['fluxo']),
                'std': np.std(d['fluxo'])
            }
        return dados
    
    pipeline.adicionar_etapa("Validação", etapa_validacao)
    pipeline.adicionar_etapa("Estatísticas", etapa_estatisticas)
    
    resultados = pipeline.executar(dados_entrada)
    
    print(pipeline.gerar_relatorio())
    
    return resultados


__all__ = ['PipelineProcessamento', 'executar_modulo_36']
