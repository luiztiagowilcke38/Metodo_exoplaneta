"""
Módulo 39: Exportação de Resultados
Sistema Estatístico para Detecção de Exoplanetas
Autor: Luiz Tiago Wilcke

Exportação de resultados em múltiplos formatos para análise posterior.
"""

import numpy as np
import json
import csv
from typing import Dict, List, Optional, Any
from datetime import datetime
import os


class ExportadorResultados:
    """
    Exportador de resultados de análise de exoplanetas.
    
    Formatos suportados:
    - CSV: Para dados tabulares
    - JSON: Para estruturas complexas
    - NPZ: Para arrays NumPy
    - TXT: Para resumos legíveis
    """
    
    def __init__(self, diretorio_saida: str = "resultados"):
        """
        Inicializa o exportador.
        
        Parâmetros:
            diretorio_saida: Diretório para salvar os arquivos
        """
        self.diretorio_saida = diretorio_saida
        os.makedirs(diretorio_saida, exist_ok=True)
    
    def exportar_csv(self, dados: Dict, nome_arquivo: str) -> str:
        """
        Exporta dados tabulares para CSV.
        
        Parâmetros:
            dados: Dicionário com chaves como colunas
            nome_arquivo: Nome do arquivo (sem extensão)
            
        Retorna:
            Caminho do arquivo salvo
        """
        caminho = os.path.join(self.diretorio_saida, f"{nome_arquivo}.csv")
        
        # Converter para formato tabular
        if isinstance(dados, dict):
            # Verificar se é um dicionário de arrays
            if all(isinstance(v, np.ndarray) for v in dados.values()):
                with open(caminho, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(dados.keys())
                    for row in zip(*dados.values()):
                        writer.writerow(row)
            else:
                # Dicionário de escalares ou misto
                with open(caminho, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['chave', 'valor'])
                    for k, v in dados.items():
                        if isinstance(v, np.ndarray):
                            v = v.tolist()
                        writer.writerow([k, v])
        
        return caminho
    
    def exportar_json(self, dados: Dict, nome_arquivo: str, 
                       indent: int = 2) -> str:
        """
        Exporta dados para JSON.
        
        Parâmetros:
            dados: Dicionário de dados
            nome_arquivo: Nome do arquivo (sem extensão)
            indent: Indentação para legibilidade
            
        Retorna:
            Caminho do arquivo salvo
        """
        caminho = os.path.join(self.diretorio_saida, f"{nome_arquivo}.json")
        
        # Converter arrays NumPy para listas
        def converter(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: converter(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [converter(v) for v in obj]
            return obj
        
        dados_json = converter(dados)
        
        with open(caminho, 'w', encoding='utf-8') as f:
            json.dump(dados_json, f, indent=indent, ensure_ascii=False)
        
        return caminho
    
    def exportar_npz(self, dados: Dict, nome_arquivo: str) -> str:
        """
        Exporta arrays NumPy comprimidos.
        
        Parâmetros:
            dados: Dicionário de arrays
            nome_arquivo: Nome do arquivo (sem extensão)
            
        Retorna:
            Caminho do arquivo salvo
        """
        caminho = os.path.join(self.diretorio_saida, f"{nome_arquivo}.npz")
        
        # Filtrar apenas arrays
        arrays = {k: np.array(v) for k, v in dados.items() 
                  if isinstance(v, (np.ndarray, list))}
        
        np.savez_compressed(caminho, **arrays)
        
        return caminho
    
    def exportar_resumo_txt(self, resultados: Dict, nome_arquivo: str) -> str:
        """
        Exporta resumo em texto legível.
        
        Parâmetros:
            resultados: Dicionário de resultados
            nome_arquivo: Nome do arquivo (sem extensão)
            
        Retorna:
            Caminho do arquivo salvo
        """
        caminho = os.path.join(self.diretorio_saida, f"{nome_arquivo}.txt")
        
        with open(caminho, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("RESUMO DE RESULTADOS - DETECÇÃO DE EXOPLANETAS\n")
            f.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Autor: Luiz Tiago Wilcke\n")
            f.write("=" * 70 + "\n\n")
            
            for nome, dados in resultados.items():
                f.write(f"\n## {nome}\n")
                f.write("-" * 50 + "\n")
                
                self._escrever_recursivo(f, dados, nivel=0)
        
        return caminho
    
    def _escrever_recursivo(self, f, obj, nivel: int = 0) -> None:
        """Escreve objeto recursivamente com indentação."""
        indent = "  " * nivel
        
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    f.write(f"{indent}{k}:\n")
                    self._escrever_recursivo(f, v, nivel + 1)
                elif isinstance(v, np.ndarray):
                    if len(v) > 10:
                        f.write(f"{indent}{k}: array({len(v)} elementos, μ={np.mean(v):.4f})\n")
                    else:
                        f.write(f"{indent}{k}: {v}\n")
                else:
                    f.write(f"{indent}{k}: {v}\n")
        elif isinstance(obj, list):
            if len(obj) > 10:
                f.write(f"{indent}[{len(obj)} elementos]\n")
            else:
                for i, item in enumerate(obj):
                    f.write(f"{indent}[{i}]: {item}\n")
    
    def exportar_completo(self, resultados: Dict, prefixo: str = "analise") -> Dict[str, str]:
        """
        Exporta resultados em todos os formatos disponíveis.
        
        Retorna:
            Dicionário com caminhos de todos os arquivos gerados
        """
        caminhos = {}
        
        # Dados numéricos para NPZ
        dados_arrays = {}
        for nome, dados in resultados.items():
            if 'tempo' in dados and 'fluxo' in dados:
                dados_arrays[f"{nome}_tempo"] = dados['tempo']
                dados_arrays[f"{nome}_fluxo"] = dados['fluxo']
        
        if dados_arrays:
            caminhos['npz'] = self.exportar_npz(dados_arrays, f"{prefixo}_dados")
        
        # Resumo em JSON
        caminhos['json'] = self.exportar_json(resultados, f"{prefixo}_resultados")
        
        # Resumo em texto
        caminhos['txt'] = self.exportar_resumo_txt(resultados, f"{prefixo}_resumo")
        
        return caminhos


def executar_modulo_39(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    """Exporta resultados da análise."""
    print("=" * 60)
    print("MÓDULO 39: EXPORTAÇÃO DE RESULTADOS")
    print("Sistema de Detecção de Exoplanetas")
    print("Autor: Luiz Tiago Wilcke")
    print("=" * 60)
    
    exportador = ExportadorResultados(diretorio_saida)
    caminhos = exportador.exportar_completo(dados_entrada, "exoplanetas")
    
    print("\n>>> Arquivos exportados:")
    for formato, caminho in caminhos.items():
        print(f"    {formato.upper()}: {caminho}")
    
    print("\n" + "=" * 60)
    print("MÓDULO 39 CONCLUÍDO")
    print("=" * 60)
    
    return dados_entrada


__all__ = ['ExportadorResultados', 'executar_modulo_39']
