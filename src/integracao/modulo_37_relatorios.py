"""
Módulo 37: Geração de Relatórios
Sistema Estatístico para Detecção de Exoplanetas
Autor: Luiz Tiago Wilcke

Geração automática de relatórios em formato texto e HTML.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from datetime import datetime
import os


class GeradorRelatorios:
    """
    Gera relatórios detalhados da análise de exoplanetas.
    
    Formatos suportados:
    - Texto (.txt)
    - Markdown (.md)
    - HTML (.html)
    """
    
    def __init__(self, titulo: str = "Relatório de Detecção de Exoplanetas"):
        """
        Inicializa o gerador.
        
        Parâmetros:
            titulo: Título do relatório
        """
        self.titulo = titulo
        self.secoes = []
        self.data_geracao = datetime.now()
    
    def adicionar_secao(self, titulo: str, conteudo: str) -> None:
        """Adiciona uma seção ao relatório."""
        self.secoes.append({'titulo': titulo, 'conteudo': conteudo})
    
    def adicionar_tabela(self, titulo: str, dados: Dict, headers: List[str]) -> None:
        """Adiciona uma tabela ao relatório."""
        linhas = []
        for chave, valores in dados.items():
            linha = [chave] + [str(v) for v in valores]
            linhas.append(linha)
        
        self.secoes.append({
            'titulo': titulo,
            'tipo': 'tabela',
            'headers': headers,
            'linhas': linhas
        })
    
    def gerar_texto(self) -> str:
        """Gera relatório em formato texto."""
        linhas = []
        linhas.append("=" * 70)
        linhas.append(self.titulo.center(70))
        linhas.append(f"Gerado em: {self.data_geracao.strftime('%Y-%m-%d %H:%M:%S')}")
        linhas.append("Autor: Luiz Tiago Wilcke")
        linhas.append("=" * 70)
        
        for secao in self.secoes:
            linhas.append(f"\n## {secao['titulo']}")
            linhas.append("-" * 50)
            
            if secao.get('tipo') == 'tabela':
                # Formatar tabela
                headers = secao['headers']
                larguras = [max(len(h), max(len(l[i]) for l in secao['linhas'])) 
                           for i, h in enumerate(headers)]
                
                linha_header = " | ".join(h.ljust(l) for h, l in zip(headers, larguras))
                linhas.append(linha_header)
                linhas.append("-" * len(linha_header))
                
                for linha in secao['linhas']:
                    linhas.append(" | ".join(c.ljust(l) for c, l in zip(linha, larguras)))
            else:
                linhas.append(secao['conteudo'])
        
        return '\n'.join(linhas)
    
    def gerar_markdown(self) -> str:
        """Gera relatório em formato Markdown."""
        linhas = []
        linhas.append(f"# {self.titulo}")
        linhas.append(f"\n**Gerado em:** {self.data_geracao.strftime('%Y-%m-%d %H:%M:%S')}")
        linhas.append("**Autor:** Luiz Tiago Wilcke\n")
        linhas.append("---\n")
        
        for secao in self.secoes:
            linhas.append(f"\n## {secao['titulo']}\n")
            
            if secao.get('tipo') == 'tabela':
                headers = secao['headers']
                linhas.append("| " + " | ".join(headers) + " |")
                linhas.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for linha in secao['linhas']:
                    linhas.append("| " + " | ".join(linha) + " |")
            else:
                linhas.append(secao['conteudo'])
        
        return '\n'.join(linhas)
    
    def gerar_html(self) -> str:
        """Gera relatório em formato HTML."""
        html = []
        html.append("""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Relatório - Detecção de Exoplanetas</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
        th { background-color: #3498db; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .meta { color: #7f8c8d; font-size: 0.9em; }
        .highlight { background-color: #ffffcc; padding: 2px 5px; }
    </style>
</head>
<body>
""")
        html.append(f"<h1>{self.titulo}</h1>")
        html.append(f'<p class="meta">Gerado em: {self.data_geracao.strftime("%Y-%m-%d %H:%M:%S")}<br>')
        html.append('Autor: Luiz Tiago Wilcke</p>')
        
        for secao in self.secoes:
            html.append(f"<h2>{secao['titulo']}</h2>")
            
            if secao.get('tipo') == 'tabela':
                html.append("<table>")
                html.append("<tr>" + "".join(f"<th>{h}</th>" for h in secao['headers']) + "</tr>")
                for linha in secao['linhas']:
                    html.append("<tr>" + "".join(f"<td>{c}</td>" for c in linha) + "</tr>")
                html.append("</table>")
            else:
                html.append(f"<p>{secao['conteudo']}</p>")
        
        html.append("</body></html>")
        return '\n'.join(html)
    
    def salvar(self, diretorio: str, nome_base: str = "relatorio") -> Dict[str, str]:
        """Salva relatório em múltiplos formatos."""
        caminhos = {}
        
        # Texto
        caminho_txt = os.path.join(diretorio, f"{nome_base}.txt")
        with open(caminho_txt, 'w', encoding='utf-8') as f:
            f.write(self.gerar_texto())
        caminhos['txt'] = caminho_txt
        
        # Markdown
        caminho_md = os.path.join(diretorio, f"{nome_base}.md")
        with open(caminho_md, 'w', encoding='utf-8') as f:
            f.write(self.gerar_markdown())
        caminhos['md'] = caminho_md
        
        # HTML
        caminho_html = os.path.join(diretorio, f"{nome_base}.html")
        with open(caminho_html, 'w', encoding='utf-8') as f:
            f.write(self.gerar_html())
        caminhos['html'] = caminho_html
        
        return caminhos


def executar_modulo_37(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    """Gera relatórios da análise."""
    print("=" * 60)
    print("MÓDULO 37: GERAÇÃO DE RELATÓRIOS")
    print("Sistema de Detecção de Exoplanetas")
    print("Autor: Luiz Tiago Wilcke")
    print("=" * 60)
    
    os.makedirs(diretorio_saida, exist_ok=True)
    
    gerador = GeradorRelatorios("Análise de Curvas de Luz para Detecção de Exoplanetas")
    
    # Resumo
    gerador.adicionar_secao("Resumo Executivo", 
        "Este relatório apresenta os resultados da análise estatística " +
        "de curvas de luz para detecção de exoplanetas usando métodos avançados.")
    
    # Dados analisados
    tabela_dados = {}
    for nome, dados in dados_entrada.items():
        tabela_dados[nome] = [
            len(dados['fluxo']),
            f"{np.mean(dados['fluxo']):.6f}",
            f"{np.std(dados['fluxo']):.6f}"
        ]
    
    gerador.adicionar_tabela("Dados Analisados", tabela_dados, 
                             ["Curva de Luz", "N pontos", "Média", "Desvio Padrão"])
    
    # Salvar
    caminhos = gerador.salvar(diretorio_saida, "relatorio_exoplanetas")
    
    for formato, caminho in caminhos.items():
        print(f"    Relatório {formato.upper()} salvo: {caminho}")
    
    return dados_entrada


__all__ = ['GeradorRelatorios', 'executar_modulo_37']
