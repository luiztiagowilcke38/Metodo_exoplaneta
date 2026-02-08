"""
Módulo 12: Filtro de Kalman
Sistema Estatístico para Detecção de Exoplanetas
Autor: Luiz Tiago Wilcke

Implementação completa do Filtro de Kalman para estimação
de estados ocultos em curvas de luz estelares.

Modelo de espaço de estados:
    x_k = F_k * x_{k-1} + B_k * u_k + w_k    (Equação de transição)
    z_k = H_k * x_k + v_k                     (Equação de observação)

Onde:
    x_k: Vetor de estado (fluxo verdadeiro, tendência, derivada)
    z_k: Observação (fluxo medido)
    F_k: Matriz de transição de estado
    H_k: Matriz de observação
    Q_k: Covariância do ruído de processo
    R_k: Covariância do ruído de medição
    w_k ~ N(0, Q_k): Ruído de processo
    v_k ~ N(0, R_k): Ruído de medição
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
from scipy.linalg import inv, cholesky
import os


class FiltroKalman:
    """
    Filtro de Kalman completo com suporte a:
    - Modelos de múltiplos estados
    - Suavização RTS (Rauch-Tung-Striebel)
    - Estimação adaptativa de parâmetros
    - Detecção de anomalias
    """
    
    def __init__(self, dim_estado: int = 3, dim_observacao: int = 1):
        """
        Inicializa o filtro de Kalman.
        
        Parâmetros:
            dim_estado: Dimensão do vetor de estado
            dim_observacao: Dimensão do vetor de observação
        """
        self.n = dim_estado
        self.m = dim_observacao
        
        # Estado inicial
        self.x = np.zeros(self.n)
        self.P = np.eye(self.n) * 1000  # Covariância inicial alta (incerteza)
        
        # Matrizes do modelo
        self.F = None  # Transição de estado
        self.H = None  # Observação
        self.Q = None  # Covariância do processo
        self.R = None  # Covariância da medição
        
        # Histórico
        self.estados_filtrados = []
        self.covariancias_filtradas = []
        self.inovacoes = []
        self.covariancias_inovacao = []
        self.ganhos_kalman = []
        self.log_verossimilhanca = 0
    
    def configurar_modelo_curva_luz(self, dt: float = 1.0, 
                                     sigma_processo: float = 1e-5,
                                     sigma_medicao: float = 1e-4):
        """
        Configura modelo de espaço de estados para curvas de luz.
        
        Estado: [fluxo, tendência, curvatura]
        O modelo assume que o fluxo verdadeiro segue um processo
        com tendência suave e possíveis trânsitos.
        
        Parâmetros:
            dt: Intervalo de tempo entre observações
            sigma_processo: Desvio padrão do ruído de processo
            sigma_medicao: Desvio padrão do ruído de medição
        """
        # Matriz de transição (modelo de movimento com aceleração constante)
        # x_k = F * x_{k-1}
        # [fluxo]      [1  dt  dt²/2] [fluxo]
        # [tendência] = [0  1   dt   ] [tendência]
        # [curvatura]   [0  0   1    ] [curvatura]
        self.F = np.array([
            [1, dt, 0.5 * dt**2],
            [0, 1, dt],
            [0, 0, 1]
        ])
        
        # Matriz de observação (observamos apenas o fluxo)
        self.H = np.array([[1, 0, 0]])
        
        # Covariância do ruído de processo
        # Modelo de ruído de jerk constante
        q = sigma_processo**2
        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4
        self.Q = q * np.array([
            [dt4/4, dt3/2, dt2/2],
            [dt3/2, dt2, dt],
            [dt2/2, dt, 1]
        ])
        
        # Covariância do ruído de medição
        self.R = np.array([[sigma_medicao**2]])
    
    def configurar_modelo_transito(self, dt: float = 1.0,
                                    sigma_fluxo: float = 1e-5,
                                    sigma_transito: float = 1e-4,
                                    sigma_medicao: float = 1e-4):
        """
        Configura modelo específico para detecção de trânsitos.
        
        Estado: [fluxo_base, profundidade_transito, taxa_variacao, fase]
        """
        self.n = 4
        self.x = np.zeros(4)
        self.x[0] = 1.0  # Fluxo base normalizado
        self.P = np.diag([1e-6, 1e-4, 1e-6, 1e-2])
        
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, 0],
            [0, 0, 0.99, 0],
            [0, 0, 0, 1]
        ])
        
        self.H = np.array([[1, -1, 0, 0]])  # Fluxo = base - profundidade * indicador
        
        self.Q = np.diag([sigma_fluxo**2, sigma_transito**2, 
                          (sigma_fluxo/dt)**2, 1e-6])
        self.R = np.array([[sigma_medicao**2]])
    
    def predizer(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Etapa de predição do filtro de Kalman.
        
        x̂_{k|k-1} = F_k * x̂_{k-1|k-1}
        P_{k|k-1} = F_k * P_{k-1|k-1} * F_k^T + Q_k
        
        Retorna:
            Tupla com (estado predito, covariância predita)
        """
        # Estado predito
        x_pred = self.F @ self.x
        
        # Covariância predita
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        return x_pred, P_pred
    
    def atualizar(self, z: np.ndarray, x_pred: np.ndarray, 
                  P_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Etapa de atualização do filtro de Kalman.
        
        Inovação: ỹ_k = z_k - H_k * x̂_{k|k-1}
        Covariância da inovação: S_k = H_k * P_{k|k-1} * H_k^T + R_k
        Ganho de Kalman: K_k = P_{k|k-1} * H_k^T * S_k^{-1}
        Estado atualizado: x̂_{k|k} = x̂_{k|k-1} + K_k * ỹ_k
        Covariância atualizada: P_{k|k} = (I - K_k * H_k) * P_{k|k-1}
        
        Parâmetros:
            z: Vetor de observação
            x_pred: Estado predito
            P_pred: Covariância predita
            
        Retorna:
            Tupla com (estado atualizado, covariância atualizada, métricas)
        """
        # Garantir que z seja array
        z = np.atleast_1d(z)
        
        # Inovação (resíduo de medição)
        y = z - self.H @ x_pred
        
        # Covariância da inovação
        S = self.H @ P_pred @ self.H.T + self.R
        
        # Ganho de Kalman
        # K = P_pred * H^T * S^{-1}
        try:
            S_inv = inv(S)
        except:
            S_inv = np.linalg.pinv(S)
        
        K = P_pred @ self.H.T @ S_inv
        
        # Estado atualizado
        x_upd = x_pred + K @ y
        
        # Covariância atualizada (forma de Joseph para estabilidade numérica)
        # P = (I - KH)P(I - KH)^T + KRK^T
        I_KH = np.eye(self.n) - K @ self.H
        P_upd = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        
        # Log-verossimilhança incremental
        # log p(z_k | z_{1:k-1}) = -0.5 * (log|S| + y^T S^{-1} y + m*log(2π))
        log_det_S = np.log(np.linalg.det(S))
        mahalanobis = y.T @ S_inv @ y
        ll_increment = -0.5 * (log_det_S + mahalanobis + self.m * np.log(2 * np.pi))
        
        metricas = {
            'inovacao': y.flatten(),
            'cov_inovacao': S,
            'ganho_kalman': K,
            'mahalanobis': mahalanobis.item() if hasattr(mahalanobis, 'item') else mahalanobis,
            'll_increment': ll_increment.item() if hasattr(ll_increment, 'item') else ll_increment
        }
        
        return x_upd, P_upd, metricas
    
    def filtrar(self, observacoes: np.ndarray) -> Dict:
        """
        Executa o filtro de Kalman em toda a série de observações.
        
        Parâmetros:
            observacoes: Array de observações [N x m]
            
        Retorna:
            Dicionário com resultados da filtragem
        """
        N = len(observacoes)
        
        # Resetar histórico
        self.estados_filtrados = np.zeros((N, self.n))
        self.covariancias_filtradas = np.zeros((N, self.n, self.n))
        self.estados_preditos = np.zeros((N, self.n))
        self.covariancias_preditas = np.zeros((N, self.n, self.n))
        self.inovacoes = np.zeros((N, self.m))
        self.covariancias_inovacao = np.zeros((N, self.m, self.m))
        self.ganhos_kalman = np.zeros((N, self.n, self.m))
        self.log_verossimilhanca = 0
        
        # Inicializar com primeira observação
        self.x[0] = observacoes[0]
        
        for k in range(N):
            # Predição
            x_pred, P_pred = self.predizer()
            self.estados_preditos[k] = x_pred
            self.covariancias_preditas[k] = P_pred
            
            # Atualização
            x_upd, P_upd, metricas = self.atualizar(observacoes[k], x_pred, P_pred)
            
            # Salvar resultados
            self.x = x_upd
            self.P = P_upd
            self.estados_filtrados[k] = x_upd
            self.covariancias_filtradas[k] = P_upd
            self.inovacoes[k] = metricas['inovacao']
            self.covariancias_inovacao[k] = metricas['cov_inovacao']
            self.ganhos_kalman[k] = metricas['ganho_kalman']
            self.log_verossimilhanca += metricas['ll_increment']
        
        return {
            'estados': self.estados_filtrados,
            'covariancias': self.covariancias_filtradas,
            'inovacoes': self.inovacoes,
            'log_verossimilhanca': self.log_verossimilhanca
        }
    
    def suavizar_rts(self) -> Dict:
        """
        Suavização Rauch-Tung-Striebel (RTS).
        
        Propaga informação de trás para frente para obter
        estimativas ótimas dado TODAS as observações.
        
        x̂_{k|N} = x̂_{k|k} + C_k * (x̂_{k+1|N} - x̂_{k+1|k})
        P_{k|N} = P_{k|k} + C_k * (P_{k+1|N} - P_{k+1|k}) * C_k^T
        
        Onde: C_k = P_{k|k} * F^T * P_{k+1|k}^{-1}
        
        Retorna:
            Dicionário com estados e covariâncias suavizados
        """
        N = len(self.estados_filtrados)
        
        estados_suavizados = np.zeros_like(self.estados_filtrados)
        covariancias_suavizadas = np.zeros_like(self.covariancias_filtradas)
        
        # Condição inicial: último estado filtrado
        estados_suavizados[-1] = self.estados_filtrados[-1]
        covariancias_suavizadas[-1] = self.covariancias_filtradas[-1]
        
        # Propagação reversa
        for k in range(N - 2, -1, -1):
            # Ganho de suavização
            try:
                P_pred_inv = inv(self.covariancias_preditas[k + 1])
            except:
                P_pred_inv = np.linalg.pinv(self.covariancias_preditas[k + 1])
            
            C = self.covariancias_filtradas[k] @ self.F.T @ P_pred_inv
            
            # Estado suavizado
            estados_suavizados[k] = (self.estados_filtrados[k] + 
                                     C @ (estados_suavizados[k + 1] - self.estados_preditos[k + 1]))
            
            # Covariância suavizada
            covariancias_suavizadas[k] = (self.covariancias_filtradas[k] + 
                                          C @ (covariancias_suavizadas[k + 1] - 
                                               self.covariancias_preditas[k + 1]) @ C.T)
        
        return {
            'estados': estados_suavizados,
            'covariancias': covariancias_suavizadas
        }
    
    def detectar_anomalias(self, threshold_mahalanobis: float = 3.0) -> np.ndarray:
        """
        Detecta anomalias usando a distância de Mahalanobis das inovações.
        
        Trânsitos geram inovações grandes (fluxo observado << predito).
        
        Parâmetros:
            threshold_mahalanobis: Threshold para classificar como anomalia
            
        Retorna:
            Array booleano indicando anomalias
        """
        N = len(self.inovacoes)
        mahalanobis = np.zeros(N)
        
        for k in range(N):
            try:
                S_inv = inv(self.covariancias_inovacao[k])
            except:
                S_inv = np.linalg.pinv(self.covariancias_inovacao[k])
            
            mahalanobis[k] = np.sqrt(self.inovacoes[k] @ S_inv @ self.inovacoes[k])
        
        anomalias = mahalanobis > threshold_mahalanobis
        
        return anomalias, mahalanobis
    
    def estimar_parametros_em(self, observacoes: np.ndarray, 
                               max_iter: int = 50,
                               tolerancia: float = 1e-6) -> Dict:
        """
        Estimação de parâmetros via algoritmo EM (Expectation-Maximization).
        
        Estima Q e R maximizando a verossimilhança marginal.
        
        E-step: Filtrar e suavizar
        M-step: Atualizar Q e R
        
        Q̂ = (1/N) Σ (x̂_{k|N} - F*x̂_{k-1|N})(...)^T + F*P_{k-1|N}*F^T - P_{k|N}
        R̂ = (1/N) Σ (z_k - H*x̂_{k|N})(...)^T + H*P_{k|N}*H^T
        """
        ll_anterior = -np.inf
        
        for iteracao in range(max_iter):
            # E-step
            resultado = self.filtrar(observacoes)
            suavizado = self.suavizar_rts()
            
            N = len(observacoes)
            x_suav = suavizado['estados']
            P_suav = suavizado['covariancias']
            
            # M-step: Atualizar Q
            Q_novo = np.zeros_like(self.Q)
            for k in range(1, N):
                diff = x_suav[k] - self.F @ x_suav[k-1]
                Q_novo += np.outer(diff, diff)
                Q_novo += self.F @ P_suav[k-1] @ self.F.T
                Q_novo -= P_suav[k]
            Q_novo /= (N - 1)
            
            # M-step: Atualizar R
            R_novo = np.zeros_like(self.R)
            for k in range(N):
                diff = observacoes[k] - self.H @ x_suav[k]
                R_novo += np.outer(diff, diff)
                R_novo += self.H @ P_suav[k] @ self.H.T
            R_novo /= N
            
            # Garantir positividade
            self.Q = 0.5 * (Q_novo + Q_novo.T) + 1e-10 * np.eye(self.n)
            self.R = 0.5 * (R_novo + R_novo.T) + 1e-10 * np.eye(self.m)
            
            # Verificar convergência
            ll = resultado['log_verossimilhanca']
            if abs(ll - ll_anterior) < tolerancia:
                break
            ll_anterior = ll
        
        return {
            'Q': self.Q,
            'R': self.R,
            'log_verossimilhanca': ll,
            'iteracoes': iteracao + 1
        }
    
    def plotar_resultados(self, tempo: np.ndarray, observacoes: np.ndarray,
                          titulo: str, salvar: Optional[str] = None) -> plt.Figure:
        """Gera visualização completa dos resultados do filtro."""
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
        # Estado filtrado vs observado
        ax = axes[0, 0]
        ax.scatter(tempo, observacoes, s=1, alpha=0.3, c='gray', label='Observado')
        ax.plot(tempo, self.estados_filtrados[:, 0], 'b-', lw=1, label='Filtrado')
        
        # Intervalo de confiança
        std = np.sqrt(self.covariancias_filtradas[:, 0, 0])
        ax.fill_between(tempo, self.estados_filtrados[:, 0] - 2*std,
                       self.estados_filtrados[:, 0] + 2*std,
                       alpha=0.2, color='blue', label='IC 95%')
        ax.set_xlabel('Tempo (BJD)'); ax.set_ylabel('Fluxo')
        ax.set_title('Filtragem de Kalman'); ax.legend()
        
        # Inovações
        ax = axes[0, 1]
        ax.scatter(tempo, self.inovacoes[:, 0], s=2, alpha=0.5)
        ax.axhline(0, color='r', ls='--')
        std_inov = np.sqrt(self.covariancias_inovacao[:, 0, 0])
        ax.fill_between(tempo, -2*std_inov, 2*std_inov, alpha=0.2, color='red')
        ax.set_xlabel('Tempo'); ax.set_ylabel('Inovação')
        ax.set_title('Inovações (Resíduos de Predição)')
        
        # Detecção de anomalias
        ax = axes[1, 0]
        anomalias, mahalanobis = self.detectar_anomalias(3.0)
        ax.plot(tempo, mahalanobis, 'b-', lw=0.5)
        ax.axhline(3.0, color='r', ls='--', label='Threshold')
        ax.scatter(tempo[anomalias], mahalanobis[anomalias], c='red', s=10, label='Anomalias')
        ax.set_xlabel('Tempo'); ax.set_ylabel('Distância de Mahalanobis')
        ax.set_title('Detecção de Anomalias'); ax.legend()
        
        # Tendência estimada
        ax = axes[1, 1]
        if self.n >= 2:
            ax.plot(tempo, self.estados_filtrados[:, 1], 'g-', lw=1)
            ax.set_ylabel('Tendência (derivada)')
        ax.set_xlabel('Tempo')
        ax.set_title('Tendência Estimada')
        
        # Ganho de Kalman
        ax = axes[2, 0]
        ax.plot(tempo, self.ganhos_kalman[:, 0, 0], 'b-', lw=0.5)
        ax.set_xlabel('Tempo'); ax.set_ylabel('K[0,0]')
        ax.set_title('Ganho de Kalman (primeiro elemento)')
        
        # ACF das inovações
        ax = axes[2, 1]
        inov = self.inovacoes[:, 0]
        n_lags = min(50, len(inov) // 4)
        acf = np.correlate(inov, inov, 'full')
        acf = acf[len(inov)-1:len(inov)+n_lags] / acf[len(inov)-1]
        ax.bar(range(len(acf)), acf, alpha=0.7)
        ax.axhline(1.96/np.sqrt(len(inov)), color='r', ls='--')
        ax.axhline(-1.96/np.sqrt(len(inov)), color='r', ls='--')
        ax.set_xlabel('Lag'); ax.set_ylabel('ACF')
        ax.set_title('ACF das Inovações (deve ser ruído branco)')
        
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{titulo}\nLog-verossimilhança: {self.log_verossimilhanca:.2f}',
                    fontweight='bold')
        plt.tight_layout()
        
        if salvar:
            plt.savefig(salvar, dpi=150, bbox_inches='tight')
        
        return fig


def executar_modulo_12(dados_entrada: Dict, diretorio_saida: str = "resultados") -> Dict:
    """Executa análise com Filtro de Kalman."""
    print("=" * 60)
    print("MÓDULO 12: FILTRO DE KALMAN")
    print("Sistema de Detecção de Exoplanetas")
    print("Autor: Luiz Tiago Wilcke")
    print("=" * 60)
    
    os.makedirs(diretorio_saida, exist_ok=True)
    resultados = {}
    
    for nome, dados in dados_entrada.items():
        print(f"\n>>> Processando: {nome}")
        
        tempo = dados['tempo']
        fluxo = dados['fluxo']
        erro = dados.get('erro_fluxo', np.ones_like(fluxo) * 0.0001)
        
        # Calcular dt médio
        dt = np.median(np.diff(tempo))
        sigma_obs = np.median(erro)
        
        # Configurar e executar filtro
        filtro = FiltroKalman(dim_estado=3, dim_observacao=1)
        filtro.configurar_modelo_curva_luz(
            dt=dt,
            sigma_processo=sigma_obs * 0.1,
            sigma_medicao=sigma_obs
        )
        
        # Filtrar
        resultado_filtro = filtro.filtrar(fluxo)
        
        # Suavizar (RTS)
        resultado_suavizado = filtro.suavizar_rts()
        
        # Detectar anomalias
        anomalias, mahalanobis = filtro.detectar_anomalias(3.0)
        n_anomalias = np.sum(anomalias)
        
        print(f"    dt médio: {dt:.6f} dias")
        print(f"    Log-verossimilhança: {resultado_filtro['log_verossimilhanca']:.2f}")
        print(f"    Anomalias detectadas: {n_anomalias}")
        print(f"    Ganho de Kalman médio: {np.mean(filtro.ganhos_kalman[:, 0, 0]):.4f}")
        
        # Plotar
        arquivo_grafico = os.path.join(
            diretorio_saida,
            f"kalman_{nome.replace(' ', '_').lower()}.png"
        )
        filtro.plotar_resultados(tempo, fluxo, nome, arquivo_grafico)
        print(f"    Gráfico salvo em: {arquivo_grafico}")
        
        resultados[nome] = {
            **dados,
            'kalman': {
                'estados_filtrados': resultado_filtro['estados'],
                'estados_suavizados': resultado_suavizado['estados'],
                'inovacoes': resultado_filtro['inovacoes'],
                'log_verossimilhanca': resultado_filtro['log_verossimilhanca'],
                'anomalias': anomalias,
                'mahalanobis': mahalanobis
            }
        }
    
    plt.close('all')
    print("\n" + "=" * 60)
    print("MÓDULO 12 CONCLUÍDO")
    print("=" * 60)
    
    return resultados


__all__ = ['FiltroKalman', 'executar_modulo_12']
