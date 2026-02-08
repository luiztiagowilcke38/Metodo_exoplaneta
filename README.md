# Sistema Estatístico para Detecção de Exoplanetas

**Autor:** Luiz Tiago Wilcke

Sistema Python com 40 módulos para detecção de exoplanetas utilizando técnicas de séries temporais, modelos lineares e estatística Bayesiana com dados reais das missões Kepler e TESS.

## Descrição

Este projeto implementa um pipeline completo para análise de curvas de luz estelares e detecção de exoplanetas através do método de trânsito. O sistema utiliza dados reais obtidos do arquivo NASA/MAST através da biblioteca `lightkurve`.

## Fundamentos Matemáticos

### Modelo de Curva de Luz

O fluxo observado de uma estrela com planeta em trânsito é modelado por:

```math
F(t) = F_0 \cdot \left[1 - \delta \cdot T(t)\right] + \varepsilon(t) + V(t) + S(t)
```

Onde:
- $F_0$ : Fluxo base normalizado
- $\delta$ : Profundidade do trânsito (razão de raios ao quadrado: $(R_p/R_\star)^2$)
- $T(t)$ : Função de trânsito com limb darkening
- $\varepsilon(t)$ : Ruído fotométrico gaussiano
- $V(t)$ : Variabilidade estelar intrínseca
- $S(t)$ : Sistemáticas instrumentais

### Limb Darkening Quadrático

A intensidade do disco estelar segue a lei de limb darkening:

```math
I(\mu) = 1 - c_1(1 - \mu) - c_2(1 - \mu)^2
```

Onde $\mu = \cos(\theta)$ é o cosseno do ângulo entre a linha de visada e a normal à superfície.

### Periodograma Lomb-Scargle

Para dados irregularmente amostrados, utilizamos o periodograma Lomb-Scargle:

```math
P(\omega) = \frac{1}{2} \left\{ \frac{\left[\sum_j (x_j - \bar{x}) \cos\omega(t_j - \tau)\right]^2}{\sum_j \cos^2 \omega(t_j - \tau)} + \frac{\left[\sum_j (x_j - \bar{x}) \sin\omega(t_j - \tau)\right]^2}{\sum_j \sin^2 \omega(t_j - \tau)} \right\}
```

### Box-Fitting Least Squares (BLS)

O algoritmo BLS ajusta um modelo de caixa retangular:

```math
\chi^2 = \sum_{i=1}^{N} \frac{(f_i - m_i)^2}{\sigma_i^2}
```

Onde o modelo $m_i$ é:

```math
m_i = \begin{cases} 1 - \delta & \text{se } t_i \text{ em trânsito} \\ 1 & \text{caso contrário} \end{cases}
```

### Filtro de Kalman

O filtro de Kalman estima estados ocultos através de:

**Predição:**

```math
\hat{x}_{k|k-1} = F_k \hat{x}_{k-1|k-1}
```

```math
P_{k|k-1} = F_k P_{k-1|k-1} F_k^T + Q_k
```

**Atualização:**

```math
K_k = P_{k|k-1} H_k^T (H_k P_{k|k-1} H_k^T + R_k)^{-1}
```

```math
\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k (z_k - H_k \hat{x}_{k|k-1})
```

### Inferência Bayesiana

A distribuição a posteriori dos parâmetros do modelo segue o Teorema de Bayes:

```math
P(\theta | D) = \frac{P(D | \theta) \cdot P(\theta)}{P(D)} = \frac{\mathcal{L}(\theta) \cdot \pi(\theta)}{\int \mathcal{L}(\theta) \pi(\theta) d\theta}
```

### Verossimilhança Gaussiana

A função de verossimilhança para dados fotométricos:

```math
\mathcal{L}(\theta) = \prod_{i=1}^{N} \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp\left[-\frac{(f_i - m_i(\theta))^2}{2\sigma_i^2}\right]
```

Log-verossimilhança:

```math
\ln \mathcal{L} = -\frac{1}{2} \sum_{i=1}^{N} \left[ \frac{(f_i - m_i)^2}{\sigma_i^2} + \ln(2\pi\sigma_i^2) \right]
```

### MCMC - Metropolis-Hastings

A probabilidade de aceitação no algoritmo Metropolis-Hastings:

```math
\alpha = \min\left(1, \frac{P(\theta')}{P(\theta)} \cdot \frac{q(\theta | \theta')}{q(\theta' | \theta)}\right)
```

### Fator de Bayes

Para comparação de modelos:

```math
B_{12} = \frac{P(D | M_1)}{P(D | M_2)} = \frac{\int P(D | \theta_1, M_1) P(\theta_1 | M_1) d\theta_1}{\int P(D | \theta_2, M_2) P(\theta_2 | M_2) d\theta_2}
```

### Critérios de Informação

**AIC (Akaike):**

```math
\text{AIC} = 2k - 2\ln(\hat{\mathcal{L}})
```

**BIC (Bayesiano):**

```math
\text{BIC} = k\ln(n) - 2\ln(\hat{\mathcal{L}})
```

Onde $k$ é o número de parâmetros e $n$ o número de observações.

## Estrutura do Projeto

```
ExoplanetasMetodo/
├── main.py                   # Script principal
├── requirements.txt          # Dependências
├── dados/                    # Dados baixados
├── resultados/               # Gráficos e relatórios
└── src/
    ├── dados/                # Módulos 01-05
    ├── series_temporais/     # Módulos 06-15
    ├── modelos_lineares/     # Módulos 16-25
    ├── bayesianos/           # Módulos 26-35
    └── integracao/           # Módulos 36-40
```

## Módulos

### Dados (01-05)
| # | Módulo | Descrição |
|---|--------|-----------|
| 01 | `carregamento_dados` | Download de curvas de luz via lightkurve |
| 02 | `preprocessamento` | Remoção de outliers (sigma-clipping) |
| 03 | `normalizacao` | Normalização e detrending |
| 04 | `validacao_dados` | Testes estatísticos de qualidade |
| 05 | `exploracao_dados` | Análise exploratória |

### Séries Temporais (06-15)
| # | Módulo | Descrição |
|---|--------|-----------|
| 06 | `decomposicao_temporal` | Decomposição STL |
| 07 | `analise_fourier` | FFT e espectro de potência |
| 08 | `periodograma` | Lomb-Scargle |
| 09 | `autocorrelacao` | ACF/PACF |
| 10 | `arima` | Modelos ARIMA |
| 11 | `sarima` | Modelos sazonais |
| 12 | `filtros_kalman` | Filtro de Kalman |
| 13 | `wavelets` | Análise tempo-frequência |
| 14 | `deteccao_transitos` | Algoritmo BLS |
| 15 | `box_fitting` | BLS otimizado |

### Modelos Lineares (16-25)
| # | Módulo | Descrição |
|---|--------|-----------|
| 16 | `regressao_linear` | OLS com diagnósticos |
| 17 | `regressao_ridge` | Regularização L2 |
| 18 | `regressao_lasso` | Regularização L1 |
| 19 | `elastic_net` | L1 + L2 |
| 20 | `pca` | Componentes Principais |
| 21 | `regressao_robusta` | Huber |
| 22 | `modelos_mistos` | Efeitos mistos |
| 23 | `regressao_quantilica` | Quantis |
| 24 | `splines` | B-splines |
| 25 | `gam` | Modelos Aditivos Generalizados |

### Estatística Bayesiana (26-35)
| # | Módulo | Descrição |
|---|--------|-----------|
| 26 | `inferencia_bayesiana` | Teorema de Bayes |
| 27 | `mcmc` | Monte Carlo via Cadeias de Markov |
| 28 | `metropolis_hastings` | Algoritmo M-H |
| 29 | `gibbs_sampling` | Amostragem de Gibbs |
| 30 | `modelos_hierarquicos` | Multinível |
| 31 | `selecao_modelos` | AIC/BIC/WAIC |
| 32 | `priori_informativas` | Elicitação de prioris |
| 33 | `posteriori_analise` | Análise posteriori |
| 34 | `credibilidade` | Intervalos HPD |
| 35 | `fator_bayes` | Comparação de modelos |

### Integração (36-40)
| # | Módulo | Descrição |
|---|--------|-----------|
| 36 | `pipeline_completo` | Pipeline integrado |
| 37 | `validacao_cruzada` | K-fold temporal |
| 38 | `metricas_avaliacao` | ROC, precisão, recall |
| 39 | `visualizacao` | Visualizações científicas |
| 40 | `relatorio_final` | Geração de relatórios |

## Instalação

```bash
pip install -r requirements.txt
```

## Uso

```bash
python main.py
```

Os resultados serão salvos em `resultados/`.

## Licença

MIT License
