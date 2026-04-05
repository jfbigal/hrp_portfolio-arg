# HRP Portfolio — Mercado Argentino

Implementación de Hierarchical Risk Parity (HRP) aplicada al panel del índice Merval, basada en:

> López de Prado, M. (2016). *Building Diversified Portfolios that Outperform Out-of-Sample*. Journal of Portfolio Management.

El proyecto compara tres enfoques de construcción de portfolios:

| Método  | Descripción                                                                             |
| ------- | --------------------------------------------------------------------------------------- |
| **HRP** | Hierarchical Risk Parity — clustering jerárquico + recursive bisection                  |
| **IVP** | Inverse-Variance Portfolio — considera solo varianzas, ignorando correlaciones cruzadas |
| **EW**  | Equal Weight                                                                            |

---

## Metodología

El algoritmo HRP opera en tres etapas:

### **Stage 1 – Tree Clustering**

Convierte la matriz de correlaciones en una métrica de distancia:

`d(i,j) = sqrt((1 - rho(i,j)) / 2)`

y aplica clustering jerárquico (single linkage) para construir un dendrograma.

---

### **Stage 2 – Cuasi diagonalización**

Reorganiza filas y columnas de la matriz de covarianza para que activos similares queden contiguos, sin necesidad de cambio de base.

---

### **Stage 3 – Recursive Bisection**

Asigna pesos top-down bisectando recursivamente el árbol, distribuyendo capital en proporción inversa a la varianza de cada subcluster (usando IVP dentro de cada cluster).

---

## Resultados

### Backtest out-of-sample

* **Lookback:** 252 días hábiles (1 año)
* **Rebalanceo:** cada 21 días hábiles (mensual)
* **Metodología:** rolling window sin lookahead bias

#### Métricas

| Método | Retorno (%) | Vol (%) | Sharpe | Max DD (%) | Valor Final |
| ------ | ----------- | ------- | ------ | ---------- | ----------- |
| HRP    | 122.5       | 37.0    | 2.35   | -37.4      | 54.5        |
| IVP    | 125.7       | 38.2    | 2.32   | -37.6      | 58.6        |
| EW     | 126.1       | 40.8    | 2.20   | -39.3      | 59.1        |


* HRP presenta **menor volatilidad** que IVP y Equal Weight
* Logra un **Sharpe Ratio superior**, indicando mejor eficiencia riesgo-retorno
* Reduce el drawdown respecto a Equal Weight
* Equal Weight obtiene mayor retorno, pero a costa de mayor riesgo

Esto es consistente con la literatura: HRP mejora la estabilidad out-of-sample al incorporar la estructura de correlaciones sin requerir la inversión de la matriz de covarianza.

---

## Graficos

### Stage 1 — Dendrograma


![Dendrograma](results/hrp_dendrogram.png)

---

### Stage 2 — Correlación

[`Ver heatmap`](results/hrp_heatmaps.html)

![Heatmaps](results/hrp_heatmaps_screenshot.png)

---

### Stage 3 — Pesos del portfolio

Comparación entre HRP, IVP y Equal Weight.

[`Ver gráfico`](results/hrp_weights.html)
![Pesos](results/hrp_weights_screenshot.png)

---

### Equity Curve + Volatilidad

[`Ver backtest`](results/hrp_backtest.html)
![Backtest](results/hrp_backtest_screenshot.png)

---

## Estructura del repositorio

```
hrp_portfolio-arg/
├── hrp.py
├── requirements.txt
├── results/
│   ├── hrp_dendrogram.png
│   ├── hrp_weights.html
│   ├── hrp_heatmaps.html
│   └── hrp_backtest.html
└── README.md
```

---

## Instalación y uso

```bash
git clone https://github.com/jfbigal/hrp_portfolio-arg.git
cd hrp_portfolio-arg
pip install -r requirements.txt
python hrp.py
```

Los gráficos se guardan automáticamente en la carpeta `results/`.

---

## Universo de activos

20 acciones del panel Merval vía `yfinance`:

`ALUA` · `BBAR` · `BYMA` · `CEPU` · `COME` · `CRES` · `EDN` · `GGAL` · `IRSA` · `LOMA` · `MIRG` · `PAMP` · `SUPV` · `TECO2` · `TGNO4` · `TGSU2` · `TRAN` · `TXAR` · `VALO` · `YPFD`

---


## Referencia

López de Prado, M. (2016). *Building Diversified Portfolios that Outperform Out-of-Sample*.
Journal of Portfolio Management, 42(4).
https://ssrn.com/abstract=2708678

---
