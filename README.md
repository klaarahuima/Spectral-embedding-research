# Spectral embedding of random circulant-type graphs

Python code for a research project on how well the **Laplacian spectral embedding** (2nd and 3rd eigenvectors) of a graph recovers the circular structure it was built from, when that structure is perturbed at random.

Bachelor thesis project, École Polytechnique, carried out at Dalhousie University (spring 2025) under the supervision of Prof. Jeannette Janssen.
[Report (PDF)](#) · [Interactive demo](#) · [Author: Klaara Huima](https://github.com/klaarahuima)

## The goal

Take vertices arranged around a circle and connect those that are close. The 2nd and 3rd Laplacian eigenvectors, plotted against each other, should place the vertices on a closed curve in the same circular order. What happens to this embedding when

* edges are removed at random,
* vertices are sampled at random instead of regularly,
* the vertex density is not uniform (here: twice as many vertices in the first quadrant)?

Embedding quality is measured by counting **inversions**: pairs of vertices whose circular order in the embedding differs from their order on the circle.

## Main observation

For the non-uniform model graph, each of the 2nd and 3rd eigenvectors splits into **two segments that are each an almost exact sinusoid, with different parameters**: one on the first quadrant (Q1), one on the other three (Q2–Q4).

With ψ(k) = (A/√n) · cos or sin(2π f k / n + φ) and k the index inside the segment, the ideal circulant model graph with n = 1000 gives (reproduce with `python reproduce\_figure.py --model circulant --n 1000`):

|Segment|ψ₂ (cos): A, f|ψ₃ (sin): A, f|
|-|-|-|
|Q1|1.05, 0.50|0.96, 0.64|
|Q2–Q4|1.61, 1.12|1.65, 1.43|

**Why two frequencies.** In the ideal model graph, a vertex in Q2–Q4 has degree 2 (offsets ±1) and a vertex in Q1 has degree 4 (offsets ±1, ±2). Away from the two interfaces, the eigenvalue equation Lψ = λψ then gives plain sinusoids whose frequency ω satisfies

* Q2–Q4: 2 − 2 cos ω = λ
* Q1: 4 − 2 cos ω − 2 cos 2ω = λ

so for small λ, ω ≈ √λ in Q2–Q4 and ω ≈ √(λ/5) in Q1: in vertex-index units the two frequencies differ by a factor √5. This matches the fits (n = 1000, f = ωn / 2π):

||Q1, predicted / fitted|Q2–Q4, predicted / fitted|
|-|-|-|
|ψ₂|0.502 / 0.501|1.123 / 1.123|
|ψ₃|0.641 / 0.641|1.433 / 1.433|

(checked for n = 500, 1000 and 2000). Amplitudes and phases come from matching the two pieces at the interfaces and from periodicity; they are measured here, not derived. Frequencies are expressed per vertex index: expressed per unit angle, the two segments differ by only √5 / 2 ≈ 1.12, because vertices are twice as dense in Q1.

## Graph models (`graphfunctions.py`)

|Function|Model|
|-|-|
|`random\_graph\_unit\_circle`|n vertices at random angles; two vertices closer than `threshold` (degrees) are joined with probability `p`, others with probability `q`. Angle distributions: `uniform\_angles` (default), `non\_uniform\_angles` (twice as many vertices in Q1), `any\_angle` (chosen density ratio)|
|`nearest\_neighbours`|each vertex joined to its k nearest neighbours on the circle, non-uniform vertices|
|`removing\_random\_edges`|circulant graph from which each edge is removed independently (random edge removal as in Richter and Rocha; \[add full citation])|
|`ideal\_circulant\_denseQ1`|deterministic model graph: Q1 holds ⌊n/5⌋·2 vertices with degree 4, the rest degree 2|

## Repository contents

|File|Role|
|-|-|
|`graphfunctions.py`|graph models, eigenvector computation (`eigsh`), polar coordinates, inversion counting, plotting helpers|
|`fittingeigenvectors.py`|fits sinusoids to the 2nd and 3rd eigenvectors on Q1 and Q2–Q4 for a list of orders, and plots how A, f, φ and the offset depend on n|
|`simulatinginversions.py`|Monte-Carlo estimate of the proportion of inversions against the order n|

**Parameter study** (`fittingeigenvectors.py`):

```python
import fittingeigenvectors as fe
orders = \[500, 700, 1000]
params = fe.get\_parameters(orders)           # fits ψ₂, ψ₃ on Q1 and Q2–Q4 for each order
fe.visualise\_parameters(params, orders)      # 4 × 4 grid of parameters against n
```

`get\_parameters` opens one plot per fit; close each window to continue. To study the ideal model graph, use the commented `fun.ideal\_circulant\_denseQ1(n)` line in the script. If you use `random\_graph\_unit\_circle`, switch it to non-uniform vertices first (the Q1 / Q2–Q4 split assumes it).

**Inversions against order** (`simulatinginversions.py`): `python simulatinginversions.py` runs 20 graphs per order for n = 500…2000 (edit `orders`, `p`, `q` and `num\_simulations` at the top), keeps connected graphs only, writes one row per graph to `simulation\_results.csv`, and plots the average inversion proportion against n. It is slow for large n; results are written as they are produced.

## Notes and limitations

* **Fit initialisation.** `fittingeigenvectors.py` starts curve fitting at a fixed frequency suited to n ≳ 500. For smaller orders the first-quadrant fit can land in a wrong local minimum.
* **Signs.** Eigenvectors are defined up to sign, so fitted amplitudes and frequencies can come out negative; compare absolute values.
* **Short segments.** On Q1 each eigenvector spans only about a fifth to a quarter of a period, so on noisy graphs the fitted amplitude, frequency and phase are poorly constrained. On the ideal model graph the data are exact sinusoids and the fits are well determined.
* **Random graphs.** The Q1 / Q2–Q4 boundary is the number of vertices actually sampled in Q1 (random, about 2n/5).
* **Disconnected graphs.** The embedding is meaningless when the graph is disconnected.

## Acknowledgements

Supervised by Prof. Jeannette Janssen (Dalhousie University).

