# Phase 3 prototype and correctness gate

**Verdict** **PASS WP 3.1–3.3.** Proceed to the preregistered Phase 4 simulations with the Phase 2 interior-root scope and failure accounting. Use Colab for the full sweep. This phase establishes prototype correctness and a timing budget.

## Implementation

The [Python package](../src/bw_causal/) implements rank-support BW transport, Procrustes distance, exact score derivatives, and the signed-score Newton solver from [Gate G2](gate2_memo.md). Newton acceptance uses the actual signed Jacobian, a residual check, and PSD-domain safeguards. At dimensions above 15 the solver uses matrix-free Hessian actions and iterative solves. Those paths passed a derivative comparison through dimension 35. Their full runtime and convergence remain Phase 4 measurements.

The matched methods share a single anchor, design, and estimation partition, cross-fitting folds, propensity predictions, and honest-forest configuration. Only the BW methods use the learned fixed anchor. The same square-root coordinate map and guarded arm backtransform serve the Layer A identity-anchor ablation and the square-root baseline. Separate comparisons include log, log-Cholesky, and elementwise DR, a matched BW T-learner without causal augmentation, a global Fréchet T-learner, and an arm-stratified BW adaptation of the local-constant RFWLFR. The arm-stratified method trains the configured number of trees in each arm and evaluates direct BW impurity at the two-means proposal. The global T-learner uses the signed global regression objective with the common guarded solver. The one-step, positive IPW, unaugmented, and entropy variants are selectable through the BW forest's prediction interface. Cross-fit regressions use the conditional mean of scores rather than transports of fitted barycenters.

Joint inference uses fitted root influences and shared Gaussian multipliers across the declared matrix contrast, three individual eigenvalues, and leading projector in each subgroup. Statuses and abstentions prevent those queries from counting toward coverage of the original full family. Weak spectral gaps are checked against a prespecified estimated uncertainty diagnostic. This first-order fitted subgroup check supports no general continuous-covariate coverage claim. See the [enabling specification](phase2_enabling_spec.md).

## Verification

| Check | Result |
|---|---|
| Deterministic tests | 13 passed. Coverage includes rank 0, 1, 4 and 5 derivatives, rotation at condition ratio 10,000, repeated internal eigenvalues, exact identity anchor reduction, outcome holdout honesty, boundary-target failure, calibration failure and a known interior root with rank-one observed outcomes. |
| Main smoke | 30 cases on seeds 0–4, two sample sizes (200 and 600), three regimes, dimension 5 and 24 trees. All six required methods returned an estimate for every arm/query. Zero materially negative eigenvalues were recorded among accepted estimates. |
| Noiseless | Maximum BW PEHE was 0.00449 for the primary method and 0.00512 across the six required methods. Both are below the planned 0.05 threshold. |
| Null | The 95% joint region covered the entire declared subgroup contrast family in 4/5 seeds at each sample size, with zero abstentions. This small check is too imprecise to establish calibration. |
| Runtime | The 30 cases and three small pilots took 33.59 seconds total. The slowest registered 600-observation case, including additional comparators, took 1.34 seconds. Peak memory was 110,192 KiB, with no swap. |

[Deterministic test output](../results/phase3_smoke/pytest.txt), [complete smoke results](../results/phase3_smoke/smoke_summary.json), [resource measurements](../results/phase3_smoke/runtime.txt).

The separate rank-four observation pilot had positive-definite ground-truth arm barycenters. Seven of eight primary arm/query fits converged. The eighth had 14 local treated observations, a kernel effective sample size of 19.4, a residual of 0.606, and a nonpositive empirical signed Hessian at the IPW start. Arithmetic, isotropic, and held-out-anchor starts under the same solver also reached nonpositive signed Hessians before convergence. This diagnosis leaves the original pilot and failure intact. Phase 4 must count it as a failure if repeated, and must keep the exact-zero outcome rank separate from the target covariance rank. The entropy ablation separately failed the declared balance tolerance at 17 of 40 arm/query fits across the main anisotropic smoke cases. These are logged failures. See the [alternate-start results](../results/phase3_smoke/singular_pilot_starts.json).

The anisotropic smoke and pilots share two binary strata and mean-identity noncommuting transport noise. They provide exact BW roots and test that the code executes, but the resulting handful of errors does not decide comparative performance.

## Full-sweep budget

One thread measured **1.23 seconds at dimension 5** and **4.44 seconds at dimension 15** for six methods, 800 observations, 200 trees, and six queries. With 2,400 tasks of each dimension, that is about **3.8 CPU-hours** or **38 minutes at an ideal six-way parallel efficiency**, before additional queries, stress experiments, uncertainty construction, extra comparators or notebook startup. At 24 trees, the arm-stratified direct-BW RFWLFR fit alone took 1.60 seconds at dimension 5 and 5.66 seconds at dimension 15. The complete 24-tree, six-query comparison, including additional methods and ablations, took 2.16 and 14.01 seconds respectively. These measurements support sharding even though memory was modest at the two crossover dimensions. [Full-tree budget pilot](../results/phase3_smoke/budget_pilot_core_200_trees.json), [resource profile](../results/phase3_smoke/budget_pilot_runtime.txt).

Run the package from this project directory with a local environment and one BLAS thread.

```bash
python -m pip install -e '.[test]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
python scripts/run_phase3_smoke.py
```
