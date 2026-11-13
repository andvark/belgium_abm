# Scenario comparison: `corporate_tax_plus_5pp` vs baseline

> **ILLUSTRATIVE STYLISED POLICY - NOT BELGIAN LAW; fiscal results are mechanical demonstrations, not estimates of Belgian policy effects | SCENARIO OVERRIDES: corporate_income_tax.rate=0.3**

> Results are mechanical demonstrations on a stylised model, not estimates of real policy effects.

- Baseline year 2025, horizon 30 years, 10 paired replicate(s) with common random numbers (seeds [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]).
- **static_fixed_behaviour**: the scenario policy applied to the unchanged baseline trajectory (direct mechanical effect).
- **dynamic_total**: the ABM re-run under the scenario. **behavioural_residual_NOT_CAUSAL** = dynamic - static; it is only a descriptive gap, not a causal decomposition.

## Differences (scenario - baseline) at year 30

| metric | effect | mean diff | p05 | p95 | baseline |
|---|---|---|---|---|---|
| Gini, market income (pensions as transfers) | dynamic_total | 0.0001085 | 4.406e-05 | 0.0001806 | 0.4992 |
| Gini, market income (pensions as transfers) | static_fixed_behaviour | -3.234e-05 | -4.61e-05 | -2.28e-05 | 0.4992 |
| Gini, market income (pensions as transfers) | behavioural_residual_NOT_CAUSAL | 0.0001409 | 7.725e-05 | 0.0002217 | 0.4992 |
| Gini, market income (pensions as deferred income) | dynamic_total | 1.72e-06 | -5.087e-05 | 5.464e-05 | 0.3617 |
| Gini, market income (pensions as deferred income) | static_fixed_behaviour | -8.018e-05 | -9.625e-05 | -6.936e-05 | 0.3617 |
| Gini, market income (pensions as deferred income) | behavioural_residual_NOT_CAUSAL | 8.19e-05 | 3.476e-05 | 0.0001453 | 0.3617 |
| Gini, disposable income | dynamic_total | -4.878e-05 | -9.099e-05 | -6.99e-06 | 0.2853 |
| Gini, disposable income | static_fixed_behaviour | -0.0001146 | -0.0001337 | -0.0001031 | 0.2853 |
| Gini, disposable income | behavioural_residual_NOT_CAUSAL | 6.585e-05 | 2.424e-05 | 0.0001215 | 0.2853 |
| Gini, after VAT | dynamic_total | -6.314e-05 | -0.0001048 | -1.814e-05 | 0.2906 |
| Gini, after VAT | static_fixed_behaviour | -0.0001153 | -0.0001346 | -0.0001035 | 0.2906 |
| Gini, after VAT | behavioural_residual_NOT_CAUSAL | 5.218e-05 | 1.115e-05 | 0.000111 | 0.2906 |
| At-risk-of-poverty rate | dynamic_total | -1.719e-05 | -0.0003404 | 0.00034 | 0.1774 |
| At-risk-of-poverty rate | static_fixed_behaviour | -5.666e-05 | -0.0004268 | 0.0002013 | 0.1774 |
| At-risk-of-poverty rate | behavioural_residual_NOT_CAUSAL | 3.947e-05 | -0.0001138 | 0.0002322 | 0.1774 |
| S80/S20 | dynamic_total | -0.001143 | -0.002601 | 0.0004753 | 4.856 |
| S80/S20 | static_fixed_behaviour | -0.002856 | -0.003343 | -0.002494 | 4.856 |
| S80/S20 | behavioural_residual_NOT_CAUSAL | 0.001713 | 0.0003118 | 0.003673 | 4.856 |
| P90/P10 | dynamic_total | -0.001986 | -0.00581 | 0.002824 | 4.297 |
| P90/P10 | static_fixed_behaviour | -0.002803 | -0.005341 | 0.001338 | 4.297 |
| P90/P10 | behavioural_residual_NOT_CAUSAL | 0.0008174 | -0.001816 | 0.0057 | 4.297 |
| Gini, net worth | dynamic_total | -2.952e-06 | -4.851e-05 | 6.231e-05 | 0.5928 |
| Gini, net worth | static_fixed_behaviour | 0 | 0 | 0 | 0.5928 |
| Gini, net worth | behavioural_residual_NOT_CAUSAL | -2.952e-06 | -4.851e-05 | 6.231e-05 | 0.5928 |
| Top-10% wealth share | dynamic_total | 7.278e-05 | -1.277e-05 | 0.0001472 | 0.3729 |
| Top-10% wealth share | static_fixed_behaviour | 0 | 0 | 0 | 0.3729 |
| Top-10% wealth share | behavioural_residual_NOT_CAUSAL | 7.278e-05 | -1.277e-05 | 0.0001472 | 0.3729 |
| Top-1% wealth share | dynamic_total | 8.689e-05 | 6.214e-05 | 0.0001035 | 0.06842 |
| Top-1% wealth share | static_fixed_behaviour | 0 | 0 | 0 | 0.06842 |
| Top-1% wealth share | behavioural_residual_NOT_CAUSAL | 8.689e-05 | 6.214e-05 | 0.0001035 | 0.06842 |
| Median equivalised disposable income (EUR) | dynamic_total | -18.8 | -37.01 | 0.5344 | 3.523e+04 |
| Median equivalised disposable income (EUR) | static_fixed_behaviour | -7.371 | -24.95 | 0 | 3.523e+04 |
| Median equivalised disposable income (EUR) | behavioural_residual_NOT_CAUSAL | -11.43 | -23.35 | 2.391 | 3.523e+04 |
| Unemployment rate | dynamic_total | -3.831e-05 | -0.0001374 | 1.105e-05 | 0.08212 |
| Unemployment rate | static_fixed_behaviour | 0 | 0 | 0 | 0.08212 |
| Unemployment rate | behavioural_residual_NOT_CAUSAL | -3.831e-05 | -0.0001374 | 1.105e-05 | 0.08212 |
| Net fiscal balance (EUR) | dynamic_total | 3.889e+08 | 3.371e+08 | 4.673e+08 | 1.832e+11 |
| Net fiscal balance (EUR) | static_fixed_behaviour | 5.56e+08 | 5.191e+08 | 5.958e+08 | 1.832e+11 |
| Net fiscal balance (EUR) | behavioural_residual_NOT_CAUSAL | -1.67e+08 | -2.223e+08 | -1.122e+08 | 1.832e+11 |

Group-level winners and losers: `group_differences.csv`, `winners_losers_by_baseline_wealth_quintile.csv`; fiscal flows: `fiscal_differences.csv`.
