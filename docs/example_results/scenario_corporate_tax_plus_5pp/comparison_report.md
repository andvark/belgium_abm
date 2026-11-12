# Scenario comparison: `corporate_tax_plus_5pp` vs baseline

> **ILLUSTRATIVE STYLISED POLICY - NOT BELGIAN LAW; fiscal results are mechanical demonstrations, not estimates of Belgian policy effects | SCENARIO OVERRIDES: corporate_income_tax.rate=0.3**

> Results are mechanical demonstrations on a stylised model, not estimates of real policy effects.

- Baseline year 2025, horizon 30 years, 10 paired replicate(s) with common random numbers (seeds [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]).
- **static_fixed_behaviour**: the scenario policy applied to the unchanged baseline trajectory (direct mechanical effect).
- **dynamic_total**: the ABM re-run under the scenario. **behavioural_residual_NOT_CAUSAL** = dynamic - static; it is only a descriptive gap, not a causal decomposition.

## Differences (scenario - baseline) at year 30

| metric | effect | mean diff | p05 | p95 | baseline |
|---|---|---|---|---|---|
| Gini, market income (pensions as transfers) | dynamic_total | 9.492e-05 | 1.714e-07 | 0.0001919 | 0.4982 |
| Gini, market income (pensions as transfers) | static_fixed_behaviour | -1.502e-05 | -2.836e-05 | -2.219e-06 | 0.4982 |
| Gini, market income (pensions as transfers) | behavioural_residual_NOT_CAUSAL | 0.0001099 | 2.279e-05 | 0.0002057 | 0.4982 |
| Gini, market income (pensions as deferred income) | dynamic_total | -1.431e-05 | -0.0001157 | 8.429e-05 | 0.3677 |
| Gini, market income (pensions as deferred income) | static_fixed_behaviour | -6.042e-05 | -7.336e-05 | -4.866e-05 | 0.3677 |
| Gini, market income (pensions as deferred income) | behavioural_residual_NOT_CAUSAL | 4.611e-05 | -4.676e-05 | 0.000143 | 0.3677 |
| Gini, disposable income | dynamic_total | -7.178e-05 | -0.0001383 | -5.731e-06 | 0.2877 |
| Gini, disposable income | static_fixed_behaviour | -9.772e-05 | -0.0001125 | -8.431e-05 | 0.2877 |
| Gini, disposable income | behavioural_residual_NOT_CAUSAL | 2.593e-05 | -2.923e-05 | 9.005e-05 | 0.2877 |
| Gini, after VAT | dynamic_total | -8.765e-05 | -0.0001537 | -2.146e-05 | 0.2927 |
| Gini, after VAT | static_fixed_behaviour | -9.836e-05 | -0.0001132 | -8.482e-05 | 0.2927 |
| Gini, after VAT | behavioural_residual_NOT_CAUSAL | 1.071e-05 | -4.397e-05 | 7.485e-05 | 0.2927 |
| At-risk-of-poverty rate | dynamic_total | -7.355e-05 | -0.0004851 | 0.0003216 | 0.1799 |
| At-risk-of-poverty rate | static_fixed_behaviour | -9.077e-05 | -0.0004141 | 0.0001568 | 0.1799 |
| At-risk-of-poverty rate | behavioural_residual_NOT_CAUSAL | 1.722e-05 | -0.0002004 | 0.0002012 | 0.1799 |
| S80/S20 | dynamic_total | -0.002034 | -0.005235 | 0.0001401 | 4.78 |
| S80/S20 | static_fixed_behaviour | -0.002353 | -0.002808 | -0.002049 | 4.78 |
| S80/S20 | behavioural_residual_NOT_CAUSAL | 0.0003195 | -0.002431 | 0.002396 | 4.78 |
| P90/P10 | dynamic_total | -0.004682 | -0.01588 | 0.001449 | 4.316 |
| P90/P10 | static_fixed_behaviour | -0.003814 | -0.01129 | 0.001414 | 4.316 |
| P90/P10 | behavioural_residual_NOT_CAUSAL | -0.0008679 | -0.006442 | 0.002255 | 4.316 |
| Gini, net worth | dynamic_total | -0.0001522 | -0.0002303 | -7.983e-05 | 0.5857 |
| Gini, net worth | static_fixed_behaviour | 0 | 0 | 0 | 0.5857 |
| Gini, net worth | behavioural_residual_NOT_CAUSAL | -0.0001522 | -0.0002303 | -7.983e-05 | 0.5857 |
| Top-10% wealth share | dynamic_total | -0.0001017 | -0.0001992 | 1.911e-05 | 0.3714 |
| Top-10% wealth share | static_fixed_behaviour | 0 | 0 | 0 | 0.3714 |
| Top-10% wealth share | behavioural_residual_NOT_CAUSAL | -0.0001017 | -0.0001992 | 1.911e-05 | 0.3714 |
| Top-1% wealth share | dynamic_total | 3.665e-05 | -3.741e-05 | 9.709e-05 | 0.06887 |
| Top-1% wealth share | static_fixed_behaviour | 0 | 0 | 0 | 0.06887 |
| Top-1% wealth share | behavioural_residual_NOT_CAUSAL | 3.665e-05 | -3.741e-05 | 9.709e-05 | 0.06887 |
| Median equivalised disposable income (EUR) | dynamic_total | -25.54 | -58.78 | -4.35 | 3.848e+04 |
| Median equivalised disposable income (EUR) | static_fixed_behaviour | -14.13 | -49.1 | 0 | 3.848e+04 |
| Median equivalised disposable income (EUR) | behavioural_residual_NOT_CAUSAL | -11.41 | -25.37 | -0.01489 | 3.848e+04 |
| Unemployment rate | dynamic_total | -1.081e-05 | -7.354e-05 | 1.845e-05 | 0.07384 |
| Unemployment rate | static_fixed_behaviour | 0 | 0 | 0 | 0.07384 |
| Unemployment rate | behavioural_residual_NOT_CAUSAL | -1.081e-05 | -7.354e-05 | 1.845e-05 | 0.07384 |
| Net fiscal balance (EUR) | dynamic_total | 3.76e+08 | 3.018e+08 | 4.656e+08 | 2.197e+11 |
| Net fiscal balance (EUR) | static_fixed_behaviour | 5.795e+08 | 5.469e+08 | 6.128e+08 | 2.197e+11 |
| Net fiscal balance (EUR) | behavioural_residual_NOT_CAUSAL | -2.035e+08 | -3.016e+08 | -1.182e+08 | 2.197e+11 |

Group-level winners and losers: `group_differences.csv`, `winners_losers_by_baseline_wealth_quintile.csv`; fiscal flows: `fiscal_differences.csv`.
