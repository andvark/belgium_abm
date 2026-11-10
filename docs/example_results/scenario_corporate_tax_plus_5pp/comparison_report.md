# Scenario comparison: `corporate_tax_plus_5pp` vs baseline

> **ILLUSTRATIVE STYLISED POLICY - NOT BELGIAN LAW; fiscal results are mechanical demonstrations, not estimates of Belgian policy effects | SCENARIO OVERRIDES: corporate_income_tax.rate=0.3**

> Results are mechanical demonstrations on a stylised model, not estimates of real policy effects.

- Baseline year 2025, horizon 30 years, 10 paired replicate(s) with common random numbers (seeds [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]).
- **static_fixed_behaviour**: the scenario policy applied to the unchanged baseline trajectory (direct mechanical effect).
- **dynamic_total**: the ABM re-run under the scenario. **behavioural_residual_NOT_CAUSAL** = dynamic - static; it is only a descriptive gap, not a causal decomposition.

## Differences (scenario - baseline) at year 30

| metric | effect | mean diff | p05 | p95 | baseline |
|---|---|---|---|---|---|
| Gini, market income (pensions as transfers) | dynamic_total | 0.0001233 | 6.379e-05 | 0.0001735 | 0.4996 |
| Gini, market income (pensions as transfers) | static_fixed_behaviour | -1.287e-05 | -2.165e-05 | -3.065e-06 | 0.4996 |
| Gini, market income (pensions as transfers) | behavioural_residual_NOT_CAUSAL | 0.0001362 | 7.351e-05 | 0.0001899 | 0.4996 |
| Gini, market income (pensions as deferred income) | dynamic_total | 1.348e-05 | -5.373e-05 | 6.252e-05 | 0.3694 |
| Gini, market income (pensions as deferred income) | static_fixed_behaviour | -5.888e-05 | -7.017e-05 | -5.187e-05 | 0.3694 |
| Gini, market income (pensions as deferred income) | behavioural_residual_NOT_CAUSAL | 7.235e-05 | 5.889e-06 | 0.0001238 | 0.3694 |
| Gini, disposable income | dynamic_total | -4.539e-05 | -9.327e-05 | 1.784e-05 | 0.2887 |
| Gini, disposable income | static_fixed_behaviour | -9.805e-05 | -0.000113 | -9.002e-05 | 0.2887 |
| Gini, disposable income | behavioural_residual_NOT_CAUSAL | 5.267e-05 | 1.294e-07 | 0.0001138 | 0.2887 |
| Gini, after VAT | dynamic_total | -6.106e-05 | -0.0001097 | 6.397e-06 | 0.2937 |
| Gini, after VAT | static_fixed_behaviour | -9.872e-05 | -0.000114 | -9.063e-05 | 0.2937 |
| Gini, after VAT | behavioural_residual_NOT_CAUSAL | 3.766e-05 | -1.663e-05 | 0.000103 | 0.2937 |
| At-risk-of-poverty rate | dynamic_total | -0.0001025 | -0.0004597 | 0.0002013 | 0.1802 |
| At-risk-of-poverty rate | static_fixed_behaviour | -4.539e-05 | -0.0003778 | 0.0001504 | 0.1802 |
| At-risk-of-poverty rate | behavioural_residual_NOT_CAUSAL | -5.706e-05 | -0.000409 | 0.0002015 | 0.1802 |
| S80/S20 | dynamic_total | -0.001137 | -0.002934 | 0.0008144 | 4.835 |
| S80/S20 | static_fixed_behaviour | -0.002378 | -0.002777 | -0.00208 | 4.835 |
| S80/S20 | behavioural_residual_NOT_CAUSAL | 0.001241 | -0.0005733 | 0.003326 | 4.835 |
| P90/P10 | dynamic_total | 0.001302 | -0.002584 | 0.006494 | 4.336 |
| P90/P10 | static_fixed_behaviour | -2.152e-05 | -0.003189 | 0.002676 | 4.336 |
| P90/P10 | behavioural_residual_NOT_CAUSAL | 0.001324 | -0.0009584 | 0.004958 | 4.336 |
| Gini, net worth | dynamic_total | -0.000323 | -0.000363 | -0.000266 | 0.586 |
| Gini, net worth | static_fixed_behaviour | 0 | 0 | 0 | 0.586 |
| Gini, net worth | behavioural_residual_NOT_CAUSAL | -0.000323 | -0.000363 | -0.000266 | 0.586 |
| Top-10% wealth share | dynamic_total | -0.000292 | -0.0003953 | -0.0001547 | 0.3784 |
| Top-10% wealth share | static_fixed_behaviour | 0 | 0 | 0 | 0.3784 |
| Top-10% wealth share | behavioural_residual_NOT_CAUSAL | -0.000292 | -0.0003953 | -0.0001547 | 0.3784 |
| Top-1% wealth share | dynamic_total | 1.382e-05 | -2.788e-05 | 5.689e-05 | 0.06833 |
| Top-1% wealth share | static_fixed_behaviour | 0 | 0 | 0 | 0.06833 |
| Top-1% wealth share | behavioural_residual_NOT_CAUSAL | 1.382e-05 | -2.788e-05 | 5.689e-05 | 0.06833 |
| Median equivalised disposable income (EUR) | dynamic_total | -23.31 | -42.88 | -7.954 | 3.863e+04 |
| Median equivalised disposable income (EUR) | static_fixed_behaviour | -10.72 | -29.12 | -0.2417 | 3.863e+04 |
| Median equivalised disposable income (EUR) | behavioural_residual_NOT_CAUSAL | -12.59 | -18.7 | -4.151 | 3.863e+04 |
| Unemployment rate | dynamic_total | 1.367e-05 | -8.719e-06 | 7.763e-05 | 0.07336 |
| Unemployment rate | static_fixed_behaviour | 0 | 0 | 0 | 0.07336 |
| Unemployment rate | behavioural_residual_NOT_CAUSAL | 1.367e-05 | -8.719e-06 | 7.763e-05 | 0.07336 |
| Net fiscal balance (EUR) | dynamic_total | 3.808e+08 | 3.126e+08 | 4.81e+08 | 2.261e+11 |
| Net fiscal balance (EUR) | static_fixed_behaviour | 5.947e+08 | 5.518e+08 | 6.393e+08 | 2.261e+11 |
| Net fiscal balance (EUR) | behavioural_residual_NOT_CAUSAL | -2.14e+08 | -2.721e+08 | -1.414e+08 | 2.261e+11 |

Group-level winners and losers: `group_differences.csv`, `winners_losers_by_baseline_wealth_quintile.csv`; fiscal flows: `fiscal_differences.csv`.
