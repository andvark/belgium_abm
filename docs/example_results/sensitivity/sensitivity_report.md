# Sensitivity analysis (one-at-a-time)

> **ILLUSTRATIVE STYLISED POLICY - NOT BELGIAN LAW; fiscal results are mechanical demonstrations, not estimates of Belgian policy effects**

Year 15, 8,000 simulated persons, 2 paired replicate(s), scenario `corporate_tax_plus_5pp`. Each parameter is moved to a low and a high value with all others at baseline. Ranges are judgments (public proxies +-30%; plausible intervals for pure assumptions). First-order swings only; no interactions. Few replicates: treat small swings as noise.

## wealth_top10_share (baseline 0.3916)

| rank | parameter | low | high | value at low | value at high | swing |
|---|---|---|---|---|---|---|
| 1 | marginal_propensity_to_consume | 0.7 | 0.95 | 0.3716 | 0.4082 | +0.03668 |
| 2 | separation_rate | 0.006 | 0.024 | 0.3862 | 0.4023 | +0.01608 |
| 3 | real_wage_growth | 0 | 0.02 | 0.3936 | 0.3895 | -0.00413 |
| 4 | business_shock_sigma | 0.15 | 0.35 | 0.39 | 0.3941 | +0.004091 |
| 5 | bv_director_pay_share | 0.4 | 0.8 | 0.3902 | 0.3938 | +0.003684 |
| 6 | bv_payout_ratio | 0.1 | 0.6 | 0.3909 | 0.3932 | +0.002262 |

## gini_disposable (baseline 0.2756)

| rank | parameter | low | high | value at low | value at high | swing |
|---|---|---|---|---|---|---|
| 1 | real_wage_growth | 0 | 0.02 | 0.2648 | 0.2878 | +0.02301 |
| 2 | marginal_propensity_to_consume | 0.7 | 0.95 | 0.2777 | 0.2727 | -0.005046 |
| 3 | separation_rate | 0.006 | 0.024 | 0.2734 | 0.2783 | +0.004942 |
| 4 | business_shock_sigma | 0.15 | 0.35 | 0.2737 | 0.2781 | +0.004397 |
| 5 | wage_shock_sigma | 0.05 | 0.12 | 0.2759 | 0.2801 | +0.004187 |
| 6 | bv_productivity_multiplier | 0.8 | 1.2 | 0.2745 | 0.2775 | +0.00298 |

## poverty_rate_60pct_median (baseline 0.188)

| rank | parameter | low | high | value at low | value at high | swing |
|---|---|---|---|---|---|---|
| 1 | real_wage_growth | 0 | 0.02 | 0.1862 | 0.1937 | +0.007445 |
| 2 | parttime_share_f | 0.2723 | 0.5057 | 0.1852 | 0.1895 | +0.004219 |
| 3 | bv_productivity_multiplier | 0.8 | 1.2 | 0.1862 | 0.1892 | +0.002975 |
| 4 | bv_payout_ratio | 0.1 | 0.6 | 0.187 | 0.1893 | +0.002325 |
| 5 | education_wealth_sensitivity | 0 | 3 | 0.1875 | 0.1896 | +0.002196 |
| 6 | bv_director_pay_share | 0.4 | 0.8 | 0.187 | 0.1891 | +0.002063 |

## median_eq_disposable_income (baseline 3.287e+04)

| rank | parameter | low | high | value at low | value at high | swing |
|---|---|---|---|---|---|---|
| 1 | real_wage_growth | 0 | 0.02 | 3.071e+04 | 3.529e+04 | +4582 |
| 2 | fin_return | 0 | 0.03 | 3.227e+04 | 3.349e+04 | +1218 |
| 3 | separation_rate | 0.006 | 0.024 | 3.33e+04 | 3.242e+04 | -881.9 |
| 4 | marginal_propensity_to_consume | 0.7 | 0.95 | 3.342e+04 | 3.259e+04 | -834.8 |
| 5 | business_shock_sigma | 0.15 | 0.35 | 3.262e+04 | 3.285e+04 | +230.6 |
| 6 | wage_shock_sigma | 0.05 | 0.12 | 3.262e+04 | 3.281e+04 | +188.4 |

## effect_net_fiscal_balance_dynamic (baseline 5.01e+08)

| rank | parameter | low | high | value at low | value at high | swing |
|---|---|---|---|---|---|---|
| 1 | bv_director_pay_share | 0.4 | 0.8 | 7.847e+08 | 2.622e+08 | -5.226e+08 |
| 2 | bv_productivity_multiplier | 0.8 | 1.2 | 4.282e+08 | 6.257e+08 | +1.975e+08 |
| 3 | real_wage_growth | 0 | 0.02 | 4.613e+08 | 6.362e+08 | +1.749e+08 |
| 4 | entry_wealth_sensitivity | 0 | 2 | 4.419e+08 | 6.098e+08 | +1.679e+08 |
| 5 | fin_return | 0 | 0.03 | 5.779e+08 | 5.042e+08 | -7.372e+07 |
| 6 | wage_shock_sigma | 0.05 | 0.12 | 5.134e+08 | 5.781e+08 | +6.471e+07 |

## effect_bv_owner_eq_income (baseline -233.5)

| rank | parameter | low | high | value at low | value at high | swing |
|---|---|---|---|---|---|---|
| 1 | bv_payout_ratio | 0.1 | 0.6 | -33.67 | -580.8 | -547.2 |
| 2 | bv_director_pay_share | 0.4 | 0.8 | -404.6 | -145.7 | +258.8 |
| 3 | bv_productivity_multiplier | 0.8 | 1.2 | -237.8 | -359 | -121.2 |
| 4 | separation_rate | 0.006 | 0.024 | -291.7 | -411.5 | -119.8 |
| 5 | real_wage_growth | 0 | 0.02 | -238 | -341.8 | -103.9 |
| 6 | job_separation_rate | 0.01906 | 0.03539 | -139 | -233.6 | -94.55 |

## effect_wealth_top10_share (baseline 0.000121)

| rank | parameter | low | high | value at low | value at high | swing |
|---|---|---|---|---|---|---|
| 1 | bv_productivity_multiplier | 0.8 | 1.2 | 0.0001872 | 3.807e-05 | -0.0001491 |
| 2 | marginal_propensity_to_consume | 0.7 | 0.95 | 6.408e-05 | 0.0001665 | +0.0001024 |
| 3 | bv_payout_ratio | 0.1 | 0.6 | 5.395e-05 | 0.0001298 | +7.584e-05 |
| 4 | wage_shock_sigma | 0.05 | 0.12 | 9.57e-05 | 0.0001611 | +6.536e-05 |
| 5 | entry_wealth_sensitivity | 0 | 2 | 0.000183 | 0.000135 | -4.8e-05 |
| 6 | bv_director_pay_share | 0.4 | 0.8 | 5.947e-05 | 0.0001052 | +4.574e-05 |
