# Real Belgian data: what was acquired, how, and what is derived

Reference year **2025** (population on 1 January 2025, EU-LFS annual 2025). Rebuild everything with

```bash
abm fetch-data --year 2025            # downloads (or reuses data/real/cache) and writes data/real/belgium_2025/
abm fetch-data --year 2025 --offline  # rebuild from the committed raw cache only
```

Every raw API response is cached under `data/real/cache/` and its sha256 is recorded in
`data/real/belgium_2025/build_manifest.json`; an offline rebuild is tested to reproduce the tables.

## Access findings (Statbel)

* `statbel.fgov.be/en/open-data` and its download pages answer HTTP 200 but serve an **anti-bot
  challenge page** (F5 "TSPD"), not the data. Working around it would mean impersonating a browser,
  which this project deliberately does **not** do. If you want Statbel's own files (for example the
  population-by-municipality or household-type downloads), download them in a normal browser, and
  convert them to the layouts in `data/templates/`.
* **BeStat** (`bestat.statbel.fgov.be/bestat/api`, Statbel's own open API) works with plain
  requests, but its published "views" are mostly coarse indicator tables (e.g. 3 age groups, or rates
  without counts). The detailed cubes (`datasources`) cannot be queried through that API, so BeStat is
  used only where a view is adequate (VAT-enterprise legal forms by region, 2024).
* **Eurostat's dissemination API** republishes the Belgian national statistical institute's
  figures (population register, Belgian Labour Force Survey, EU-SILC) at the needed detail, openly. It is
  the main source. The numbers are Statbel / NBB-produced; Eurostat is the distributor.

## Tables and provenance

| table (`data/real/belgium_2025/`) | status | sources | notes |
|---|---|---|---|
| `population_age_sex_region.csv` | **observed** | `demo_r_d2jan` | NUTS-2 summed to Brussels / Flanders / Wallonia; bands split at 18 so adults/children are exact; 100+ merged into 85-99 |
| `education_by_age_region.csv` | **observed** | `lfst_r_lfsd2pop` | ISCED 0-2 / 3-4 / 5-8 -> LOW / MEDIUM / HIGH; 'unknown' dropped and renormalised |
| `household_templates_region.csv` | derived | `lfst_hhnhtych`, `lfst_r_lfsd2hh` | national composition x children seed, max-entropy fit per region to the regional adult / child totals. Fitted household counts are 94% (Brussels), 102% (Flanders), 98% (Wallonia) of the published private-household counts (institutional population) |
| `labour_state_by_age_education_region.csv` | derived | `lfsa_pgaed`, `lfsa_egaed`, `lfsa_agaed`, `lfst_r_lfe2emprt`, `lfst_r_lfu3rt`, `edat_lfse_18` | IPF of national age x education x status seeds to regional margins (education by age: observed; employment and unemployment rates by age: observed) |
| `work_arrangement_by_age_region.csv` | derived | `lfsa_esgan`, `lfsa_egan`, `lfst_r_lfe2estat`, BeStat `efedbaa1-...` | self-employed share of employment: national age profile scaled by the regional / national ratio at 15-64 (Brussels 1.00, Flanders 1.07, Wallonia 0.85); **the BV share is a proxy** (private limited companies / (companies + natural persons) among VAT-subject enterprises, 2024) |
| `hfcs_wealth_targets.csv` | **observed** (wave 2023) | ECB HFCS statistical tables, BE column (`abm import-hfcs`) | validation only; net wealth median, mean, percentiles, Gini, top shares, homeownership, medians by age. Definitions differ from the model's net worth |
| `crosscheck_*_vs_statbel.csv` | observed (cross-check) | Statbel LFS rates by region x education x age; Statbel education series (`abm crosscheck`) | independent test of the derived labour and education tables; never fed back into the generator |
| `tenure_by_household_type_region.csv` | derived | `ilc_lvho02` (EU-SILC published aggregate) | national shares applied to all regions; person shares used as household shares |
| `demography_rates.csv` | **observed** (2024) | `demo_mlifetable`, `demo_frate` | qx by sex and age; fertility per woman by age; used by the dynamic model |
| `validation_targets.csv` | **observed** | `ilc_di03`, `ilc_di12`, `ilc_li02`, `ilc_di11`, `ilc_di01` | EU-SILC 2025 (income year 2024); used only to validate (the median also anchors the level calibration) |

The 12 documented assumptions of the build (e.g. regional household mix = national mix; students only
aged 15-24; retired = inactive aged 65+) are listed in `build_manifest.json` and in
`docs/model_assumptions.md`.

## Still not available (restricted or unpublished)

EU-SILC research microdata, HFCS microdata, BELMOD administrative data, longitudinal LFS transitions,
BV owner counts, person-to-firm links, institutional population by age. See `abm sources`.

## Files supplied by the project owner and how they were used

| file | content | use |
|---|---|---|
| `HFCS_Statistical_Tables_Wave_2023_October_2026.xlsx` | ECB HFCS wave 2023 aggregate tables, all countries | Belgian column parsed into `hfcs_wealth_targets.csv`; wealth **validation** in every `abm simulate` run (`validation_wealth_vs_hfcs.csv`) |
| `unemployment_per_age_region_education.xlsx` | Statbel LFS employment, unemployment and activity rates by region x education x age (3 bands) x year | independent cross-check of the derived labour table |
| `Niveau_d_instruction1987-2016_5_FR.xls` | Statbel education attainment 15+, national, 1987-2025 | cross-check of the education table (2025 series); the 1987-2025 trend could later drive a cohort education trend |
| `self_employed_per_region.xlsx` | Eurostat `lfst_r_lfe2estat` extract, employment by professional status, NUTS 2, 2025 | identical to the API pull; confirmed the regional self-employment factors |
| `estat_ilc_lvho02defaultview_filtered.tsv.gz` | `ilc_lvho02` total tenure, all countries | already used through the API (Belgium owner share 70.9% of persons in 2025) |
| `statbel_age_region_nationality.zip` (`TF_SOC_POP_STRUCT_2026.xlsx`) | Statbel population 1 January **2026** by municipality, sex, nationality, marital status, age | consistency check only: 11.868 million residents vs 11.883 million in the 2025 Eurostat series (Brussels 1.256 m, Flanders 6.898 m, Wallonia 3.713 m). Could become the 2026 baseline year |
| `Publication_EAK_Module_2017_def_en.xlsx` | Belgian LFS 2017 ad-hoc module on self-employment (attitudes, reasons, dependence) | not used yet; background for the entry/exit behaviour of the self-employed |
| `LFS_EmpR_Alt_age_EN.xlsx` | employment rates for alternative age groups (20-64, 20-65, ..., by sex and region) | not used yet; useful to validate the `retirement_age_67` scenario |

## Independent checks of the derived tables

* **Labour state** (derived by IPF from national seeds and regional margins) against Statbel's published rates for 26 region x education x
  age cells (2025): mean absolute difference 1.8 pp for employment and 1.1 pp for unemployment. Worst cell: low-educated 25-49 in Brussels
  (model employment 37% vs Statbel 47%, unemployment 30% vs 23%). Details: `crosscheck_labour_vs_statbel.csv`.
* **Education** (15+, three groups) against Statbel's 2025 series: LOW 25.4% vs 25.3%, MEDIUM 37.2% vs 37.3%, HIGH 37.4% vs 37.5%.
* **Wealth** against HFCS 2023 (validation, not calibration): see the table in any run report. With retired households seeded from career
  earnings, the net-worth Gini (0.59 vs 0.604), homeownership (67% vs 66.7%) and the medians of households aged 65+ are close; the overall
  median is about 20% low and the wealth of households under 35 is far too low. Part of the gap is definitional.

Rebuild the cross-checks with `abm crosscheck --lfs-xlsx <file> --education-xls <file>` and the wealth targets with
`abm import-hfcs --xlsx <file>`. The raw workbooks are not committed (size, redistribution terms); their sha256 is recorded in the CSV headers.
