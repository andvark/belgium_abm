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
| `work_arrangement_by_age_region.csv` | derived | `lfsa_esgan`, `lfsa_egan`, BeStat `efedbaa1-...` | self-employed share of employment by age is national; **the BV share is a proxy** (private limited companies / (companies + natural persons) among VAT-subject enterprises, 2024) |
| `tenure_by_household_type_region.csv` | derived | `ilc_lvho02` (EU-SILC published aggregate) | national shares applied to all regions; person shares used as household shares |
| `demography_rates.csv` | **observed** (2024) | `demo_mlifetable`, `demo_frate` | qx by sex and age; fertility per woman by age; used by the dynamic model |
| `validation_targets.csv` | **observed** | `ilc_di03`, `ilc_di12`, `ilc_li02`, `ilc_di11`, `ilc_di01` | EU-SILC 2025 (income year 2024); used only to validate (the median also anchors the level calibration) |

The 12 documented assumptions of the build (e.g. regional household mix = national mix; students only
aged 15-24; retired = inactive aged 65+) are listed in `build_manifest.json` and in
`docs/model_assumptions.md`.

## Still not available (restricted or unpublished)

EU-SILC research microdata, HFCS microdata, BELMOD administrative data, longitudinal LFS transitions,
BV owner counts, person-to-firm links, institutional population by age. See `abm sources`.
