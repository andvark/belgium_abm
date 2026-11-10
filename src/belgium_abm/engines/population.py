"""Synthetic population generator (household-first; spec section 7).

Pipeline
    1. candidate pool  - for each (region, household template) draw many candidate households
                         whose members' sexes/ages follow the region's age-sex structure under
                         explicit couple / parent-child age rules;
    2. IPU             - re-weight candidates so that person cells (region x sex x age band)
                         and household templates match the targets;
    3. integerise      - systematic rounding of the weights to household counts per template,
                         then within-template swap hill-climbing on the person-cell error;
    4. attributes      - education, work state, work arrangement and tenure are allocated from
                         the published conditional shares by controlled count refinement.
No income or wealth is created here (Phase 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from belgium_abm.agents.columns import coerce_frame
from belgium_abm.agents.enums import EducationLevel, HouseholdType, Role
from belgium_abm.agents.household import HOUSEHOLD_COLUMNS
from belgium_abm.agents.person import PERSON_COLUMNS
from belgium_abm.config import ModelConfig
from belgium_abm.engines.integerise import objective, refine_by_swaps
from belgium_abm.engines.ipu import kish_effective_size, run_ipu
from belgium_abm.engines.sampling import (
    age_pmf_from_bands,
    assign_grouped_labels,
    band_index,
    sample_ages_in_ranges,
    systematic_counts,
)
from belgium_abm.io.loaders import TargetBundle
from belgium_abm.rng import RngFactory

SEXES = ("F", "M")


@dataclass
class Population:
    persons: pd.DataFrame
    households: pd.DataFrame
    diagnostics: dict[str, Any]


@dataclass
class _Block:
    """Candidate households of one (region, template)."""

    tix: int
    region_ix: int
    household_type: str
    n_adults: int
    n_children: int
    has_partner: bool
    offset: int
    ages: np.ndarray  # (n, m)
    sexes: np.ndarray  # (n, m) 0 = F, 1 = M

    @property
    def n(self) -> int:
        return self.ages.shape[0]

    @property
    def m(self) -> int:
        return self.ages.shape[1]


def _sex_given_age(rng: np.random.Generator, pmf2: np.ndarray, ages: np.ndarray) -> np.ndarray:
    f, m = pmf2[0][ages], pmf2[1][ages]
    tot = f + m
    p_f = np.where(tot > 0, f / np.where(tot > 0, tot, 1.0), 0.5)
    return (rng.random(ages.shape[0]) >= p_f).astype(np.int8)


def _joint_draw(rng: np.random.Generator, pmf2: np.ndarray, n: int, min_age: int) -> tuple[np.ndarray, np.ndarray]:
    flat = pmf2[:, min_age:].ravel()
    if flat.sum() <= 0:
        raise ValueError(f"population table gives no support for ages >= {min_age}")
    width = pmf2.shape[1] - min_age
    idx = rng.choice(flat.size, size=n, p=flat / flat.sum())
    return (idx // width).astype(np.int8), (idx % width + min_age).astype(np.int64)


def _draw_candidates(
    rng: np.random.Generator, pmf2: np.ndarray, household_type: str, n_adults: int, n_children: int, cfg: ModelConfig, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_age = pmf2.shape[1] - 1
    adult_age = cfg.adult_age
    m = n_adults + n_children
    ages = np.zeros((n, m), dtype=np.int64)
    sexes = np.zeros((n, m), dtype=np.int8)
    ok = np.ones(n, dtype=bool)
    has_partner = household_type in (HouseholdType.COUPLE.value, HouseholdType.COUPLE_WITH_CHILDREN.value)

    sexes[:, 0], ages[:, 0] = _joint_draw(rng, pmf2, n, adult_age)
    if has_partner:
        diff = np.rint(rng.normal(0.0, cfg.couple_age_difference_sd, n)).astype(np.int64)
        ages[:, 1] = np.clip(ages[:, 0] + diff, adult_age, max_age)
        sexes[:, 1] = _sex_given_age(rng, pmf2, ages[:, 1])
        first_other = 2
    else:
        first_other = 1
    for col in range(first_other, n_adults):
        sexes[:, col], ages[:, col] = _joint_draw(rng, pmf2, n, adult_age)

    if n_children:
        parent_cols = [0, 1] if has_partner else [0]
        pa = ages[:, parent_cols]
        lo = np.maximum(0, pa.max(axis=1) - cfg.max_parent_age_at_birth)
        hi = np.minimum(adult_age - 1, pa.min(axis=1) - cfg.min_parent_age_at_birth)
        pmf_all = pmf2.sum(axis=0)
        for col in range(n_adults, m):
            a, good = sample_ages_in_ranges(rng, pmf_all, lo, hi)
            ages[:, col] = a
            sexes[:, col] = _sex_given_age(rng, pmf2, a)
            ok &= good
    return ages, sexes, ok


def _make_block(
    rng: np.random.Generator, pmf2: np.ndarray, household_type: str, a: int, c: int, cfg: ModelConfig, n: int
) -> tuple[np.ndarray, np.ndarray]:
    got_a: list[np.ndarray] = []
    got_s: list[np.ndarray] = []
    remaining = n
    for _ in range(cfg.generator.max_candidate_attempts):
        ages, sexes, ok = _draw_candidates(rng, pmf2, household_type, a, c, cfg, remaining)
        got_a.append(ages[ok])
        got_s.append(sexes[ok])
        remaining -= int(ok.sum())
        if remaining <= 0:
            break
    if remaining > 0:
        raise RuntimeError(
            f"could not generate {n} valid candidate households for template {household_type} ({a} adults, {c} children): "
            "the age/parent-age rules leave too little support in the population table"
        )
    return np.vstack(got_a)[:n], np.vstack(got_s)[:n]


def _pivot(df: pd.DataFrame, index: list[str], column: str, levels: list[str], index_values: list[tuple[Any, ...]]) -> np.ndarray:
    p = df.pivot_table(index=index, columns=column, values="share", aggfunc="sum", fill_value=0.0)
    p = p.reindex(columns=levels, fill_value=0.0)
    p = p.reindex(pd.MultiIndex.from_tuples(index_values, names=index) if len(index) > 1 else pd.Index([v[0] for v in index_values], name=index[0]), fill_value=0.0)
    return p.to_numpy(dtype=float)


def _group_members(keys: np.ndarray, base_idx: np.ndarray, n_groups: int) -> list[np.ndarray]:
    order = np.argsort(keys, kind="stable")
    counts = np.bincount(keys, minlength=n_groups)
    bounds = np.concatenate([[0], np.cumsum(counts)])
    sorted_idx = base_idx[order]
    return [sorted_idx[bounds[g] : bounds[g + 1]] for g in range(n_groups)]


def generate_population(cfg: ModelConfig, targets: TargetBundle, seed: int) -> Population:
    rf = RngFactory(seed)
    gen = cfg.generator
    regions = targets.regions
    n_reg = len(regions)
    pop = targets.population
    f = gen.n_persons / targets.total_persons
    starts = np.array(sorted(pop["age_min"].unique()), dtype=np.int64)
    n_band = len(starts)
    max_age = targets.max_age
    adult_age, working_min = cfg.adult_age, cfg.working_age_min

    # single-year age pmfs per region and sex ("uniform within band")
    pmf = np.zeros((n_reg, 2, max_age + 1))
    for ri, region in enumerate(regions):
        for si, sex in enumerate(SEXES):
            sub = pop[(pop["region"] == region) & (pop["sex_or_demographic_stratum"] == sex)]
            pmf[ri, si] = age_pmf_from_bands(sub[["age_min", "age_max", "count"]], max_age)

    # ---- 1. candidate pool ------------------------------------------------------------------
    hh_t = targets.households.reset_index(drop=True)
    n_cell = n_reg * 2 * n_band
    blocks: list[_Block] = []
    rows_l: list[np.ndarray] = []
    cols_l: list[np.ndarray] = []
    offset = 0
    for tix, row in enumerate(hh_t.itertuples(index=False)):
        ri = regions.index(row.region)
        a, c = int(row.n_adults), int(row.n_children)
        n_cand = max(gen.min_pool_per_template, int(np.ceil(gen.pool_multiplier * row.count * f)))
        rng = rf.stream(f"pool:{row.region}:{row.household_type}:{a}:{c}")
        ages, sexes = _make_block(rng, pmf[ri], row.household_type, a, c, cfg, n_cand)
        blk = _Block(tix, ri, row.household_type, a, c, row.household_type in ("COUPLE", "COUPLE_WITH_CHILDREN"), offset, ages, sexes)
        blocks.append(blk)
        cell = ri * 2 * n_band + sexes.astype(np.int64) * n_band + band_index(ages, starts)
        rows_l.append(cell.ravel())
        cols_l.append(np.repeat(np.arange(offset, offset + n_cand), blk.m))
        rows_l.append(np.full(n_cand, n_cell + tix))
        cols_l.append(np.arange(offset, offset + n_cand))
        offset += n_cand
    n_cand_total = offset
    n_tmpl = len(hh_t)
    M_all = sp.coo_matrix(
        (np.ones(sum(r.size for r in rows_l)), (np.concatenate(rows_l), np.concatenate(cols_l))),
        shape=(n_cell + n_tmpl, n_cand_total),
    ).tocsr()
    tmpl_of = np.concatenate([np.full(b.n, b.tix) for b in blocks])

    # ---- targets ------------------------------------------------------------------------------
    cell_target = np.zeros(n_cell)
    for r in pop.itertuples(index=False):
        ri = regions.index(r.region)
        si = SEXES.index(r.sex_or_demographic_stratum)
        bi = int(np.searchsorted(starts, r.age_min))
        cell_target[ri * 2 * n_band + si * n_band + bi] += r.count * f
    tmpl_expected = hh_t["count"].to_numpy(dtype=float) * f
    targets_all = np.concatenate([cell_target, tmpl_expected])

    # ---- 2. IPU -------------------------------------------------------------------------------
    ipu = run_ipu(M_all, targets_all, max_iter=gen.ipu_max_iterations, tol=gen.ipu_tolerance)
    w = ipu.weights
    unsupported = []
    for c in ipu.unsupported:
        unsupported.append(f"person cell #{c}" if c < n_cell else f"template {hh_t.iloc[c - n_cell].to_dict()}")

    # ---- 3. integerise ------------------------------------------------------------------------
    rng_int = rf.stream("integerise")
    h_total = int(round(float(tmpl_expected.sum())))
    k_per_tmpl = systematic_counts(rng_int, h_total, tmpl_expected)
    selection_parts = []
    fallback_templates = []
    for b in blocks:
        k = int(k_per_tmpl[b.tix])
        if k == 0:
            continue
        w_slice = w[b.offset : b.offset + b.n]
        if w_slice.sum() <= 0:
            fallback_templates.append(b.tix)
            w_slice = np.ones(b.n)
        mult = systematic_counts(rng_int, k, w_slice)
        selection_parts.append(np.repeat(np.arange(b.offset, b.offset + b.n), mult))
    selection = np.concatenate(selection_parts)
    M_cells = M_all[:n_cell]
    x0 = np.asarray(M_cells[:, selection].sum(axis=1)).ravel()
    obj_rounding = objective(x0, cell_target)
    swaps = refine_by_swaps(
        rf.stream("swaps"), selection, tmpl_of, M_cells, cell_target,
        n_proposals=gen.refine_swaps_per_household * selection.size, proposal_weights=w,
    )
    selection = swaps.selection

    # ---- build households / persons from the selection ----------------------------------------
    hh_tix = tmpl_of[selection]
    hh_local = np.empty(selection.size, dtype=np.int64)
    for b in blocks:
        mask = hh_tix == b.tix
        hh_local[mask] = selection[mask] - b.offset
    perm = rf.stream("household_order").permutation(selection.size)
    hh_tix, hh_local = hh_tix[perm], hh_local[perm]
    n_hh = hh_tix.size
    hh_ids = np.arange(1, n_hh + 1)

    part_hid, part_col, part_age, part_sex, part_tix = [], [], [], [], []
    for b in blocks:
        pos = np.flatnonzero(hh_tix == b.tix)
        if pos.size == 0:
            continue
        loc = hh_local[pos]
        for col in range(b.m):
            part_hid.append(hh_ids[pos])
            part_col.append(np.full(pos.size, col))
            part_age.append(b.ages[loc, col])
            part_sex.append(b.sexes[loc, col])
            part_tix.append(np.full(pos.size, b.tix))
    hid = np.concatenate(part_hid)
    col = np.concatenate(part_col)
    order = np.lexsort((col, hid))
    hid, col = hid[order], col[order]
    age = np.concatenate(part_age)[order]
    sex_i = np.concatenate(part_sex)[order]
    tix_p = np.concatenate(part_tix)[order]
    n_p = hid.size
    pid = np.arange(1, n_p + 1)

    block_by_tix = {b.tix: b for b in blocks}
    sizes_hh = np.array([block_by_tix[int(t)].m for t in hh_tix])
    start = np.concatenate([[0], np.cumsum(sizes_hh)[:-1]])
    head_pid = start + 1

    role = np.empty(n_p, dtype=object)
    partner = np.full(n_p, -1, dtype=np.int64)
    parent1 = np.full(n_p, -1, dtype=np.int64)
    parent2 = np.full(n_p, -1, dtype=np.int64)
    for b in blocks:
        mask = tix_p == b.tix
        c_col = col[mask]
        r = np.where(c_col == 0, Role.HEAD.value, np.where(c_col < b.n_adults, Role.OTHER_ADULT.value, Role.CHILD.value)).astype(object)
        if b.has_partner:
            r[c_col == 1] = Role.PARTNER.value
        role[mask] = r
        h0 = hid[mask] - 1
        if b.has_partner:
            partner[mask] = np.where(c_col == 0, start[h0] + 2, np.where(c_col == 1, start[h0] + 1, -1))
        is_child = c_col >= b.n_adults
        parent1[mask] = np.where(is_child, start[h0] + 1, -1)
        if b.has_partner:
            parent2[mask] = np.where(is_child, start[h0] + 2, -1)

    region_ix_p = np.array([block_by_tix[int(t)].region_ix for t in tix_p], dtype=np.int64)
    region_p = np.asarray(regions, dtype=object)[region_ix_p]
    sex_p = np.asarray(SEXES, dtype=object)[sex_i]

    # ---- 4. attributes ------------------------------------------------------------------------
    prop = gen.refine_counts_proposals_per_cell
    working_idx = np.flatnonzero(age >= working_min)

    edu_t = targets.education
    edu_levels = [lv for lv in (EducationLevel.LOW.value, EducationLevel.MEDIUM.value, EducationLevel.HIGH.value) if lv in set(edu_t["education_level"])]
    e_starts = np.array(sorted(edu_t["age_min"].unique()), dtype=np.int64)
    n_eb = len(e_starts)
    edu = np.full(n_p, EducationLevel.NOT_ASSESSED.value, dtype=object)
    keys = region_ix_p[working_idx] * n_eb + np.maximum(band_index(age[working_idx], e_starts), 0)
    members = _group_members(keys, working_idx, n_reg * n_eb)
    shares = _pivot(edu_t, ["region", "age_min"], "education_level", edu_levels, [(r, int(s)) for r in regions for s in e_starts])
    assign_grouped_labels(rf.stream("education"), edu, members, shares, edu_levels, [np.arange(n_reg * n_eb) // n_eb], prop)

    lab_t = targets.labour_state
    states = [s for s in ("STUDENT", "EMPLOYED", "UNEMPLOYED", "INACTIVE", "RETIRED") if s in set(lab_t["work_state"])]
    l_starts = np.array(sorted(lab_t["age_min"].unique()), dtype=np.int64)
    n_lb, n_el = len(l_starts), len(edu_levels)
    state = np.full(n_p, None, dtype=object)
    e_ix = np.array([edu_levels.index(v) for v in edu[working_idx]], dtype=np.int64)
    keys = (region_ix_p[working_idx] * n_lb + np.maximum(band_index(age[working_idx], l_starts), 0)) * n_el + e_ix
    members = _group_members(keys, working_idx, n_reg * n_lb * n_el)
    shares = _pivot(lab_t, ["region", "age_min", "education_level"], "work_state", states, [(r, int(s), e) for r in regions for s in l_starts for e in edu_levels])
    g = np.arange(n_reg * n_lb * n_el)
    assign_grouped_labels(rf.stream("labour_state"), state, members, shares, states, [g // n_el, g // (n_el * n_lb)], prop)

    arr_t = targets.work_arrangement
    arrs = [a for a in ("EMPLOYEE", "SOLE_PROPRIETOR", "BV_OWNER_MANAGER") if a in set(arr_t["primary_work_arrangement"])]
    a_starts = np.array(sorted(arr_t["age_min"].unique()), dtype=np.int64)
    n_ab = len(a_starts)
    arrangement = np.full(n_p, "NONE", dtype=object)
    emp_idx = np.flatnonzero(state == "EMPLOYED")
    keys = region_ix_p[emp_idx] * n_ab + np.maximum(band_index(age[emp_idx], a_starts), 0)
    members = _group_members(keys, emp_idx, n_reg * n_ab)
    shares = _pivot(arr_t, ["region", "age_min"], "primary_work_arrangement", arrs, [(r, int(s)) for r in regions for s in a_starts])
    assign_grouped_labels(rf.stream("work_arrangement"), arrangement, members, shares, arrs, [np.arange(n_reg * n_ab) // n_ab], prop)

    student: list[Any] = [None] * n_p
    for i in working_idx:
        student[i] = state[i] == "STUDENT"

    n_adults_hh = np.array([block_by_tix[int(t)].n_adults for t in hh_tix], dtype=np.int64)
    n_children_hh = np.array([block_by_tix[int(t)].n_children for t in hh_tix], dtype=np.int64)
    type_hh = np.array([block_by_tix[int(t)].household_type for t in hh_tix], dtype=object)
    region_hh = np.asarray(regions, dtype=object)[np.array([block_by_tix[int(t)].region_ix for t in hh_tix], dtype=np.int64)]

    tenure = np.full(n_hh, None, dtype=object)
    if targets.tenure is not None:
        ten_t = targets.tenure
        tenures = [t for t in ("RENTER", "MORTGAGED_OWNER", "OUTRIGHT_OWNER", "OTHER") if t in set(ten_t["housing_tenure"])]
        types = [t for t in (h.value for h in HouseholdType) if t in set(hh_t["household_type"])]
        keys = np.array([regions.index(r) * len(types) + types.index(t) for r, t in zip(region_hh, type_hh, strict=True)], dtype=np.int64)
        members = _group_members(keys, np.arange(n_hh), n_reg * len(types))
        shares = _pivot(ten_t, ["region", "household_type"], "housing_tenure", tenures, [(r, t) for r in regions for t in types])
        assign_grouped_labels(rf.stream("tenure"), tenure, members, shares, tenures, [np.arange(n_reg * len(types)) // len(types)], prop)

    # ---- assemble tables ------------------------------------------------------------------------
    def opt_int(a: np.ndarray) -> pd.array:
        return pd.array([None if v < 0 else int(v) for v in a], dtype="Int64")

    persons = pd.DataFrame(
        {
            "person_id": pid,
            "household_id": hid,
            "age": age,
            "sex_or_demographic_stratum": sex_p,
            "region": region_p,
            "education_level": edu,
            "student_status": pd.array(student, dtype="boolean"),
            "work_state": state,
            "primary_work_arrangement": arrangement,
            "alive": True,
            "weight": targets.total_persons / n_p,
            "role": role,
            "partner_id": opt_int(partner),
            "parent1_id": opt_int(parent1),
            "parent2_id": opt_int(parent2),
        }
    )
    persons = coerce_frame(persons, PERSON_COLUMNS)

    members_str = [";".join(str(int(s) + k + 1) for k in range(int(m))) for s, m in zip(start, sizes_hh, strict=True)]
    households = pd.DataFrame(
        {
            "household_id": hh_ids,
            "member_ids": members_str,
            "region": region_hh,
            "household_type": type_hh,
            "n_adults": n_adults_hh,
            "n_children": n_children_hh,
            "head_person_id": head_pid,
            "housing_tenure": tenure,
            "num_dependents": n_children_hh,
            "weight": targets.total_households / n_hh,
        }
    )
    households = coerce_frame(households, HOUSEHOLD_COLUMNS)

    diagnostics: dict[str, Any] = {
        "scale_factor": f,
        "candidate_pool_size": int(n_cand_total),
        "ipu_iterations": ipu.iterations,
        "ipu_converged": bool(ipu.converged),
        "ipu_max_rel_error": ipu.max_rel_error,
        "ipu_weight_kish_n_eff": kish_effective_size(w),
        "ipu_unsupported_constraints": unsupported,
        "templates_without_ipu_support": [hh_t.iloc[t].to_dict() for t in fallback_templates],
        "integerise_objective_after_rounding": obj_rounding,
        "integerise_objective_after_swaps": swaps.objective_after,
        "swaps_accepted": swaps.accepted,
        "n_persons": int(n_p),
        "n_households": int(n_hh),
    }
    return Population(persons, households, diagnostics)
