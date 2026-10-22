#!/usr/bin/env python3
"""
physarum_sim.py

Adaptive transport-network simulation of Physarum polycephalum on a
heterogeneous two-dimensional nutrient landscape.

Core transport model (Tero, Kobayashi and Nakagaki, 2007; Tero et al., 2010):

    L(t) p = I                              pressures from the conductivity-weighted Laplacian
    Q_ij = (D_ij / l_ij) (p_i - p_j)        Poiseuille flux through tube ij
    dD_ij/dt = alpha |Q_ij / Q0|^mu - gamma D_ij
                                            flux-driven reinforcement and decay of conductivity

Phenomenological extensions (NOT part of the Tero model, introduced here to
obtain spatial growth and nutrient coupling):

    1. A biomass field B(x, y, t) on the grid with nutrient-driven growth and a
       maintenance loss term.
    2. A stochastic, nutrient-biased colonization rule that lets biomass move into
       unoccupied cells adjacent to occupied cells only.
    3. A source/sink allocation rule that produces a current vector I whose sum is
       exactly zero inside every connected component of the active network.
    4. Flux-mediated redistribution of biomass along the network (biomass follows
       the net current, with exact conservation inside each component).
    5. Nutrient depletion by uptake in occupied cells, optional regeneration, and
       scripted nutrient events (new patches appearing at a given step).
    6. Soft pruning: tubes whose conductivity falls below a threshold are reset to
       a small floor value and no longer counted as network tubes.

Run `python physarum_sim.py --help` for the command-line options.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import gaussian_filter
from scipy.sparse.csgraph import connected_components

# Patch definition: (fractional y in [0,1], fractional x in [0,1], amplitude, sigma in cells)
Patch = Tuple[float, float, float, float]


# =============================================================================
# 1. Configuration
# =============================================================================
@dataclass
class Config:
    """All model and numerical parameters. Values are illustrative, not calibrated."""

    # --- domain and time ---
    H: int = 140                    # grid rows
    W: int = 180                    # grid columns
    connectivity: int = 8           # 4 or 8 neighbours
    seed: int = 7                   # random seed (nutrient noise and colonization)
    dt: float = 0.5                 # time step (dimensionless time units)
    n_steps: int = 2500             # number of simulation steps
    scale: float = 1.0              # W / 90; set by make_config, used to scale length parameters
    obstacles: List[Tuple[float, float, float, float]] = field(default_factory=list)
                                    # impassable rectangles (fy0, fx0, fy1, fx1), fractional coordinates

    # --- nutrient landscape S(x, y, t) ---
    patches: List[Patch] = field(default_factory=list)
    baseline: float = 0.03          # uniform background nutrient
    noise_amp: float = 0.06         # amplitude of correlated noise
    noise_corr: float = 4.0         # correlation length of the noise (cells)
    uptake_rate: float = 0.03       # u: maximal specific uptake rate
    uptake_K: float = 0.3           # K: half-saturation biomass for uptake
    regen_rate: float = 0.0         # r: relaxation rate of S towards the initial field
    events: List[Tuple[int, Patch]] = field(default_factory=list)   # (step, patch)

    # --- bacterial prey and inhibitors (extension, scenario E) ---
    prey: List[Tuple[float, float, float, float, bool]] = field(default_factory=list)
                                    # colonies (fy, fx, amp, sigma, releases_inhibitor)
    prey_growth: float = 0.01       # r_p: logistic regrowth rate of a colony toward its capacity
    prey_floor: float = 0.02        # fraction of capacity below which a colony cannot be grazed
    inhib_range: float = 8.0        # range (cells, scaled) of the diffusing inhibitor
    inhib_tol: float = 0.2          # inhibitor level that reduces colonization by a factor e
    inhib_toxicity: float = 0.03    # biomass loss rate per unit inhibitor (R = 1 at an intact colony)

    # --- consolidation of the sheet into veins (extension) ---
    consolidate_age: int = 50       # cells older than this (steps) start draining into tubes
    consolidate_rate: float = 0.04  # fraction of their biomass drained per unit time
    b_sheet: float = 0.03           # thin-film biomass that old sheet cells keep
    retract_age: int = 200          # film cells older than this with no vein retract completely
    scar_steps: int = 200           # vacated cells are not recolonized or counted as front for this long

    # --- organism and colonization (extension) ---
    inoc_fy: float = 0.5            # inoculum position, fractional row
    inoc_fx: float = 0.13           # inoculum position, fractional column
    inoc_radius: float = 3.0        # inoculum radius (cells)
    inoc_biomass: float = 1.0       # initial biomass per inoculated cell
    inoc_food: float = 0.4          # nutrient patch amplitude placed under the inoculum
    inoc_food_sigma: float = 3.0
    yield_coeff: float = 1.0        # Y: biomass produced per unit nutrient taken up
    maintenance: float = 0.001      # m: biomass loss rate
    b_min: float = 0.015            # occupancy threshold on B
    b_keep: float = 0.04            # a donor cell keeps at least this much biomass
    b_colonize: float = 0.05        # biomass moved into a newly colonized cell
    k_col: float = 1.5              # colonization rate constant
    eta: float = 2.0                # exponent of nutrient sensitivity of the front
    eps_explore: float = 0.005      # nutrient-independent exploration term
    lam_grad: float = 20.0          # weight of the relative attractant-gradient term
    sense_diff: float = 1.0         # D_a: diffusivity of the chemoattractant (cells^2 / time)
    sense_decay: float = 0.0025     # k_a: decay rate of the chemoattractant; range sqrt(D_a/k_a) = 20 cells
    sense_threshold: float = 0.15   # only nutrient above this level releases attractant
    B_ref: float = 0.5              # biomass at which the resource / demand factors saturate
    kappa_T: float = 0.5            # flux-mediated biomass redistribution coefficient
    diff_B: float = 0.1             # D_B: local spreading of biomass along active edges (sheet)

    # --- source and sink allocation (extension) ---
    q_flow: float = 1.0             # injected current per unit of total source weight
    source_eps: float = 0.1         # nutrient-independent source weight of established biomass
    front_sink_c0: float = 0.1      # nutrient-independent sink weight of a front cell
    front_sink_wA: float = 2.0      # sink weight of a front cell from the attractant it senses
    demand_sink_weight: float = 1.0 # sink weight of under-occupied nutrient-rich cells
    n_sources: int = 4              # source sites drawn per step (0 = diffuse allocation)
    n_sinks: int = 2                # sink sites drawn per step (0 = diffuse allocation)
    f_max: float = 0.5              # max fraction of a cell's biomass moved by flow per step
    solver: str = "direct"          # "direct" (sparse LU) or "cg" (conjugate gradient)

    # --- adaptive conductivity (Tero model) ---
    alpha: float = 1.0              # reinforcement coefficient
    mu: float = 1.5                 # flux exponent
    gamma: float = 0.1              # decay coefficient
    Q0: float = 1.0                 # reference flux that makes |Q/Q0| dimensionless
    D_init: float = 0.2             # conductivity of a newly activated tube
    D_floor: float = 1e-4           # lower bound on conductivity (pruned state)
    D_max: float = 1e3              # upper bound on conductivity
    prune_threshold: float = 0.05   # tubes below this are reset to D_floor
    tube_D: float = 1.0             # conductivity above which a tube protects its wall cells
    D_noise: float = 0.05           # relative noise on D_init to break symmetry

    # --- visualization ---
    steps_per_frame: int = 2
    new_age: int = 40               # cells colonized within this many steps are "new"
    quiver_stride: int = 4


def make_config(scenario: str, **overrides) -> Config:
    """
    Return the configuration of scenario A, B, C, D or E with optional overrides.

    Scenario geometry is written for a reference width of 90 cells; every length
    parameter (patch widths, inoculum radius, attractant and inhibitor ranges, noise
    correlation) is multiplied by scale = W / 90, and the rate parameters that act
    per cell (colonization rate, gradient weight, flux reference) are rescaled so that
    the dynamics in physical units do not depend on the grid resolution.
    """
    s = scenario.upper()
    if s == "A":
        cfg = Config(inoc_fy=0.5, inoc_fx=0.12,
                     patches=[(0.5, 0.84, 1.0, 6.0)])
    elif s == "B":
        cfg = Config(inoc_fy=0.52, inoc_fx=0.47, noise_amp=0.08,
                     patches=[(0.12, 0.12, 1.0, 4.5), (0.14, 0.86, 0.9, 5.0),
                              (0.86, 0.20, 0.8, 4.0), (0.84, 0.80, 1.0, 5.0),
                              (0.50, 0.93, 0.6, 3.0), (0.30, 0.68, 0.45, 2.5),
                              (0.72, 0.52, 0.5, 2.5), (0.52, 0.07, 0.7, 3.5),
                              (0.08, 0.50, 0.55, 3.0)])
    elif s == "C":
        cfg = Config(inoc_fy=0.5, inoc_fx=0.2,
                     patches=[(0.2, 0.73, 1.0, 5.0), (0.8, 0.73, 0.9, 5.0)],
                     uptake_rate=0.05,
                     events=[(500, (0.17, 0.15, 1.0, 5.0)), (1500, (0.85, 0.30, 0.9, 4.5))])
    elif s == "D":
        # labyrinth: three vertical walls with offset gaps, plus two baffles
        cfg = Config(inoc_fy=0.5, inoc_fx=0.07, noise_amp=0.04,
                     patches=[(0.5, 0.94, 1.0, 5.0), (0.10, 0.44, 0.55, 3.0),
                              (0.90, 0.70, 0.7, 3.5), (0.88, 0.12, 0.45, 2.5)],
                     obstacles=[(0.00, 0.30, 0.72, 0.33), (0.86, 0.30, 1.00, 0.33),
                                (0.28, 0.55, 1.00, 0.58), (0.00, 0.55, 0.14, 0.58),
                                (0.00, 0.78, 0.58, 0.81), (0.72, 0.78, 1.00, 0.81),
                                (0.50, 0.33, 0.53, 0.46), (0.45, 0.62, 0.48, 0.72)])
    elif s == "E":
        # bacterial prey: six colonies, three of which release a diffusing inhibitor
        cfg = Config(inoc_fy=0.5, inoc_fx=0.12, baseline=0.02, noise_amp=0.03,
                     uptake_rate=0.04,
                     prey=[(0.50, 0.42, 1.0, 4.0, True),    # blocks the direct route
                           (0.20, 0.40, 0.9, 4.0, False),
                           (0.80, 0.40, 0.9, 4.0, False),
                           (0.18, 0.80, 1.0, 4.5, False),
                           (0.82, 0.80, 1.0, 4.5, True),
                           (0.50, 0.88, 0.8, 3.5, True)])
    else:
        raise ValueError(f"Unknown scenario '{scenario}' (use A, B, C, D or E)")
    cfg = replace(cfg, **overrides)
    sc = cfg.W / 90.0
    cfg.scale = sc
    cfg.patches = [(fy, fx, a, sg * sc) for (fy, fx, a, sg) in cfg.patches]
    cfg.events = [(st, (fy, fx, a, sg * sc)) for (st, (fy, fx, a, sg)) in cfg.events]
    cfg.prey = [(fy, fx, a, sg * sc, inh) for (fy, fx, a, sg, inh) in cfg.prey]
    cfg.inoc_radius *= sc
    cfg.inoc_food_sigma *= sc
    cfg.noise_corr *= sc
    cfg.sense_decay = cfg.sense_diff / (20.0 * sc) ** 2
    cfg.inhib_range *= sc
    cfg.lam_grad *= sc
    cfg.k_col *= sc
    cfg.quiver_stride = max(3, int(round(cfg.quiver_stride * sc)))
    return cfg


# =============================================================================
# 2. Grid and graph topology
# =============================================================================
class Grid:
    """Regular H x W lattice with a fixed list of potential edges (tubes)."""

    def __init__(self, H: int, W: int, connectivity: int = 4,
                 obstacle: Optional[np.ndarray] = None):
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        self.H, self.W, self.N = H, W, H * W
        ys, xs = np.mgrid[0:H, 0:W]
        self.y = ys.ravel().astype(float)
        self.x = xs.ravel().astype(float)
        idx = np.arange(self.N).reshape(H, W)
        self.obstacle = np.zeros(self.N, dtype=bool) if obstacle is None else obstacle.astype(bool)

        pairs = [(idx[:, :-1], idx[:, 1:], 1.0),      # horizontal
                 (idx[:-1, :], idx[1:, :], 1.0)]      # vertical
        if connectivity == 8:
            r2 = float(np.sqrt(2.0))
            pairs += [(idx[:-1, :-1], idx[1:, 1:], r2),
                      (idx[:-1, 1:], idx[1:, :-1], r2)]
        ei, ej, el = [], [], []
        for a, b, L in pairs:
            ei.append(a.ravel()); ej.append(b.ravel())
            el.append(np.full(a.size, L))
        ei, ej, el = np.concatenate(ei), np.concatenate(ej), np.concatenate(el)
        keep = ~self.obstacle[ei] & ~self.obstacle[ej]   # no tube may touch an obstacle cell
        self.ei, self.ej, self.el = ei[keep], ej[keep], el[keep]   # each undirected edge once
        self.E = self.ei.size
        # unit direction vector from i to j (used for flux visualization)
        self.ux = (self.x[self.ej] - self.x[self.ei]) / self.el
        self.uy = (self.y[self.ej] - self.y[self.ei]) / self.el

        # neighbour table (N x degree), padded with -1
        deg = connectivity
        self.nbr = np.full((self.N, deg), -1, dtype=np.int64)
        fill = np.zeros(self.N, dtype=np.int64)
        for a, b in zip(self.ei, self.ej):
            self.nbr[a, fill[a]] = b; fill[a] += 1
            self.nbr[b, fill[b]] = a; fill[b] += 1


# =============================================================================
# 3. Nutrient landscape
# =============================================================================
def gaussian_patch(grid: Grid, fy: float, fx: float, amp: float, sigma: float) -> np.ndarray:
    """Gaussian nutrient patch centred at fractional position (fy, fx)."""
    cy, cx = fy * (grid.H - 1), fx * (grid.W - 1)
    r2 = (grid.x - cx) ** 2 + (grid.y - cy) ** 2
    return amp * np.exp(-r2 / (2.0 * sigma ** 2))


def make_obstacle_mask(H: int, W: int, cfg: Config) -> np.ndarray:
    """Boolean mask of impassable cells from the rectangles in cfg.obstacles."""
    mask = np.zeros((H, W), dtype=bool)
    for (fy0, fx0, fy1, fx1) in cfg.obstacles:
        y0, y1 = int(round(fy0 * (H - 1))), int(round(fy1 * (H - 1)))
        x0, x1 = int(round(fx0 * (W - 1))), int(round(fx1 * (W - 1)))
        mask[min(y0, y1):max(y0, y1) + 1, min(x0, x1):max(x0, x1) + 1] = True
    return mask.ravel()


def make_prey_fields(grid: Grid, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrying capacity K_p of the bacterial colonies (nutrient the colony holds when
    ungrazed) and the inhibitor release footprint of the colonies flagged as
    inhibitor producers.
    """
    Kp = np.zeros(grid.N)
    inh = np.zeros(grid.N)
    for (fy, fx, amp, sig, is_inh) in cfg.prey:
        g = gaussian_patch(grid, fy, fx, amp, sig)
        Kp += g
        if is_inh:
            inh += g
    return Kp, inh


def make_nutrient_field(grid: Grid, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Baseline + Gaussian patches + correlated noise, clipped to be nonnegative."""
    S = np.full(grid.N, cfg.baseline, dtype=float)
    for (fy, fx, amp, sig) in cfg.patches:
        S += gaussian_patch(grid, fy, fx, amp, sig)
    if cfg.inoc_food > 0:
        S += gaussian_patch(grid, cfg.inoc_fy, cfg.inoc_fx, cfg.inoc_food, cfg.inoc_food_sigma)
    if cfg.noise_amp > 0:
        noise = gaussian_filter(rng.standard_normal((grid.H, grid.W)), cfg.noise_corr)
        noise /= max(noise.std(), 1e-12)
        S += cfg.noise_amp * noise.ravel()
    S = np.clip(S, 0.0, None)
    S[grid.obstacle] = 0.0
    return S


# =============================================================================
# 4. Pressure solver (Tero model) - reusable for tests and for the simulation
# =============================================================================
def solve_pressure(n: int, a: np.ndarray, b: np.ndarray, w: np.ndarray,
                   I: np.ndarray, labels: np.ndarray, ncomp: int,
                   solver: str = "direct") -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Solve L p = I on a graph with n nodes, edges (a, b) and weights w = D / l.

    The Laplacian is singular (one zero eigenvalue per connected component).
    Gauge: one reference node per component is grounded by adding 1 to its
    diagonal entry. Because the current sums to zero inside each component, the
    grounded node ends up at p = 0 and the regularised system is solved exactly
    by a solution of L p = I (the ground carries zero current).

    Returns p and a diagnostics dictionary.
    """
    # first node of each component as reference (last assignment wins, so reverse)
    ref = np.empty(ncomp, dtype=np.int64)
    ref[labels[::-1]] = np.arange(n)[::-1]

    rows = np.concatenate([a, b, a, b, ref])
    cols = np.concatenate([a, b, b, a, ref])
    data = np.concatenate([w, w, -w, -w, np.ones(ncomp)])
    L = sp.csc_matrix((data, (rows, cols)), shape=(n, n))

    diag = {}
    if not np.any(I):
        p = np.zeros(n)
    elif solver == "cg":
        Minv = spla.LinearOperator((n, n), matvec=lambda v: v / L.diagonal())
        p, info = spla.cg(L, I, M=Minv, rtol=1e-10, maxiter=5000)
        diag["cg_info"] = float(info)
    else:
        try:
            p = spla.splu(L).solve(I)
        except RuntimeError as exc:                 # singular (should not happen with the gauge)
            warnings.warn(f"sparse LU failed ({exc}); using lsqr fallback")
            p = spla.lsqr(L, I, atol=1e-12, btol=1e-12)[0]

    if not np.all(np.isfinite(p)):
        warnings.warn("non-finite pressures detected; resetting pressures to zero")
        p = np.zeros(n)

    scale = max(float(np.max(np.abs(I))) if n else 0.0, 1e-12)
    diag["residual"] = float(np.max(np.abs(L @ p - I))) / scale if n else 0.0
    diag["ref_pressure"] = float(np.max(np.abs(p[ref]))) if ncomp else 0.0
    return p, diag


def adapt_conductivity(D: np.ndarray, Q: np.ndarray, cfg: Config) -> np.ndarray:
    """
    One step of dD/dt = alpha |Q/Q0|^mu - gamma D using the exact solution of the
    linear ODE with the forcing frozen over the step (exponential integrator).
    The update is unconditionally stable and keeps D positive.
    """
    r = cfg.alpha * np.abs(Q / cfg.Q0) ** cfg.mu
    eg = np.exp(-cfg.gamma * cfg.dt)
    Dn = D * eg + r * (1.0 - eg) / cfg.gamma
    return np.clip(Dn, cfg.D_floor, cfg.D_max)


# =============================================================================
# 5. Simulation
# =============================================================================
METRIC_KEYS = ["step", "time", "occupied_area", "biomass", "network_length", "n_tubes",
               "n_active_edges", "mean_D", "consumed", "n_components", "cost_Dl",
               "dissipation", "residual", "divergence_err", "component_balance"]


class PhysarumSimulation:
    """State and update rules of the coupled transport / colonization model."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.grid = Grid(cfg.H, cfg.W, cfg.connectivity, make_obstacle_mask(cfg.H, cfg.W, cfg))
        g = self.grid

        # fields on nodes
        self.S = make_nutrient_field(g, cfg, self.rng)
        self.Kp, self.inh_src = make_prey_fields(g, cfg)
        self.prey_cells = self.Kp > cfg.prey_floor
        self.S += self.Kp
        self.S_ref = self.S.copy()                   # target for regeneration
        self.B = np.zeros(g.N)
        cy, cx = cfg.inoc_fy * (g.H - 1), cfg.inoc_fx * (g.W - 1)
        self.inoc_nodes = np.flatnonzero(((g.x - cx) ** 2 + (g.y - cy) ** 2 <= cfg.inoc_radius ** 2)
                                         & ~g.obstacle)
        self.B[self.inoc_nodes] = cfg.inoc_biomass
        self.inoc_xy = (cx, cy)
        self.age = np.full(g.N, -1, dtype=np.int64)  # step at which a cell was colonized
        self.vacated = np.full(g.N, -10**9, dtype=np.int64)   # step at which a cell was abandoned
        self.age[self.inoc_nodes] = 0
        self.p = np.full(g.N, np.nan)
        self.I = np.zeros(g.N)
        self.comp = np.full(g.N, -1, dtype=np.int64)
        self.front = np.zeros(g.N, dtype=bool)
        self.ncomp = 0

        # fields on edges
        self.D = np.full(g.E, cfg.D_init)
        self.Q = np.zeros(g.E)
        self.active = np.zeros(g.E, dtype=bool)

        # chemoattractant operator (k_a I - D_a Lap) with zero-flux boundaries, factorised once
        wa = cfg.sense_diff / g.el ** 2
        rows = np.concatenate([g.ei, g.ej, g.ei, g.ej])
        cols = np.concatenate([g.ei, g.ej, g.ej, g.ei])
        data = np.concatenate([wa, wa, -wa, -wa])
        Lap = sp.csc_matrix((data, (rows, cols)), shape=(g.N, g.N))
        self._sense_lu = spla.splu((cfg.sense_decay * sp.identity(g.N, format="csc") + Lap).tocsc())
        self.A = np.zeros(g.N)                      # attractant field
        # inhibitor operator (same form, shorter range); only built when inhibitor colonies exist
        self.has_inhibitor = bool(np.any(self.inh_src > 0))
        self.R = np.zeros(g.N)                      # inhibitor field
        if self.has_inhibitor:
            k_inh = cfg.sense_diff / cfg.inhib_range ** 2
            self._inhib_lu = spla.splu((k_inh * sp.identity(g.N, format="csc") + Lap).tocsc())
            # normalise so that R = 1 at the centre of an intact inhibitor colony
            self._R_scale = 1.0 / max(float(self._inhib_lu.solve(self.inh_src).max()), 1e-12)
        self._update_attractant()
        self.toxic_loss = 0.0

        # bookkeeping
        self.step_count = 0
        self.t = 0.0
        self.consumed = 0.0
        self.lost_biomass = 0.0
        self.metrics: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
        self.last_diag: Dict[str, float] = {}
        self.worst: Dict[str, float] = {"residual": 0.0, "divergence_err": 0.0,
                                        "component_balance": 0.0, "nutrient_accounting": 0.0,
                                        "adjacency_violations": 0.0, "negative_S": 0.0,
                                        "nonfinite_D": 0.0}
        self.n_warnings = 0
        self.last_event_step = -1
        self._check_config()

    # ------------------------------------------------------------------ setup
    def _check_config(self):
        c = self.cfg
        if not (0.0 < c.f_max <= 1.0):
            raise ValueError("f_max must be in (0, 1]")
        if c.diff_B * c.dt * c.connectivity >= 1.0:
            raise ValueError("diff_B * dt * connectivity must be below 1 for a stable diffusion step")
        if c.mu < 0 or c.gamma <= 0 or c.alpha < 0:
            raise ValueError("alpha >= 0, mu >= 0 and gamma > 0 are required")

    @property
    def occupancy(self) -> np.ndarray:
        return self.B > self.cfg.b_min

    def _update_attractant(self):
        """
        Quasi-steady chemoattractant A released by concentrated food:
            0 = D_a Lap(A) - k_a A + max(S - S_thr, 0).
        A decays with distance from food over a range sqrt(D_a / k_a); its relative
        gradient is what the growth front senses (extension; not part of the Tero model).
        The release threshold S_thr keeps the uniform background and the weak noise from
        producing a flat attractant level that would mask distant patches.
        """
        self.A = np.maximum(self._sense_lu.solve(np.maximum(self.S - self.cfg.sense_threshold, 0.0)), 0.0)
        if self.has_inhibitor:
            # living inhibitor-producing bacteria release in proportion to their current density
            living = np.where(self.Kp > 0, self.S / np.maximum(self.Kp, 1e-12), 0.0)
            self.R = self._R_scale * np.maximum(self._inhib_lu.solve(self.inh_src * np.clip(living, 0.0, 1.0)), 0.0)

    def _apply_events(self):
        for (step, patch) in self.cfg.events:
            if step == self.step_count:
                add = gaussian_patch(self.grid, *patch)
                self.S += add
                self.S_ref += add
                self.last_event_step = self.step_count
                print(f"[event] step {step}: nutrient patch added at "
                      f"(fy={patch[0]:.2f}, fx={patch[1]:.2f}), amplitude {patch[2]}")

    # --------------------------------------------------------- network state
    def _open_cells(self) -> np.ndarray:
        """Unoccupied cells that may be entered: not an obstacle, not recently vacated."""
        return ~self.occupancy & (self.step_count - self.vacated > self.cfg.scar_steps)

    def _update_front(self, occ: np.ndarray):
        """Occupied cells with at least one open neighbour (recent scars do not count)."""
        g = self.grid
        openc = self._open_cells()
        front = np.zeros(g.N, dtype=bool)
        m = occ[g.ei] & openc[g.ej]
        front[g.ei[m]] = True
        m = openc[g.ei] & occ[g.ej]
        front[g.ej[m]] = True
        self.front = front

    def _activate_edges(self, occ: np.ndarray):
        """Tubes exist only between two occupied cells; new tubes start at D_init."""
        g, c = self.grid, self.cfg
        active = occ[g.ei] & occ[g.ej]
        newly = active & ~self.active
        if np.any(newly):
            noise = 1.0 + c.D_noise * self.rng.uniform(-1.0, 1.0, newly.sum())
            self.D[newly] = c.D_init * noise
        lost = self.active & ~active
        self.D[lost] = c.D_init          # a tube that disappears loses its memory
        self.Q[~active] = 0.0
        self.active = active

    # --------------------------------------------------- sources and sinks
    def _allocate_currents(self, occ_idx: np.ndarray, labels: np.ndarray, ncomp: int,
                           csize: np.ndarray) -> np.ndarray:
        """
        Current vector I on the occupied nodes, balanced to zero in every component.

        Weights (extension, biologically motivated):
            source weight  s_i = B_i (eps_s + S_i)        feeding sites dominate; all established
                                                           biomass is a weak source of stored cytoplasm
            sink weight    k_i = front_i (c0 + S_i + w_A a_i^3),  a_i = (A_i - min_c A) / (max_c A - min_c A)
                                 + w_d S_i max(0, 1 - B_i / B_ref)
                                                           food-facing growth front and
                                                           under-occupied rich cells
            I0_c = q_flow * sum_c s                        total current injected in component c

        Sampled mode (n_sources > 0 and n_sinks > 0, default; the analogue of the random
        source/sink choice of Tero et al. 2010): in every component, n_sources source
        cells are drawn with probability s / sum s and n_sinks sink cells with
        probability k / sum k. Each drawn source injects I0_c / n_sources and each drawn
        sink withdraws I0_c / n_sinks, so the component sum is exactly zero. Drawing a
        few sites per step concentrates the flow along paths, which is what lets the
        Tero rule reinforce trunks and prune the rest; averaged over many steps the
        flow still follows the weights.

        Diffuse mode (n_sources = 0 or n_sinks = 0): I_i = I0_c (s_i / sum s - k_i / sum k).
        """
        c = self.cfg
        Bl, Sl, fl, Al = self.B[occ_idx], self.S[occ_idx], self.front[occ_idx], self.A[occ_idx]
        Amax = np.zeros(ncomp); Amin = np.full(ncomp, np.inf)
        np.maximum.at(Amax, labels, Al)
        np.minimum.at(Amin, labels, Al)
        Arel = (Al - Amin[labels]) / np.maximum(Amax[labels] - Amin[labels], 1e-12)
        s = Bl * (c.source_eps + Sl)
        k = fl * (c.front_sink_c0 + Sl + c.front_sink_wA * Arel ** 3) \
            + c.demand_sink_weight * Sl * np.maximum(0.0, 1.0 - Bl / c.B_ref)
        if self.has_inhibitor:
            k = k * np.exp(-self.R[occ_idx] / c.inhib_tol)
        sum_s = np.bincount(labels, s, ncomp)
        sum_k = np.bincount(labels, k, ncomp)
        I0 = c.q_flow * sum_s
        valid = (sum_s > 0) & (sum_k > 0) & (csize > 1)
        I = np.zeros(occ_idx.size)
        if c.n_sources > 0 and c.n_sinks > 0:
            for comp in np.flatnonzero(valid):
                nodes = np.flatnonzero(labels == comp)
                ps, pk = s[nodes] / sum_s[comp], k[nodes] / sum_k[comp]
                ns = min(c.n_sources, int(np.count_nonzero(ps)))
                nk = min(c.n_sinks, int(np.count_nonzero(pk)))
                src = self.rng.choice(nodes, size=ns, replace=False, p=ps)
                snk = self.rng.choice(nodes, size=nk, replace=False, p=pk)
                np.add.at(I, src, I0[comp] / ns)
                np.add.at(I, snk, -I0[comp] / nk)
        else:
            v = valid[labels]
            I[v] = I0[labels[v]] * (s[v] / sum_s[labels[v]] - k[v] / sum_k[labels[v]])
        return I

    # ------------------------------------------------------------- flow step
    def _solve_flow(self, occ: np.ndarray):
        g, c = self.grid, self.cfg
        occ_idx = np.flatnonzero(occ)
        n = occ_idx.size
        self.p[:] = np.nan
        self.I[:] = 0.0
        self.comp[:] = -1
        self.Q[:] = 0.0
        self.last_diag = {"residual": 0.0, "divergence_err": 0.0, "component_balance": 0.0}
        if n == 0:
            self.ncomp = 0
            return
        loc = np.full(g.N, -1, dtype=np.int64)
        loc[occ_idx] = np.arange(n)
        ae = np.flatnonzero(self.active)
        a, b = loc[g.ei[ae]], loc[g.ej[ae]]
        w = self.D[ae] / g.el[ae]

        adj = sp.coo_matrix((np.ones(ae.size), (a, b)), shape=(n, n))
        ncomp, labels = connected_components(adj, directed=False)
        csize = np.bincount(labels, minlength=ncomp)
        I = self._allocate_currents(occ_idx, labels, ncomp, csize)

        p, diag = solve_pressure(n, a, b, w, I, labels, ncomp, c.solver)
        Q = w * (p[a] - p[b])
        self.Q[ae] = Q
        self.p[occ_idx] = p
        self.I[occ_idx] = I
        self.comp[occ_idx] = labels
        self.ncomp = ncomp

        # --- diagnostics: component balance, residual, local mass conservation
        bal = float(np.max(np.abs(np.bincount(labels, I, ncomp)))) if ncomp else 0.0
        div = np.bincount(a, Q, n) - np.bincount(b, Q, n)     # net outflow per node
        scale = max(float(np.max(np.abs(I))), 1e-12)
        self.last_diag = {"residual": diag["residual"],
                          "divergence_err": float(np.max(np.abs(div - I))) / scale,
                          "component_balance": bal / scale}
        if np.any(labels[a] != labels[b]):
            warnings.warn("an active edge joins two different components (should be impossible)")
            self.n_warnings += 1
        for key in ("residual", "divergence_err", "component_balance"):
            self.worst[key] = max(self.worst[key], self.last_diag[key])
        if self.last_diag["residual"] > 1e-6 or self.last_diag["divergence_err"] > 1e-6:
            warnings.warn(f"step {self.step_count}: linear-solve residual "
                          f"{self.last_diag['residual']:.2e}, divergence error "
                          f"{self.last_diag['divergence_err']:.2e}")
            self.n_warnings += 1

    # -------------------------------------------------------- adaptation
    def _adapt(self):
        c = self.cfg
        m = self.active
        self.D[m] = adapt_conductivity(self.D[m], self.Q[m], c)
        pruned = m & (self.D < c.prune_threshold)
        self.D[pruned] = c.D_floor
        if not np.all(np.isfinite(self.D)) or np.any(self.D < 0):
            self.worst["nonfinite_D"] += 1
            warnings.warn("non-finite or negative conductivity detected; clipping")
            self.n_warnings += 1
            self.D = np.nan_to_num(self.D, nan=c.D_floor, posinf=c.D_max, neginf=c.D_floor)
            self.D = np.clip(self.D, c.D_floor, c.D_max)

    # ------------------------------------------ nutrient uptake and growth
    def _consume_and_grow(self, occ: np.ndarray) -> float:
        """
        Uptake  U_i = u S_i B_i / (K + B_i) dt      (saturating in biomass, occupied cells only)
        dS_i    = -U_i + r (S_ref_i - S_i) dt
        dB_i    = Y U_i - m B_i dt
        Returns the nutrient consumed in this step.
        """
        c = self.cfg
        U = c.uptake_rate * self.S * self.B / (c.uptake_K + self.B) * c.dt
        U[~occ] = 0.0
        U = np.minimum(U, self.S)
        S_before = self.S.sum()
        self.S -= U
        taken = float(U.sum())
        self.consumed += taken
        if c.regen_rate > 0:
            self.S += c.regen_rate * (self.S_ref - self.S) * c.dt
        if np.any(self.prey_cells):
            # logistic regrowth of grazed colonies; a residual fraction of the capacity
            # cannot be grazed, so a colony recovers after the organism leaves
            pc = self.prey_cells
            self.S[pc] = np.maximum(self.S[pc], c.prey_floor * self.Kp[pc])
            self.S[pc] += c.prey_growth * self.S[pc] * (1.0 - self.S[pc] / self.Kp[pc]) * c.dt
        if np.any(self.S < 0):
            self.worst["negative_S"] += 1
            warnings.warn("negative nutrient concentration detected; clipping")
            self.n_warnings += 1
            self.S = np.clip(self.S, 0.0, None)
        if c.regen_rate == 0 and self.last_event_step != self.step_count and not np.any(self.prey_cells):
            acc = abs((S_before - self.S.sum()) - taken)
            self.worst["nutrient_accounting"] = max(self.worst["nutrient_accounting"], acc)
        self.B += c.yield_coeff * U - c.maintenance * self.B * c.dt
        if self.has_inhibitor:
            tox = c.inhib_toxicity * self.R * self.B * c.dt
            tox = np.minimum(tox, self.B)
            self.B -= tox
            self.toxic_loss += float(tox.sum())
        self.B = np.clip(self.B, 0.0, None)
        return taken

    # --------------------------------------- flux-mediated redistribution
    def _redistribute(self, occ: np.ndarray):
        """
        Flux-mediated biomass transport (extension). Each source cell releases
        min(kappa_T I_i dt, f_max B_i), so no cell loses more than the fraction f_max of
        its biomass; the biomass actually released in a component is given to that
        component's sink cells in proportion to their withdrawal -I_i. Sources lose,
        sinks gain, and the biomass of every component is conserved exactly.
        """
        c = self.cfg
        if self.ncomp == 0:
            return
        I = self.I
        src = occ & (I > 0)
        snk = occ & (I < 0)
        if not np.any(src) or not np.any(snk):
            return
        loss = np.minimum(c.kappa_T * I[src] * c.dt, c.f_max * self.B[src])
        released = np.bincount(self.comp[src], loss, self.ncomp)
        demand = np.bincount(self.comp[snk], -I[snk], self.ncomp)
        self.B[src] -= loss
        js = np.flatnonzero(snk)
        self.B[js] += released[self.comp[js]] * (-I[js]) / np.maximum(demand[self.comp[js]], 1e-300)

    # ------------------------------------------------------- colonization
    def _colonize(self, occ_prev: np.ndarray) -> int:
        """
        Candidate cells: open cells (unoccupied, not an obstacle, not vacated within the
        last scar_steps) with at least one occupied neighbour.
        Colonization rate of candidate j:
            lambda_j = k_col (eps + S_j^eta + lam_grad g_j) min(1, B_donor,max / B_ref)
            g_j = max over occupied neighbours i of max(0, A_j - A_i) / A_i
                                                             (relative attractant gain)
        Probability over dt: P_j = 1 - exp(-lambda_j dt).
        On colonization, b_colonize is moved from the richest occupied neighbour
        (which must keep at least b_keep) into cell j. Biomass is conserved.
        """
        g, c = self.grid, self.cfg
        occ = self.occupancy
        openc = self._open_cells()
        m1 = occ[g.ei] & openc[g.ej]
        m2 = openc[g.ei] & occ[g.ej]
        donors = np.concatenate([g.ei[m1], g.ej[m2]])
        targets = np.concatenate([g.ej[m1], g.ei[m2]])
        if targets.size == 0:
            return 0
        Bdmax = np.zeros(g.N)
        np.maximum.at(Bdmax, targets, self.B[donors])
        # relative attractant gain of the target with respect to its best donor
        gain = np.maximum(0.0, self.A[targets] - self.A[donors]) / (self.A[donors] + 1e-12)
        gmax = np.zeros(g.N)
        np.maximum.at(gmax, targets, gain)
        tu = np.unique(targets)
        St = self.S[tu]
        grad = gmax[tu]
        lam = c.k_col * (c.eps_explore + St ** c.eta + c.lam_grad * grad) \
            * np.minimum(1.0, Bdmax[tu] / c.B_ref)
        if self.has_inhibitor:
            lam = lam * np.exp(-self.R[tu] / c.inhib_tol)
        prob = 1.0 - np.exp(-lam * c.dt)
        chosen = tu[self.rng.random(tu.size) < prob]
        self.rng.shuffle(chosen)
        n_new = 0
        for tnode in chosen:
            nb = g.nbr[tnode]
            nb = nb[nb >= 0]
            nb = nb[occ[nb]]
            if nb.size == 0:
                continue
            d = nb[np.argmax(self.B[nb])]
            if self.B[d] < c.b_colonize + c.b_keep:
                continue
            self.B[d] -= c.b_colonize
            self.B[tnode] = c.b_colonize
            self.age[tnode] = self.step_count
            n_new += 1
        # validation: every newly occupied cell must touch a cell occupied at step start
        new = self.occupancy & ~occ_prev
        if np.any(new):
            nbo = np.where(g.nbr[new] >= 0, occ_prev[np.clip(g.nbr[new], 0, None)], False)
            viol = int(np.sum(~np.any(nbo, axis=1)))
            if viol:
                self.worst["adjacency_violations"] += viol
                warnings.warn(f"{viol} cells colonized without an occupied neighbour")
                self.n_warnings += 1
        return n_new

    # ------------------------------------------------------- sheet spreading
    def _diffuse_biomass(self):
        """
        Local spreading of cytoplasm within the sheet (extension): explicit diffusion of
        B along active edges whose two cells are younger than consolidate_age,
        dB_i = D_B dt sum_j (B_j - B_i) / l_ij^2. Exactly
        conservative; stable because D_B dt times the maximal degree is below 1.
        """
        g, c = self.grid, self.cfg
        young = (self.age >= 0) & (self.step_count - self.age <= c.consolidate_age)
        ae = self.active & young[g.ei] & young[g.ej]      # spreading only inside the young fan
        i, j = g.ei[ae], g.ej[ae]
        f = c.diff_B * c.dt * (self.B[j] - self.B[i]) / g.el[ae] ** 2
        self.B += np.bincount(i, f, g.N) - np.bincount(j, f, g.N)

    # ------------------------------------------------------- consolidation
    def _node_tube_strength(self) -> np.ndarray:
        """Largest conductivity among the active tubes touching each cell."""
        g, c = self.grid, self.cfg
        Dnode = np.zeros(g.N)
        strong = self.active & (self.D >= c.tube_D)
        np.maximum.at(Dnode, g.ei[strong], self.D[strong])
        np.maximum.at(Dnode, g.ej[strong], self.D[strong])
        return Dnode

    def _consolidate(self):
        """
        Consolidation of the sheet into veins (extension). Cells colonized more than
        consolidate_age steps ago that do not carry a reinforced tube drain a fraction
        consolidate_rate dt of their biomass per step down to the thin-film value
        b_sheet; cells older than retract_age with no tube that do not touch a vein cell
        give up all their biomass and are abandoned (they become scars), so veins keep a
        one-cell film flank. The released biomass of each component goes
        to that component's vein cells in proportion to the conductivity of the tube
        they carry, or to its front cells when the component has no reinforced tube.
        Biomass is conserved exactly. The organism is therefore left with veins and an
        advancing fan, as in the living plasmodium, while the film between veins
        disappears behind the fan.
        """
        g, c = self.grid, self.cfg
        if self.ncomp == 0:
            return
        occ = self.occupancy
        Dnode = self._node_tube_strength()
        agec = self.step_count - self.age
        film = occ & (self.age >= 0) & (agec > c.consolidate_age) & (Dnode <= 0.0) & (self.comp >= 0)
        if not np.any(film):
            return
        halo = np.zeros(g.N, dtype=bool)                  # cells touching a vein cell
        veinc = Dnode > 0.0
        m = self.active & (veinc[g.ei] | veinc[g.ej])
        halo[g.ei[m]] = True; halo[g.ej[m]] = True
        retract = film & (agec > c.retract_age) & ~halo
        give = np.where(retract[film], self.B[film],
                        np.minimum(c.consolidate_rate * c.dt * self.B[film],
                                   np.maximum(self.B[film] - c.b_sheet, 0.0)))
        rec = occ & (Dnode > 0.0) & (self.comp >= 0)
        wsum = np.bincount(self.comp[rec], Dnode[rec], self.ncomp)
        nofront = wsum <= 0
        rec_f = occ & self.front & (self.comp >= 0) & nofront[np.clip(self.comp, 0, None)] & (Dnode <= 0.0)
        wf = np.bincount(self.comp[rec_f], np.ones(int(rec_f.sum())), self.ncomp)
        can = (wsum > 0) | (wf > 0)
        idx_film = np.flatnonzero(film)
        take = can[self.comp[idx_film]]
        idx_film, give = idx_film[take], give[take]
        if idx_film.size == 0:
            return
        self.B[idx_film] -= give
        pool = np.bincount(self.comp[idx_film], give, self.ncomp)
        ir = np.flatnonzero(rec)
        self.B[ir] += pool[self.comp[ir]] * Dnode[ir] / np.maximum(wsum[self.comp[ir]], 1e-12)
        jf = np.flatnonzero(rec_f)
        self.B[jf] += pool[self.comp[jf]] / np.maximum(wf[self.comp[jf]], 1e-12)
        gone = idx_film[retract[np.flatnonzero(film)][take]]
        self.B[gone] = 0.0
        self.age[gone] = -1
        self.vacated[gone] = self.step_count

    # ------------------------------------------------------ dead fragments
    def _remove_dead_fragments(self):
        """
        A component that has no reinforced tube and no cell younger than consolidate_age
        is a stranded fragment of film: it is abandoned and its biomass counted as lost
        (extension; a detached fragment of plasmodium dies or becomes a separate
        organism, which this model does not follow).
        """
        c = self.cfg
        if self.ncomp <= 1:
            return
        occ = self.occupancy
        Dnode = self._node_tube_strength()
        young = (self.age >= 0) & (self.step_count - self.age <= c.consolidate_age)
        has_vein = np.zeros(self.ncomp, dtype=bool)
        has_young = np.zeros(self.ncomp, dtype=bool)
        has_vein[self.comp[occ & (Dnode > 0)]] = True
        has_young[self.comp[occ & young]] = True
        # the largest component is never removed
        sizes = np.bincount(self.comp[occ], minlength=self.ncomp)
        keep_main = np.zeros(self.ncomp, dtype=bool); keep_main[np.argmax(sizes)] = True
        dead = ~has_vein & ~has_young & ~keep_main
        cells = occ & dead[np.clip(self.comp, 0, None)] & (self.comp >= 0)
        if np.any(cells):
            self.lost_biomass += float(self.B[cells].sum())
            self.B[cells] = 0.0
            self.age[cells] = -1
            self.vacated[cells] = self.step_count

    # ---------------------------------------------------------- retraction
    def _retract(self, occ_prev: np.ndarray):
        """
        Cells whose biomass fell below b_min are abandoned and release their remainder
        to their richest occupied neighbour (biomass conserved; lost only when no
        neighbour is occupied). A cell carrying a reinforced tube (adjacent active edge
        with D >= tube_D) is kept: it draws biomass from its richest neighbour up to
        b_keep when that neighbour can spare it (tube-wall maintenance, extension).
        """
        g, c = self.grid, self.cfg
        occ = self.occupancy
        lost = occ_prev & ~occ
        if not np.any(lost):
            return
        Dnode = self._node_tube_strength()
        for node in np.flatnonzero(lost):
            nb = g.nbr[node]
            nb = nb[nb >= 0]
            nb = nb[occ[nb]]
            if nb.size == 0:
                self.lost_biomass += self.B[node]
                self.B[node] = 0.0
                self.age[node] = -1
                self.vacated[node] = self.step_count
                continue
            rich = nb[np.argmax(self.B[nb])]
            need = c.b_keep - self.B[node]
            if Dnode[node] > 0 and self.B[rich] - need >= c.b_keep:
                self.B[rich] -= need
                self.B[node] += need
            else:
                self.B[rich] += self.B[node]
                self.B[node] = 0.0
                self.age[node] = -1
                self.vacated[node] = self.step_count

    # ------------------------------------------------------------ metrics
    def _record(self):
        g, c = self.grid, self.cfg
        occ = self.occupancy
        tubes = self.active & (self.D >= c.prune_threshold)
        nA = int(self.active.sum())
        m = self.metrics
        m["step"].append(self.step_count)
        m["time"].append(self.t)
        m["occupied_area"].append(int(occ.sum()))
        m["biomass"].append(float(self.B.sum()))
        m["network_length"].append(float(g.el[tubes].sum()))
        m["n_tubes"].append(int(tubes.sum()))
        m["n_active_edges"].append(nA)
        m["mean_D"].append(float(self.D[self.active].mean()) if nA else 0.0)
        m["consumed"].append(self.consumed)
        m["n_components"].append(self.ncomp)
        m["cost_Dl"].append(float((self.D[tubes] * g.el[tubes]).sum()))
        Qa, Da, la = self.Q[self.active], self.D[self.active], g.el[self.active]
        m["dissipation"].append(float((Qa ** 2 * la / Da).sum()) if nA else 0.0)
        m["residual"].append(self.last_diag.get("residual", 0.0))
        m["divergence_err"].append(self.last_diag.get("divergence_err", 0.0))
        m["component_balance"].append(self.last_diag.get("component_balance", 0.0))

    # --------------------------------------------------------------- step
    def step(self):
        """Advance the coupled model by one time step dt."""
        self._apply_events()
        occ = self.occupancy
        occ_prev = occ.copy()
        # 1. network morphogenesis: tubes exist between occupied cells
        self._update_front(occ)
        self._activate_edges(occ)
        # 2. Tero model: pressures, fluxes, conductivity adaptation, pruning
        self._solve_flow(occ)
        self._adapt()
        # 3. nutrient consumption and biomass growth (extension)
        self._consume_and_grow(occ)
        self._update_attractant()
        # 4. flux-mediated biomass redistribution (extension)
        self._redistribute(occ)
        self._diffuse_biomass()
        self._consolidate()
        self._remove_dead_fragments()
        # 5. spatial expansion of the growth front (extension)
        self._colonize(occ_prev)
        # 6. retraction from starved cells (extension)
        self._retract(occ_prev)
        self.step_count += 1
        self.t += self.cfg.dt
        self._record()

    def run(self, n_steps: Optional[int] = None, callback=None):
        n = self.cfg.n_steps if n_steps is None else n_steps
        for _ in range(n):
            self.step()
            if callback is not None:
                callback(self)

    # ------------------------------------------------------------ helpers
    def node_flux_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Mean signed flux vector at each node (for arrows)."""
        g = self.grid
        qx, qy = self.Q * g.ux, self.Q * g.uy
        vx = 0.5 * (np.bincount(g.ei, qx, g.N) + np.bincount(g.ej, qx, g.N))
        vy = 0.5 * (np.bincount(g.ei, qy, g.N) + np.bincount(g.ej, qy, g.N))
        return vx, vy

    def save_metrics(self, path: str):
        keys = METRIC_KEYS
        with open(path, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(keys)
            for row in zip(*[self.metrics[k] for k in keys]):
                wr.writerow(row)

    def report(self):
        m = self.metrics
        w = self.worst
        print("\n=== Simulation summary ===")
        print(f"steps: {self.step_count}, time: {self.t:.1f}")
        print(f"occupied cells: {m['occupied_area'][-1]}, biomass: {m['biomass'][-1]:.2f}, "
              f"biomass lost at retraction: {self.lost_biomass:.3f}")
        print(f"network tubes (D >= prune_threshold): {m['n_tubes'][-1]}, "
              f"total length: {m['network_length'][-1]:.1f}, mean D: {m['mean_D'][-1]:.3f}")
        print(f"nutrient consumed: {self.consumed:.2f}, components: {self.ncomp}, "
              f"biomass killed by inhibitor: {self.toxic_loss:.2f}")
        print("=== Numerical validation (worst case over the run) ===")
        print(f"max relative linear-solve residual     : {w['residual']:.2e}")
        print(f"max relative local mass-balance error  : {w['divergence_err']:.2e}")
        print(f"max relative component current balance : {w['component_balance']:.2e}")
        print(f"max nutrient accounting mismatch       : {w['nutrient_accounting']:.2e}")
        print(f"colonizations without neighbour        : {int(w['adjacency_violations'])}")
        print(f"negative nutrient events               : {int(w['negative_S'])}")
        print(f"non-finite conductivity events         : {int(w['nonfinite_D'])}")
        print(f"conductivity range                     : [{self.D.min():.2e}, {self.D.max():.2e}]")
        print(f"warnings raised                        : {self.n_warnings}")


# =============================================================================
# 6. Visualization
# =============================================================================
class Visualizer:
    """Four-panel figure with reusable artists; works for animation and snapshots."""

    def __init__(self, sim: PhysarumSimulation, title: str = ""):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        self.plt = plt
        self.sim, g, c = sim, sim.grid, sim.cfg
        self.fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
        (self.axA, self.axB), (self.axC, self.axD) = axes
        self.fig.suptitle(title, fontsize=12)
        kw = dict(interpolation="nearest", origin="upper",
                  extent=(-0.5, g.W - 0.5, g.H - 0.5, -0.5))
        self.Smax = max(float(sim.S.max()), 1e-6)

        # Panel A: nutrient + network
        self.imS = self.axA.imshow(sim.S.reshape(g.H, g.W), cmap="YlGn", vmin=0, vmax=self.Smax, **kw)
        self.fig.colorbar(self.imS, ax=self.axA, fraction=0.046, pad=0.02, label="nutrient S")
        self.segs = np.stack([np.stack([g.x[g.ei], g.y[g.ei]], 1),
                              np.stack([g.x[g.ej], g.y[g.ej]], 1)], 1)
        self.imOcc = self.axA.imshow(np.zeros((g.H, g.W, 4)), zorder=2, **kw)   # translucent body
        self.lc = LineCollection([], linewidths=1.0, zorder=3)
        self.axA.add_collection(self.lc)
        self.sc_new = self.axA.scatter([], [], s=2.5, c="white", edgecolors="k", linewidths=0.2,
                                       zorder=4, label="newly colonized")
        obst = np.zeros((g.H, g.W, 4)); obst[g.obstacle.reshape(g.H, g.W)] = (0.25, 0.25, 0.28, 1.0)
        self.imR = self.axA.imshow(np.zeros((g.H, g.W, 4)), zorder=2.5, **kw)   # inhibitor overlay
        for ax in (self.axA, self.axB, self.axC):
            ax.imshow(obst, zorder=6, **kw)                                      # obstacles
        for (fy, fx, amp, sig, inh) in c.prey:
            cx_, cy_ = fx * (g.W - 1), fy * (g.H - 1)
            self.axA.plot(cx_, cy_, marker="x" if inh else "o", ms=9, mew=1.6,
                          mec="#b00000" if inh else "#1b5e20", mfc="none", ls="", zorder=7)
        if c.prey:
            self.axA.plot([], [], marker="o", mec="#1b5e20", mfc="none", ls="", label="prey colony")
            self.axA.plot([], [], marker="x", mec="#b00000", mfc="none", ls="",
                          label="prey colony with inhibitor")
        self.axA.plot(*sim.inoc_xy, marker="*", ms=14, mec="k", mfc="gold", ls="", zorder=5,
                      label="inoculum")
        self.axA.legend(loc="upper right", fontsize=7, framealpha=0.8)
        self.axA.set_title("A. Nutrient landscape and transport network (width ~ conductivity)",
                           fontsize=9)

        # Panel B: biomass + front
        self.imB = self.axB.imshow(sim.B.reshape(g.H, g.W), cmap="magma", vmin=0,
                                   vmax=max(c.inoc_biomass, 0.5), **kw)
        self.fig.colorbar(self.imB, ax=self.axB, fraction=0.046, pad=0.02, label="biomass B")
        self.sc_front = self.axB.scatter([], [], s=2, c="cyan", marker="s", zorder=3,
                                         label="growth front")
        self.axB.legend(loc="upper right", fontsize=7, framealpha=0.8)
        self.axB.set_title("B. Biomass and advancing growth front", fontsize=9)

        # Panel C: pressure + flux arrows
        self.imP = self.axC.imshow(np.ma.masked_invalid(sim.p.reshape(g.H, g.W)),
                                   cmap="coolwarm", **kw)
        self.cbP = self.fig.colorbar(self.imP, ax=self.axC, fraction=0.046, pad=0.02,
                                     label="pressure p (gauge: p_ref = 0)")
        st = c.quiver_stride
        yy, xx = np.mgrid[0:g.H:st, 0:g.W:st]
        self.q_idx = (yy * g.W + xx).ravel()
        self.quiv = self.axC.quiver(xx.ravel(), yy.ravel(), np.zeros(self.q_idx.size),
                                    np.zeros(self.q_idx.size), angles="xy", scale_units="xy",
                                    scale=1.0, width=0.004, color="k", zorder=3)
        self.axC.set_title("C. Pressure field and net flux direction (arrows)", fontsize=9)
        for ax in (self.axA, self.axB, self.axC):
            ax.set_xlim(-0.5, g.W - 0.5); ax.set_ylim(g.H - 0.5, -0.5)
            ax.set_xticks([]); ax.set_yticks([])

        # Panel D: metrics
        self.metric_names = ["occupied_area", "network_length", "consumed", "mean_D",
                             "n_tubes", "n_components"]
        self.lines = {}
        for name in self.metric_names:
            (ln,) = self.axD.plot([], [], lw=1.4, label=name)
            self.lines[name] = ln
        self.axD.set_xlabel("time"); self.axD.set_ylabel("value / running maximum")
        self.axD.set_ylim(-0.02, 1.05)
        self.axD.legend(loc="lower right", fontsize=7, ncol=2)
        self.axD.grid(alpha=0.3)
        self.axD.set_title("D. Network evolution metrics (normalised)", fontsize=9)
        self.txt = self.axD.text(0.02, 0.97, "", transform=self.axD.transAxes, fontsize=7.5,
                                 va="top", family="monospace",
                                 bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        self.fig.tight_layout(rect=(0, 0, 1, 0.96))
        self.refresh()

    def refresh(self):
        sim, g, c = self.sim, self.sim.grid, self.sim.cfg
        H, W = g.H, g.W
        # A
        self.imS.set_data(sim.S.reshape(H, W))
        tubes = sim.active & (sim.D >= c.prune_threshold)
        if np.any(tubes):
            Dt = sim.D[tubes]
            Dref = max(float(np.percentile(Dt, 97)), c.D_init)
            rel = np.clip(Dt / Dref, 0.0, 1.0)
            self.lc.set_segments(self.segs[tubes])
            self.lc.set_linewidths(0.2 + 3.5 * np.sqrt(rel))
            cols = np.zeros((rel.size, 4)); cols[:, 0] = 0.45; cols[:, 1] = 0.10; cols[:, 2] = 0.05
            cols[:, 3] = 0.08 + 0.92 * rel ** 1.5
            self.lc.set_color(cols)
        else:
            self.lc.set_segments([])
        body = np.zeros((H, W, 4)); occ2 = sim.occupancy.reshape(H, W)
        bb = np.clip(sim.B.reshape(H, W) / max(c.inoc_biomass, 0.5), 0.0, 1.0)
        body[..., 0], body[..., 1], body[..., 2] = 0.95, 0.75, 0.10   # amber plasmodium
        body[..., 3] = np.where(occ2, 0.12 + 0.75 * np.sqrt(bb), 0.0)  # thin film faint, veins and fan dense
        self.imOcc.set_data(body)
        if sim.has_inhibitor:
            rr = np.zeros((H, W, 4)); rr[..., 0] = 0.85
            rr[..., 3] = 0.55 * np.clip(sim.R.reshape(H, W) / max(float(sim.R.max()), 1e-12), 0, 1)
            self.imR.set_data(rr)
        new = (sim.age >= 0) & (sim.step_count - sim.age <= c.new_age) & sim.occupancy
        self.sc_new.set_offsets(np.c_[g.x[new], g.y[new]] if np.any(new) else np.empty((0, 2)))
        # B
        self.imB.set_data(sim.B.reshape(H, W))
        fr = sim.front
        self.sc_front.set_offsets(np.c_[g.x[fr], g.y[fr]] if np.any(fr) else np.empty((0, 2)))
        # C
        pm = np.ma.masked_invalid(sim.p.reshape(H, W))
        self.imP.set_data(pm)
        if pm.count():
            lo, hi = np.percentile(pm.compressed(), [2, 98])
            if hi <= lo:
                hi = lo + 1e-9
            self.imP.set_clim(lo, hi)
        vx, vy = sim.node_flux_vectors()
        U, V = vx[self.q_idx], vy[self.q_idx]
        mag = np.hypot(U, V); vmax = float(mag.max())
        if vmax > 0:
            sc = 0.85 * c.quiver_stride / vmax
            U, V = U * sc, V * sc
        self.quiv.set_UVC(U, V)
        # D
        m = sim.metrics
        if m["time"]:
            t = np.asarray(m["time"])
            for name in self.metric_names:
                arr = np.asarray(m[name], dtype=float)
                mx = float(arr.max()) if arr.size else 0.0
                self.lines[name].set_data(t, arr / mx if mx > 0 else arr)
            self.axD.set_xlim(0, max(float(t[-1]), 1e-6) * 1.02)
            self.txt.set_text(
                f"step {sim.step_count:5d}  t={sim.t:7.1f}\n"
                f"area {m['occupied_area'][-1]:5d}  tubes {m['n_tubes'][-1]:5d}\n"
                f"length {m['network_length'][-1]:7.1f} meanD {m['mean_D'][-1]:7.2f}\n"
                f"consumed {m['consumed'][-1]:7.2f} comps {m['n_components'][-1]:3d}\n"
                f"resid {m['residual'][-1]:.1e} divErr {m['divergence_err'][-1]:.1e}")
        return [self.imS, self.imOcc, self.imR, self.lc, self.sc_new, self.imB, self.sc_front, self.imP, self.quiv,
                self.txt] + list(self.lines.values())

    def animate(self, n_frames: int, interval: int = 30, save: Optional[str] = None, fps: int = 20):
        from matplotlib.animation import FuncAnimation, PillowWriter, writers
        spf = self.sim.cfg.steps_per_frame

        def update(_frame):
            for _ in range(spf):
                self.sim.step()
            return self.refresh()

        anim = FuncAnimation(self.fig, update, frames=n_frames, interval=interval,
                             blit=False, repeat=False)
        if save:
            ext = os.path.splitext(save)[1].lower()
            if ext == ".gif":
                writer = PillowWriter(fps=fps)
            elif ext in (".mp4", ".mov", ".m4v"):
                if not writers.is_available("ffmpeg"):
                    sys.exit("ffmpeg is not available; install it or export a .gif instead")
                writer = writers["ffmpeg"](fps=fps, bitrate=2400)
            else:
                sys.exit("unsupported export format; use .gif or .mp4")
            t0 = time.time()
            anim.save(save, writer=writer, dpi=90)
            print(f"saved {save} ({n_frames} frames, {time.time() - t0:.1f} s)")
        else:
            self.plt.show()
        return anim


# =============================================================================
# 7. Validation tests
# =============================================================================
def test_two_path_competition(mu: float, n_iter: int = 300, verbose: bool = True) -> float:
    """
    Tiny graph: source node 0, sink node 1, a direct tube 0-1 (length 1) and a
    detour 0-2-3-1 (total length 3). With flux-driven adaptation the short route
    should win. Returns the ratio D_detour / D_direct at the end.
    """
    cfg = Config(mu=mu, alpha=1.0, gamma=0.1, dt=0.5, Q0=1.0, D_init=0.5, D_floor=1e-6)
    ei = np.array([0, 0, 2, 3]); ej = np.array([1, 2, 3, 1]); el = np.ones(4)
    D = np.full(4, 0.5)
    I = np.array([1.0, -1.0, 0.0, 0.0])
    labels = np.zeros(4, dtype=np.int64)
    for _ in range(n_iter):
        w = D / el
        p, diag = solve_pressure(4, ei, ej, w, I, labels, 1)
        Q = w * (p[ei] - p[ej])
        D = adapt_conductivity(D, Q, cfg)
    ratio = float(D[1] / D[0])
    if verbose:
        print(f"[two-path test] mu={mu}: D_direct={D[0]:.3f}, D_detour={D[1]:.2e}, "
              f"ratio={ratio:.2e}, residual={diag['residual']:.1e}, "
              f"flux conservation at node 0: {abs(Q[0] + Q[1] - 1.0):.1e}")
    return ratio


def test_reinforcement_and_decay(verbose: bool = True) -> bool:
    """A tube with flow should strengthen; a tube without flow should decay."""
    cfg = Config(alpha=1.0, mu=1.0, gamma=0.1, dt=0.5, D_floor=1e-4)
    D = np.array([0.5, 0.5])
    Q = np.array([3.0, 0.0])
    for _ in range(200):
        D = adapt_conductivity(D, Q, cfg)
    ok = (abs(D[0] - 30.0) < 0.1) and (D[1] <= cfg.D_floor * 1.01)
    if verbose:
        print(f"[reinforcement test] D(flow=3) -> {D[0]:.3f} (expected alpha|Q|/gamma = 30), "
              f"D(no flow) -> {D[1]:.1e}: {'PASS' if ok else 'FAIL'}")
    return ok


def test_disconnected_components(verbose: bool = True) -> bool:
    """Two disconnected pairs; each pair balanced; no flow between pairs."""
    ei = np.array([0, 2]); ej = np.array([1, 3]); w = np.array([1.0, 2.0])
    I = np.array([1.0, -1.0, 0.5, -0.5])
    labels = np.array([0, 0, 1, 1])
    p, diag = solve_pressure(4, ei, ej, w, I, labels, 2)
    Q = w * (p[ei] - p[ej])
    ok = diag["residual"] < 1e-10 and abs(Q[0] - 1.0) < 1e-10 and abs(Q[1] - 0.5) < 1e-10
    if verbose:
        print(f"[disconnected test] Q pair1={Q[0]:.3f} (1.0), Q pair2={Q[1]:.3f} (0.5), "
              f"residual={diag['residual']:.1e}: {'PASS' if ok else 'FAIL'}")
    return ok


def run_validation_suite(args) -> bool:
    print("=== Unit tests of the transport core ===")
    ok = test_reinforcement_and_decay()
    ok &= test_disconnected_components()
    r1 = test_two_path_competition(1.0)
    r2 = test_two_path_competition(2.0)
    ok &= (r1 < 1e-2) and (r2 < r1)
    print(f"[two-path test] detour suppressed for mu=1 and faster for mu=2: "
          f"{'PASS' if (r1 < 1e-2 and r2 < r1) else 'FAIL'}")
    print("\n=== Full-model smoke test (scenario A, 150 steps, 40x60 grid) ===")
    cfg = make_config("A", H=40, W=60, n_steps=150, seed=args.seed, connectivity=4)
    sim = PhysarumSimulation(cfg)
    sim.run()
    sim.report()
    w = sim.worst
    checks = {
        "linear-solve residual < 1e-8": w["residual"] < 1e-8,
        "local mass balance < 1e-8": w["divergence_err"] < 1e-8,
        "component current balance < 1e-10": w["component_balance"] < 1e-10,
        "nutrient accounting exact": w["nutrient_accounting"] < 1e-9,
        "no colonization without neighbour": w["adjacency_violations"] == 0,
        "nutrient nonnegative": w["negative_S"] == 0 and np.all(sim.S >= 0),
        "conductivity finite and positive": np.all(np.isfinite(sim.D)) and np.all(sim.D > 0),
        "organism expanded": sim.metrics["occupied_area"][-1] > sim.metrics["occupied_area"][0],
        "nutrient consumed > 0": sim.consumed > 0,
    }
    for name, passed in checks.items():
        print(f"  {name:40s}: {'PASS' if passed else 'FAIL'}")
        ok &= bool(passed)
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


def compare_mu(args):
    """Run scenario B headless for mu = 1 and mu = 2 and compare the final networks."""
    print("=== Comparison of the flux exponent mu (scenario B, headless) ===")
    rows = []
    for mu in (1.0, 2.0):
        cfg = make_config("B", H=args.H, W=args.W, n_steps=args.steps, seed=args.seed, mu=mu,
                          Q0=args.Q0)
        sim = PhysarumSimulation(cfg)
        t0 = time.time()
        sim.run()
        m = sim.metrics
        rows.append((mu, m["n_tubes"][-1], m["n_active_edges"][-1], m["network_length"][-1],
                     m["mean_D"][-1], m["cost_Dl"][-1], m["dissipation"][-1], time.time() - t0))
    print(f"{'mu':>4} {'tubes':>7} {'active':>7} {'length':>9} {'meanD':>9} {'cost':>10} "
          f"{'dissip':>10} {'sec':>6}")
    for r in rows:
        print(f"{r[0]:4.1f} {r[1]:7d} {r[2]:7d} {r[3]:9.1f} {r[4]:9.2f} {r[5]:10.1f} "
              f"{r[6]:10.3f} {r[7]:6.1f}")
    print("Interpretation: with mu = 2 fewer tubes carry a larger share of the flow "
          "(stronger winner-take-all competition), so the surviving network is sparser "
          "and has a lower dissipation per unit conductivity cost.")


# =============================================================================
# 8. Command-line interface
# =============================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Physarum polycephalum adaptive-network simulation")
    ap.add_argument("--scenario", default="B", choices=list("ABCDEabcde"))
    ap.add_argument("--H", type=int, default=140, help="grid rows")
    ap.add_argument("--W", type=int, default=180, help="grid columns")
    ap.add_argument("--steps", type=int, default=2500, help="number of simulation steps")
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mu", type=float, default=1.5, help="flux exponent of the Tero rule")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--Q0", type=float, default=1.0)
    ap.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    ap.add_argument("--solver", default="direct", choices=["direct", "cg"])
    ap.add_argument("--n-sources", type=int, default=4, help="source sites per step (0 = diffuse)")
    ap.add_argument("--n-sinks", type=int, default=2, help="sink sites per step (0 = diffuse)")
    ap.add_argument("--regen", type=float, default=0.0, help="nutrient regeneration rate")
    ap.add_argument("--steps-per-frame", type=int, default=2)
    ap.add_argument("--save", default=None, help="export animation to .gif or .mp4")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--no-anim", action="store_true", help="run headless, no animation window")
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="in headless mode save a PNG of the four panels every N steps")
    ap.add_argument("--outdir", default="physarum_output")
    ap.add_argument("--test", action="store_true", help="run the validation suite and exit")
    ap.add_argument("--compare-mu", action="store_true", help="compare mu=1 and mu=2 and exit")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.test:
        sys.exit(0 if run_validation_suite(args) else 1)
    if args.compare_mu:
        compare_mu(args)
        return

    headless = args.no_anim or args.save is not None
    import matplotlib
    if headless:
        matplotlib.use("Agg")

    cfg = make_config(args.scenario, H=args.H, W=args.W, n_steps=args.steps, dt=args.dt,
                      seed=args.seed, mu=args.mu, alpha=args.alpha, gamma=args.gamma, Q0=args.Q0,
                      connectivity=args.connectivity, solver=args.solver,
                      n_sources=args.n_sources, n_sinks=args.n_sinks, regen_rate=args.regen,
                      steps_per_frame=args.steps_per_frame)
    sim = PhysarumSimulation(cfg)
    title = (f"Physarum polycephalum adaptive network, scenario {args.scenario.upper()} "
             f"(mu={cfg.mu}, alpha={cfg.alpha}, gamma={cfg.gamma}, grid {cfg.H}x{cfg.W})")
    os.makedirs(args.outdir, exist_ok=True)

    if args.no_anim:
        viz = Visualizer(sim, title) if args.snapshot_every > 0 else None
        t0 = time.time()
        for k in range(cfg.n_steps):
            sim.step()
            if viz is not None and (sim.step_count % args.snapshot_every == 0
                                    or sim.step_count == cfg.n_steps):
                viz.refresh()
                fn = os.path.join(args.outdir,
                                  f"scenario_{args.scenario.upper()}_step{sim.step_count:05d}.png")
                viz.fig.savefig(fn, dpi=100)
                print(f"saved {fn}")
        print(f"run time: {time.time() - t0:.1f} s for {cfg.n_steps} steps")
    else:
        viz = Visualizer(sim, title)
        n_frames = max(1, cfg.n_steps // cfg.steps_per_frame)
        viz.animate(n_frames, save=args.save, fps=args.fps)

    sim.report()
    csv_path = os.path.join(args.outdir, f"metrics_scenario_{args.scenario.upper()}.csv")
    sim.save_metrics(csv_path)
    print(f"metrics written to {csv_path}")


if __name__ == "__main__":
    main()
