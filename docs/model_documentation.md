# Physarum polycephalum adaptive transport-network simulation: model documentation

Companion document to physarum_sim.py. The script implements the Tero adaptive transport-network model on a two-dimensional lattice and adds a spatial colonization layer so that the simulated organism expands from an inoculum toward nutrient patches, builds and reinforces transport tubes, prunes unused tubes, depletes nutrients and reorganizes when the nutrient landscape changes.

## A. Mathematical model

Summary:

- The transport core (pressures, fluxes, conductivity adaptation) is the model of Tero, Kobayashi and Nakagaki (2007) and Tero et al. (2010).
- Everything that moves the organism in space (biomass field, colonization rule, attractant field, source and sink allocation, biomass transport along the flow, retraction) is a phenomenological extension written for this script.
- All state variables live on a regular H by W lattice (140 by 180 by default); every lattice cell is a node and every pair of 8-neighbours (optionally 4-neighbours) is a potential tube. Cells inside obstacles have no tubes and cannot be entered.
- Scenario geometry is written for a reference width of 90 cells; make_config multiplies every length parameter by scale = W / 90 and rescales the per-cell rate parameters so that the dynamics do not depend on the grid resolution.

### A.1 Spatial domain and nutrient landscape (extension)

The nutrient field S(x, y, t) is initialized as the sum of a uniform baseline, a set of Gaussian patches with amplitude a_k, centre (y_k, x_k) and width sigma_k, and a correlated noise field obtained by Gaussian smoothing of white noise (correlation length noise_corr cells, amplitude noise_amp). Negative values are clipped to zero. A small patch is placed under the inoculum to represent the food on which the organism is seeded. The field is reproducible through the seed parameter.

Nutrient uptake happens only in occupied cells and saturates in biomass:

U_i = u S_i B_i / (K + B_i) dt, with U_i capped at S_i, and U_i = 0 where the cell is unoccupied.

S_i decreases by U_i. With regen_rate r greater than zero, S_i relaxes toward its initial value: dS_i = r (S_ref,i - S_i) dt. Scripted events add a Gaussian patch at a given step (scenario C). The script checks every step that the total decrease of S equals the total uptake when no regeneration or event is active.

Obstacles (extension): cfg.obstacles lists impassable rectangles. Edges that touch an obstacle cell are removed from the edge list at construction, so obstacles block transport, attractant diffusion, inhibitor diffusion and colonization at once; the nutrient is zero inside them.

Bacterial prey (extension, scenario E): each colony is a Gaussian footprint with a carrying capacity K_p. The colony adds its capacity to S at start and regrows logistically where it has been grazed, dS = r_p S (1 - S / K_p) dt, with S never below prey_floor times K_p so that a colony recovers after the organism leaves. Colonies flagged as inhibitor producers release an inhibitor in proportion to their living density; the inhibitor field R solves the same quasi-steady diffusion equation as the attractant with a shorter range (inhib_range) and is normalised so that R = 1 at the centre of an intact colony. The inhibitor multiplies the colonization rate and the front sink weight by exp(-R / inhib_tol) and kills biomass at the rate inhib_toxicity R B. The organism therefore senses the colony as food (attractant) and as a threat (inhibitor) at the same time; whether it grazes a colony or avoids it is an emergent outcome of that trade-off, and the biomass killed is reported.

### A.2 Graph, Laplacian, pressure and flux (Tero model)

The graph G(t) = (V, E(t)) has one node per cell. A potential edge ij has Euclidean length l_ij (1 for axial neighbours, square root of 2 for diagonal neighbours). An edge is active when both end cells are occupied. Each active edge carries a conductivity D_ij(t) and a signed flux Q_ij(t).

The conductivity-weighted Laplacian on the occupied nodes is L_ii = sum over active neighbours j of D_ij / l_ij and L_ij = -D_ij / l_ij. Each undirected edge is stored once in index arrays (ei, ej, el) and enters the sparse matrix with four entries, so no edge is double-counted. The pressure vector p solves L p = I, and the flux on edge ij is Q_ij = (D_ij / l_ij)(p_i - p_j). These three relations are the Poiseuille-flow part of the Tero model.

Gauge: L has one zero eigenvalue per connected component. The script adds 1 to the diagonal entry of one reference node per component. Because the current I sums to zero inside each component, the reference node ends at p = 0 and the regularised system has exactly the solution of L p = I (the artificial ground carries no current). Connected components are found with scipy.sparse.csgraph.connected_components on the active edges; a cell with no active edge forms its own component and receives no current. No term couples different components, so disconnected regions cannot exchange fluid. The linear system is solved with a sparse LU factorisation (scipy.sparse.linalg.splu); a conjugate-gradient option with Jacobi preconditioning exists for large grids.

### A.3 Conductivity adaptation (Tero model)

dD_ij / dt = alpha |Q_ij / Q0|^mu - gamma D_ij.

The script integrates this equation with an exponential integrator: the forcing term alpha |Q/Q0|^mu is frozen over the step and the linear decay is integrated exactly, D(t + dt) = D e^(-gamma dt) + alpha |Q/Q0|^mu (1 - e^(-gamma dt)) / gamma. The update is unconditionally stable and keeps D positive. D is then clipped to [D_floor, D_max]. Soft pruning: an active edge whose conductivity falls below prune_threshold is reset to D_floor; it stays in the Laplacian with negligible conductance, is no longer counted as a network tube, and can regrow if flux returns. A tube that becomes inactive because one of its cells is abandoned is reset to D_init and loses its memory. A newly activated tube receives D_init multiplied by a small random factor (D_noise) to break lattice symmetry.

### A.4 Source and sink allocation (extension)

The Tero model needs a current vector I; the original papers drive flow between food sources (in Tero et al. 2010, a random pair of food sources per iteration). On a growing organism the script defines two weights on occupied cells:

- source weight s_i = B_i (eps_s + S_i): cells that feed are strong sources of cytoplasm, and every established cell is a weak source of stored cytoplasm;
- sink weight k_i = front_i (c0 + S_i + w_A a_i^3) + w_d S_i max(0, 1 - B_i / B_ref): growth-front cells are sinks, with extra weight where the attractant a_i is highest (a_i is the attractant level rescaled between 0 and 1 within the component), and nutrient-rich cells with little biomass are sinks (growth demand).

The total current of a component is I0_c = q_flow times the sum of s over the component. In the default sampled mode, n_sources source cells and n_sinks sink cells are drawn per step with probabilities proportional to s and k; each drawn source injects I0_c / n_sources and each drawn sink withdraws I0_c / n_sinks, so the component sum of I is exactly zero. In the diffuse mode (n_sources = 0 or n_sinks = 0), I_i = I0_c (s_i / sum s - k_i / sum k). The sampled mode is the one that produces sparse trunks: with hundreds of sources and sinks spread over a sheet, the diffuse mode gives every lattice edge a small flux and nothing is ever pruned. Averaged over many steps the sampled flow follows the same weights.

Limitations of this rule: the choice of which cells source and sink cytoplasm is a modelling decision that no measurement supports; the flow direction is steady within a step, whereas real shuttle streaming reverses every one to two minutes; and the current magnitude q_flow is an illustrative scale.

### A.5 Chemoattractant field (extension)

The front senses distant food through a quasi-steady attractant A released by concentrated nutrient: 0 = D_a Laplacian(A) - k_a A + max(S - S_thr, 0), with zero-flux boundaries. The range of the attractant is the square root of D_a / k_a (20 cells by default). The operator is factorised once at start-up and solved every step. Only the relative gradient of A is used, so distant patches act on the front even where A is small. The release threshold S_thr stops the uniform background from producing a flat attractant level that would mask distant patches.

### A.6 Biomass, colonization, transport and retraction (extension)

Biomass field: B(x, y, t) on the lattice. A cell is occupied when B exceeds b_min. The inoculum is a disk of radius inoc_radius with B = inoc_biomass. Growth and maintenance: dB_i = Y U_i - m B_i dt.

Colonization: a candidate is an unoccupied cell j with at least one occupied neighbour. Its colonization rate is lambda_j = k_col (eps + S_j^eta + lam_grad g_j) min(1, B_donor,max / B_ref), where g_j is the largest relative attractant gain max(0, A_j - A_i) / A_i over occupied neighbours i, and B_donor,max is the largest biomass among those neighbours. The colonization probability over a step is P_j = 1 - exp(-lambda_j dt). On colonization, b_colonize is moved from the richest occupied neighbour into cell j; the donor must keep at least b_keep, otherwise the attempt fails. Biomass is conserved by construction, and the script verifies every step that no newly occupied cell lacks a neighbour that was occupied at the start of the step.

Transport-mediated supply: dB_i = -f_c kappa_T I_i dt, so sources lose biomass and sinks gain it along the solved flow. The factor f_c, at most 1, is chosen per component so that no source loses more than the fraction f_max of its biomass; since the same factor multiplies every I_i of the component and the component sum of I is zero, biomass is conserved exactly within each component.

Transport-mediated supply, capped per source: each source cell releases min(kappa_T I_i dt, f_max B_i) and the biomass released in a component is distributed to that component's sink cells in proportion to their withdrawal; biomass is conserved exactly in each component. Several source cells (n_sources, default 4) and a few sink cells (n_sinks, default 2) are drawn per step.

Sheet spreading: biomass diffuses along active edges whose two cells are younger than consolidate_age, dB_i = D_B dt sum_j (B_j - B_i) / l_ij^2. This spreads the cytoplasm delivered to the tip across the advancing fan; the step is conservative and stable because D_B dt times the degree is below 1.

Consolidation into veins: cells older than consolidate_age that carry no reinforced tube (no adjacent active edge with D at least tube_D) drain a fraction consolidate_rate dt of their biomass per step down to a thin-film value b_sheet; cells older than retract_age that still carry no tube and do not touch a vein cell give up all their biomass and are abandoned, becoming scars that are neither counted as front nor recolonized for scar_steps. The released biomass of a component goes to its vein cells in proportion to the conductivity they carry, or to its front cells when it has no vein. Components with no vein cell and no young cell are stranded fragments and are abandoned (their biomass is counted as lost). Together these rules reproduce the morphology of the living plasmodium: an advancing fan, a vein network behind it with a one-cell film flank, and no persistent sheet between the veins.

Retraction: a cell whose biomass falls below b_min is abandoned and its remainder moves to its richest occupied neighbour (lost only when no neighbour is occupied; the lost amount is reported). A cell adjacent to a reinforced tube is kept instead: it draws biomass from its richest neighbour up to b_keep when that neighbour can spare it.

Order of operations in one step: events; occupancy and front; edge activation; pressure and flux solve; conductivity adaptation and pruning; nutrient uptake, prey regrowth, biomass growth, inhibitor toxicity and attractant and inhibitor update; biomass transport along the flow; sheet spreading; consolidation; removal of stranded fragments; colonization; retraction; metrics.

Boundary conditions: zero-flux boundaries for the attractant; the lattice edge simply ends the graph for transport; nutrient does not diffuse.

## B. Python implementation

The full implementation is the single file physarum_sim.py (about 1,400 lines, NumPy, SciPy and Matplotlib only, no Numba). Its structure follows the sections above: Config (all parameters), Grid (lattice and edge arrays), make_nutrient_field, solve_pressure and adapt_conductivity (Tero core, reusable in tests), PhysarumSimulation (state and update rules, one method per process), Visualizer (four-panel figure with reusable artists, animation and snapshots), validation tests, the mu comparison and the command-line interface.

## C. Installation and execution

Summary:

- Python 3.9 or later with numpy, scipy and matplotlib. The script was run with numpy 2.5, scipy 1.18 and matplotlib 3.11.
- ffmpeg is only needed for MP4 export; GIF export uses Pillow, which Matplotlib already requires.

Install the dependencies:

pip install numpy scipy matplotlib

Run the interactive animation (scenario B by default):

python physarum_sim.py

Choose a scenario (A single patch, B nine irregular patches, C dynamic patches, D labyrinth with obstacles, E bacterial prey with inhibitors), grid size and duration:

python physarum_sim.py --scenario D --H 140 --W 180 --steps 4000

python physarum_sim.py --scenario E --steps 2500 --steps-per-frame 10

A smaller grid runs faster at the cost of shape resolution; the geometry and the dynamics are rescaled automatically:

python physarum_sim.py --scenario B --H 70 --W 90 --steps 1500

Run headless, write a PNG of the four panels every 500 steps and the metrics CSV:

python physarum_sim.py --scenario B --no-anim --snapshot-every 500 --outdir physarum_output

Export the animation (GIF needs no extra software; MP4 needs ffmpeg):

python physarum_sim.py --scenario B --save physarum_B.gif --steps-per-frame 5

python physarum_sim.py --scenario B --save physarum_B.mp4 --steps-per-frame 5 --fps 20

Run the validation suite and the mu comparison:

python physarum_sim.py --test

python physarum_sim.py --compare-mu --H 50 --W 64 --steps 1000

Other options: --mu, --alpha, --gamma, --Q0 (Tero rule), --connectivity 8, --solver cg, --n-sources and --n-sinks (0 for the diffuse allocation), --regen (nutrient regeneration), --seed, --dt. The companion script plot_metrics.py reads the metrics CSV files and writes per-scenario PNG plots of the raw time series and one Excel workbook.

Runtime on an ordinary laptop: about 60 to 90 seconds for 2,500 headless steps on the default 140 by 180 grid with 8-connectivity (about 100,000 potential tubes), and about 8 seconds for 1,500 steps on a 70 by 90 grid. The interactive animation runs at a few frames per second with ten steps per frame on the default grid.

## D. Parameter interpretation

Summary:

- alpha, mu, gamma and Q0 are the Tero adaptation parameters and set the steady-state conductivity D* = alpha |Q/Q0|^mu / gamma and the relaxation time 1 / gamma.
- The colonization, biomass, consolidation and inhibitor parameters are illustrative; none of them is calibrated against measurements.
- The default flux exponent is mu = 1.5: on a sheet with changing source and sink sites the flux distribution is heavy-tailed, mu = 1 keeps every lattice edge alive, and mu above 1 is what prunes the lattice into a vein network with a main trunk and secondary branches (mu = 2 leaves a single trunk).

Conductivity reinforcement alpha: the gain of the reinforcement. The steady state of a tube under constant flux is D* = alpha |Q/Q0|^mu / gamma, so alpha and gamma enter only through their ratio for the steady state, while gamma alone sets the speed. Default 1.

Flow-response exponent mu: sets how strongly tubes compete, and on the growing sheet it decides whether veins appear at all. With mu = 1 the reinforcement is proportional to the flux; parallel paths of equal length keep sharing the flow and the model converges to shortest paths between sources and sinks (Bonifaci, Mehlhorn and Varma 2012 prove this for the single-pair case). With mu larger than 1 a tube that carries slightly more flux gains conductivity faster than linearly, which is a winner-take-all rule: alternative routes collapse and the network becomes sparser. With mu smaller than 1 the response is sublinear and many parallel paths survive. The built-in two-path test shows the detour conductivity falling to 7 10^-6 of the direct path for mu = 1 and to 10^-7 for mu = 2. In the scenario B comparison (50 by 64 grid, 1,000 steps), mu = 2 left 1,155 tubes against 2,236 for mu = 1, with a lower dissipation.

Conductivity decay gamma: the rate at which an unused tube disappears; 1 / gamma is the memory of the network. Default 0.1, so a tube with no flux decays from D_init = 0.2 to the prune threshold 0.05 in about 14 time units (28 steps). Larger gamma prunes faster and makes the network track changes in the flow more quickly; smaller gamma preserves old routes.

Reference flux Q0: makes the term |Q/Q0|^mu dimensionless, so alpha has the units of D per time regardless of mu. The dimensionless group that governs sparsity is alpha |Q_typ/Q0|^mu / (gamma D_init): above 1 a tube carrying the typical flux grows, below 1 it decays.

Growth-front nutrient sensitivity: eta sets how strongly local nutrient attracts the front (S^eta; with eta = 2 a cell with S = 0.1 attracts 100 times less than a cell with S = 1), lam_grad weights the relative attractant gradient (the exponential attractant decays by 1 / 20 per cell, so lam_grad = 20 gives an uphill rate contribution of about 1), and eps_explore is the undirected exploration rate. The ratio lam_grad / (20 eps_explore) is the directional bias of the front; the default (200) produces fingers rather than a round blob.

Nutrient uptake rate u and half-saturation K: u is the maximal fraction of local nutrient taken per unit time at high biomass, and K is the biomass at which uptake is half-maximal. Together with the yield Y they set how much biomass a patch produces: a patch of amplitude 1 and width 5 cells holds about 157 nutrient units and yields about 157 biomass units at Y = 1.

Growth and colonization rates: k_col scales the colonization rate; b_colonize is the biomass invested per new cell and therefore the number of cells a given biomass can occupy (50 units occupy at most 1,000 cells at the default 0.05); b_keep and b_min set how thin the sheet can become before it stops donating and before it retracts; the maintenance rate m sets how fast unfed biomass disappears (half-life 693 time units at the default 0.001).

Source and sink parameters: q_flow sets the current per unit source weight and therefore the flux scale; kappa_T sets how much biomass follows the flow; n_sources and n_sinks set the concentration of the flow (1 and 1 reproduce the random-pair driving of Tero et al. 2010; 0 gives the diffuse allocation).

Numerical time step dt: the conductivity update is unconditionally stable, so dt is limited by accuracy and by the explicit couplings (uptake, colonization probabilities, biomass transport). At the default dt = 0.5, the largest per-step changes are the uptake fraction u dt = 0.015 and the capped transport fraction f_max = 0.5; halving dt changes the trajectories only through the stochastic draws.

Units: all quantities are dimensionless. Lengths are lattice cells, time is the unit of the adaptation equation, nutrient and biomass are arbitrary units related by the yield Y. No parameter is calibrated against a physical measurement of Physarum; the values were chosen so that the organism reaches patches 50 to 60 cells away within about 1,000 steps and so that trunks reach conductivities 10 to 100 times the initial value.

Consolidation parameters: consolidate_age sets how far behind the front the fan ends (cells younger than this spread biomass and are not drained); consolidate_rate sets how fast the film thins (half-life about 35 steps at the default); retract_age sets when a tube-less film cell is abandoned; scar_steps sets how long an abandoned cell stays closed; tube_D sets which tubes count as veins; diff_B sets the spreading of cytoplasm within the fan.

Inhibitor parameters: inhib_range is the distance over which the inhibitor decays, inhib_tol the level that suppresses colonization by a factor e, and inhib_toxicity the biomass loss rate at R = 1. In scenario E the organism grazes the weak central inhibitor colony at a large biomass cost and avoids the two strong colonies on the right; shifting inhib_tol or inhib_toxicity moves that outcome.

## E. Scientific limitations

Summary:

- The Tero model is a phenomenological rule for tube adaptation that reproduces network-level behaviour; it does not describe the cell biology that produces the flow.
- The colonization layer is a set of stated assumptions whose only justification is that they yield the qualitative behaviour (directed expansion, trunks, pruning, reorganization); it has not been validated against measurements.

Intracellular biochemical signalling: the model contains no calcium, ATP, or any signalling species. The attractant field replaces chemotactic signal transduction by a steady diffusion equation whose relative gradient is read directly by the colonization rule.

Actomyosin-driven contractions and oscillatory shuttle streaming: real cytoplasmic flow is driven by peristaltic contractions of the tube walls with a period of one to two minutes and reverses direction (Alim et al. 2013). The model imposes a steady pressure-driven flow between sources and sinks within a step; the random choice of source and sink sites per step is a crude surrogate for the changing flow pattern and does not model the oscillation.

Cytoplasmic mass conservation: the fluid in the Tero model is conserved only through the current balance; there is no volume stored in the tubes, no relation between tube radius and stored cytoplasm, and the biomass field B is a separate quantity moved by the flow with a capped coefficient. Biomass is conserved exactly in colonization and transport, and the small amount lost at retraction of isolated cells is reported.

Mechanical deformation: the plasmodium is represented by a set of occupied lattice cells; there is no deformable membrane, no tension, no adhesion and no volume constraint. Front shape is a result of the stochastic colonization rule on a lattice and shows lattice artefacts (axis-aligned trunks and bundles of parallel edges with 4-connectivity).

Three-dimensional tube geometry: tubes are one-dimensional edges with a scalar conductivity. Poiseuille flow in a cylindrical tube gives D proportional to the fourth power of the radius, so an increase of D by a factor 10 corresponds to a radius increase of 78 percent, but the model never represents radii, wall thickness, or the sheet that connects the tubes.

What can be interpreted as emergent network adaptation: the appearance of trunks between feeding sites and active fronts, the decay of tubes that stop carrying flow, the dependence of sparsity on mu, and the rerouting of flow after a new patch appears are consequences of the Tero rule acting on the solved flow and would occur for any colonization layer that supplies sources and sinks. These behaviours are the ones that the Tero model was shown to reproduce against experiments (maze solving, Tokyo rail network).

What should not be interpreted as validated biology: the speed and shape of the expansion, the fan and vein geometry (the consolidation rules are a phenomenological substitute for the dewetting-like contraction of the real sheet), the biomass numbers, the fraction of biomass moved by flow, the timing of retraction, and the response to the inhibitor (the toxicity and repellence functions are assumptions). These follow from the stated assumptions and the illustrative parameter values. Model validation here means the numerical checks (balanced currents, exact linear solve, local mass conservation, finite positive conductivities, nonnegative nutrient, colonization only by contact); empirical validation against Physarum observations has not been done and would require calibrating the colonization and transport parameters to time-lapse data.

## F. Further improvements

- Reaction-diffusion chemotaxis: replace the quasi-steady attractant by a time-dependent diffusion equation with consumption by the organism, and let repellents (depleted or toxic regions) enter with negative weight.
- Oscillatory flow: drive the pressure solve with time-periodic source and sink strengths or with a wall-contraction phase field so that the flux reverses on a one to two minute period, and adapt D on the time-averaged absolute flux, which is closer to the shuttle streaming of the organism.
- Adaptive tube radii: store a radius per tube, set D from the fourth power of the radius, and conserve the cytoplasm volume held in the tubes, which couples pruning to biomass.
- Biomass transport by the flow with an explicit advection scheme on the network, including storage in tubes, instead of the capped per-step redistribution.
- Agent-based growth front: represent the front by discrete agents (as in the particle model of Jones 2010) that sense the attractant, move, deposit biomass and create tubes, which removes the lattice artefacts of the cell-based colonization rule.
- Mechanical sheet model: add a membrane with surface tension and a volume constraint so that the shape of the plasmodium results from forces rather than from a stochastic rule.
- Parameter calibration: fit the colonization and transport parameters to time-lapse experiments (front speed, tube widths, time to connect two food sources) and report the fitted values with uncertainties.

## References

Verified references (the author of this document has read these works; the reader should still confirm the bibliographic details before citing):

- Tero A., Kobayashi R., Nakagaki T. (2007). A mathematical model for adaptive transport network in path finding by true slime mold. Journal of Theoretical Biology 244, 553 to 564. This paper introduces the pressure-driven network with flux-dependent conductivity adaptation used here.
- Tero A., Takagi S., Saigusa T., Ito K., Bebber D. P., Fricker M. D., Yumiki K., Kobayashi R., Nakagaki T. (2010). Rules for biologically inspired adaptive network design. Science 327, 439 to 442. This paper applies the model to the Tokyo rail network with random source and sink pairs among food sources and introduces the sigmoid variant of the reinforcement function.
- Nakagaki T., Yamada H., Tóth Á. (2000). Maze-solving by an amoeboid organism. Nature 407, 470. The experimental observation that motivated the model.
- Tero A., Kobayashi R., Nakagaki T. (2006). Physarum solver: a biologically inspired method of road-network navigation. Physica A 363, 115 to 119.
- Bonifaci V., Mehlhorn K., Varma G. (2012). Physarum can compute shortest paths. Journal of Theoretical Biology 309, 121 to 133. Proof of convergence of the mu = 1 dynamics to the shortest path.
- Jones J. (2010). Characteristics of pattern formation and evolution in approximations of Physarum transport networks. Artificial Life 16, 127 to 153. The agent-based particle model mentioned under further improvements.
- Alim K., Amselem G., Peaudecerf F., Brenner M. P., Pringle A. (2013). Random network peristalsis in Physarum polycephalum organizes fluid flows across an individual. Proceedings of the National Academy of Sciences 110, 13306 to 13311. Measurements of the peristaltic flow that the model does not represent.

No unverified citations are used in this document. Digital object identifiers were intentionally omitted to avoid reporting them from memory.
