# Experiment pipeline

The experiment directories follow the dependency order below.

| Stage | Directory | Purpose | Consumes | Main output |
|---|---|---|---|---|
| H1 | `h1_direction/` | Fit and validate refusal directions | refusal corpus | `h1_report.json`, `candidates.pt` |
| H2 | `h2_jailbreak/` | Generate paired attacks and controls; judge compliance | jailbreak dataset | `h2_report.json`, local `judged.json` |
| F1 | `f1_geometry/` | Record refusal-direction projections across layers | H2 judged rows | `f1_report.json`, `projections.json.gz` |
| F1 class | `f1_categories/` | Summarize trajectories by attack class | F1 projections, H2 labels | `f1_categories_report.json.gz` |
| Foundation causal | `arditi_gap/` | Sweep all candidate layer-position cells and validate selected sites | H1 directions | `proxy_grid_report.json`, `generation_validation_report.json` |
| H3 | `h3_causal/` | Test sustained removal and strong direction addition | H1, H2 | `h3_causal_report.json` |
| Closure | `paper_closure/` | Compare intervention scopes and transfer measured class profiles | H1, H2, F1 | `paper_closure_report.json` |

`scripts/reproduce_paper.sh` invokes these stages in dependency order. Each runner also exposes `--help` and may be called independently. Full judged generations are deliberately local-only; the public artifact includes aggregate reports and text-free per-example labels.

`f3_clamp/` and `f3_transplant/` contain the precursor measured-state interventions whose lessons motivated the matched, paired closure experiment. They are retained for implementation provenance and unit tests; the workshop paper reports the superseding `paper_closure/` results.
