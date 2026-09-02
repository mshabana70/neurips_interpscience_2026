# Included result artifacts

The paper figures and tables read the aggregate reports in this directory directly.

- `h1/`: held-out refusal-direction discrimination.
- `h2/`: attack and matched-control compliance by class. `judgments.json` is a compact label-only export used for audit, not as input to downstream reruns.
- `f1_geometry/`: across-layer projections and geometry summaries.
- `f1_categories/`: class-level trajectory summaries.
- `arditi_gap/`: full layer-by-position proxy grids and full-generation validation.
- `h3_causal/`: sustained removal and strong-addition summaries.
- `paper_closure/`: request-only versus sustained removal, profile transfer, and restoration summaries with paired intervals and corrected tests.
- `public_labels/`: per-example condition and outcome records with prompts, prefills, and responses removed.
- `precomputed/`: an archival visualization used for the H1 appendix figure.

Raw model completions are excluded intentionally. Reproducing GPU experiments creates them under the ignored `reproduced/` directory. The aggregate reports remain sufficient to rebuild every quantitative paper figure.
