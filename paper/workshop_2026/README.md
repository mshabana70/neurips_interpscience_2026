# Paper source and figures

`main.tex` is the anonymous workshop manuscript. All quantitative figures are either rebuilt from the public aggregate results or copied from a clearly named archival result by `build_figures.py`.

From the repository root:

```bash
uv sync --extra plots
uv run python paper/workshop_2026/build_figures.py
```

The builder regenerates these 16 files in `figures/`:

```text
fig1_method_overview.png
fig2_refusal_trajectories.png
fig3_scope_points.png
fig4_causal_contrast.png
fig1_direction_auc.png
fig2_generation_validation.png
fig3_jailbreak_dataset.png
fig4_refusal_signal_by_class.png
fig4_closure_scope.png
fig5_closure_transplant.png
fig6_closure_restoration.png
fig6_sustained_addition.png
figA1_gemma_causal_sweep.png
figA2_qwen_causal_sweep.png
figA3_jailbreak_examples.png
figA4_intervention_examples.png
```

The method overview is conceptual. The jailbreak and intervention example panels are deterministic redacted displays. The other figures read the JSON or JSON.GZ paths listed in `docs/ARTIFACT_MANIFEST.md`.
