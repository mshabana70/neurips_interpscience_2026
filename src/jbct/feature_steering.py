"""P2: steer a GROUP of transcoder features and measure the per-class effect.

H4 gave us three feature groups — a shared backbone and one disjoint set per jailbreak class.
Those are correlational: they are what attribution selected, not what the model needs. This
module makes them causal claims that can fail:

    steer the SHARED group   -> both classes move
    steer PREFILL-only       -> prefill moves, fiction does not
    steer FICTION-only       -> fiction moves, prefill does not

**Why this basis is safe (MENTEE_NOTE Bug 3).** Both registry transcoders are per-layer
(`type="plt"`), so a feature has ONE decoder vector `W_dec[f]` of shape (d_model,), and
`decode` is `acts @ W_dec + b_dec` reconstructing the layer's `feature_output_hook`. That hook
is Gemma's `hook_mlp_out` / Qwen's `mlp.hook_out` — and TransformerLens applies Gemma's
post-MLP norm BEFORE `hook_mlp_out` (`transformer_block.py:229-231`), so in both models the
reconstruction target is the tensor that is summed into the residual stream. The decoder
vector is therefore already in residual-stream basis and the matching application point is the
block output. Refusal-Lens' Bug 3 was extracting in the residual stream and applying at the
transcoder's *input* hook (post-RMSNorm, pre-MLP) — ~1700x apart. Steering at
`mlp.hook_in` here would repeat it exactly.

**Two intervention kinds, and why clamp is the default.**

`clamp_interventions` + `generate_with_clamp` SET a feature's activation to a value, reusing
circuit-tracer's `feature_intervention_generate` — the same code path that produced the graphs
being interpreted. Zero is the ablation; a multiple of peak is amplification. It reads each
feature's real activation before writing `(value - a_observed) * W_dec[f]`, so a zero-clamp is
a no-op wherever the feature was already silent.

`FeatureSteering` is the additive approximation: `+coeff * scale_f * W_dec[f]` at every
position unconditionally. It needs no transcoder at run time (just the cached decoder rows),
which makes it cheap, but at positions where the feature never fired it pushes the residual
somewhere the feature could not have put it. Kept as a diagnostic, not the measurement.

**Why the additive coefficient is in activation units.** A bare `coeff * W_dec[f]` is
uninterpretable: decoder norms vary across features and residual norms vary ~7x across depth,
which is what made H3's first activation-addition sweep collapse the model. The unit is the
feature's own observed activation on the graph that selected it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import get_layers
from .registry import ModelSpec


@dataclass
class FeatureVector:
    """One feature's decoder vector plus the activation that sets its natural scale."""

    layer: int
    index: int
    vec: "object"  # torch.Tensor (d_model,), residual-stream basis
    scale: float  # observed max activation on the source graph
    label: str = ""

    @property
    def key(self) -> tuple[int, int]:
        return (self.layer, self.index)


def graph_activations(pt_path, cache: bool = True) -> dict[tuple[int, int], float]:
    """{(layer, feature): max activation over positions} from a raw graph.

    Max rather than mean: the scale we want is "how hard does this feature fire when it
    fires", and averaging over a long prompt where it is silent almost everywhere would
    shrink every scale toward zero and make the sweep meaningless.

    Cached beside the graph — `Graph.from_pt` reads a ~115 MB adjacency matrix to produce a
    dict of a few tens of thousands of floats. The cache suffix deliberately does not end in
    `.pt`, or a `glob("*.pt")` over the sample dir would feed it back to `Graph.from_pt`.
    """
    import torch

    cpath = Path(str(pt_path) + ".acts")
    if cache and cpath.exists():
        d = torch.load(cpath, weights_only=True)
        return {(int(a), int(b)): float(v)
                for (a, b), v in zip(d["keys"].tolist(), d["vals"].tolist())}

    from circuit_tracer import Graph

    g = Graph.from_pt(str(pt_path))
    acts: dict[tuple[int, int], float] = {}
    values = g.activation_values.tolist()
    for (layer, _pos, feat), val in zip(g.active_features.tolist(), values):
        k = (int(layer), int(feat))
        v = abs(float(val))
        if v > acts.get(k, 0.0):
            acts[k] = v
    if cache:
        items = sorted(acts.items())
        torch.save({"keys": torch.tensor([k for k, _ in items], dtype=torch.int32),
                    "vals": torch.tensor([v for _, v in items], dtype=torch.float32)}, cpath)
    return acts


def _layer_weight_file(spec: ModelSpec, layer: int) -> str:
    """Download ONE layer's transcoder weights and return the local path.

    circuit-tracer's own loader calls `snapshot_download` for `layer_*.safetensors`, which for
    Gemma is **23.7 GB of weights** — to read 36 rows. Neither registry repo lists explicit
    per-file URIs in its config, so the paths are reconstructed from the same
    `{subfolder}/layer_{i}.safetensors` convention `resolve_transcoder_paths` globs for, and
    fetched one layer at a time (~350 MB each, and only for layers a group actually touches).
    """
    from huggingface_hub import hf_hub_download

    from .transcoders import parse_hf_ref

    repo_id, subfolder, revision = parse_hf_ref(spec.transcoder.hf_ref)
    return hf_hub_download(repo_id, f"layer_{layer}.safetensors",
                           subfolder=subfolder, revision=revision)


def load_feature_vectors(spec: ModelSpec, features, scales=None, labels=None,
                         dtype=None) -> list[FeatureVector]:
    """Decoder vectors for `[(layer, index), ...]`, read as slices from the transcoder repo.

    `get_slice("W_dec")[idxs]` pulls just those rows off disk, so nothing materializes the
    16k x d_model matrix. The result is a few hundred KB and gets cached — the GPU steering
    run then needs the base model only, never a transcoder.
    """
    import torch
    from safetensors import safe_open

    if spec.transcoder is None:
        raise ValueError(f"{spec.name} has no transcoder in the registry")
    if spec.transcoder.type != "plt":
        raise NotImplementedError(
            f"{spec.name} uses a {spec.transcoder.type!r} transcoder. Steering assumes ONE "
            f"decoder vector per feature; a cross-layer transcoder writes to every downstream "
            f"layer, so its steering vector is a per-layer family, not a vector."
        )
    scales, labels = scales or {}, labels or {}
    torch_dtype = torch.float32 if dtype is None else dtype

    by_layer: dict[int, list[int]] = {}
    for layer, index in features:
        by_layer.setdefault(int(layer), []).append(int(index))

    out: list[FeatureVector] = []
    for layer, idxs in sorted(by_layer.items()):
        path = _layer_weight_file(spec, layer)
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            # gemma-scope-2 checkpoints name it `w_dec`; the relu-transcoder format `W_dec`.
            # Only W_enc is transposed on load, so either way row `i` IS feature i's vector.
            key = "W_dec" if "W_dec" in keys else "w_dec"
            if key not in keys:
                raise KeyError(f"{path} has no W_dec/w_dec (keys: {sorted(keys)})")
            rows = f.get_slice(key)[idxs]
        rows = rows.detach().to("cpu", torch_dtype)
        for i, index in enumerate(idxs):
            out.append(FeatureVector(
                layer=layer, index=index, vec=rows[i].clone(),
                scale=float(scales.get((layer, index), 1.0)),
                label=labels.get((layer, index), ""),
            ))
    return out


def sample_sham_features(activations, group, seed: int = 0, tol: float = 0.5):
    """Random features matched to `group` on LAYER and activation magnitude.

    The specificity control, carried straight from H3: "steering the prefill group changed
    behaviour" means nothing until we know that steering *anything* of the same size at the
    same depth does not. Matching on layer matters as much as magnitude — a perturbation at
    L15 and one at L0 do not have comparable downstream reach.
    """
    import random

    rng = random.Random(seed)
    taken = {fv.key for fv in group}
    pool_by_layer: dict[int, list] = {}
    for (layer, feat), act in activations.items():
        pool_by_layer.setdefault(layer, []).append((feat, act))

    sham = []
    for fv in group:
        pool = [(f, a) for f, a in pool_by_layer.get(fv.layer, [])
                if (fv.layer, f) not in taken
                and abs(a - fv.scale) <= tol * max(fv.scale, 1e-6)]
        if not pool:  # no magnitude match at this layer — fall back to any feature there
            pool = [(f, a) for f, a in pool_by_layer.get(fv.layer, [])
                    if (fv.layer, f) not in taken]
        if not pool:
            continue
        feat, act = rng.choice(pool)
        taken.add((fv.layer, feat))
        sham.append((fv.layer, feat, act))
    return sham


def clamp_interventions(features, mode: str = "zero", factor: float = 1.0) -> list[tuple]:
    """circuit-tracer interventions: `(layer, position, feature_idx, value)`.

    **This is the same operation as Neuronpedia's steering slider, not a rival to it.** A
    clamp SETS a feature's activation to `value`; the slider is which value you pick:

        value = 0                -> the feature is off        (the "negative"/ablate end)
        value = factor * peak    -> the feature fires harder  (the "positive" end)

    The difference from an activation *addition* is where the number comes from. An addition
    applies `+c * W_dec[f]` unconditionally, including at positions where the feature never
    fired — which pushes the residual somewhere the feature could not have put it. A clamp
    reads the feature's real activation first and writes `(value - a_observed) * W_dec[f]`,
    so at a position where the feature was already silent, a zero-clamp is a no-op. That is
    why it is the faithful ablation: it removes what the feature contributed, no more.

    (For JumpReLU features, activations are non-negative, so "steer negative" past zero is
    not a state the feature can represent. Zero is the floor, and clamping to it is the
    well-defined version of turning the feature off.)

    `position` is `slice(0, None)` — **open-ended**, which is the flag
    `_convert_open_ended_interventions` looks for to keep the clamp applied to tokens the
    model generates. A bounded slice would silently stop at the prompt, and H3 already paid
    for that lesson with directional ablation.
    """
    if mode not in ("zero", "amplify"):
        raise ValueError(f"mode must be 'zero' or 'amplify', got {mode!r}")
    return [
        (fv.layer, slice(0, None), fv.index,
         0.0 if mode == "zero" else float(factor) * fv.scale)
        for fv in features
    ]


def feature_activity(rmodel, spec: ModelSpec, tokenizer, texts, prefills, features) -> dict:
    """Do the target features actually FIRE on the evaluation prompts?

    This is the precondition a clamp experiment cannot skip, and skipping it wasted an
    8-hour run. The features are selected from a handful of attribution graphs, but the
    behavioural arms run on different prompts of the same class. Where a feature is already
    silent, `value - a_observed` is zero and a zero-clamp is a **no-op by construction** — so
    the experiment returns "no effect" whether or not the feature matters, and the two
    outcomes are indistinguishable in the ASR.

    Costs one forward pass per prompt and no generation, so it belongs before the arms, not
    after them.
    """
    import torch

    from .generate import build_input

    per_feature = {fv.key: {"layer": fv.layer, "index": fv.index, "graph_scale": fv.scale,
                            "n_prompts_active": 0, "max_act": 0.0, "total_positions": 0,
                            "active_positions": 0, "label": fv.label}
                   for fv in features}
    for text, prefill in zip(texts, prefills):
        prompt = build_input(spec, tokenizer, text, prefill)
        with torch.inference_mode():
            _logits, cache = rmodel.get_activations(prompt)
        for fv in features:
            col = cache[fv.layer, :, fv.index].to(torch.float32)
            rec = per_feature[fv.key]
            active = int((col > 0).sum())
            rec["total_positions"] += int(col.numel())
            rec["active_positions"] += active
            rec["n_prompts_active"] += int(active > 0)
            rec["max_act"] = max(rec["max_act"], float(col.max()))
        del cache
    return per_feature


def generate_with_clamp(rmodel, spec: ModelSpec, tokenizer, texts, prefills,
                        interventions, max_new_tokens: int = 512,
                        freeze_attention: bool = False) -> list[str]:
    """Greedy continuations under a feature clamp, one prompt at a time.

    `feature_intervention_generate` takes a single input, so this loops rather than batches.
    The BASELINE must come through this same call with `interventions=[]` — TransformerLens
    and HuggingFace do not decode bit-identically, so a baseline taken from `jbct.generate`
    would fold a framework difference into the measured ASR shift.

    `freeze_attention` defaults to **False** here, unlike circuit-tracer's default. Freezing
    exists to make a single-token intervention match attribution's linear approximation; it
    costs an extra full forward with all-layer caching on every prompt. A behavioural arm
    wants the effect to propagate through attention anyway, so the freeze buys nothing and
    was a large part of why the first run took 8.5 hours.
    """
    from .generate import build_input

    out = []
    for text, prefill in zip(texts, prefills):
        prompt = build_input(spec, tokenizer, text, prefill)
        generation, _logits, _acts = rmodel.feature_intervention_generate(
            prompt, interventions, max_new_tokens=max_new_tokens, do_sample=False,
            verbose=False, return_activations=False, freeze_attention=freeze_attention,
        )
        # `generate` returns prompt + continuation; the judge must see the prefill as part
        # of the assistant message, so it is re-attached rather than dropped.
        response = generation[len(prompt):] if generation.startswith(prompt) else generation
        out.append((prefill or "") + response)
    return out


def measure_resid_norms(model, spec: ModelSpec, input_ids) -> dict[int, float]:
    """Mean residual norm at each block's output — the denominator for the Bug 3 check."""
    import torch

    blocks = get_layers(model, spec)
    norms: dict[int, float] = {}

    def make(layer):
        def hook(_m, _i, output):
            resid = output[0] if isinstance(output, tuple) else output
            norms[layer] = float(resid.to(torch.float32).norm(dim=-1).mean())

        return hook

    handles = [b.register_forward_hook(make(i)) for i, b in enumerate(blocks)]
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in handles:
            h.remove()
    return norms


class FeatureSteering:
    """Add `coeff * scale_f * W_dec[f]` to the residual stream at each feature's own layer.

    Applies at EVERY position and stays live across generation steps — the same choice H3
    made for directional ablation, and for the same reason: an intervention confined to the
    prompt leaves the model free to rebuild the feature's contribution while it writes.

    Features at the same layer are pre-summed into one vector, so the cost is one add per
    touched layer regardless of group size.
    """

    def __init__(self, model, spec: ModelSpec, features: list[FeatureVector], coeff: float):
        import torch

        self.model = model
        self.spec = spec
        self.blocks = get_layers(model, spec)
        self.coeff = coeff
        self.features = features
        self._delta: dict[int, object] = {}
        for fv in features:
            if not 0 <= fv.layer < len(self.blocks):
                raise ValueError(f"layer {fv.layer} out of range for {spec.name}")
            v = fv.vec.detach().to(torch.float64) * (coeff * fv.scale)
            self._delta[fv.layer] = (
                v if fv.layer not in self._delta else self._delta[fv.layer] + v
            )
        self._handles: list = []
        self.n_calls = 0

    def _make_hook(self, layer: int):
        def hook(_module, _inp, output):
            is_tuple = isinstance(output, tuple)
            resid = output[0] if is_tuple else output
            delta = self._delta[layer].to(device=resid.device, dtype=resid.dtype)
            new = resid + delta
            self.n_calls += 1
            return (new, *output[1:]) if is_tuple else new

        return hook

    def relative_magnitude(self, resid_norms: dict[int, float]) -> dict[int, float]:
        """‖delta‖ / ‖resid‖ per layer — the Bug 3 smoke alarm.

        A correct residual-basis steer sits within an order of magnitude of the residual it
        edits. Ratios near 1e-3 or 1e3 mean the vector is in the wrong space, and no
        coefficient sweep can rescue that: RMSNorm is not a rescaling you can undo.
        """
        return {layer: float(d.norm()) / max(resid_norms.get(layer, 1.0), 1e-9)
                for layer, d in self._delta.items()}

    def __enter__(self):
        for layer in self._delta:
            self._handles.append(self.blocks[layer].register_forward_hook(self._make_hook(layer)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False
