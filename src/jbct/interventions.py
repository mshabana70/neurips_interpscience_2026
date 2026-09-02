"""H3 causal arm: intervene on the refusal direction during generation.

Two interventions, testing the two halves of mediation:

- **Directional ablation** (Arditi 2024) — remove the direction from the residual stream:
  ``x' = x - r_hat (r_hat . x)``. If refusal is mediated by this direction, ablating it makes
  the model comply with requests it would otherwise refuse. Applied at EVERY layer, because
  the direction is present throughout the stream, not just where it was measured.
- **Activation addition** — push the residual back along the direction:
  ``x' = x + coeff * r_hat``. Applied at ONE layer. If a jailbreak works *through* the
  direction, restoring it should undo the jailbreak.

Three things here are load-bearing, each a lesson already paid for:

1. **Basis (MENTEE_NOTE Bug 3).** The direction was extracted from the block OUTPUT
   (residual stream). It must be applied there too. Refusal-Lens computed r in the residual
   stream but injected it at the transcoder's post-RMSNorm pre-MLP point — ~1700x apart in
   magnitude, across a non-linear normalization that rescaling cannot undo.
2. **Every generated token, not just the prompt.** The hook stays live across generation
   steps, so it fires on the KV-cache forward for each new token. Ablating only the prompt
   would leave the model free to re-introduce the direction while it writes.
3. **dtype/device.** ``r_hat`` arrives as float64 on CPU (that is how directions are stored);
   the residual is bf16/fp32 on GPU. The projection is computed in the residual's dtype after
   an explicit cast — never by relying on promotion rules.
"""

from __future__ import annotations

from .models import get_layers
from .registry import ModelSpec


def _unit(v):
    import torch

    x = v.detach().to(dtype=torch.float64)
    return x / torch.linalg.vector_norm(x)


def _clamp_edit(resid, r, w, target: float, position: int | None, record=None,
                prompt_only: bool = False):
    """``x' = x + (target - <x, r>) w`` — the clamp arithmetic, shared by both clamp hooks.

    Factored out on purpose. `ProjectionClamp` (one layer) and `ProfileClamp` (a target per
    layer) must apply IDENTICAL arithmetic; two copies would be free to drift, and a silent
    divergence between the single-layer and multi-layer results is exactly the failure this
    experiment has already paid for twice.

    `r` and `w` are unit vectors already cast to the residual's device/dtype. `record`, if
    given, receives one shift per prompt on the prompt pass only.

    **`prompt_only` exists because a NONZERO target has no meaning on a generated token.**
    The target is measured at one position of the prompt; holding a generated token to it
    imposes a value that token never naturally carries. Zeroing survives this (ablation is
    position-agnostic), which is why the single-layer F3 runs and the `zero` arm looked
    fine while the multi-layer `null` arm — prompt-pass shift 64.45 against targets up to
    8,157, i.e. already on target — degenerated into repetition loops in 149/150 responses
    and hit the token cap 150/150. Damage scaled with |target| in both models.
    """
    is_prompt_pass = resid.shape[-2] > 1
    if prompt_only and not is_prompt_pass:
        return resid
    if position is None:
        shift = target - (resid @ r)
        if record is not None and is_prompt_pass:
            record.extend(float(x) for x in shift[..., -1].flatten())
        return resid + shift.unsqueeze(-1) * w
    # Only the target token. During generation the sequence is length 1, so that single
    # position IS the token being written and the constraint stays live.
    idx = position if is_prompt_pass else -1
    cur = resid[:, idx, :]
    shift = target - (cur @ r)
    if record is not None and is_prompt_pass:
        record.extend(float(x) for x in shift.flatten())
    out = resid.clone()
    out[:, idx, :] = cur + shift.unsqueeze(-1) * w
    return out


class _ResidualHook:
    """Base: rewrite each decoder block's output residual in place."""

    def __init__(self, model, spec: ModelSpec, r_hat, layers=None):
        self.model = model
        self.spec = spec
        self.blocks = get_layers(model, spec)
        self.layers = list(range(len(self.blocks))) if layers is None else list(layers)
        for layer in self.layers:
            if not 0 <= layer < len(self.blocks):
                raise ValueError(f"layer {layer} out of range for {spec.name}")
        # `r_hat=None` is for subclasses carrying a direction PER LAYER (`ProfileClamp`),
        # which cannot be expressed as one vector. Everything else normalizes defensively:
        # a caller passing an unnormalized r would otherwise scale the intervention by
        # ||r||, which varies ~7x across depth.
        self._r = None if r_hat is None else _unit(r_hat)
        self._handles: list = []
        self.n_calls = 0

    def _edit(self, resid):  # pragma: no cover - overridden
        raise NotImplementedError

    def _make_hook(self, layer: int):
        """One hook per layer. `layer` is passed so per-layer subclasses can close over it;
        single-direction hooks ignore it."""

        def hook(_module, _inp, output):
            is_tuple = isinstance(output, tuple)
            resid = output[0] if is_tuple else output
            new = self._edit(resid)
            self.n_calls += 1
            return (new, *output[1:]) if is_tuple else new

        return hook

    def _r_like(self, resid):
        return self._r.to(device=resid.device, dtype=resid.dtype)

    def __enter__(self):
        for layer in self.layers:
            self._handles.append(
                self.blocks[layer].register_forward_hook(self._make_hook(layer)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


class DirectionalAblation(_ResidualHook):
    """``x' = x - r_hat (r_hat . x)`` — zero the component along the refusal direction.

    Defaults to all decoder-block outputs. This repository's H3 experiments usually
    restrict the edit to one layer; use ``ArditiAblation`` for the paper's stronger
    all-write intervention.

    ``prompt_only`` restricts the ablation to the prompt forward pass (all prompt
    positions) and leaves generated tokens untouched. Default False preserves the
    established all-token behavior.
    """

    def __init__(self, model, spec: ModelSpec, r_hat, layers=None,
                 prompt_only: bool = False):
        super().__init__(model, spec, r_hat, layers=layers)
        self.prompt_only = prompt_only
        # ``n_calls`` counts hook callbacks, including generation callbacks that a
        # prompt-only hook deliberately leaves untouched.  Keep a second counter so
        # experiment reports can distinguish callbacks from actual edits.
        self.n_edits = 0

    def _edit(self, resid):
        if self.prompt_only and resid.shape[-2] <= 1:
            return resid  # generation pass (seq len 1): leave generated tokens alone
        self.n_edits += 1
        r = self._r_like(resid)
        proj = (resid @ r).unsqueeze(-1) * r
        return resid - proj


class ArditiAblation:
    """The original paper's all-write ablation, optionally restricted to named layers.

    ``DirectionalAblation`` edits the decoder block output, which is the intervention used
    by this repository's H3 experiments.  Arditi et al.'s released implementation is more
    aggressive: for every selected decoder layer it projects the direction out of the block
    input and out of both the attention and MLP contributions.  That prevents any residual
    writer in the layer from reintroducing the direction.

    Keeping this as a separate class is intentional.  Retrofitting the stronger intervention
    into ``DirectionalAblation`` would silently change the meaning of all existing results.
    The Arditi-gap experiment reports both this exact scope and the established block-output
    scope explicitly.
    """

    def __init__(self, model, spec: ModelSpec, r_hat, layers=None):
        self.model = model
        self.spec = spec
        self.blocks = get_layers(model, spec)
        self.layers = list(range(len(self.blocks))) if layers is None else list(layers)
        for layer in self.layers:
            if not 0 <= layer < len(self.blocks):
                raise ValueError(f"layer {layer} out of range for {spec.name}")
            block = self.blocks[layer]
            if not hasattr(block, "self_attn") or not hasattr(block, "mlp"):
                raise AttributeError(
                    f"{type(block).__name__} must expose self_attn and mlp for exact "
                    "Arditi ablation"
                )
        self._r = _unit(r_hat)
        self._handles: list = []
        self.n_calls = 0

    def _edit(self, activation):
        r = self._r.to(device=activation.device, dtype=activation.dtype)
        self.n_calls += 1
        return activation - (activation @ r).unsqueeze(-1) * r

    def _pre_hook(self, _module, inputs):
        if not isinstance(inputs, tuple) or not inputs:
            return self._edit(inputs)
        return (self._edit(inputs[0]), *inputs[1:])

    def _output_hook(self, _module, _inputs, output):
        if isinstance(output, tuple):
            return (self._edit(output[0]), *output[1:])
        return self._edit(output)

    def __enter__(self):
        for layer in self.layers:
            block = self.blocks[layer]
            self._handles.append(block.register_forward_pre_hook(self._pre_hook))
            self._handles.append(block.self_attn.register_forward_hook(self._output_hook))
            self._handles.append(block.mlp.register_forward_hook(self._output_hook))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False


class ActivationAddition(_ResidualHook):
    """``x' = x + coeff * r_hat`` at one layer — push refusal back in.

    ``coeff`` is in units of the residual norm at the target layer unless
    ``absolute=True``, because raw residual norms differ by ~7x across depth and by orders
    of magnitude between models; a bare coefficient would not transfer.
    """

    def __init__(self, model, spec, r_hat, layer: int, coeff: float, absolute: bool = False):
        super().__init__(model, spec, r_hat, layers=[layer])
        self.coeff = coeff
        self.absolute = absolute

    def _edit(self, resid):
        r = self._r_like(resid)
        scale = self.coeff
        if not self.absolute:
            # Norm of the LAST position, matching where the decision is read.
            scale = self.coeff * float(resid[..., -1, :].norm())
        return resid + scale * r


class ProjectionClamp(_ResidualHook):
    """``x' = x + (target - <x, r_hat>) w`` — SET the component, don't remove it.

    F3's intervention, and the one the other two hooks cannot express. `DirectionalAblation`
    is the special case ``target = 0``; `ActivationAddition` moves by a fixed amount
    regardless of where the prompt started. Neither can put a prompt at a *specified*
    magnitude, which is what "reduce refused prompts to jailbroken level" asks for.

    **``position`` is load-bearing, and defaulting it to "everywhere" cost a run.** The target
    is measured at ONE token, so clamping every position overwrites the whole sequence with a
    value only that token naturally holds. At gemma L15, where the projection sits near
    -30,000, that left 150/150 responses incoherent in every arm — including the alpha=0 arm
    that clamps to the population's own mean and should have been a near-null. Pass the
    site's negative index to edit only that token; `generate` forces left padding
    (`generate.py:89`), so a negative index aligns across a padded batch. On generation steps
    the sequence is one token long and that token is the one clamped, which keeps the
    constraint live without touching the prompt again.

    ``apply_along`` writes the correction along a DIFFERENT unit vector than the one the
    projection is read from. That is what makes an exactly matched sham possible: the shift
    is computed from ``r_hat`` per prompt, so the control moves each prompt by the same
    amount the real arm did and differs only in direction.
    """

    def __init__(self, model, spec, r_hat, layer: int, target: float, position: int | None = None,
                 apply_along=None, record: list | None = None, prompt_only: bool = False):
        super().__init__(model, spec, r_hat, layers=[layer])
        self.target = float(target)
        self.position = position
        self.record = record
        self.prompt_only = prompt_only
        self._w = None if apply_along is None else _unit(apply_along)

    def _w_like(self, resid):
        w = self._r if self._w is None else self._w
        return w.to(device=resid.device, dtype=resid.dtype)

    def _edit(self, resid):
        return _clamp_edit(resid, self._r_like(resid), self._w_like(resid),
                           self.target, self.position, self.record, self.prompt_only)


class ProfileClamp(_ResidualHook):
    """``x'_L = x_L + (target[L] - <x_L, r_hat[L]>) w_hat[L]`` at EVERY named layer.

    The multi-layer transplant. `ProjectionClamp` sets one layer to one value; this sets a
    whole per-layer *profile*, which is what "give this prompt the refusal geometry of a
    successful jailbreak" actually requires — a jailbreak does not shift one site, it
    reshapes the trajectory.

    **Every layer carries its own direction.** H1/F1 fit a separate ``r`` at each
    (layer, position); reusing one layer's vector across the stack would be a basis error of
    the same family as reading a target along ``r_raw`` and imposing it along ``r_comp``,
    which cost this experiment two rounds of GPU time. `directions` and `targets` must name
    exactly the same layers, checked at construction rather than discovered mid-run.

    Clamps are applied in the forward order the blocks run, so layer L+1's clamp acts on a
    residual that L's clamp has already rewritten. That is the intent: the refusal-direction
    trajectory ends up on the target profile regardless of what upstream layers recompute.

    ``apply_along`` (per layer) and ``position`` behave exactly as in `ProjectionClamp` —
    the sham derives its shift from ``r_hat`` and writes it along ``u_hat``, and the target
    is imposed only at the token it was measured at. ``record`` is a dict keyed by layer.
    """

    def __init__(self, model, spec, directions: dict, targets: dict,
                 position: int | None = None, apply_along: dict | None = None,
                 record: dict | None = None, prompt_only: bool = False):
        if set(directions) != set(targets):
            raise ValueError(
                "directions and targets must cover the same layers; "
                f"directions has {sorted(set(directions) - set(targets))} extra, "
                f"targets has {sorted(set(targets) - set(directions))} extra"
            )
        super().__init__(model, spec, None, layers=sorted(directions))
        self._rs = {int(k): _unit(v) for k, v in directions.items()}
        self._targets = {int(k): float(v) for k, v in targets.items()}
        self.position = position
        self.record = record
        self.prompt_only = prompt_only
        if apply_along is None:
            self._ws = None
        else:
            if set(apply_along) != set(directions):
                raise ValueError("apply_along must cover the same layers as directions")
            self._ws = {int(k): _unit(v) for k, v in apply_along.items()}
        if record is not None:
            for layer in self._rs:
                record.setdefault(layer, [])

    def _edit_at(self, resid, layer: int):
        r = self._rs[layer].to(device=resid.device, dtype=resid.dtype)
        w = r if self._ws is None else self._ws[layer].to(device=resid.device,
                                                          dtype=resid.dtype)
        rec = None if self.record is None else self.record[layer]
        return _clamp_edit(resid, r, w, self._targets[layer], self.position, rec,
                           self.prompt_only)

    def _make_hook(self, layer: int):
        def hook(_module, _inp, output):
            is_tuple = isinstance(output, tuple)
            resid = output[0] if is_tuple else output
            new = self._edit_at(resid, layer)
            self.n_calls += 1
            return (new, *output[1:]) if is_tuple else new

        return hook


class ShiftAlong(_ResidualHook):
    """``x' = x + delta * u`` for a FIXED scalar delta — the matched sham for a clamp.

    A clamp's magnitude of change varies by prompt, so a sham that adds a constant is not
    matched. This takes the shift the real clamp computed and applies it along another
    direction, so the two arms differ in direction alone.
    """

    def __init__(self, model, spec, u_hat, layer: int, delta: float):
        super().__init__(model, spec, u_hat, layers=[layer])
        self.delta = float(delta)

    def _edit(self, resid):
        return resid + self.delta * self._r_like(resid)
