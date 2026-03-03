---
layout: page
title: Gauge-like invariances in Transformers
nav_exclude: true
---

This note is about a *useful analogy*: many degrees of freedom in modern deep nets behave like **gauge choices**—parameters or internal representations that can change without changing the model’s input–output behavior (or change it only in tightly controlled ways). Thinking in these terms helps you:

- separate “real” function changes from reparameterizations,
- reason about normalization / scaling,
- design cleaner experiments, and
- avoid over-interpreting internal activations.

The goal here is not to claim Transformers are literally gauge theories (although they might be... tbd later); it’s to catalogue **invariances / near-invariances** that are practically relevant.

---

## 1. What I mean by “gauge-like”

A *gauge-like freedom* is a transformation of internal variables or parameters that:

1. leaves the computed function (approximately) unchanged, and
2. is “local” in the sense that it can vary per layer, per channel, or per token without requiring a global constraint.

In deep learning, these show up as:
- **reparameterization symmetries** (same function, different parameters),
- **normalization-induced scale freedoms**, and
- **basis changes** in representation space that downstream components can absorb.

---

## 2. Two concrete invariances you can verify today

### 2.1 Softmax is invariant to adding a constant (per row)

For a vector $\(z \in \mathbb{R}^n\)$,
\[
\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}.
\]
For any constant \(c\),
\[
\mathrm{softmax}(z + c\mathbf{1}) = \mathrm{softmax}(z).
\]

**Transformer consequence:** attention weights are unchanged if you add the same scalar to every logit in a row. This is why subtracting \(\max(z)\) is numerically safe: it’s a **gauge fixing** of the softmax “offset”.

This is a real, exact invariance.

---

### 2.2 LayerNorm largely removes global scale (per token)

Define LayerNorm on a token vector \(x \in \mathbb{R}^d\):
\[
\mathrm{LN}(x) = \gamma \odot \frac{x - \mu(x)}{\sigma(x)} + \beta,
\]
where \(\mu(x)\) and \(\sigma(x)\) are the mean/std over features of that token.

If you scale inputs \(x \mapsto ax\) with \(a>0\), then:
- the standardized term \(\frac{x-\mu}{\sigma}\) is **invariant** to global scale \(a\),
- up to numerical eps and the learned \(\gamma,\beta\).

So LN tends to make the network insensitive to certain amplitude rescalings of intermediate representations (especially pre-LN architectures).

This is not a perfect invariance in every circumstance, but it is strong enough that you can often treat “activation magnitude” as partially gauge-like unless you control for normalization.

---

## 3. “Gauge transformations” as basis changes in representation space

A recurring pattern in deep nets:

- one module produces some representation \(h\),
- a downstream module consumes \(h\) through a linear map.

If we insert an invertible linear transform \(G\) (think: a change of basis),
\[
h' = Gh,
\]
and compensate downstream with
\[
W' = WG^{-1},
\]
then the composed mapping \(Wh\) stays identical:
\[
W'h' = (WG^{-1})(Gh) = Wh.
\]

This is an **exact functional symmetry** for linear stacks. In practice, nonlinearities and normalization break it partially, but the core idea survives: many internal coordinates are not identifiable.

**Interpretation:** a lot of “feature directions” are not unique. Claims like “this neuron represents X” can be unstable under innocuous reparameterizations.

---

## 4. Attention: what’s invariant vs what’s not

With attention logits (single head, single token-to-token row):
\[
\ell_j = \frac{q^\top k_j}{\sqrt{d}}.
\]

Two points:

1. **Offset invariance** (Section 2.1) is exact: \(\ell \mapsto \ell + c\mathbf{1}\) doesn’t matter.
2. **Scale changes** \(\ell \mapsto a\ell\) *do* matter: they change softmax sharpness. This is a knob that behaves like an inverse temperature.

So:
- “logit offset” is gauge-like,
- “logit scale” is a physically meaningful parameter (temperature).

This distinction is a good habit: *separate redundant degrees of freedom from meaningful ones.*

---

## 5. Residual streams and “gauge fixing” intuition

In pre-LN Transformers, you often see:
\[
h_{l+1} = h_l + \mathrm{Block}(\mathrm{LN}(h_l)).
\]

LN acts like a local standardization (a partial gauge fixing) before the block computes a correction.

The residual path keeps an “unfixed” stream \(h_l\) accumulating information, while each block reads a normalized view of it.

One mental model:
- residual stream = “raw field”
- LN = “choose a local gauge”
- block output = “gauge-dependent computation”
- addition = “update field”

Again: analogy, not identity—but it helps organize what changes are meaningful.

---

## 6. Practical takeaways for experiments

If you are probing internal states or training dynamics:

1. **Don’t compare raw activation magnitudes across layers/models** unless you account for LN/RMSNorm. Normalize first or probe post-norm consistently.
2. When analyzing attention logits, separate:
   - offset (irrelevant),
   - temperature/scale (relevant),
   - relative differences between tokens (relevant).
3. When claiming interpretability of directions, test stability under:
   - random orthogonal rotations in a subspace (if feasible),
   - reparameterizations that preserve the function (or nearly do).

---

## 7. Minimal experiment checklist (1–2 evenings)

Here are “doable” sanity checks that make the note concrete:

- **Softmax gauge fixing demo:** show \(\mathrm{softmax}(z)\) unchanged under \(z+c\).
- **LN scale demo:** sample random token vectors and show \(\mathrm{LN}(x)\approx \mathrm{LN}(ax)\) for a range of \(a\).
- **Temperature demo:** scale attention logits by \(a\) and measure entropy of the resulting attention distribution.
- **Basis-change stress test (toy):** in a 2-layer linear network, apply \(G\) in the hidden layer and compensate with \(G^{-1}\) in the next layer; verify identical outputs.

---

## 8. Where this is going

Next notes in this sequence could be:

- **Gauge-like symmetries as a taxonomy:** exact vs approximate, parameter-space vs activation-space.
- **Equivariance vs gauge freedom:** when “symmetry” constrains the function vs when it creates non-identifiability.
- **A practical “gauge-invariant” probing protocol:** how to test whether a claimed feature is robust to reparameterizations.

---

*Status:* draft. I’ll revise this as I add sharper examples and a clean set of definitions.
