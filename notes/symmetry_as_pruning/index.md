---
layout: page
title: Symmetries and pruning in neural networks
nav_exclude: true
---

## Symmetries and pruning in neural networks

In this note I explore how symmetries of a neural network architecture can be used to reduce the parameter count, while leaving the model invariant. As a practical example, I will use a very explicit symmetry of the transformer architecture, and I am implementing my experiments in Kaparthy's nanoGPT, in this [fork](https://github.com/lvnierop/nanoGPT).

First I will define my terms for clarity: I want to avoid misunderstandings due to different terminology in math, physics, and computer science. 
* <b> "Transformation" </b>: A transformation of a model is a map of the space of valid parameters (weights, biases) to itself. The original and transformed parameters both define valid instances of the network architecture, but in general will have different activations.
* <b> "Invariant" </b>: A model is invariant under a transformation, if the architecture with the transformed weights gives the same activations in the output layer as the one with the original weights. Internal activations may vary. This has to hold for any conceivable input, so it is a property of the model (both weight sets model the same probability distribution). In practice, we take invariance up to numerical precision errors.
* <b> Symmetry </b>: A symmetry is the set of transformations that leaves a model invariant, regardless of what the trained weights are. The symmetries naturally form a group: if two transformations each leave the model invariant, then so does their composition. The symmetries of the architecture create equivalence classes of weights that all model the same function. The equivalence classes are sometimes called the orbit(s) of the symmetry group.

### The core idea
When we identify a symmetry of an architecture, we can ask the question: Is there a reason to prefer certain members of the equivalence class over others? From the point of view of the model quality (test scores, generalization, etc.) the answer is no: by definition all the members of the equivalence class are, well... equivalent. However, there is potentially a difference in terms of efficiency. I am exploring two ways we may unlock efficiencies.

First off, there is a potential efficiency in the forward pass of the model (which also would improve serving speed), if we can replace some of the weight matrices by identity blocks, and remove the compute required by those blocks. In this note I will show an explicit construction that creates such identity blocks within all equivalence classes, making it a symmetry justified parameter reduction (spoiler: my current implementation does not realize a speed up, the gains are small and get lost by the overhead of splitting and concatenating weight matrices. Still looking for a better way to split)

Secondly, the training phase can be made more efficient, because the identity blocks mentioned above are not learned: each equivalence class has a member with the same block of weights replaced by the identity, so instead of learning those weights and transforming them away after the fact, they can be set as untrainable identity operators right from the beginning. This experiment I still need to do, and it will either be part of a future note or an update on the current note.

### An explicit realization in transformers

In the transformer architecture, the attention is calculated using the product of the key and query matrices. Importantly, there is no activation on the key and query calculations, so we can insert the identity matrix in the form $M^{-1}M$, with M an arbitrary invertible $d_head\times d_head$ matrix. We can then apply them to the left and right, getting the following mapping:
$$
K^h = W_k^h X, \\ \hat K^h = (M^h W_k^h) X = \hat W_k^h X, \\ 
Q^h = W_q^h X, \\ \hat Q^h = (((M^h)^{-1})^T W_q^h) X = \hat W_q^h X \\
$$
Invariance follows because those weights only enter in the combination
$$
(\hat Q^h)^T K^h = X^T (W_q^h)^T (M^h)^{-1} M^h W_k^h X = X^T (W_q^h)^T W_k^h X = (Q^h)^T K^h
$$

