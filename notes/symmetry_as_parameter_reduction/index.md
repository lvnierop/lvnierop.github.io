---
layout: base
title: Symmetries and parameter reduction in neural networks
nav_exclude: true
---

## Symmetries and parameter reduction in neural networks

In this note I explore how symmetries of a neural network architecture can be used to reduce the parameter count, while leaving the model invariant. As a practical example, I will use a very explicit symmetry of the transformer architecture, and I am implementing my experiments in Kaparthy's nanoGPT, in this [fork](https://github.com/lvnierop/nanoGPT).

First I will define my terms for clarity: I want to avoid misunderstandings due to different terminology in math, physics, and computer science. 
* <b> "Transformation" </b>: A transformation of a model is a map of the space of valid parameters (weights, biases) to itself. The original and transformed parameters both define valid instances of the network architecture, but in general will have different activations.
* <b> "Invariant" </b>: A model is invariant under a transformation, if the architecture with the transformed weights gives the same activations in the output layer as the one with the original weights. Internal activations may vary. This has to hold for any conceivable input, so it is a property of the model (both weight sets model the same probability distribution). In practice, we take invariance up to numerical precision errors.
* <b> "Symmetry" </b>: A symmetry is the set of transformations that leaves a model invariant, regardless of what the trained weights are. The symmetries naturally form a group: if two transformations each leave the model invariant, then so does their composition. The symmetries of the architecture create equivalence classes of weights that all model the same function. The equivalence classes are sometimes called the orbit(s) of the symmetry group.
* <b> "gauge" </b> In physics, symmetries of model parameters that leave the physics of the model invariant are often refered to as gauge symmetries. This note is based on those ideas, so in some places (especially in the code) the word gauge appears. Just noting it down here, it means rougly the same thing as symmetry. "Gauge fixing" refers to picking a condition that the symmetry can always satisfy by some transformation, and imposing the condition on the dynamical system.

### The core idea
When we identify a symmetry of an architecture, we can ask the question: Is there a reason to prefer certain members of the equivalence class over others? From the point of view of the model quality (test scores, generalization, etc.) the answer is no: by definition all the members of the equivalence class are, well... equivalent. However, there is potentially a difference in terms of efficiency. I am exploring two ways we may unlock efficiencies.

First off, there is a potential efficiency in the forward pass of the model (which also would improve serving speed), if we can replace some of the weight matrices by identity blocks, and remove the compute required by those blocks. In this note I will show an explicit construction that creates such identity blocks within all equivalence classes, making it a symmetry justified parameter reduction (spoiler: my current implementation does not realize a speed up, the gains are small and get lost by the overhead of splitting and concatenating weight matrices. Still looking for a better way to split)

Secondly, the training phase can be made more efficient, because the identity blocks mentioned above are not learned: each equivalence class has a member with the same block of weights replaced by the identity, so instead of learning those weights and transforming them away after the fact, they can be set as untrainable identity operators right from the beginning (more fun spoiler: small training speedup on cpu).

### An explicit realization in transformers

In the transformer architecture, the attention is calculated using the product of the key and query matrices. Importantly, there is no activation on the key and query calculations, so we can insert the identity matrix in the form $M^{-1}M$, with M an arbitrary invertible $d_{head}\times d_{head}$ matrix. We can then apply them to the left and right, getting the following mapping:

$$ K^h = W_k^h X,$$ 

$$ \hat K^h = (M^h W_k^h) X = \hat W_k^h X,$$ 

$$Q^h = W_q^h X,$$

$$\hat Q^h = (((M^h)^{-1})^T W_q^h) X = \hat W_q^h X$$

Invariance follows because those weights only enter in the combination

$$
(\hat Q^h)^T \hat K^h = X^T (W_q^h)^T (M^h)^{-1} M^h W_k^h X = X^T (W_q^h)^T W_k^h X = (Q^h)^T K^h
$$

### Experiments in code
Here I will describe the experiment I have done, including files/scripts to run for reproducing my results. The code i used is in [this commit](https://github.com/lvnierop/nanoGPT/commit/38c167affb30e58e37fd9746a108444146e4fd8e).

First, we need a trained model. I used the basic no-gpu instructions:

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

Next, we create 3 new model checkpoints based off the checkpoint created by this file. One of them applies a random invariant transformation to each head as specified above. Another one applies a random transformation to the keys and values independently (without imposing the inverse transpose relationship). A third one applies a symmetry transformation that explicitly inverts the first $(d_{head}\times d_{head})$ block of each head's key projection, rather than picking the matrix at random. All this is done with the command:
```sh
python key_value_rotations.py
```
If you are following along, note that the base and output directories are hard-coded, to be fixed later. Now that we have the 4 models, we want to assert the following:

*Given identical inputs, both the max and mean difference in output logits is small when comparing the base model to the two symmetry transformed models, but much larger when comparing to the randomly transformed model that doesn't respect the symmetry condition.*

This step is performed by
```sh 
python compare_chkpts.py /path/to/checkpoint_dir /path/to/other_dir
```

|base model | comparison model            | diff max | diff mean|
| --- |-----------------------------| --- | --- |
|out-shakespeare-char| out-shakespeare-char-gauged |3.6955e-05|6.1509e-06|
|out-shakespeare-char| out-shakespeare-char-gauge-fixed  |9.4414e-05|2.6329e-05|
|out-shakespeare-char| out-shakespeare-char-random |5.9480|1.9566|

This clearly shows that the idea of gauge transformations works as expected. We do see slightly higher changes on the gauge fixed version (where we set a block to the identity). This comes from the inverse typically requireing some large-ish eigenvalues compared to a random transformation. That causes larger cutoffs on the float precision, leading to slightly larger output differences (although still negligible compared to the random change).

### Implementing an identity block passthrough

An alternative attention calculation is done in model.py line 19, skipping the first $n_{head}$ columns of the key matrix. The key projection is created by concatenating the first $d_{head}$ entries of X and the output of the smaller projection matrix applied to X. To benchmark the forward pass I ran 
```sh 
python bench_attn.py
```
Unfortunately, this implementation does not actually give any speed increase (actually it slows down a little bit). It seems that the reduction in computation is swamped by the overhead of stitching tensors. 

### Training with the identity block passthrough

In order to test training gains, I ran training on [this commit](https://github.com/lvnierop/nanoGPT/commit/35c4c43e990e9860b041c001dcaf7b1478863af1). I trained shakespeare-char on cpu and gpu, with and without the passthrough blocks enabled. Note that the model settings are the default choices in Kaparthy's readme for the given devices

| config | device | train_time | val_loss |
| --- | --- | --- |
| train_shakespear_char.py | cpu | 49.97 | 1.8857 |
| train_shakespear_char_passthrough.py | cpu | 47.57 | 1.8857 |
| train_shakespear_char.py | gpu | 2958.0 | 1.7009 |
| train_shakespear_char_passthrough.py | gpu | 2904.3 | 1.7009 |

The speed up on cpu is on the order of 5%. The loss is identical, not just in the final result but also at the intermediate checkpoints. This points to the improvement being purely a speed up due to reduced work (by not computing gradients on part of the key projection matrix). The image also shows that the time difference grows linearly with the iterations, so this is a per-step improvement rather than an overhead or random noise quirk. On the gpu, on the other hand, we have a slowdown of about 2%. This points again to the case where the overhead of stitching tensors is more expensive than the computational gains. Here, too, the difference grows with epochs linearly so it is a per-loop overhead. 

<p align="center">
  <img src="/assets/images/time_cpu.png" width="48%">
  <img src="/assets/images/time_gpu.png" width="48%">
</p>

A final note on the loss at the various checkpoints: They are not just "similar", but identical to the 4 decimal places that nanoGPT records them. One thing that implies is that the symmetry doesn't just preserve the function modeled by the neural network identically: it even preserves the loss landscape. 

<p align="center">
  <img src="/assets/images/val_loss.png" width="48%">
  <img src="/assets/images/val_loss_gpu.png" width="48%">
</p>

### Conclusions

* The symmetry transformations leave the end to end model invariant as predicted (up to floating point precision)
* The expected gains of removing a block of key weights fail to materialize on the forward pass: orchestration/tensor stitching overhead swamps gains in the forward pass
* Training on the cpu benefits with about a 5% speed up
* Training on the gpu actually loses 2% speed. Crucially, this holds *in the current implementation*. Potentially, we can find a way to implement a gauge choice like this (or a better one) while avoiding too much tensor stitching to maintain the speed win on the gpu.

<b>Final takeaway: Yes, symmetries can be used to systematically remove parameters from a neural network. Wether it is beneficial depends on if we can do so in a GPU friendly way (which may be on a case by case basis).</b>

### Future work

* Attempt different implementation that needs less tensor stitching (eg. directly in the attention block rather than in the qkv projection only)
* Identify other symmetries that can benefit from this treatment
