---
layout: base
title: Skip Connections induce a differential equation
nav_exclude: true
date: 2026-06-12
---

In this note I explore a potentially useful connection between neural network architectures with skip connections, and differential equations. The key observation is that any neural network block with a skip connection is mapping a vector space to itself. This means that such blocks define a flow vector field, and take a finite size step along that flow field. When such blocks are stacked, as is often the case, we have a flow field that changes as we move along the depth of the stack. I will show that there is a sense in which any such stack can be viewed as a numerical approximation to an underlying differential equation. The parameters of the differential equation are not unique, but there is a unique smoothest version that I derive.

---

## 1. Formal expression of skip connections

Let's consider an input vector space $V$, and an operator family parametrized by weights: $M(w): V\rightarrow V = I + \delta M(w)$. For example, the delta operator in the case of transformers is the full attention mechanism as well as the fully connected piece. 
The parameters $w$ are different for each transformer block. In order to consider the depth of the stack as a formal dimension for the purpose of the differential equation, consider a stack of $n_b$ blocks. Choosing the initial state to sit at $t=0$, and the final state at $t=1$, the layers now look like:

$$
x_{n+1} = x_{n} + \delta M(w(t=n/n_b)) x_{n}
$$
$$
x\bigl(t=(n+1)/n_b\bigr) = x\bigl(t=n/n_b\bigr) + \delta M\bigl(w(t=n/n_b)\bigr) x\bigl(t=n/n_b\bigr)
$$

Note that this expression looks exactly like the Euler approximation of a differential equation, which is the basis for this post. 

## 2. Finding appropriate continuous weight functions

The formal connection between neural networks with skip connections and differential equations makes the leap from a discrete set of weights, $w_i$, to a continuous weight function, $w(t)$. At face value this is a dramatic increase in parametrization. However, there is a minimal choice that has the same parameter count as the original: take the fourier series on the interval 0-1, and only keep the first n terms, where n is the depth of the original stack. 

Of course, we can have any choice for the higher order terms in the series, and alias them to agree with the original function on the only points that matter. However, choosing the lowest order terms has the benefit of minimizing gradients of the weight function, so it that sense it is the smoothest choice for the weights. Explicitly, we write:

$$
w_k = \sum_{n=0}^{n_b-1} w(t=n/N) e^{-2\pi i kn/N}
$$

For $n_b$ odd, we have

$$
w(t) = \frac1{n_b}\sum_{k=-\frac{N-1}{2}}^{\frac{N-1}{2}}w_ke^{i2\pi kt}
$$

while for $n_b$ even we get

$$
w(t) = \frac1{n_b}\left[\sum_{k=-\frac{N-1}{2}+1}^{\frac{N-1}{2}-1}w_ke^{i2\pi kt} + \frac{w_{n_b/2}}{2}\left(e^{i\pi n_bt}+e^{-i\pi n_bt}\right)\right]
$$

where negative k are found by periodic extension, $w_k=w_{k+n_b}$

