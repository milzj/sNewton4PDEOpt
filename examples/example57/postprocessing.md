---
date: 'July 17, 2022'
title: Postprocessing
---

We consider

$$\min\_{u \in {U\_{\text{ad}}}}\, 
    G(u) := F(u) + \beta {\\|u\\|\_{L^1({D})}},$$ where
$F : L^2({D}) \to {\mathbb{R}}$ is strictly convex and continuously
differentiable, and $\beta \geq 0$. We choose
${U\_{\text{ad}}}= \{\, u \in L^2({D}) :\, a \leq u \leq b\,\}$ with
$-\infty < a  < 0 < b < \infty$.

An approximate solution to the above problem can be obtained via the
solution $u\_\alpha^*$ to the regularized problem

$$\min\_{u \in {U\_{\text{ad}}}}\, 
G(u) + (\alpha/2){\\|u\\|\_{L^2({D})}}^2,$$ where $\alpha > 0$.

If $\nabla F(u\_\alpha^*) \in H\_0^1({D})$, then
$u\_\alpha^* \in H^1({D})$. However, the solution $u^*$ to the
nonregularized problem fulfills $u^*(x) = \{a, 0, b\}$ for almost every
$x \in {D}$, provided that the measure of
$\{x \in {D}: \|\nabla F(u^*)(x)\| = \beta\}$ is zero.

Given the solution $u\_\alpha^*$ to the regularized problem, we may use
the following control as an approximate solution to nonregularized
problem:

$$\widetilde{u}(x) = 
    \begin{cases}
    a & \text{if} \quad \nabla F(u\_\alpha^*)(x) > \beta \\
    b & \text{if} \quad \nabla F(u\_\alpha^*)(x) < -\beta \\
    0 & \text{else}
    \end{cases}$$

Using Lemma 5.6 in [K. Kunisch and D. Walter
(2021)](https://arxiv.org/abs/2109.15217), we find that $\widetilde{u}$
solves

$$\min\_{v \in {U\_{\text{ad}}}}\, 
    (\nabla F(u\_\alpha^*),v)\_{L^2({D})}
    +\beta {\\|v\\|\_{L^1({D})}}.$$

A natural question is whether

$$G(\widetilde{u}) \leq G(u\_\alpha^*)$$ holds true. We verify this
inequality empirically one an example.
