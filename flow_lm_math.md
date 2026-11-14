# The Mathematics of Inference in a Flow Language Model (Lafa), and What We Test

This note writes out exactly how Lafa generates text, following the paper (Chavan et al., 2026, arXiv:2610.28559, §2 and App. B) **and** its released inference code (`flow_model.py`, `_euler_denoise_block`). It then shows where, mathematically, the parallel positions of a block can "talk to each other", and what our experiments measure.

---

## 1. Setup and Notation

| Symbol | Meaning |
|---|---|
| $V$ | vocabulary; $d$ = embedding dimension (896 for Lafa-0.5B) |
| $E \in \mathbb{R}^{\lvert V\rvert \times d}$ | token embedding matrix (row $E_v$ = embedding of token $v$; norms fixed to their pretrained values) |
| $c$ | the prefix: prompt + everything generated so far (clean tokens) |
| $B = 4$ | block size: number of tokens generated **in parallel** |
| $y = (y^1,\dots,y^B)$ | the next block of tokens |
| $i, j \in \{1,\dots,B\}$ | positions inside the block |
| $t \in [0,1]$ | base time; $t=0$ is pure noise, $t=1$ is clean data |
| $\tau^i(t)$ | **tokenwise time** of position $i$ (learned; $\tau^i(0)=0$, $\tau^i(1)=1$) |

### Blockwise factorization
Text is generated block by block. Each block is conditioned on everything before it:

$$
p(y_{1:M}\mid c)=\prod_{r=0}^{M/B-1} p\big(y^{(r)} \,\big|\, c,\, y^{(<r)}\big).
$$

Between blocks the model behaves like an ordinary autoregressive (AR) model: the prefix is encoded once with causal attention and cached (KV cache). **Inside** a block, all $B$ tokens are produced **simultaneously** by a flow.

---

## 2. The Flow Inside One Block

### 2.1 Data and noise
The clean block is represented by its token embeddings:

$$
x_1 = (E_{y^1},\dots,E_{y^B}) \in \mathbb{R}^{B\times d}.
$$

Noise is Gaussian: $x_0 \sim \mathcal{N}(0,\sigma^2 I)$ with $\sigma = s\cdot\sigma_{\text{emb}}$, where $\sigma_{\text{emb}}$ is the standard deviation of the embedding table.
- Training uses $s = 4.0$.
- **Inference uses $s = 0.1$**, so generation starts from almost exactly zero.

### 2.2 Probability path (asynchronous interpolation)
Each position $i$ moves from noise to data on its own clock:

$$
I^i_t=\big(1-\tau^i(t)\big)\,x_0^i+\tau^i(t)\,x_1^i .
$$

The schedules are learned so that **earlier positions are denoised faster**: $\tau^1(t)\ge\tau^2(t)\ge\dots$ This gives a left-to-right bias inside the block.

### 2.3 The denoiser (a per-position classifier)
The network does not output a velocity. For every position it outputs a distribution over the clean token:

$$
D_t(x;c)^i = p\big(y^i=\cdot \,\big|\, c,\ I_t=x\big)\in\Delta^{\lvert V\rvert-1},\qquad i=1,\dots,B .
$$

Its expected embedding is the "predicted clean block":

$$
\hat x_1^i = D_t(x;c)^i\,E \quad(\text{a probability-weighted average of embeddings}).
$$

**Each position gets its own distribution (a marginal).** The network sees the whole block $x$ through bidirectional attention, but it never outputs a joint distribution over the 4 tokens. Keep this in mind for §4.

### 2.4 From denoiser to velocity
The velocity that moves the block toward data is recovered as

$$
b_t(x;c)^i=\frac{\dot\tau^i(t)}{1-\tau^i(t)}\big(D_t(x;c)^i E - x^i\big).
$$

---

## 3. The Inference Algorithm (what the code actually does)

Use a grid of $K$ steps ($K=32$ in the paper): $t_k = k/K$. Write $\tau^i_k=\tau^i(t_k)$ (the code precomputes this per-position grid).

### 3.1 Initialization and the self-conditioning bootstrap
$$
x_0\sim\mathcal{N}(0,\sigma^2 I),\qquad q_{-1} = D_{0}(x_0;c)\quad(\text{one extra pass, without self-conditioning}).
$$

### 3.2 Self-conditioning: the model reads its own previous guess
At step $k$ the network's input is not $x_k$ alone:

$$
\boxed{\ \tilde x_k^i = x_k^i + x_k^i W_1^\top + \big(q_{k-1}^i E\big)W_2^\top\ }
$$

Here $q_{k-1}^i$ is the distribution position $i$ predicted at the previous step, and $W_1, W_2\in\mathbb{R}^{d\times d}$ are learned (zero-initialized).
- In training, this is used half the time, with $q$ from a no-gradient pass at the **same** $t$.
- Otherwise the input is plain $x$, which is the model's trained "self-conditioning off" state.

### 3.3 The denoiser call with prefix guidance
Two forward passes per step:
- **Conditional:** $q_k = D_{t_k}(\tilde x_k; c)$. The block attends to the prefix and to itself.
- **Prefix-free:** $q^u_k = D_{t_k}(\tilde x^u_k;\varnothing)$. The prefix keys are masked out, and this pass has its **own** self-conditioning state.

Guided prediction (classifier-free guidance, $g=8$):

$$
\hat x_1^{(k)} = q^u_k E + g\,\big(q_k E - q^u_k E\big).
$$

### 3.4 Euler step (per position)
$$
x_{k+1}^i = x_k^i + \frac{\tau^i_{k+1}-\tau^i_k}{1-\tau^i_k}\,\big(\hat x_1^{(k),i}-x_k^i\big)
\;=\;(1-\alpha^i_k)\,x_k^i+\alpha^i_k\,\hat x_1^{(k),i},
\qquad \alpha^i_k=\frac{\tau^i_{k+1}-\tau^i_k}{1-\tau^i_k}.
$$

Then $q_k$ (the **unguided** conditional prediction) is stored as next step's feedback $q_k \to$ step $k+1$.

### 3.5 Readout
After the last step, each position independently takes the most likely token:

$$
\hat y^i=\arg\max_v\, q_{K-1}^i(v),\qquad i=1,\dots,B .
$$

The block is appended to the prefix, cached, and the next block starts from fresh noise.

### 3.6 A consequence worth knowing: the state is a running average of past predictions
Unrolling the Euler step (§3.4):

$$
x_k^i=\Big(\prod_{m<k}(1-\alpha^i_m)\Big)x_0^i+\sum_{m<k}\Big(\alpha^i_m\!\!\prod_{m<l<k}\!\!(1-\alpha^i_l)\Big)\hat x_1^{(m),i}.
$$

At inference $x_0\approx 0$ ($s=0.1$), so **the state $x_k$ is essentially a recency-weighted average of the model's past (guided) predictions.** Within a couple of steps the state therefore contains the same "which answer" information as the fed-back prediction $q_{k-1}$. We confirmed this directly (`phase4.md` §0.2).

At training noise ($s=4$) the noise term dominates for much longer, so the fed-back prediction carries information the state doesn't yet have.

---

## 4. Where Contradictions Come From

**Readout is per-position** (§3.5): each token is the argmax of its *own* marginal $q^i$. Suppose the right answer has two modes, A = "New York" and B = "San Francisco". Then:
- if position 1's marginal leans to "San" and position 2's leans to "York", the readout is **"San York"**, a **hybrid** that is neither valid answer;
- nothing in the readout itself forbids this.

A continuous flow avoids hybrids **only if, by the end of the trajectory, all positions have committed to the same mode.** Commitment has to be *coordinated* across positions during the 32 steps. The only places where position $j$ can learn what position $i$ is committing to are the **attention reads** inside the denoiser. Through the input $\tilde x^i$ (§3.2), $j$ sees three components of position $i$:

$$
\tilde x^i=\underbrace{x^i + x^iW_1^\top}_{\text{state channel}}\;+\;\underbrace{(q^i_{k-1}E)W_2^\top}_{\text{feedback channel}}.
$$

| Route by which position $j$ ends up agreeing with $i$ | Name in our project |
|---|---|
| $j$ reads $i$'s **feedback** $q^i_{k-1}$ ("I'm going with *San*") | **M2: feedback coordination** (our headline hypothesis) |
| $j$ reads $i$'s **state** $x^i$; SC only makes states better | **R0: state coordination** |
| SC helps each position via its **own** feedback; agreement isn't SC's doing | **M1 / M5: own-position stabilization** |

These routes are summed into one input vector, so the architecture doesn't separate them. **Only interventions can.**

---

## 5. What We Measure and Do (the math of the experiments)

### 5.1 Is mixing real? (phenomenon)
- **Hybrid rate:** on prompts with two valid answers A, B, the fraction of outputs whose tokens come from both A and B (aligned token-wise; includes sub-word mixes like "Albert"+"aac").
- Compared against the **same model decoding autoregressively**, which writes token $i$ conditioned on tokens $<i$ and so cannot mix modes this way.

### 5.2 Does SC repair it? (coherence)
- For a generated block $\hat y$, the in-block **coherence** at position $j$ is $\text{coh}_j=\log p_{\text{AR}}(\hat y^j \mid c,\hat y^{<j})$, under the model's own AR head and under an independent judge (Mistral-7B).
- **Dependence** (does token $j$ fit *its own* predecessors, beyond being a typical token?):

$$
\text{dep}_j=\text{coh}_j-\log\frac1S\sum_{s=1}^S p_{\text{AR}}\big(\hat y^j \mid c,\,y'^{(s),<j}\big),\qquad y'^{(s)}\sim p_{\text{AR}}(\cdot\mid c).
$$

### 5.3 Which channel carries it? (mechanism)
**Mode margin** at a position $j$ where the two answers differ:

$$
\mu_j=\log q^j(a_j)-\log q^j(b_j).
$$

**Single-step direct effect of neighbour feedback.** At step $k$, keep the states $x_k$ fixed. Remove the feedback of source positions $S$ (they get the plain state, i.e. the trained "off" input) and measure the change at $j\notin S$:

$$
\delta^{D}_{S\to j}=\mu_j(\text{full})-\mu_j(\text{feedback of }S\text{ off}).
$$

Within a single step the states can't change, so any effect must travel through $j$'s attention reads of the feedback channel.

**Beyond the state (C1′).** Before measuring $\delta^D$, erase the "which answer" information from the source's **state** $x_k^i$ with linear concept erasure (LEACE), and verify a probe can no longer read it.
- If removing $q^i$ still moves $j$, the feedback carried information the state didn't. **(M2)**
- If not, the information lived in the state. **(R0)**

**Share decomposition (H2).** Over whole trajectories, switch three factors on/off independently:
- $O$: $j$'s own feedback;
- $N$: $j$'s reads of the neighbours' feedback channel (implemented by giving $j$ the neighbours' keys/values from a pass with or without their feedback);
- $F$: the neighbours' own feedback driving their own evolution (the state route).

The 8 cells give an exact decomposition of SC's total effect,

$$
\Delta_{\text{SC}}=\underbrace{m_O+m_N+m_F}_{\text{main effects}}+\underbrace{m_{ON}+m_{OF}+m_{NF}+m_{ONF}}_{\text{interactions}},
$$

and the **N-share** $m_N/\Delta_{\text{SC}}$ is the fraction of SC's benefit carried by feedback coordination.

**Content (C3).** Replace neighbour $i$'s feedback with the feedback it produces on a closely matched prompt that steers to the *other* answer. Does $j$ follow? Compare against equally confident feedback that carries no answer content.

### 5.4 Why we test the mechanism at training noise
§3.6 shows that at inference noise the state absorbs the feedback within ~2 steps, so the two channels carry the same information and can't be separated. At training noise ($s=4$) the feedback leads the state by ~15–20 steps (measured), so M2 and R0 make different predictions there. We also test the inference setting, restricted to its first two steps.

---

## 6. One-Paragraph Summary
Lafa writes 4 tokens at once by moving a block of embeddings from noise to data in 32 steps, reading out each token independently at the end. Self-conditioning feeds each position's previous prediction back into the network. Independent per-position readout can produce hybrids ("San York") unless positions commit to the same answer, and the only way they can coordinate is by attending to each other's state or fed-back prediction. We measure how often hybrids happen and how much self-conditioning reduces them. We then intervene on each route separately (cutting reads, erasing state information, swapping feedback) to find which one actually keeps the positions consistent.
