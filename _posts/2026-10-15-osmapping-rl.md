---
layout: post
title: "Geolocating Planning documents with Agents and Reinforcement Learning"
date: 2026-10-15 18:10:00
description: "Matching planning documents to the entirety of Greater Manchester using tiny models and reinforcemment learning"
tags: machine-learning computer-vision mapping sovereign-ai reinforcement-learning
---


# Beyond Static Embeddings: Training a 6,000-Parameter RL Agent to Actively Hunt Down Historical Planning Maps

**Created on October 15, 2026**
`2026` · `reinforcement-learning` · `markov-decision-processes` · `computer-vision` · `sovereign-ai`

*Why throw a huge LLM or brute-force grid search at a spatial business problem when you can frame it as a Markov Decision Process (MDP) and solve it in milliseconds with a tiny policy network in pure NumPy?*

```html
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/agent_search.gif" title="Agent Search in Action" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 2: The agent actively searches the map to locate the correct property.
</div>
```

*Figure 1: Our 6,000-parameter Decision Encoder actively exploring real Ordnance Survey geometry in Trafford. Starting from a rough street-level geocode (blue cross), it zooms out, spots a candidate peak across the neighbourhood, flies over, zooms in to verify, and commits to the exact Land Registry plot (green).*

---

Back in April, I wrote about training a 49-million-parameter contrastive ResNet50 model to geolocate planning documents across Greater Manchester. That approach treated map localization like a giant one-shot classification problem: chop Greater Manchester into 3,478 reference tiles, embed the document, run a matrix multiplication, and hope the nearest tile embedding is the right one.

It worked well as a proof of concept, but as I dug deeper into real-world council archives and started following Sergey Levine’s Deep Reinforcement Learning course (CS285) at UC Berkeley, I realized that **one-shot static perception is the wrong abstraction for how real localization works**.

When a human caseworker—or an intelligent system—tries to locate a messy, historical planning document on a map, they don’t take a single glance at the entire city and guess. They **investigate**:

1. They start at a rough clue (an address, a postcode, or a street name).
2. They look around at a medium zoom[cite: 1, 7]. If the street layout doesn't match, they **zoom out** to get their bearings.
3. If the title block says `Scale 1:1250`but nothing lines up, they suspect the OCR or title block is wrong and **test a `1:2500` scale hypothesis**.
4. In repetitive suburban estates where three cul-de-sacs look almost identical, they **jump between rival candidates** at fine resolution to compare them side by side.
5. Most importantly, if the scan is too degraded or ambiguous, they **know when to escalate to a human** rather than corrupting the database with a bad guess.

Instead of jumping on the LLM wrapper bandwagon, I wanted to see if we could turn this exact operational workflow into a **Markov Decision Process (MDP)** and train a tiny neural network via **Imitation Learning + Reinforcement Learning (REINFORCE)** to actively explore the map.

---

## Step 1: Framing Business Economics as an MDP

In reinforcement learning, an agent interacts with an environment over discrete time steps: observing a state $s_t$, picking an action $a_t$ from a menu, and receiving a reward $r_t$.

To test this properly without relying on clean synthetic grids, I built an environment on top of **real Ordnance Survey OpenMap Local buildings, Open Roads, and HM Land Registry INSPIRE title polygons** across Trafford and Greater Manchester.

### The Harder Reality: Hand-Drawn Scans & Demolished Buildings

Lets be honest modern digital plans are easy. The real headache for councils is historical plans they are documents traced by hand decades ago against older maps, where terraced houses were merged into single blocks, pen lines wobble, and **10% to 30% of the buildings on the map have since been demolished or newly built**.

On top of visual degradation, the starting address prior is often unreliable:

* **UPRN (45%):** Near the property, though 12% of the time it geocodes to the wrong house up to 200m away.
* **Postcode (25%):** Anywhere within a 100m to 350m radius of the site.
* **Street (15%):** Dropped at a random midpoint along the named road.
* **None (15%):** Dropped at a completely random coordinate in a 2.4km $\times$ 2.4km town.

### The Action Space (17 Discrete Decisions)

At each step (up to a budget of 16 looks), the agent views a local crop of the map through a fast Normalized Cross-Correlation (NCC) template matcher and chooses from 17 actions:

* **Navigate:** `pan_N`, `pan_E`, `pan_S`, `pan_W`, `zoom_in`, `zoom_out` (switching between 960m coarse, 480m medium, and 240m fine viewports), or `goto_peak` (center on the highest correlation peak in the current view).
* **Investigate Rivals:** The environment keeps a short-term memory of the top 3 distinct match locations seen so far. Picking `check_0`, `check_1`, or `check_2` jumps straight to that candidate at the finest zoom level to verify it.
* **Gather Clues & Test Hypotheses:** `read_street` runs OCR on the plan to unlock street jump candidates (`jump_0..2`), while `switch_scale` flips the agent's scale hypothesis between `1:1250` and `1:2500` (since 25% of plans have misread or missing scale text).
* **Terminal Actions (`commit` vs. `escalate`):** Committing runs a sub-pixel registration and checks if the plan's red boundary polygon overlaps the true Land Registry plot ($\text{IoU} \ge 0.5$). Escalating stops the search and flags the document for human review.

### Encoding Operational Economics in the Reward Function

Why use RL instead of a simple threshold? Because business workflows have asymmetric costs:

* **Each step/look (`-0.02`):** Compute and latency budget. Looking at 50 windows is expensive.
* **Correct commit (`+1.0`):** Automated throughput.
* **Wrong commit (`-1.0`):** A false positive in a statutory register is a costly failure.
* **Escalate (`-0.35`):** Passing a tough document to a human costs time, but is far preferable to recording a wrong location.

---

## Step 2: The Brain (A 6,000-Parameter Decision Encoder in Pure NumPy)

Does the RL policy need to be a massive Vision Transformer? **No.**

The template matcher already turns image comparisons into spatial correlation surfaces. What the agent actually needs is a **sequential decision policy** that looks at the *summary statistics* of the search so far and decides what to do next.

I built a `DecisionEncoder` in pure NumPy—a two-layer, 64-unit `tanh` Multi-Layer Perceptron (MLP) with a companion Value Network baseline:

* At each step, it takes a 48-dimensional global situation vector $G$ (current zoom, remaining step budget, current peak score, peak distinctiveness over the runner-up, distance from the geocode, scale state, and the score lead of the top remembered candidates) concatenated with a 21-dimensional per-action feature vector $A[a]$.
* Because it operates on structured state-action features `[situation; option]`, the entire policy network is roughly **6,000 parameters**, runs in microseconds on a CPU, and requires zero deep learning frameworks.

---

## Step 3: SFT Then RL (Why RL From Scratch Fails)

To train the policy, I used the two-stage pipeline standard in modern RL:

1. **Supervised Fine-Tuning / Behavior Cloning (SFT):** First, I wrote two rule-based heuristic agents (`coarse_to_fine` and a more cautious `smart_expert` that double-checks rival peaks) and recorded 2,400 episodes (~20,500 expert decisions) on a 2.4km $\times$ 2.4km training area in Trafford. The MLP is trained via cross-entropy to imitate the expert (`96.8%` action accuracy).
2. **REINFORCE with a Learned Value Baseline:** Starting from the cloned weights, the agent plays its own rollouts. For every trajectory $\tau$, we compute the reward-to-go $R_t = \sum_{t'=t}^{T} r_{t'}$ and subtract the Value Network's expected return $V_\phi(s_t)$ to get the **Advantage**:

$$A_t = R_t - V_\phi(s_t)$$

We then update the policy weights along the policy gradient, combined with an entropy bonus $\mathcal{H}(\pi_\theta)$ to encourage exploration and annealed potential-based distance shaping:

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i, t} \nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t}) \cdot A_{i,t} + \beta \nabla_\theta \mathcal{H}(\pi_\theta(\cdot \mid s_{i,t}))$$

### The "Always Escalate" Trap When Training RL From Scratch

What happens if you skip Behavior Cloning and train REINFORCE from random weights (`--scratch`)?

In our experiments, a randomly initialized agent wanders around the map and makes random commits, losing `-1.0` eighty percent of the time (for an average starting return of `-0.66`). Within 50 iterations, **RL-from-scratch discovers a clever, degenerate local optimum**: if it simply presses `escalate` on Step 1 of every single episode, it collects a guaranteed `-0.35` reward with `0%` wrong commits! Once it collapses into 100% instant escalation, it never experiences a `+1.0` commit again.

Warm-starting with Behavior Cloning (`SFT`) places the policy into the region of state space where it actually finds maps, and **REINFORCE (`SFT+RL`)** then refines its decision boundaries far beyond the expert it copied.


*Figure 2: Step-by-step filmstrip of the SFT+RL agent locating a degraded hand-drawn plan (`hand` style, where buildings have changed since the plan was drawn). Each panel shows the agent's local viewport crop, the matcher peak (orange crosshair), and the Decision Encoder's live action probability distribution.*

---

## Results: Evaluating on Unseen Trafford Map Geometry

I trained the system on 1,200 plans in one 2.4km $\times$ 2.4km area of Trafford (`BNG 374000, 393000`) and evaluated every method on the exact same 400 unseen test episodes in a separate 2.4km $\times$ 2.4km town (`BNG 377000, 391000`).

| Method | Correct (Recall) | Wrong (Bad Commit) | Escalate | Avg Looks | Precision | Total Reward |
| --- | --- | --- | --- | --- | --- | --- |
| **Geocode Commit** (no search) | 32.8% | 17.0% | 50.2% | 1.0 | 65.8% | `-0.018` |
| **Exhaustive Grid** (1 scale) | 29.5% | 1.3% | 69.3% | 26.0 | 95.9% | `-0.460` |
| **Exhaustive Grid** (both scales) | 30.0% | 1.8% | 68.2% | 51.0 | 94.5% | `-0.956` |
| **Coarse-to-Fine** (heuristic) | 65.7% | 5.2% | 29.0% | 6.7 | 92.6% | `+0.355` |
| **Smart Expert** (heuristic) | 58.5% | 2.5% | 39.0% | 8.3 | 95.9% | `+0.267` |
| **Pure SFT** (Behavior Cloning) | 59.0% | 2.5% | 38.5% | 8.2 | 95.9% | `+0.265` |
| **SFT + RL (REINFORCE)** | **76.2%** | **7.0%** | **16.8%** | **5.2** | **91.6%** | **`+0.548`** |

*Paired 95% bootstrap confidence interval of **SFT + RL** over the best hand-written baseline (`coarse_to_fine`): **Reward `+0.193` `[+0.126, +0.254]**`, **Correct `+10.5%` `[+6.5%, +14.2%]**`, while taking **1.5 fewer looks** per plan.*

### Where RL Crushes Rules: Hand-Drawn & Historical Plans (`hand` style)

When we break down the results by plan style and starting address quality, we see why learned sequential policies beat hand-tuned thresholds:

| Subset | Coarse-to-Fine | Smart Expert | Pure SFT (Cloned) | **SFT + RL (Ours)** |
| --- | --- | --- | --- | --- |
| **Hand-Drawn + Demolished Buildings (`hand`)** | 66% *(7.0 steps)* | 60% *(8.4 steps)* | 56% *(8.8 steps)* | **78% *(5.3 steps, 92% prec)*** |
| **Modern Colour (`os_colour`)** | 65% *(6.6 steps)* | 55% *(8.6 steps)* | 59% *(8.0 steps)* | **78% *(4.9 steps, 89% prec)*** |
| **Modern Monochrome (`os_mono`)** | 66% *(6.6 steps)* | 60% *(8.1 steps)* | 60% *(7.9 steps)* | **75% *(5.3 steps, 92% prec)*** |
| **Exact / Near Address (`uprn`)** | 81% *(4.6 steps)* | 70% *(6.4 steps)* | 70% *(6.3 steps)* | **91% *(3.1 steps, 94% prec)*** |
| **Postcode Only (100–350m off)** | 64% *(7.0 steps)* | 55% *(8.8 steps)* | 57% *(8.4 steps)* | **83% *(5.5 steps, 91% prec)*** |
| **Standard `1:1250` Plans** | 71% *(6.2 steps)* | 65% *(7.5 steps)* | 66% *(7.4 steps)* | **81% *(5.0 steps, 96% prec)*** |

Look at what happened between **Pure SFT (`59.0%` overall, `56%` on hand-drawn)** and **SFT + RL (`76.2%` overall, `78%` on hand-drawn)**:

* The hand-written `smart_expert` was tuned to be cautious (`95.9%` precision), which caused it to second-guess itself on degraded hand-drawn plans—burning **8.4 steps** checking rival peaks and escalating 40% of the time.
* Pure Behavior Cloning (`sft_decision_encoder`) faithfully copied that hesitation (`56%` correct in `8.8` steps).
* Once unleashed with **REINFORCE**, the policy discovered that it didn't need a rigid correlation threshold. By combining the current peak score, local distinctiveness (`score - second`), and the lead over remembered candidates in `obs()`, **SFT + RL boosted recall on hand-drawn historical plans by +22% over the cloned expert (`56%` $\rightarrow$ `78%`) while cutting search time from 8.8 steps down to 5.3 steps!**

---

## The Takeaway: Small MDPs Over Giant Models

There is a temptation in AI engineering right now to treat every multi-step business problem as a prompt-engineering exercise for a massive vision-language model.

But when you strip a problem down to its actual structure—**what is the state, what are the actions, and what are the business costs of time, errors, and human escalation**—you can often solve it with a 6,000-parameter network that trains on a laptop CPU in 15 minutes and executes in microseconds.

By pairing fast classical signal processing (Normalized Cross-Correlation) with a lightweight RL decision policy (`SFT + REINFORCE`), our agent doesn't just recognize maps—it **actively hunts them down**, adapts to hand-drawn historical drift, verifies its own hypotheses, and knows when to ask a human for help.