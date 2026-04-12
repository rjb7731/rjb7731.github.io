---
layout: post
title: "Training a 49-Million-Parameter Model to Match Planning Docs to OS Maps"
date: 2026-04-11 12:00:00
description: "Matching planning documents to the entirety of Greater Manchester using contrastive learning and synthetic data."
tags: machine-learning computer-vision mapping sovereign-ai
---

**Training a single 49-million-parameter model to match planning documents to OS Maps for the entirety of Greater Manchester!**

Most plans within the past 20 years rely on data taken from Ordnance Survey (OS) mapping. Before this, people used to hand-draw buildings, roads, and trees with pencils and pens—hatching lines, hand-signing certain sections, and smudging ink. The great thing with "new" mapping is that, generally, people no longer go through this effort and use these maps instead.

This does raise the question: can we train a system on these actual maps for the purpose of being able to place plans exactly where they belong on an official map? Any such system would need to match the visual patterns of roads, buildings, and rivers as a form of spatial fingerprinting. It feels like this is something that should be possible with relatively small compute, without the need to rely on huge multi-billion parameter models and squashing their capabilities into this niche task.

Richard Sutton’s **"The Bitter Lesson"** comes to mind here; a lot of these advanced prompting strategies and "agent" frameworks that many AI Engineers put together are trying to encode complexity into a process which should just be a job for deep learning to learn by itself.

First thing to think about when starting any AI projectm and this one is no different, is where the data is going to come from to prove this out. One exciting area is the possibility of synthesising new forms of data for the purposes of training intelligent systems that generalise. Inspired by early DeepMind experiments such as AlphaGo (which learned through self-play) and their work on Atari games combined with reinforcement learning, I thought it would be interesting to gamify the process of matching planning documents back to their OS Map location. The plans should reflect what we would see in actual council plans, which might be poor scans or have distractors such as stamps and large white spaces on the page.

If I could train a model to recognize spatial fingerprints associated with roads, rivers, and buildings, it should be able to locate a planning document simply by looking at the map. This would mean no text reading, no ground control points, and no complex georeferencing transforms. Just visual pattern matching—a field where deep learning models have excelled since the "AlexNet moment" way back in 2012.

---

## Step 1: Building the Environment

I started with OS Open Map Local data, which is free vector data covering all of Great Britain, including roads, buildings, water, woodland, and railways. Using this, I rendered a grid of clean reference tiles covering Greater Manchester: **3,478 tiles**, each representing a 1km × 1km area. Each tile captures the raw spatial structure of its location: road layout, building footprints, and water features.

I built a synthetic document generator that produces realistic planning documents from the OS vector data. Each document includes:

- The map rendered in color or monochrome OS styles.
- Title blocks ("HM Land Registry", "Planning Application", council names).
- Scale bars, north arrows, copyright notices.
- Road name labels at appropriate angles.
- Building numbers.
- Subject property highlighting with red outlines and hatching.
- Randomized page layouts (portrait/landscape, varying margins).

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/synth_plan.png" title="Synthetic Document Generation" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 1: Examples of synthetic planning documents generated with perfect ground truth.
</div>

The key point, though, is that I know exactly where each document is located, giving me perfect ground truth for training!

---

## Step 2: Contrastive Learning

The model architecture is conceptually simple: a dual encoder trained with contrastive loss, similar to CLIP but for maps. Two ResNet50 encoders (pretrained on ImageNet) map images into a shared 256-dimensional embedding space:

1. **The query encoder** processes planning documents.
2. **The tile encoder** processes reference map tiles.

During training, matching pairs (a planning document and its corresponding tile) are pushed together in embedding space, while non-matching pairs are pushed apart. The model learns to ignore rendering styles, title blocks, labels, and annotations, focusing instead on the spatial features that identify a location.

I utilized **ResNet50** as it meant I did not have to go through the pain and effort of teaching the model to recognize shapes and patterns from scratch. At inference time, encoding a new planning document and finding the nearest tile embedding is a single matrix multiplication—essentially instant. All my results and training have been conducted on a single MacBook Pro.

---

## Results: Urmston (Proof of Concept)

On a 4km × 4km area with 225 tiles:

| Metric             | Value                        |
| :----------------- | :--------------------------- |
| **Top-1 Accuracy** | 65%                          |
| **Top-5 Accuracy** | **98.6%**                    |
| **Random Chance**  | 0.4%                         |
| **Parameters**     | 49M (ResNet50 backbone)      |
| **Training Time**  | ~50 minutes on a MacBook Pro |

The correct tile is in the model's top 5 guesses 98.6% of the time.

---

## Results: Greater Manchester (Scale Test)

Scaling to all of Greater Manchester—**3,478 tiles** covering a 30km × 30km area—with only 1,993 training documents (less than one per tile on average):

| Metric                 | Value     |
| :--------------------- | :-------- |
| **Top-1 Accuracy**     | 51.4%     |
| **Top-5 Accuracy**     | **79.8%** |
| **Random Chance**      | 0.03%     |
| **Training Documents** | 1,993     |
| **Total Tiles**        | 3,478     |

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/geolocation_1.png" title="Greater Manchester Scale Test" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 2: The model identifying correct tiles across a major metropolitan area.
</div>

The model correctly identifies the exact 1km square more than half the time, and the correct location is in the top 5 guesses four-fifths of the time across a major metropolitan area. This is achieved with completely different rendering styles between the planning documents and reference tiles.

For context, this is achieved with less than one training example per tile. The model is genuinely learning spatial features, not memorizing.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/geolocation_2.png" title="Greater Manchester Scale Test 2" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 3: The model identifying correct tiles across a major metropolitan area.
</div>

---

## The Takeaway

The big takeaway here is the synthetic data pipeline. The ability to generate unlimited, perfectly-labeled planning documents from OS vector data is a powerful idea. The approach is simple enough that a single developer can build it on a laptop, and the pipeline can be retrained for any council area using freely available OS data.

For councils with backlogs of modern planning documents, a tool that takes a scanned plan and says "this is located here" could be built, trained, and deployed locally for effectively zero cost. No API subscriptions, no cloud dependencies, and no per-document charges.
