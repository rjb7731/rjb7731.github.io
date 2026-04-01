---
layout: post
title: Building a GAIA-1 Style World Model for Indoor Robot Navigation
date: 2026-04-01
description: How I trained an action-conditioned video world model from scratch on a home robot dataset.
tags: [robotics, world-models, deep-learning]
---

## What is a World Model?

A world model is a neural network that learns to predict *what the world will look like next* given a current observation and an action. Instead of directly learning a policy (what action to take), you first learn the dynamics of the environment — then you can plan, imagine counterfactuals, and reason about consequences.

This is the core idea behind [GAIA-1](https://wayve.ai/thinking/scaling-gaia-1/) from Wayve, which does this for autonomous driving at scale. I wanted to build something similar for indoor robot navigation.

## The Setup

I built a small wheeled robot that drives around my house and collects data — RGB frames at ~10fps paired with motor commands (left/right wheel speeds). Each trajectory is a few hundred frames of the robot navigating through rooms.

The pipeline has two stages:

**Stage 1 — DINO VQ-VAE**: Compress each 128×128 frame into 64 discrete tokens using a Vector Quantised VAE with a DINO distillation loss. This gives semantically meaningful tokens — similar scene regions get similar token IDs.

**Stage 2 — GAIA Transformer**: A causal GPT-style transformer trained on sequences of `[frame_tokens, action, frame_tokens, action, ...]`. Given N context frames and an action, it predicts the next frame's tokens autoregressively.

## Key Insight: Positional Encoding

The biggest breakthrough came from fixing a subtle inference bug. The model was trained to predict frame N from frames 1..N-1 at specific positional ranges. At inference, naively appending a new action and generating at position N+1 puts tokens at positions the model *never saw during training* — out-of-range positional embeddings produce noise.

The fix: drop the oldest context frame at each step, so generated tokens always land at the exact positional range the model was trained on. This turned smudgy blobs into coherent room scenes overnight.

## Results

The model learned real spatial structure from trajectory data alone — no 3D supervision, no depth sensors. Applying TURN_LEFT causes the scene to pan right. Applying TURN_RIGHT enough steps eventually reveals a completely different part of the room — the model has internalised the room's geometry.

Here's a video showing the imagined rollout with left and right turn actions from the same starting frame:

*(video coming soon)*

## What's Next

- Better action conditioning (continuous motor values instead of 5 discrete actions)
- Longer rollouts with less drift
- Using the world model for planning and policy learning
