---
layout: post
title: Building a GAIA-1 Style World Model for Indoor Robot Navigation
date: 2026-04-01
description: How I trained an action-conditioned video world model from scratch on a home robot dataset.
tags: [robotics, world-models, deep-learning]
---

## What is a World Model?

A world model is a neural network that learns to predict _what the world will look like next_ given a current observation and an action. Instead of directly learning a policy (what action to take), you first learn the dynamics of the environment and then you can plan, imagine counterfactuals, and reason about consequences.

This is the core idea behind [GAIA-1](https://wayve.ai/thinking/scaling-gaia-1/) from Wayve, which does this for autonomous driving at scale. I wanted to build something similar for indoor robot navigation. Wayve as a company I think we should be really proud of in the UK, originally starting in Cambridge they are now leading the way with autonomous vehicles and their cars are on our roads in the UK today. Home robotics is an area ripe with research and investment and is one area we are seeing huge advancements. 

The best way to achieve my aim here and learn, as always, is to get hands on and try and solve a problem after all "What I cannot create, I do not understand" - Feynman.

## The Setup

First things first I set about buying and building a raspberry PI controlled car. This would work as my data collection engine and through tele-operation, I would collect information about my home. It collects data in RGB frames at ~10fps paired with motor commands (left/right wheel speeds). Each trajectory is a few hundred frames of the robot navigating through rooms.

Did some tinkering and got the rig together, I then was able to SSH into the car and upload some .py files for it to take commands and send visuals from its camera back using pyzmq. 

<div class="row justify-content-center">
    <div class="col-sm-8 col-md-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/pi_car.jpeg" title="Pi Car" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption text-center">
    Figure 1: PI car after finishing the build.
</div>


The pipeline has two stages:

**Stage 1 — DINO VQ-VAE**: Compress each 128×128 image frame into 64 discrete tokens using a Vector Quantised VAE with a DINO distillation loss. This gives semantically meaningful tokens — similar scene regions get similar token IDs. These tokens become useful as my Transformer that I want to train can use these tokens as a representation of say a couch or a rug as a means to view how it changes with the respective robot actions. We know through GPT models how powerful these models can use language tokens to help users and perform coding tasks etc. Instead of just minimizing pixel-wise error (which leads to blurry averages), the DINO distillation forces the VQ-VAE to group pixels into semantic clusters. It learns that a 'rug' is a single entity, making the Transformer's job of moving the rug with actions much easier.

**Stage 2 — GAIA Transformer**: A causal GPT-style transformer trained on sequences of `[frame_tokens, action, frame_tokens, action, ...]`. Given N context frames and an action, it predicts the next frame's tokens autoregressively. The transformer here is trained on the trajectories of these compressed latent representations of my home and learns how to recreate these latents frame by frame so when I feed the latent back into the decoder portion of my above mentioned VQ-VAE it should be able to build the images back up!

You could say I am creating a mini 'world' model of my own home here....we have Wayve at home!

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/gaiai_showcase_2.png" title="Example rollouts" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption text-center">
    Figure 2: Rollout trajectories examples, notice the rugs and cabinets changing with turns.
</div>

## Results

The model learned real spatial structure from trajectory data alone there is no 3D supervision, no depth sensors. Applying TURN_LEFT causes the scene to pan right; applying TURN_RIGHT enough steps eventually reveals a completely different part of the room.

Most impressively, the model maintains "object permanence." If it pans away from the couch and then turns back, the couch reappears in the correct relative position. It hasn't just learned a video loop; it has internalized the room's geometry.

<div class="row justify-content-center">
    <div class="col-sm-10 col-md-8 mt-3 mt-md-0">
        <video width="100%" height="auto" autoplay loop muted playsinline class="img-fluid rounded z-depth-1">
            <source src="{{ '/assets/video/gaia_rollout.mp4' | relative_url }}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
</div>
<div class="caption text-center">
    Figure 3: Imagined rollout. The model "dreams" a 360-degree turn based on consecutive actions.
</div>

## What's Next

- Better action conditioning (continuous motor values instead of 5 discrete actions)
- Longer rollouts with less drift
- Using the world model for planning and policy learning