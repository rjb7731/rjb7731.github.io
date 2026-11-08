---
layout: post
title: The Generalisation Myth in Enterprise AI, and Why Your Models Don't Need to Be Massive
date: 2026-11-08
description: Why in-context learning is not adaptation, why production models remain frozen, and why domain intelligence requires far fewer parameters than you think
tags: [machine-learning, llms, enterprise-ai, reinforcement-learning]
---

<style>
  #markdown-content h2, .post-content h2 { margin-top: 3.25rem; margin-bottom: 1rem; }
  #markdown-content h3, .post-content h3 { margin-top: 2.25rem; margin-bottom: 0.75rem; }
  #markdown-content table, .post-content table { margin-top: 1rem; margin-bottom: 1.75rem; }
  #markdown-content ul, .post-content ul { margin-bottom: 1.5rem; }
  #markdown-content blockquote, .post-content blockquote { margin: 1.5rem 0; }
  #markdown-content details summary, .post-content details summary { cursor: pointer; color: var(--global-theme-color, #2a78d6); }
</style>

The prevailing assumption around modern language models is that in-context learning alone will seamlessly adapt them to any novel domain. People feed proprietary documents into a prompt and expect the system to function like a seasoned expert right out of the box.

The trouble is that deployed models are fundamentally static. When you interact with them in production, their weights are frozen. They do not retain memories, they do not update their parameters, and they do not learn from their mistakes within your operating environment.

## The onboarding problem

A great analogy here is onboarding a new hire. You would never expect an employee to operate at full capacity on day one. They need a genuine learning curve to settle in, absorb the company culture, make mistakes, and steadily improve over time.

Handing an LLM a context window is much more like handing someone who has never played chess a copy of the official rulebook. Simply reading the rules does not turn someone into a grandmaster capable of beating seasoned players or even a half decent player! Real competence requires trial and error, feedback, and the experience of winning and losing.

## The notebook versus the grandmaster

To get around the fact that models are frozen, leading labs and frameworks often market external memory layers. Whether it is agent skills, persistent memory files, or dynamic scratchpads, the setup is essentially the same. The model jots notes down into an external file, and then it is trained to look up those notes later to keep its context fresh.

Think about how absurd that would look over a chessboard.

Instead of actually learning from your past games and updating your neural circuits, you sit at the board with a spiral notebook next to you. Every time your opponent makes a move, you pause, flip through your notes to find what you did wrong three games ago, and try to apply it.

That is an incredibly brittle way to play. It burns time, it clutters your attention, and the moment you fail to look up the right page, your entire game falls apart. A grandmaster does not play world-class chess by constantly checking a notebook mid-match. They play world-class chess because thousands of hours of trial, error, and feedback have ingrained deep pattern recognition directly into their brain.

External memory files are helpful for static references, like looking up an API key or checking documentation. But for actual operational competence, procedural skill cannot remain an external lookup. It has to be baked into the system itself.

## Engineered breakthroughs vs spontaneous magic

Part of this confusion stems from the impressive breakthroughs we see from frontier labs. When models pull off remarkable feats in fields like fluid dynamics, 3D modeling, or higher mathematics, it is easy to assume the model taught itself through sheer generalised intelligence. From there, it is tempting to believe it will adapt to your business processes with that same effortless magic.

In reality, those breakthroughs are not spontaneous. They are the direct result of months of deliberate engineering. Dedicated teams design custom reinforcement learning environments, build synthetic training pipelines, and actively update the model weights to instill those exact skills. The models advance because engineers specifically trained them to do so.

> It is worth stating cleanly, models do not learn while you are using them. In-context learning can only take you so far, and building AI that adds serious enterprise value requires deliberate training and feedback systems, not passive prompting.

## You do not need a trillion parameters

The upside to this reality is liberating. True, practical intelligence tailored to your domain does not require a monolithic frontier model. You do not need a system trained on fluid dynamics, 3D assets, advanced Python, and protein folding just to process your data, triage communication, or execute workflows in your environment.

When you focus on deliberate adaptation for the problem right in front of you, the actual model you need is far smaller, faster, and more efficient than most people think.

## Key takeaways

- **In-context learning is transient.** Feeding documents to an API prompt gives transient reference material, not institutional memory or persistent parameter updates.
- **Mastery comes from environments, not rulebooks.** High-end model capabilities are built by engineers designing feedback loops and reward systems, not by spontaneous general intelligence.
- **Smaller models win on focused tasks.** Once you stop treating AI as a magical generalist and start designing deliberate adaptation pipelines, compact, task-focused models outperform unwieldy giants at a fraction of the compute cost.
