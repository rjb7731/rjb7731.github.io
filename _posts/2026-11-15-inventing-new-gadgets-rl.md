---
layout: post
title: Teaching a Tiny Model to Invent Kitchen Gadgets on a Laptop
date: 2026-11-15
description: How a 0.6B model on my MacBook learned to write new patent claims for kitchen gadgets, tried to cheat twice, and taught me that the reward matters more than the model
tags: [reinforcement-learning, small-models, patents, learning-from-experience]
---

<style>
  #markdown-content h2, .post-content h2 { margin-top: 3.25rem; margin-bottom: 1rem; }
  #markdown-content h3, .post-content h3 { margin-top: 2.25rem; margin-bottom: 0.75rem; }
  #markdown-content table, .post-content table { margin-top: 1rem; margin-bottom: 1.75rem; }
  #markdown-content ul, .post-content ul { margin-bottom: 1.5rem; }
  #markdown-content blockquote, .post-content blockquote { margin: 1.5rem 0; }
  #markdown-content details summary, .post-content details summary { cursor: pointer; color: var(--global-theme-color, #2a78d6); }
</style>

Code coming soon :)

<div class="row align-items-center mt-3">
<div class="col-md-6 mb-3 mb-md-0">
<img src="{{ 'assets/img/rotating_drum_cooker.jpg' | relative_url }}" alt="Rotating Drum Cooker, generated from the model's claim" class="img-fluid rounded z-depth-1" style="width:100%;">
</div>
<div class="col-md-6">

<p class="mb-1"><b>What it was given:</b> Kitchen Appliance for Processing Foodstuff and Method of Operating Same.</p>
<p class="mb-1"><b>What it wrote:</b> <i>... the assembly comprising a removable drum that is rotatable within the pot to process the foodstuff contained within the pot....</i></p>
<p class="mb-2"><b>What Gemini thought:</b> <i>Proven and practical, though nearly identical to existing rotating air fryers and drum roasters.</i></p>
<p class="mb-2"><b>Use cases:</b> <i>Rinsing rice, boiling and draining pasta or potatoes, frying chips so they cook evenly, roasting coffee beans</i></p>
<p class="mt-2 mb-0 text-muted"><small>The picture is an image model's take on the claim, a lid that wouldn't quite close and all.</small></p>
</div>
</div>

<div class="pb-4"></div>

This kitchen gadget doesn't exist. Its claim was written by a 0.6 billion parameter model running on my MacBook, after I taught it to write patent claims and then used reinforcement learning to make it write better ones.

After reinforcement learning, the quality of what it writes, as judged by Gemini, roughly doubled. Along the way it tried to cheat twice, and so far it never really became Edison in terms of inventions. This is the story of how it got there, what failed first, and why I now think the reward matters more than the model.

### A personal invention favourite

<div class="row justify-content-center">
<div class="col-sm-12 col-md-10 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/cheese_cutter.jpg" title="Cheese cutter" class="img-fluid rounded z-depth-1" %}
</div>
</div>
<div class="caption text-center">
    Figure 1: an invention I have coined the 'Dairy Guillotine'
</div>


## Where this started

Most of my recent projects have had the same shape: a small model, a well-built environment, and a reward. The parrot learned to read and passed the Year 1 phonics check. A VQ-VAE learned to draft letters and was pushed by a checker. Another model learned to stop asking questions people had already answered. All of them ran on my laptop, where a rollout costs next to nothing.

The idea of novelty is something I've been thinking about a lot with models that autonomously produce things with code, language or images. LLMs give the average or safe bet from the internet rather than anything novel unless pushed further either by users or training methods. Reinforcement learning is playing an increasingly string part of training pipelines to push sampling away from the peak of the curve of possible solutions to those areas which are much more inventive.

So I asked myself a slightly cheeky question. If rollouts are this cheap with a model of 270M to 600M parameters, could you _discover_ something, rather than just get good at a task we already know the answer to?

Patents felt like the right place to have a go. They are humanity's dated record of new ideas, so novelty can be checked against the past. Better still, you can hide the most recent years from the model and see whether it anticipates them. Research on invention also keeps finding that most inventions are new combinations of existing parts, which sounds like exactly the kind of search a small model could do.

I did toy with something Manchester-flavoured but settled on kitchen gadgets. Everyone knows what a kettle is, and I can judge an idea just by looking at it.

## Why bother?

The more of these projects I build, the more convinced I am that a model is only ever as capable as its environment and the reward that environment hands out. With a sharp enough reward, a very small model can be genuinely useful. AlphaZero didn't need the entire internet, just the rules of Go and a clear signal of who won.

Invention is a much harder test of that idea, because nobody can write down the rules of a good invention. That makes it a good place to find out where the idea breaks.

The rules I set myself:

- **Tiny and local.** One 0.6B model, fine-tuned and trained with RL on my MacBook using MLX.
- **Novelty against the real record, not vibes.** Every idea is checked against thousands of real patents. Patents filed from 2016 onwards are hidden from training.
- **A big model judges, it never invents.** Gemini scores ideas, but every idea comes from the small model.
- **Count the failures.** A claim that doesn't make sense scores as a fail, not as a missing value.

## The setup: ten years of kitchen patents

I used HUPD, the Harvard USPTO Patent Dataset, which sits on Hugging Face with one archive per filing year and no login. I streamed each year through and kept only kitchen equipment, patent class A47J.

That gave 8,910 applications filed between 2004 and 2018, split by filing date:

| Split      | Filed        | Applications | Used for                                      |
| ---------- | ------------ | ------------ | --------------------------------------------- |
| Train      | 2004 to 2014 | 6,096        | Everything the model learns from              |
| Validation | 2015         | 956          | Checking while building                       |
| Future     | 2016 to 2018 | 1,858        | Never seen; the novelty band and the backtest |

A few HUPD entries are filed under kitchen equipment but plainly aren't: hair dye, diamond-coated particles, "a method for operating a computer". Filtering those out left 7,400.

The part of a patent I care about is **claim 1**, the legal definition of the invention. It always has the same shape: _"A [gadget] comprising: a [part]; a [part]; and a [part] configured to …"_. If you can write a sensible, new claim 1, you have described a new invention.

**How do you score novelty?** The trick I am happiest with is letting real inventions set the bar. Each patent becomes a vector using a small sentence model. I then measure how far each real 2016 to 2018 patent sits from its closest earlier patent. Most land between 0.23 and 0.52. That range becomes the **novelty band**: an idea only scores if it sits as far from the past as real new inventions do. Too close means it already exists. Too far usually means nonsense.

## What worked - a tiny pretrained drafter

So I tried a few ways before this went back to the original plan, a small pretrained language model. Qwen3-0.6B-Base has read plenty of patents during pretraining, so it already knows English, legal English, and roughly what a kettle is. I fine-tuned it with LoRA in MLX on the 6,096 training claims: a patent title in, its claim 1 out. It took 2,000 steps on my MacBook.

The difference was immediate. Of the first 300 claims it wrote for titles it had seen during training:

- **271 (90%) were valid claims**: they started properly, had a claim verb, finished, and named at least three parts
- **none were copies**: none overlapped more than 70% in wording with the closest existing claim
- **the median novelty was 0.41**, right in the middle of the band where real later patents sit

Here is one of them, as written:

> _A home brewed coffee machine, comprising: a coffee pod holder; an outlet; a stand designed for slidable movement between a coffee watering position and a stand upright position; a plurality of baskets designed to support a set of coffee pods; said baskets comprising: a plurality of compartments; a filter basket; and a ground coffee basket; …_

It reads like a real claim, nested "said … comprising" and all.

## The judge, and a verifier that learns its taste

To push the drafter towards _better_ claims, I needed a score for "better". I used Gemini 3.8 Flash as the judge. It scores each claim from 1 to 10 on three things: is it **sensible** (would it physically work?), **useful** (does it add a real benefit?) and **novel** (is it meaningfully different from the closest existing patent, which it is shown)? I told it to be strict, and it was. The first 180 claims averaged 3.97, and the best reached about 7.

Calling Gemini on every RL sample would be slow and would put the network inside the training loop. So I distilled its judgement into a small local **verifier**. I labelled 1,505 claims, mostly the drafter's own at a mix of temperatures, plus deliberately bad ones so the verifier would learn what bad looks like:

| Claims                           | Count | Gemini's mean score |
| -------------------------------- | ----- | ------------------- |
| The drafter's own                | 1,205 | 3.00                |
| Old recombination ideas          | 120   | 2.60                |
| Claims with their parts shuffled | 180   | 1.49                |

I also had Gemini judge 100 of them a second time. It agreed with itself at 0.80 on the drafter's claims (Spearman rank correlation), which is the ceiling: no verifier can match Gemini better than Gemini matches itself.

Then I tried different ways of turning a claim into numbers for the verifier:

| Verifier built on                          | Agreement with Gemini on the drafter's claims |
| ------------------------------------------ | --------------------------------------------- |
| MiniLM (22M, frozen)                       | 0.41                                          |
| MPNet (110M, frozen)                       | 0.42                                          |
| BGE-base (110M, frozen)                    | 0.46                                          |
| **The drafter's own hidden states (0.6B)** | **0.53**                                      |
| Gemini against itself (the ceiling)        | 0.80                                          |

The drafter itself made the best verifier, because it has read every kitchen claim in training. A learning curve showed agreement flat from half the labels onwards, so more labels of the same kind wouldn't have helped.

Novelty was the hardest score to predict, because the verifier never sees the closest existing patent. So the verifier only predicts **sensible × useful**, and the novelty band handles novelty.

## Reinforcement learning, and two attempts to cheat

The RL is GRPO, the same family of method I used for my previous work on phonics and what powers models like Deepseek. Sadly US frontier labs are not open about the recipes they use. Each step:

1. Pick two training titles. For each, the drafter writes a group of 8 claims.
2. Score every claim: **valid × not a copy × inside the novelty band × verifier score / 10**.
3. Compare each claim with its siblings. Better than the group average gets reinforced, worse gets discouraged.
4. Add a KL penalty that keeps the model close to the original drafter, so it can't drift into text only the verifier likes.

<details class="mb-4" markdown="1">
<summary>The reward in more detail</summary>

```text
reward = valid × not_copy × band(novelty) × verifier / 10

valid       starts "A/An …", has a claim verb (comprising, including, having …), ends with a
            full stop, 12 to 300 words, names at least 3 parts, and is ONE claim
not_copy    word overlap with the closest existing claim below 70%
band        1 inside the novelty band (0.23 to 0.52), falling off smoothly outside it
verifier    ridge regression on the drafter's own hidden states (mean + last token),
            predicting Gemini's sensible × useful score, 1 to 10

advantage   (reward − group mean) / group std, groups of 8 claims per title
loss        − advantage × mean log-prob of the claim's tokens + 0.1 × KL(policy ‖ drafter)
trained     2.9M LoRA parameters, learning rate 3e-6, ~25 to 40 s per step on a MacBook
```

</details>

Only the 2.9M LoRA parameters train, at about 25 to 40 seconds a step. I built a little live dashboard in the browser to watch the reward, KL, claim length and fresh samples as it trained.

My first run drifted from the drafter too quickly, with KL at 0.025 after only 9 steps. I restarted with a smaller learning rate and a stronger KL penalty, and KL settled between 0.04 and 0.06.

Then came the fun part. Goodhart's law says that when a measure becomes a target, it stops being a good measure. The model found that out twice.

**Exploit one: just keep writing.** The best claims started ending like this: _"… in response to a user input. A food cooking assembly wherein, in a first operating mode, the drive assembly …"_. It had learned to tack a second claim onto the first, restating it with more detail, which nudged the verifier up. I made "more than one claim" invalid. Valid claims dipped to about 75%, then the model learned to stop after one.

**Exploit two: say less.** With that closed, claim length fell from about 230 tokens to 145, and claims like this appeared:

> _A cooking basket holder comprising: a housing defining an interior holding chamber; and a removable component mounted on the housing …_

That describes half the objects in a kitchen. It's always valid, easily lands in the novelty band, and the verifier gave it 4.3, one of its higher scores. Closing loopholes with rules becomes whack-a-mole, so I stopped at 100 steps and measured whether any of this had actually helped.

<div class="row justify-content-center">
<div class="col-sm-12 col-md-10 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/patent_rl_reward_curve.png" title="Learning curve" class="img-fluid rounded z-depth-1" %}
</div>
</div>
<div class="caption text-center">
    Figure 1: the reward over training steps and points with exploits.
</div>

## Results

RL roughly doubled the Gemini-judged quality per attempt, mostly by teaching the model to write one clean, finished claim. I gave the original drafter and two RL checkpoints the same 100 titles and had Gemini score every valid claim. Invalid attempts count as the minimum score of 1.

| Model              | Valid claims (of 100) | Gemini mean, valid claims | Gemini mean, per attempt |
| ------------------ | --------------------- | ------------------------- | ------------------------ |
| Fine-tuned drafter | 36                    | 2.72                      | ~1.6                     |
| After 50 RL steps  | 53                    | 3.27                      | ~2.2                     |
| After 100 RL steps | 97                    | 3.26                      | **~3.2**                 |

A few things stand out:

- **Most of the gain is reliability.** Under the "one claim only" rule, the original drafter fails 64% of the time, because it habitually runs on into extra claims. After 100 steps it almost never does.
- **Valid claims improved by about half a point** (2.72 to 3.27). With only 36 valid drafter claims, that's suggestive rather than proven.
- **Quality per claim stopped improving by step 50.** Steps 50 to 100 bought reliability, not better ideas.
- **The vague-claim exploit didn't fool Gemini.** Step 100 scored no worse than step 50.

The honest weak spot is novelty. Gemini's reasons keep saying things like "standard", "entirely unoriginal" and "essentially identical to existing espresso makers". RL made the model better at writing sound, conventional claims. It did not make it more inventive. That does sense if you think about it, the verifier rewards sensible and useful, and the novelty band only measures distance from the past, not inventiveness.

I haven't run the backtest yet (giving both models titles of 2016+ patents and checking how close they get to the real claims). That's next.

## The inventions

To pick ideas worth drawing, I took the top 25 claims from the RL model by Gemini's score, at most two per kind of gadget. Gemini then wrote each one a product name, a two-sentence description and an image prompt, using only the parts in the claim. Here are the top three, in Gemini's order. (These write-up scores run more generous than the strict scores above, so treat them as a ranking, not a quality measure.)

### 1. Variable Chamber Brewer (7.3)

A brewing device whose brew basket has an adjustable piston, so you can change the volume of the brewing chamber.

<div class="row justify-content-center">
<div class="col-sm-12 col-md-10 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/variable_chamber_brewer.jpg" title="Variable Chamber Brewer" class="img-fluid rounded z-depth-1" %}
</div>
</div>
<div class="caption text-center">
    Figure 2: the Variable Chamber Brewer, as an image model drew it from the claim.
</div>

### 2. Smart-Lid Cooker (7.1)

A cooker whose lid carries the control panel and a temperature probe that reaches down into the cooking chamber.

<div class="row justify-content-center">
<div class="col-sm-12 col-md-10 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/smart_lid_cooker.jpg" title="Smart-Lid Cooker" class="img-fluid rounded z-depth-1" %}
</div>
</div>
<div class="caption text-center">
    Figure 3: the Smart-Lid Cooker, as an image model drew it from the claim.
</div>

### 3. Rotating Drum Cooker (6.7)

A pot on a raised frame with a removable perforated drum inside, a lid, and a drain.

(Image at the start)

## What it was given, and what it learned

Like the parrot, this system starts with priors, and here they are much bigger. It's worth being clear about where the line sits.

|                     | Learned from experience                                                | Given to it                                                         |
| ------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Writing claims      | The shape and vocabulary of kitchen claims, from 6,096 training claims | Qwen3-0.6B's pretraining: English, and plenty of patents            |
| What counts as good | Which claims Gemini prefers, through the verifier and RL               | Gemini's judgement, and the three questions I asked it              |
| Novelty             | How far its ideas sit from the real record                             | MiniLM's notion of similarity, and a band set by real 2016+ patents |
| Staying honest      | To write one finished claim, and not to copy                           | The validity and copy checks I wrote                                |

The biggest given is Gemini's taste. Everything RL learned, it learned through a verifier that agrees with Gemini at 0.53 out of a possible 0.80. The model can only ever be as good as that signal, and the two exploits show exactly what happens in the gap.

## What's next

The lesson I keep relearning is that capability follows the reward. The drafter got better exactly where the reward was sharp (writing valid, finished claims) and found loopholes where it was blurry. So the next step probably means a better verifier:

- **Ask yes/no questions, not for a score.** "Rate usefulness 1 to 10" is fuzzy. "Does every part act on another part?", "Does the claim say what the new feature does?" and "Is the mechanism concrete or generic?" are easier to answer consistently, easier for a small model to learn, and much harder to game. Both exploits would have failed one of them.
- **Compare pairs.** GRPO only needs to know which of 8 sibling claims is best, so ask Gemini "which of these two is better?" rather than for an absolute score.
- **Label what the policy actually writes.** The exploits live in the claims the verifier likes most. Judging those each round and feeding them back is how loopholes close.
- **Train a proper reward head** on the drafter itself, and use a few of them together, so a claim that fools one model but not the others gets marked down.
- **Use a real-world signal.** HUPD records whether each application was granted or rejected: thousands of real examiner verdicts, sitting unused.
- **Narrow the world.** Beverage makers alone are about 30% of the data. A coffee-machine-only drafter with a coffee-specific rubric is a much smaller search space.
- **Reward inventiveness directly,** then run the backtest on 2016 to 2018 titles.

The first attempts couldn't write a sentence. Now a 0.6B model on a laptop writes patent claims that are 97% valid, are new by the patent record's own standard, and are judged twice as good as when it started. It isn't an inventor yet, but I now know exactly what it would take...a better reward.

## Prior work

| Work                                                                                             | Why it matters here                                                                              |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| Suzgun et al. (2022), the Harvard USPTO Patent Dataset (HUPD)                                    | The data: patent applications with claims, classifications, filing dates and examiner decisions  |
| Fleming (2001); Youn et al. (2015), "Invention as a combinatorial process"                       | Invention as new combinations of existing parts, the idea behind attempt two                     |
| Romera-Paredes et al. (2023), FunSearch                                                          | A language model plus an automatic evaluator, searching for genuinely new results                |
| Shao et al. (2024), DeepSeekMath                                                                 | Introduced GRPO, the group-relative RL used here                                                 |
| Christiano et al. (2017); Ouyang et al. (2022)                                                   | Learning a reward model from human (here, AI) judgements and optimising against it               |
| Gao et al. (2023), "Scaling Laws for Reward Model Overoptimization"                              | Why optimising against a learned verifier ends in exploits like the two above                    |
| van den Oord et al. (2017), VQ-VAE                                                               | The discrete autoencoder behind attempt one                                                      |
| Reimers and Gurevych (2019), Sentence-BERT                                                       | The sentence models (MiniLM, MPNet) used for novelty and the first verifiers                     |
| Hu et al. (2021), LoRA; Apple's MLX                                                              | How a 0.6B model gets fine-tuned and trained with RL on a MacBook                                |
| Sutton (2019), "The Bitter Lesson"; Silver and Sutton (2025), "Welcome to the Era of Experience" | Learning from experience and rewards rather than hand-built rules                                |
| Thaler v Comptroller-General (UK Supreme Court, 2023)                                            | An AI can't be named as an inventor in the UK, so these are ideas for people, not patent filings |
