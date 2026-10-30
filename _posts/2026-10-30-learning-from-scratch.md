---
layout: post
title: Teaching a Tiny Model to Read Like a Child, and Sit the Year 1 Phonics Check
date: 2026-10-30
description: How I trained a tiny model to teach itself to pass the Year 1 phonics check with no labels, no answer key
tags: [reinforcement-learning, world-models, learning-from-experience]
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
<img src="{{ 'assets/img/how_is_it_going.png' | relative_url }}" alt="The page the model reads: pictures with the words how is it going printed underneath" class="img-fluid rounded z-depth-1" style="width:100%; image-rendering: pixelated;">
</div>
<div class="col-md-6">
<p class="mb-1"><b>What it sees:</b> a page with the words "how is it going" printed underneath, in pixels.</p>
<p class="mb-2"><b>What it says:</b> /haʊ ɪs ɪt gɔɪŋ/</p>
<audio controls preload="none" style="width:100%">
  <source src="{{ 'assets/audio/how_is_it_going.wav' | relative_url }}" type="audio/wav">
</audio>
<p class="mt-2 mb-1"><small>It has never seen these words. It sounds them out with the phonics it learned, so "is" comes out with an s, and it spots the "oi" inside "going", just like a child who has just learned "oi" as in coin.</small></p>
<p class="mb-0 text-muted"><small>It does sound a bit creepy, but you would too if you didn't have vocal cords ha! I am reminded a little bit of Rocky from Project Hail Mary "Amaze! Amaze! Amaze!"</small></p>
</div>
</div>

<div class="pb-4"></div>

## Where this started

I remember being in primary school and must have been around the age of five, the class I was in we were all collectively reading a banner of the alphabet that was hung up on the wall with the teachers direction. We went through it from a to z, but saying each letter in the simplistic phonetic way to help us learn to read. We made the sounds: /æ/ for apple, /b/ for ball, /k/ for cat. Years later that memory got me thinking. We teach children that way as its the building blocks for being able to read. Could a small model learn to do the same thing, actually speaking the sound it is being taught, and then use those sounds to read words it has never seen before? And how much teaching would it need?

It turns out this idea is older than I am:

> "Instead of trying to produce a programme to simulate the adult mind, why not rather try to produce one which simulates the child's? If this were then subjected to an appropriate course of education one would obtain the adult brain." - Alan Turing, "Computing Machinery and Intelligence" (1950)

With that idea in mind, and inspired by David Silver's talk on a recent podcast, I set out to create a machine that can read and speak. I ended up calling the little model **the parrot**, putting it through a simplified version of a UK primary school reading education. It learns words by listening, learns to talk by copying, learns phonics by imitating a teacher reading aloud, and then teaches itself new words by reading picture books. The final exam is the same one every Year 1 child in England sits **the Phonics Screening Check**. Spoiler the model passes it.

## Why bother?

The tools we use every day get their skill from enormous amounts of human-labelled data. Whisper learned speech from hundreds of thousands of hours of transcribed audio, and GPT learned language from a huge share of the written internet. A five-year-old has none of that. They learn by listening, babbling, copying the people around them, and being rewarded when it works.

I have been getting more and more into reinforcement learning, and this felt like a great battleground for it. Rich Sutton and David Silver argue in "Welcome to the Era of Experience" that agents should learn from their own interaction with a world, rewarded by consequences rather than by copying human judgements. Silver has since started Ineffable Intelligence around this idea, here in the UK, which I think is really exciting. Sutton's "Bitter Lesson" makes a related point on this too, the methods that win in the long run are the general ones that learn, not the clever ones we hand-build.

The rules I set myself:

- **No answer keys.** The model is never told what a letter sounds like. It hears the teacher, tries to copy, and judges itself with its own ear.
- **Rewards from consequences.** Sounding like the teacher, pointing at the right picture, being understood by a listener.
- **Tiny.** A few megabytes, trained on my MacBook Pro for quick experimentation.
- **Judged like a child.** Not by a phonetic checklist, but by whether our listener hears the right word.

## What is phonics?

By the time children start school they can already talk. They know thousands of words by ear. Phonics teaches the bridge from print to those sounds: which letters, or groups of letters (graphemes), make which sounds (phonemes). "sh" says /ʃ/, "igh" says /aɪ/, and in "cake" the e at the end changes the a. Once you know the code, you can sound out "f-r-o-g", blend it, and recognise a word you already know by ear.

That last step is the clever part. Another David called David Share called it the self-teaching hypothesis (1995), the idea that every time a child successfully sounds out a word, they have taught themselves that spelling without anyone marking it. Sounding out acts as a built-in teacher. That idea sits at the heart of this project.

In England, every Year 1 child (age 5 to 6) sits the **Phonics Screening Check** each June. It is 40 words read aloud to a teacher:

|             | Section 1                                                 | Section 2                                                                                              |
| ----------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Alien words | 12                                                        | 8                                                                                                      |
| Real words  | 8                                                         | 12                                                                                                     |
| Code tested | Single letters, ch, sh, th, ng, ck, ar, ee, oi, oo, or... | Adds ph, wh, a-e, i-e, ay, ou, ie, ea, aw, ew, ir, ur..., longer clusters (str, spl) and two syllables |

The alien words are made up ("blarsh", "quoag") and shown with a picture of an alien so children cannot rely on words they have just memorised. The pass mark is 32 out of 40. If the model can pass that, it can genuinely decode. We have a target to work towards.

## The setup: a little world of picture books

First things first, the model needed a world to grow up in. Rather than a dataset, I built a simulation, a bit like AlphaZero's board where the world is designed, but the skills have to be learned. Building this type of simulations has become a real interest of mine, if we have perfect simulators there is a lot that even small systems can learn just from the environment.

- **Picture books.** Every page is a 240×320 image with 3 to 5 emoji pictures and a caption like "spot the frog" in a simple pixel font. There are 120 picture words, chosen from the UK Letters and Sounds phases: 72 the model sees in print, 28 it only ever hears until the test (frog, pie, dolphin...), and 20 tricky words where sounding out gives the wrong answer (glove, bread, house, bear, rainbow...).
- **Teacher voices.** Every English voice on my Mac reads every word and line aloud using the built-in `say` command, with three voices kept back so we can test on voices it has never heard. A second teacher, espeak, can read anything at all, including alien words, which is what phonics lessons need.
- **The supervisor.** A separate listener network that learns the words from the teacher voices only. It never hears the model during training, then acts as the examiner to check whether it understands what the model says.

The model only ever receives pixels and audio. Sounds never arrive as labels.

<div class="row align-items-center mt-3">
<div class="col-md-6 mb-3 mb-md-0">
<img src="{{ 'assets/img/spot_the_frog.png' | relative_url }}" alt="An example page: emoji pictures with the caption spot the frog" class="img-fluid rounded z-depth-1" style="width:100%; image-rendering: pixelated;">
</div>
<div class="col-md-6">
<p class="mb-2"><b>A teacher reading the caption</b></p>
<audio controls preload="none" style="width:100%">
  <source src="{{ 'assets/audio/teacher_spot_the_frog.wav' | relative_url }}" type="audio/wav">
</audio>
<p class="mt-2 mb-0"><small>One of the Mac voices it learns from. During listening it only hears this and has to point at the frog; the print means nothing to it yet.</small></p>
</div>
</div>
<div class="caption text-center">
    Figure 1: a page from the model's picture books. Pictures on top, the caption underneath, all one 240×320 image.
</div>

## The model

The model is two small networks: a **brain** of about 2M parameters and a **voice** of about 2.2M, under 10 MB together.

<div class="row justify-content-center">
<div class="col-12 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/architecture.png" title="How the parrot reads and learns" class="img-fluid rounded z-depth-1" %}
</div>
</div>
<div class="caption text-center">
    Figure 2: how the model reads (solid arrows) and the four ways its reading head learns (dotted arrows). The pink path is the sounding-out loop: it commits to real sounds, says them in its own voice, and hears itself.
</div>

<details class="mb-4" markdown="1">
<summary>The layers in more detail</summary>

```text
THE BRAIN (d = 128, ~2M parameters)
  pictures ─► Scene encoder: 5-layer CNN → 220 patch tokens → 2-layer transformer
  print ────► Print encoder: CNN keeping 4-px columns → 80 column tokens → 2-layer transformer
                ├─► Reading head: per-column classifier over 44 classes (42 sounds, pause, blank), CTC
                │     → a sound sequence /f r ɒ g/ → THE VOICE → spectrogram
                │     → Ear: two strided 1-D convs (40 ms steps) → 2-layer transformer → attention pool
                │     → query · picture patches → points at a picture
                └─► Sight-word route: attention pool over the print columns → query · patches → points too
  Namer: picture features → its spoken name (GRU)
  Echo:  ear frames → which of its own sounds made them (CTC), learned from babbling

THE VOICE (d = 192, ~2.2M parameters)
  sounds + whose voice → 3-layer transformer encoder → duration predictor (alignment learned by
  monotonic alignment search) → repeat each sound for its frames → 3 conv blocks + 2-layer transformer
  → 64-band log-mel spectrogram, 10 ms frames → Griffin-Lim → audio (only for our ears)
```

</details>

A few design choices matter more than the rest:

- **The bottleneck is discrete.** The reading head has to commit to a sequence of real sounds, which go out through the voice and back in through the ear. Rewards for finding the right picture cannot sneak through as some private code, so what it says always stays real speech that a listener can mark. This is the sounding-out loop.
- **It hears itself with the ear it learned words with.** Reading only works if what it says sounds like a word it already knows by ear, exactly like a child blending "f-r-o-g" and recognising it.
- **Two routes to meaning.** Sounding out, and recognising familiar words on sight, following the dual-route model of human reading (Coltheart et al.). The sight route cannot reshape the eyes, so memorising words cannot corrupt the decoding skill.
- **The echo is its own inverse model.** It learns which of its sounds made what it heard, purely from its own babbling, never from teacher labels. That is what makes phonics without an answer key possible.

## Growing up

The model grows up in stages, a bit like a child, each one learning from a different kind of experience. The whole thing runs end to end on my MacBook in a few hours.

**1. Listening (the early years).** Teacher voices talk about the pictures on the page: "spot the frog". The ear learns to point at the right picture, using a contrastive objective where the query from the audio has to match the right picture patch out of 220. It also learns to name pictures and repeat what it heard. No printed words yet.

**2. Finding its voice.** The neural voice learns to talk by mimicking recordings. Given the sounds that were said, it draws the spectrogram and learns from the difference (L1 loss), working out where each sound starts and stops on its own. Its own voice starts as Jamie, a natural-sounding British Mac voice, and is then polished two ways. A small critic network learns to tell its spectrograms apart from real ones, teaching the voice to fool the critic (a GAN-style loss that removes the smeared, averaged sound), and it practises saying words until the supervisor understands them.

**3. Babbling.** Like an infant, it makes random strings of sounds and listens to itself. From pairs of "what I said" and "what I heard" it learns the echo, an inverse model of its own mouth. No teacher involved.

**4. Phonics by imitation.** Now for the alphabet frieze. The teacher reads a letter card, then words, then whole lines aloud. The model reads the same print aloud, trying 4 different readings. Each attempt is spoken, and the reward is how close it sounds to the teacher, measured by dynamic time warping over MFCC features so it matches sound rather than symbols. Better-than-average attempts are reinforced and worse ones discouraged through a group-relative REINFORCE update, similar in spirit to GRPO. On top of that, when the echo is confident it can transcribe what the teacher said into its own sounds, that becomes a target to copy. It is learning phonics with no answer key, judged only by its own ear.

**5. Reading practice.** It reads picture books aloud to itself in its own voice, hears itself, and points at the picture it thinks the words describe. The reward is pointing at the right picture (REINFORCE again). This is where self-teaching kicks in: when it points at the bread, its namer knows the bread is called /bred/, and that becomes a lesson for how that print should be read. Phonics practice carries on alongside so it does not forget the code.

## What broke along the way

Most of what I learned came from things failing, and nearly every fix replaced something I had hand-built with something learned. The bitter lesson, in miniature. Claude Opus 5.5 was a big help too.

- **Pure trial and error stalled.** My first go at phonics was pure RL: try sounds, get scored. It got 1.6% of letter cards right. Adding babbling first, so it knows what its own mouth sounds like before it tries to copy anyone, took letter cards to 72%, and alien-word decoding later reached 97%. Infants babble for months before they copy words, and it turns out there is a good reason.
- **My hand-built voice box sounded awful.** I built a little source-filter synthesiser (voicing, hiss, pitch, mouth resonances) and had the model learn to drive it by trial and error. The supervisor understood about half its words, and it sounded like a broken robot. A learned neural voice copying real recordings was far clearer straight away.
- **Switching voices wiped out phonics.** When I retrained phonics through the new voice, decoding dropped from 97% to 3%. Its ear retuned to the new voice and lost the code. The fix was to learn phonics hearing itself through a clear synthetic voice, then switch to its own voice for reading practice while continuing phonics drills.
- **Words ran together.** Early readings came out as one long word (/ðəhen/ for "the hen"). Learning pauses between words in its own speech fixed it.
- **A learned vocoder did not help yet.** I trained a small Vocos-style network to turn spectrograms into audio instead of Griffin-Lim. It sounded great on real voices, but no better on the model's, because the blur came from the voice itself. That is what the critic fixed.

## Results

### The first model: 48 words, Phase 2 to 3

My first model learned a smaller world: 48 picture words and the early phonics code. Every test was on first sight, with no learning during the test.

| Test                                                                                   | Result                |
| -------------------------------------------------------------------------------------- | --------------------- |
| Alien words it had never seen (blarsh, quoag), read exactly right                      | 91.3%                 |
| A listener picks the right alien word from near-misses one sound away                  | 98.0%                 |
| Words it had only ever heard, read aloud correctly the first time it saw them in print | 93.8%                 |
| The supervisor understands its reading aloud (practised / tricky / never printed)      | 96.5% / 98.0% / 85.3% |

Two checks convinced me it is really reading. With the caption removed, it points at the right picture 25.8% of the time, roughly chance with 3 to 5 pictures, showing it is not guessing from the artwork. Also, the sight route on its own reads familiar words at 91% but new words at 16%. Familiar words move to recognition on sight, while new words can only be sounded out. That two-route split is what reading research describes in people, and it appeared naturally.

### My favourite moment: it taught itself the tricky words

| Page           | After listening | After phonics          | After reading practice |
| -------------- | --------------- | ---------------------- | ---------------------- |
| spot the bread | babble          | /bræd/, wrong picture  | /bred/, right picture  |
| spot the glove | babble          | /glɒvə/, wrong picture | /glʌv/, right picture  |

After phonics it read these the way a beginner does, strictly by the rules, and got them wrong. After reading practice it read them right. Nobody told it they were exceptions. /bræd/ matched no word it knew by ear, /bred/ did, and the bread was on the page. That is Share's self-teaching loop happening inside a model.

<div class="row mt-3">
<div class="col-md-4 mb-3">
<div class="p-3 rounded z-depth-1 h-100">
<p class="mb-1"><b>1. After listening</b></p>
<p class="mb-2"><code>babble</code></p>
<audio controls preload="none" style="width:100%">
  <source src="{{ 'assets/audio/glove_after_listening.wav' | relative_url }}" type="audio/wav">
</audio>
</div>
</div>
<div class="col-md-4 mb-3">
<div class="p-3 rounded z-depth-1 h-100">
<p class="mb-1"><b>2. After phonics</b></p>
<p class="mb-2"><code>by the rules: /gləʊv/</code> <small>(magic e, like "clove")</small></p>
<audio controls preload="none" style="width:100%">
  <source src="{{ 'assets/audio/glove_after_phonics.wav' | relative_url }}" type="audio/wav">
</audio>
</div>
</div>
<div class="col-md-4 mb-3">
<div class="p-3 rounded z-depth-1 h-100">
<p class="mb-1"><b>3. After reading practice</b></p>
<p class="mb-2"><code>self-taught: /glʌv/</code></p>
<audio controls preload="none" style="width:100%">
  <source src="{{ 'assets/audio/glove_after_reading_practice.wav' | relative_url }}" type="audio/wav">
</audio>
</div>
</div>
</div>
<div class="caption text-center">
    The same page, "spot the glove", read by the model at each stage of growing up.
</div>

### The phonics check

I sat the first model down for the check before teaching it the rest of the code. It scored **26.8/40** on average over 10 papers: 18.4/20 on section 1, which uses the code it had learned, but 8.4/20 on section 2, which it had never been taught. Not a pass.

So I grew a new model in a bigger world: all 120 words and the full Phase 2 to 5 code the check covers. Each paper is freshly generated to the official structure so it cannot be memorised, and marked like a teacher would: it only scores if it chose the right sounds **and** said them clearly enough to be heard as the right word.

|                         | Section 1 | Section 2 | Total       | Passed              |
| ----------------------- | --------- | --------- | ----------- | ------------------- |
| First model (Phase 2-3) | 18.4/20   | 8.4/20    | 26.8/40     | 0 of 10 papers      |
| After phonics           | 19.8/20   | 16.8/20   | **36.6/40** | **10 of 10 papers** |
| After reading practice  | 19.6/20   | 17.0/20   | **36.6/40** | **10 of 10 papers** |

It passed every one of the 10 papers. Its lowest paper was 34/40 and its best 38/40, which is about as high as it can realistically go: each paper has two real words with an alternative pronunciation (like "snow" or "head") that you can only get right if you already know the word, and the model has never heard them. Its voice never cost it a mark either: every word it chose the right sounds for was heard as the right word. The spellings it still slips on are y, the split digraph i-e and igh, which are the same ones that trip up a lot of children.

### How much teaching did it need?

This is the number I care about most for what comes next. During training I kept a snapshot every 500 steps and sat the check on each one, plotting the score against how many times the model had heard the teacher read aloud.

<div class="row justify-content-center">
<div class="col-sm-12 col-md-10 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/learning_curve.png" title="Learning curve" class="img-fluid rounded z-depth-1" %}
</div>
</div>
<div class="caption text-center">
    Figure 3: the phonics check score against how many times the model had heard the teacher read aloud. It passes the 32/40 mark after 60,000 readings.
</div>

It passed after hearing the teacher read aloud **60,000 times**. For the first 24,000 the score sits at zero: those lessons are mostly letter cards, learning single sounds, and the check needs whole words. Then it climbs steeply, passes at 60,000, and plateaus at around 36/40 by 84,000. The remaining 60,000 readings in the phonics stage added about one mark, and reading practice afterwards added a little more on top (37.3/40 on the curve's papers), mostly helping it point at pictures and read the tricky words rather than the check itself. 60,000 is the baseline I now want to drive down.

## What it was given, and what it learned

Every learning system starts with priors. AlphaZero was handed the rules of Go, and this model lives in a world I designed around English. Let's look at where that line actually is.

|                | Learned from experience                                               | Given to it                                                                                                                                                                                    |
| -------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Hearing        | What words sound like, in any voice                                   | None                                                                                                                                                                                           |
| Phonics        | Which letters make which sounds, by imitating the teacher             | None                                                                                                                                                                                           |
| Reading        | Sounding out, sight words, the tricky words                           | None                                                                                                                                                                                           |
| Speaking       | What each of its sounds sounds like, how long it lasts, its own voice | 42 sound "slots" to choose between: it learned what each one sounds like, but not that English has 42. The voice also learned to talk from teacher recordings labelled with the sounds in them |
| Hearing itself | None                                                                  | During phonics it heard itself through a clear synthetic voice, then switched to its own                                                                                                       |
| Sound for us   | None                                                                  | Griffin-Lim, a fixed method that turns its spectrograms into audio for our ears. The model itself only ever hears spectrograms                                                                 |

The biggest given is the sound inventory. A baby discovers the sound categories of its language by listening in its first year, before any school. The model skips that. Babies are probably born expecting speech to be made of reusable units, but which units, and how many, is learned. Letting the model discover its own units is the obvious next frontier.

Of course it is small: 120 words, synthetic voices, one font. Nothing here shows it would cope with a real children's book yet.

## What's next

The part I am most excited about is efficiency. A child arrives at school already talking, and learns the phonics code in a year or so of short daily lessons. The model needed to hear the teacher read aloud 60,000 times to pass, and was given 144,000 in total. Now that the learning curve measures this, the aim is to drive that number down in human-like ways:

- **Teach in order, like the alphabet frieze.** s, a, t, p, i, n first, instead of everything mixed from day one.
- **Practise what it gets wrong.** Take the time to focus on trouble spots, the way a teacher spends longer on "ay" than on "s". This is where the RL and self-play angle really comes in.
- **One creature, one life.** Today the brain and voice are separate networks trained in separate stages. Joining them, so it learns to speak through its own ear and reading and speaking improve each other, is much closer to how a child develops.
- **Discover its own sounds** from babbling and listening, instead of being handed 42 slots.
- **Read a whole page,** moving its eyes across lines, and eventually a real decodable reading book.

Turing suggested building a child machine and educating it. Seventy-five years later, a few megabytes on a laptop can sit the same reading check as a six-year-old. And it passes, 36.6 out of 40. What I really want to know now is how close it can get to learning like one.

## Prior work

Links still to add before posting.

| Work                                                           | Why it matters here                                                                                                                |
| -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Silver and Sutton (2025), "Welcome to the Era of Experience"   | The framing: agents learning from their own experience, rewarded by consequences                                                   |
| Silver, Singh, Precup and Sutton (2021), "Reward is Enough"    | Rewards from the world (being understood, the right picture) instead of answer keys                                                |
| Sutton (2019), "The Bitter Lesson"                             | Why the hand-built voice box lost to a learned one                                                                                 |
| Share (1995), the self-teaching hypothesis                     | The core idea: sounding out acts as a built-in teacher                                                                             |
| Coltheart et al. (2001), the dual-route cascaded model         | Two routes, sounding out vs sight words, which the results reproduce                                                               |
| Seidenberg and McClelland (1989), Plaut et al. (1996)          | Earlier neural reading models, which learned spelling-to-sound from labels                                                         |
| Kim et al. (2020), Glow-TTS; Siuzdak (2023), Vocos             | The alignment trick behind the voice, and the vocoder design                                                                       |
| Letters and Sounds (DfE, 2007), Year 1 Phonics Screening Check | The UK phonics phases the words follow, and the alien-word test                                                                    |
| Turing (1950), "Computing Machinery and Intelligence"          | The child machine: build a child's mind and educate it                                                                             |
| Ziegler, Perry and Zorzi (2014), Pritchard et al. (2018)       | The closest predecessors: computational models of self-teaching, working on letters and sound symbols rather than pixels and audio |
| Dehaene, "Reading in the Brain" (2009)                         | Neuronal recycling: reading repurposes visual cortex and wires it to spoken language                                               |
