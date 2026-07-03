# KVAE Adapter Effort Note

This repository captures exploratory work toward adapting Kandinsky 5 Pro / HVAE-compatible video DiT latents for decoding through `KVAE-3D-2.0-t4s8`.

## What We Tried

The initial rapid prototype trained a small spatiotemporal adapter from high-quality Kandinsky 5 Pro SFT video DiT latents into KVAE t4s8 latent space. Training used overlapping latent crops, cached KVAE targets, decoded finetuning, hard-example replay, and temporal diagnostic metrics.

The adapter produced recognizable and sometimes promising KVAE decodes, but the generated videos still showed important failures:

- temporal shimmer and fade/strobe artifacts,
- blur on detailed texture regions,
- grid-like pixel artifacts,
- inconsistent in-between frame behavior.

A phase-aware adapter variant was also tested. It improved some detail-retention cases but did not improve the held-out generated-latent temporal metrics enough to justify scaling the same approach.

## Current Interpretation

The main difficulty is not just model capacity. For generated Kandinsky 5 Pro DiT latents, there is no ground-truth KVAE t4s8 decode to supervise against. HVAE decode can preserve content and motion, but using it as a full visual teacher risks limiting KVAE to HVAE-style color, detail, and temporal behavior.

## Future Adapter Direction

If this adapter path resumes, the next version should move to real video pairs and use a KVAE-positive training objective:

- train `H2K` from K5/HVAE-compatible latents to KVAE latents,
- use triplet-style decoded supervision comparing HVAE round trip, KVAE round trip, and adapter-to-KVAE decode,
- train `K2H` as a production reverse direction for K5-compatible encode workflows,
- use cycle losses carefully so `K2H` remains K5-compatible while preserving information recoverable by `H2K`.

This would make KVAE-native quality the positive target instead of treating HVAE decode as the detail ceiling.

## Status

The direct generated-latent adapter work is paused and treated as lower priority compared to other active work directions.
