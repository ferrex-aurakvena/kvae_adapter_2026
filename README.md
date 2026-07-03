# KVAE Adapter 2026

Exploratory work toward adapting Kandinsky 5 Pro / HVAE-compatible video latents for `KVAE-3D-2.0-t4s8` decoding.

## Current Status

This is not production-ready. A direct adapter trained on high-quality Kandinsky 5 Pro SFT video DiT latents produced usable structure in test decodes, but still showed blur, temporal shimmer, fade/strobe behavior, and grid-like artifacts.

The most recent phase-aware training did not improve held-out generated-latent temporal metrics enough to justify scaling that same approach.

## What Was Tried

- Cached overlapping crops from generated K5 Pro video latents.
- KVAE t4s8 latent target caching and decoded-target finetuning.
- Hard-example replay for difficult texture and temporal cases.
- Phase-aware conditioning and temporal artifact metrics.
- Side-by-side decoded crop evaluation on detailed generated samples.

## Next Adapter Direction

The likely next adapter approach is real-video triplet/cycle training:

- `H2K`: K5/HVAE-compatible latent to KVAE latent.
- `K2H`: KVAE latent back to K5/HVAE-compatible latent for encode and image-to-video workflows.
- Triplet supervision: HVAE round trip, KVAE round trip, and adapter-to-KVAE decode.
- Carefully weighted cycle losses so the reverse path remains K5-compatible without reducing KVAE quality.

## Priority

This adapter work is currently paused as a lower-priority research path compared to other active work directions.
