# Basis files

The transformations used by `mozza_mesh`. Each file is a set of named fields;
a field gives, for each of MediaPipe's 468 face-mesh landmarks, its
displacement at amplitude 1 in face units (see `../README.md`).

| File | Fields | Derived from |
|---|---|---|
| `au_basis_v1.json` | AU1, AU2, AU4, AU5, AU7, AU12, AU15, AU20, AU43 | Landmark motion of 8 RAVDESS actors (480 video clips), regressed on MediaPipe blendshape scores; AU15 is hand-made |
| `trait_basis_v1.json` | TRUST_o, DOM_o, TRUST, DOM, THREAT | Landmark motion between the SD levels of the Oosterhof & Todorov (2008) trait face databases |

How they were built and validated: [face-transforms](https://github.com/Pablo-Arias/face-transforms)
(`docs/au_transformations.md`, `docs/trait_transformations.md`).

These files contain only averaged landmark displacements (numbers). They
contain no images, video or other material from the source databases.

## Attribution and use

**For non-commercial research only.** If you use these bases, cite:

- **RAVDESS** (AU basis): Livingstone, S. R., & Russo, F. A. (2018). The Ryerson
  Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic,
  multimodal set of facial and vocal expressions in North American English.
  *PLoS ONE*, 13(5), e0196391. https://doi.org/10.1371/journal.pone.0196391
  (database licence: CC BY-NC-SA 4.0)
- **Oosterhof & Todorov** (trait basis): Oosterhof, N. N., & Todorov, A. (2008).
  The functional basis of face evaluation. *PNAS*, 105(32), 11087-11092.
  https://doi.org/10.1073/pnas.0805664105 (face databases: Perception and
  Judgment Lab, for non-profit academic research; faces generated with
  FaceGen Modeller)
