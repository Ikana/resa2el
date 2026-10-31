# Third-party code

- `event-linking/`: EVELINK (built on BLINK), a copy of
  https://github.com/CogComp/event-linking at commit bc2b27f. It is a copy, not a
  submodule, because we change it: first for the new torch (see `env/README.md`),
  later only its negative sampling. The commit "Add EVELINK code, unchanged" has
  the original files, so `git diff` from that commit shows all our changes.
- ReSa2: reference code (firstSampling.py, secondRetrieve.py) at
  https://github.com/ad32q/ReSa2. Not added yet. Our sampler in
  `src/resa2el/samplers.py` must give the same results as this code.
