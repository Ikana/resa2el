# Third-party code

Add these as git submodules, pinned to one commit, so the runs are repeatable:

```
git submodule add https://github.com/CogComp/event-linking third_party/event-linking
git submodule add https://github.com/ad32q/ReSa2 third_party/ReSa2
```

- event-linking: EVELINK (built on BLINK). We change only its negative sampling.
- ReSa2: reference code (firstSampling.py, secondRetrieve.py). Our sampler in
  `src/resa2el/samplers.py` must give the same results as this code.
