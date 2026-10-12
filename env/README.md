# GPU environment

EVELINK needs Python 3.8, torch 1.6, transformers 3.1, and faiss.
Make this environment on the GPU machine after the submodules are added:

```
conda create -n evelink python=3.8
conda activate evelink
pip install -r third_party/event-linking/requirements.txt
pip install -e .   # our samplers (they need only numpy)
```

Our sampler package says Python 3.10+. If it fails to install on 3.8, we lower that limit.
