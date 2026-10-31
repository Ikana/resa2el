"""Smoke test for EVELINK on the new torch. Run it on the GPU machine first.

It makes a tiny random BERT (no download), builds the EVELINK bi-encoder,
and does 3 training steps with the EVELINK optimizer and scheduler.
It also saves and loads the model. If the loss is a number and nothing
fails, the torch change works on this machine.

    cd third_party/event-linking && python ../../scripts/smoke_evelink.py
"""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.getcwd())
from pytorch_transformers.modeling_bert import BertConfig, BertModel  # noqa: E402
from pytorch_transformers.optimization import WarmupLinearSchedule  # noqa: E402
from blink.biencoder.biencoder import BiEncoderRanker  # noqa: E402
from blink.candidate_ranking.utils import save_model  # noqa: E402

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
tmp = tempfile.mkdtemp()
vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"w{i}" for i in range(95)]
with open(os.path.join(tmp, "vocab.txt"), "w") as f:
    f.write("\n".join(vocab))
cfg = BertConfig(vocab_size_or_config_json_file=len(vocab), hidden_size=32,
                 num_hidden_layers=2, num_attention_heads=2, intermediate_size=64)
BertModel(cfg).save_pretrained(tmp)

params = {"bert_model": tmp, "out_dim": 1, "pull_from_layer": -1, "add_linear": False,
          "no_cuda": not torch.cuda.is_available(), "lowercase": True,
          "type_optimization": "all_encoder_layers", "learning_rate": 1e-3, "fp16": False}
ranker = BiEncoderRanker(params)
dev = str(ranker.device)
opt = ranker.get_optimizer()
sched = WarmupLinearSchedule(opt, warmup_steps=1, t_total=3)
g = torch.Generator().manual_seed(0)
ctx = torch.randint(5, len(vocab), (4, 16), generator=g)
cand = torch.randint(5, len(vocab), (4, 16), generator=g)
for step in range(3):
    loss, scores = ranker(ctx.to(dev), cand.to(dev))  # in-batch negatives
    loss.backward()
    opt.step(); sched.step(); opt.zero_grad()
    print("step", step, "loss", float(loss))
    assert torch.isfinite(loss)

save_model(ranker.model, ranker.tokenizer, os.path.join(tmp, "out"))
ranker.load_model(os.path.join(tmp, "out", "pytorch_model.bin"), cpu=True)
print("OK")
