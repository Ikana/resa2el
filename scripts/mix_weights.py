"""Mix the weights of fine-tuned bi-encoders with their start model (added for resa2el).

  theta = (1 - alpha) * theta_start + alpha * mean(theta_finetuned_1, ..., theta_finetuned_n)

alpha = 0 is the start (released) model, alpha = 1 is the fine-tuned model (WiSE-FT notation).
With more than one --models file, the fine-tuned models are averaged first (a uniform soup).
The page encoder was frozen in training, so the mix changes only the mention encoder.

Run: python scripts/mix_weights.py --start models/biencoder/pytorch_model.bin \
        --models runs/main_lr1e-5/resa2_s0/pytorch_model.bin --alpha 0.5 --out /tmp/mixed.bin
"""
import argparse

import torch


def mix(start, models, alpha):
    out = {}
    for key, s in start.items():
        if not torch.is_floating_point(s):
            out[key] = s.clone()  # integer buffers (position ids) are the same in all models
            continue
        mean = sum(m[key].to(torch.float64) for m in models) / len(models)
        out[key] = ((1 - alpha) * s.to(torch.float64) + alpha * mean).to(s.dtype)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    start = torch.load(a.start, map_location="cpu")
    models = [torch.load(m, map_location="cpu") for m in a.models]
    for m in models:
        assert m.keys() == start.keys(), "the models do not have the same weights"
    mixed = mix(start, models, a.alpha)
    changed = sum(1 for k in start if torch.is_floating_point(start[k]) and not torch.equal(start[k], mixed[k]))
    print("mixed %d model(s) at alpha %.2f: %d of %d tensors differ from the start model" % (len(models), a.alpha, changed, len(start)))
    torch.save(mixed, a.out)


if __name__ == "__main__":
    main()
