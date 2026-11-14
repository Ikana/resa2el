# Weight mixing: fine-tuned model and its start model

Date: 2026-11-13. Ticket: "How do others mix the weights of a trained model and its start model?"

## How to read this file

- Each claim has the URL that was opened. Section and table numbers are from that source.
- All sources are primary (arXiv papers, official code).
- The papers were read through a page-to-text tool. Numbers in tables were read by that tool.
  Check a number against the PDF before you put it in the report.
- "Not verified" marks a claim that the opened text did not show.
- "Our arithmetic" marks a number that we calculated. It is not in the source.
- Notation: `theta(alpha) = (1 - alpha) * theta_start + alpha * theta_finetuned`.
  `alpha = 0` is the start (released) model. `alpha = 1` is the fine-tuned model.
  This is the notation of WiSE-FT.

## 1. WiSE-FT (Wortsman et al., 2021)

Source: https://arxiv.org/pdf/2109.01903. Code: https://github.com/mlfoundations/wise-ft

- Setup: CLIP image classifiers. Fine-tune on ImageNet. Test on five shifted ImageNet sets.
  The scope is image classification only (Section 7). There is no text or retrieval experiment.
- Method: linear mix of all weights of the zero-shot model and the fine-tuned model.
  The official code does this for each key of the state dict:
  `(1-alpha) * theta_0[key] + alpha * theta_1[key]` (README of the code repository).
- Ratios tested: the paper tables use alpha from 0.00 to 1.00 in steps of 0.05 (Appendix C.1, Tables 4 and 5).
  The README command uses 11 values: `--alpha 0 0.1 0.2 ... 1.0`.
- Result (Table 1, end-to-end, ImageNet / mean of five shifts):
  - zero-shot: mean of shifts 73.4
  - fine-tuned: 86.2 / 68.6
  - alpha = 0.5: 86.8 / 76.9
  - best alpha: 87.1 / 77.4
- So fine-tuning lowered the shifted accuracy below the start model (68.6 against 73.4).
  The mix at 0.5 was better than both end models on the shifts. This is the same pattern as our news set.
- Section 5.2, Observation 2: the mixed model "may outperform both the zero-shot and fine-tuned models".
- Section 5.2, Observation 1: on ImageNet and the five shifts, for all alpha the mixed accuracy is at least the straight line between the two end accuracies. No barrier was found there.
- Choice of ratio: Section 4 recommends "α=0.5 when no domain knowledge is available".
  Appendix B, Table 3: the best alpha is only 0 to 0.4 points better than alpha = 0.5.
  The "best alpha" rows in Table 1 are chosen on the reported test column (Table 1 caption). This is an oracle number.
  The paper leaves the choice of alpha for a target distribution to future work (Section 7).
- Low start accuracy: on WILDS shifts the zero-shot model has less than 30% accuracy, and the mix still helps (Section 4).
  Not verified: the per-dataset numbers (Appendix C.2, Table 6).
- Cases where the mix is worse:
  - Linear-classifier variant, alpha = 0.5: in-distribution accuracy 83.7 against 85.2 for the fine-tuned classifier (Table 1). The mix can cost in-distribution accuracy.
  - Section 3: the mix of all layers "typically fails" for two networks that were trained from different random starts. The fine-tuned model must come from the start model.
- Learning rate and epochs change the shifted accuracy of the fine-tuned model by up to 8 points (Section 4, Figure 3). The paper says that WiSE-FT removes this trade-off.

## 2. Model soups (Wortsman et al., 2022)

Source: https://arxiv.org/pdf/2203.05482

- Method: average the weights of many models that were fine-tuned from the same start model.
  Uniform soup: average all. Greedy soup: sort by held-out accuracy, add a model only if held-out accuracy does not go down (Section 2).
- Data that chooses the ingredients: a held-out validation set. For ImageNet this is about 2% of the training set (Section 3.1). It is not the test set.
- Result under shift (Table 3, CLIP ViT-B/32, ImageNet / shifts): best single model 80.38 / 47.83. Uniform soup 79.97 / 51.45. Greedy soup 81.03 / 50.75.
  The uniform soup was best on the shifts but below the best single model in distribution.
- Text models (Table 5, GLUE validation, best single model to greedy soup):
  BERT: MRPC 88.3 to 88.3, RTE 61.0 to 61.7, CoLA 59.1 to 59.1, SST-2 92.5 to 93.0.
  Section 3.3.3: "the improvements are not as pronounced as in image classification".
- Seeds: the paper uses models that differ only in the seed for its analysis (Section 3.2, Appendix G).
  Not verified: a table with accuracy numbers for a seed-only soup. The opened text had none.
- Failure case, stated by the authors:
  - Section 3.3.1: "there can be an error barrier between fine-tuned models". They see this mainly with high learning rates.
  - Appendix J.1: with a larger learning rate, the error goes up on the path between two fine-tuned models.
  - Section 4, footnote 3: models with learning rate 1e-4 are far from the start model in weight space, and the greedy soup often rejects them.
  - Section 3.3.1: a uniform soup beats the best single model only when all models have high accuracy.
- Limit (Section 5): soups do not improve calibration as ensembles do.

## 3. PAINT (Ilharco et al., 2022)

Source: https://arxiv.org/pdf/2208.05592

- This is the closest setup to ours in one way. Only one encoder is fine-tuned. Appendix B: "we adapt only the weights of the visual encoder, maintaining the text encoder without updating its weights".
  We fine-tune only the mention encoder and keep the page encoder frozen.
- Method: mix the zero-shot weights and the fine-tuned weights with alpha in [0, 1].
- Choice of ratio (Section 2): "we pick the mixing coefficient α that optimizes average accuracy on the held-out validation sets from the supported and patching tasks".
  So the ratio comes from held-out data of both the new task and the old tasks. It does not come from test data.
- Result (Section 4): accuracy on the new task goes up 15 to 60 points for ViT-L/14. ImageNet accuracy goes down less than 1 point.
- Limits (Section 9): accuracy on the old tasks "can still decrease, especially for smaller models". Section 4.1: the method is more effective for larger models. Section 4.2: it is much worse with ResNets.

## 4. LP-FT (Kumar et al., 2022)

Source: https://arxiv.org/pdf/2202.10054

- This paper explains why fine-tuning hurts the other domain. It is not a weight-mixing paper.
- Section 4.1, Tables 1 and 2 (10 image shift benchmarks): fine-tuning gives 85.1% in distribution and 59.3% out of distribution. Linear probing gives 82.9% and 66.2%.
- Condition (abstract): fine-tuning can be worse out of distribution "when the pretrained features are good and the distribution shift is large".
  When the shift is small (CIFAR-10 to CIFAR-10.1), fine-tuning is better (Section 4.3).
- Cause (abstract, Section 4.3): fine-tuning changes ("distorts") the pretrained features. It changes them more for in-distribution examples than for out-of-distribution examples.
- LP-FT (linear probe first, then fine-tune): 85.7% in distribution, 68.9% out of distribution (Section 4.2).
- Selection data: learning rate and early stopping use the in-distribution validation set. "OOD data was only used for evaluation" (Section 4.1).
- Section 5 names WiSE-FT as concurrent work that lowers the same trade-off. The paper does not test weight mixing.

## 5. L2-SP (Li et al., 2018)

Source: https://arxiv.org/pdf/1802.01483

- Method: a penalty on the distance to the start weights, `(alpha/2) * ||w - w0||^2` (Section 3, Equation 2). This is used during training. It is not a mix after training.
- Result (Table 2): L2-SP is better than plain weight decay on all four target tasks. Example: MIT Indoors 67, 79.6 to 84.2.
- Penalty strengths: chosen by cross-validation (Section 4.2). Figure 1 shows alpha = 0, 1e-3, 1e-2, 1e-1, 1.
- The paper tests only the target task. It has no test on a second domain after fine-tuning. So it gives no number for our question.
- It is relevant only as the training-time relative of weight mixing: both keep the model near the start model.

## 6. Dense retrievers and bi-encoders

### LM-Cocktail (Xiao et al., 2023)

Source: https://arxiv.org/html/2311.13534v4. Code named in the abstract: https://github.com/FlagOpen/FlagEmbedding

- This is a direct result for a bi-encoder text retriever.
- Problem (Section 1): fine-tuning can cause "severe degeneration" of general ability outside the target domain.
- Method with one fine-tuned model (Section 2.2, Equation 3): `M_r = alpha * M_t + (1 - alpha) * M_b`. M_t is the fine-tuned model and M_b is the base model.
  The paper states that "simply setting α to 0.5 yields promising outcomes". The default alpha is 0.5.
- Setup (Sections 3.2 and 4.1.2): base encoder bge-base-v1.5. Fine-tune on one of 9 retrieval datasets. Test on 15 MTEB retrieval tasks. Metric NDCG@10.
- Result (Table 2, base / fine-tuned / mix of base and fine-tuned):
  - Fine-tuned on HotpotQA. Target task: 71.81 / 75.96 / 74.78. Other tasks: 49.81 / 47.49 / 49.98.
  - Fine-tuned on Quora. Target task: 88.90 / 90.31 / 89.93. Other tasks: 48.59 / 47.43 / 48.09.
  - Fine-tuned on MSMarco. Target task: 41.15 / 42.23 / 42.01. Other tasks: 52.00 / 51.98 / 52.71.
- Reading: the mix gets back the loss on the other tasks (about 1 to 2.5 points). It keeps most of the target gain but not all of it. For Quora the mix is still below the base model on the other tasks.
- Ratio sweep (Section 4.3, Figures 2 and 3): alpha goes from 0 to 1. Not verified: the grid values and the numbers. They are only in the figures.
- Failure in the same paper (Table 1, text classification with a decoder model): MRPC target accuracy goes from 85.78 (fine-tuned) to 73.77 (mix). The mix can cost a lot on the target task.

### Sasaki et al., CIKM 2025

Source: https://arxiv.org/html/2509.21966v1

- Setup: mix a retrieval model (e5-mistral-7b-instruct) with a domain model that is not a retriever (BioMistral-7B, or a Japanese model). Both come from Mistral-7B. This is not "fine-tuned plus its start model", so it is only a near case.
- Ratios tested: each of two layer groups takes alpha from {0.00, 0.25, 0.50, 0.75, 1.00}.
- Choice of ratio: grid search on a development set (queries sampled from the training data). The test set is not used.
- Result (Table 2, nDCG@10, source to merged): NFCorpus 39.02 to 40.59, SciFact 77.32 to 77.63, MIRACL-ja 75.49 to 77.90, JQaRA 60.80 to 63.89. Three of four are significant.
- With only 50 development queries (Table 3), the merged model still wins on three of four sets and has a smaller standard deviation than LoRA. A ratio has only one or two free numbers, so a small development set is enough.

### What we did not find

- No paper was found that applies WiSE-FT or a soup to DPR, Contriever, or BLINK. Three web searches gave no such source. This does not prove that none exists.
- No paper was found on weight mixing for entity linking or event linking.
- No paper was found for a setup with one frozen encoder in a text bi-encoder. PAINT (Section 3) is the nearest case, in image-text models.

## 7. When does linear mixing fail?

- **Different random starts.** Two networks trained from different random starts have a barrier. Two networks fine-tuned from the same pretrained weights have none (Neyshabur et al., Section 3.3, Figure 4, ResNet-50). https://arxiv.org/pdf/2008.11687
  This paper does not test the path from the pretrained model to the fine-tuned model.
- **Definition.** Barrier = the largest rise in error on the linear path above the two end models. Below 2% counts as stable. The path is checked at 30 values of alpha (Frankle et al., Section 2). For unstable pairs the error on the path goes up to random guessing (Section 3). https://arxiv.org/pdf/1912.05671
- **High learning rate, far from the start model.** See Section 2 above (model soups, Section 3.3.1, Appendix J.1, footnote 3).
- **BERT fine-tuned with different seeds.** Juneja et al. fine-tune bert-base-uncased many times. The only difference is the seed (Section 2). Abstract: "some pairs of finetuned models have large barriers of increasing loss on the linear paths between them".
  The models fall into groups. Models in one group are connected. Models in different groups have a barrier. The groups have nearly equal in-domain accuracy (MNLI 0.844 against 0.839) but different accuracy on the diagnostic set (HANS-LO 0.301 against 0.096) (Table 1, Section 4.1).
  They check 11 values of alpha on 512 examples (Appendix F.2). https://arxiv.org/pdf/2205.12411
  This matters for our 3-seed average: three BERT seeds are not sure to be in one basin.
- **Pre-trained language models, more factors.** Qin et al. (T5-base). Full fine-tuning stays connected over seeds, data order, learning rate, and batch size. Adapter tuning shows drops. Models tuned on different tasks are not connected. For different data domains of one task there is no barrier, but there is also no "valley": the score changes one way along the path (Section 4, Figure 5). Longer training moves the models apart (Section 4, Appendix B.2). They check 24 inner points plus 2 end points. https://arxiv.org/pdf/2210.14102
  So a path without a barrier does not promise a mix that beats both ends.
- **Small models.** PAINT Section 9 (see Section 3 above).
- **Loss on the target task.** WiSE-FT linear classifier and LM-Cocktail MRPC (see Sections 1 and 6 above).

## 8. Summary table

| Source | Models | Ratios tested | Data that chooses the ratio | Fine-tune domain / other domain |
|---|---|---|---|---|
| WiSE-FT | CLIP image classifiers | 0 to 1, step 0.05 (code: step 0.1) | none; 0.5 is recommended; "best" is a test oracle | 0.5 gains on both; beats both ends on shifts |
| Model soups | CLIP, ALIGN, ViT-G, BERT, T5 | equal weights | held-out validation (greedy) | gains on both for images; small for BERT |
| PAINT | CLIP, vision encoder only | continuous 0 to 1 | held-out validation of new and old tasks | large gain new; under 1 point loss old |
| LM-Cocktail | BGE bi-encoder | 0 to 1, default 0.5 | none for two models; 5 examples for many models | keeps most target gain; gets back other-task loss |
| Sasaki et al. | 7B embedding model | 0, 0.25, 0.5, 0.75, 1 | development set | gain on 3 of 4 sets |

## What this means for our decision

Our numbers: Wikipedia recall at 10 goes 77.7 to 93.3. News recall at 10 goes 33.6 to 30.9 on 769 mentions.

**Recommended ratios**

- Evaluate alpha = 0, 0.1, 0.2, ..., 1.0 (11 points). This is the grid of the official WiSE-FT code. It costs no training.
  If time is short, use 0, 0.25, 0.5, 0.75, 1 (Sasaki et al.).
- Name alpha = 0.5 as the main ratio before you look at any news result. WiSE-FT Section 4 recommends it, and LM-Cocktail Section 2.2 uses it as default for a bi-encoder.
- Mix only the mention encoder. The page encoder is the same in both models, so a mix of it changes nothing. PAINT does the same with its frozen text encoder (Appendix B).

**How to choose the ratio without the test set**

- Do not choose alpha on the 769 news mentions and then report the same 769 as the result. WiSE-FT calls its test-chosen rows "optimal α" and reports alpha = 0.5 separately (Table 1). Do the same: show the full news curve, mark its best point as an oracle, and report alpha = 0.5 as the result.
- If you want a chosen ratio, use held-out data that is not the test set. PAINT (Section 2) and model soups (Section 3.1) use held-out validation. Sasaki et al. use a development set.
  With only Wikipedia validation data, a rule is needed, because Wikipedia alone will choose a large alpha. One possible rule: the smallest alpha that keeps Wikipedia validation recall within a fixed margin of the fine-tuned model. This rule is our proposal. No source tests it.
- A small set is enough to choose one number: 50 queries worked in Sasaki et al. (Table 3), and LM-Cocktail uses 5 examples (Section 2.1). But a set held out from the news test makes the test set smaller.
- For the 3-seed average, a seed-only soup needs no selection: use equal weights (uniform soup). Check the result on Wikipedia validation data (model soups, Section 2).

**What to expect**

- On Wikipedia: alpha = 0.5 keeps most of the gain but probably not all of it. LM-Cocktail Table 2 shows a loss of 0.2 to 1.2 points of a 1.1 to 4.2 point gain on the target task. WiSE-FT shows no loss (Table 1). Our gain is much larger (15.6 points) and nobody reports a case like that, so the Wikipedia cost at 0.5 is uncertain.
- On news: expect recall between the two end models or a little above the start model. WiSE-FT got 76.9 against 73.4 and 68.6 (Table 1). LM-Cocktail got back to the base level in two of three cases (Table 2). Qin et al. saw no valley for domain shifts (Section 4), so a result between the ends is as likely as a result above both.
- Noise: the news drop of 2.7 points is about 21 mentions of 769. One standard error of a rate near 32% with n = 769 is about 1.7 points (our arithmetic; a paired test is tighter). Differences between nearby alpha values will be smaller than this. Do not read a peak at one alpha as real.
- The 3-seed average: expect a small change. Soups gave BERT 0.0 to 0.7 points on GLUE (model soups, Table 5).

**Signs of failure**

- A barrier: recall at some alpha is clearly below both end models on Wikipedia or on news. Frankle et al. use a 2% rise as the limit (Section 2). WiSE-FT saw no point below the straight line (Section 5.2, Observation 1), so a dip below the line is already unusual.
- The 3-seed average is worse than each single seed on Wikipedia validation data. This means the seeds are not in one basin (Juneja et al.). In that case, report the mean of the three single-seed scores and do not use the soup.
- A large Wikipedia loss at alpha = 0.5 with no news gain (the LM-Cocktail MRPC case, Table 1).
- A flat news curve: recall moves from 33.6 to 30.9 in one direction with no point above 33.6 (Qin et al., Section 4). This is not a barrier. It means the mix gives only a trade-off, and the honest report is the curve.
- Causes to check if it fails: learning rate too high or too many updates, so the model is far from the start model (model soups Section 3.3.1 and footnote 3; Qin et al. Appendix B.2). 1,500 updates is short, so this risk is low.
