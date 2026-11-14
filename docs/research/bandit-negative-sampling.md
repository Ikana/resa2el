# Bandits that pick negatives or training examples: what others have done

Research note for the ticket "What have others done with bandits that pick negatives?".
Date of the reading: 2026-11-13.

## How to read this note

- Each claim has the URL that was opened. Only papers and author code were used.
- The PDFs were read through a fetch tool that extracts and summarises text. Section, equation, and theorem numbers come from that extraction. Check each formula against the PDF before it goes into a proof.
- "NOT VERIFIED" marks a claim that no opened source supports.
- "Our arithmetic" marks numbers that we computed from a published bound. They are not in the source.
- A regret bound in this note is a bound on the bandit reward. It is not a bound on retrieval accuracy. No source found here proves that low bandit regret gives a better retriever.

## Short answer

1. No published method was found that uses a bandit with a proved bound to pick negatives for a dense retriever. The dense-retrieval methods (ANCE, RocketQA, SimANS, STAR/ADORE) use fixed rules. The knowledge-graph methods use REINFORCE (KBGAN) or a cache (NSCaching) and prove nothing. This is a gap that the project can fill. The search was not exhaustive.
2. Bandits with proved bounds exist for a near problem: sampling data points, coordinates, or graph neighbours to reduce gradient variance. All of them use EXP3-type rules. The reward is a gradient norm divided by a power of the sampling probability.
3. Bandits over a small number of arms (tasks, domains) exist for curriculum learning. Graves et al. (2017) use EXP3.S with learning-progress rewards. They cite the bound of Auer et al. (2002) and prove none.
4. Rewards built from loss or gradient norm push the sampler to the hardest examples. In retrieval the hardest examples are often false negatives. Authors stop the drift with a floor on each probability, a clip on the reward, a limit on the candidate set, a filter, or a reward measured on held-out data.

## Table 1. Methods that learn the sampler during training

| Method | Arms | Reward | Rule | Bound proved by the authors |
|---|---|---|---|---|
| Graves et al. 2017, automated curriculum | The N tasks of a curriculum | Learning progress. Best: prediction gain (loss before minus loss after one update on the same sample). Rescaled to [-1, 1] | EXP3.S | None. They cite Auer et al. 2002 |
| Matiisen et al. 2017, teacher-student curriculum | N subtasks | Absolute slope of the learning curve of the task | EWMA with epsilon-greedy or Boltzmann; a window rule; a Thompson-like rule | None |
| Kreutzer et al. 2021, bandits for MT data | Data groups (2 to 99: domains, languages) | Loss, prediction gain, relative prediction gain, each on train or on dev batches | EXP3 | None. They cite Auer et al. 2002 |
| Salehi et al. 2017, MABS | The n data points | a_i / p_i^2, where a_i is the squared gradient norm term | EXP3 with uniform mix eta = 0.4 | Sum of variance <= 3 x (best fixed distribution) + 50 sqrt(n^5 T a2 ln n) |
| Namkoong et al. 2017 | b blocks of coordinates or of examples | Squared block gradient norm / p_j^2, shifted | EXP3-style mirror descent with p >= p_min | Optimality gap bound against the best fixed p with p >= p_min. Bandit term decays as T^(-3/4) |
| Borsos et al. 2018, VRB | The n data points | Squared loss or gradient norm / p_i | Follow-the-regularised-leader, mixed with uniform | Regret <= 74 L n^(1/3) T^(2/3) against the best fixed distribution |
| Liu et al. 2020, GNN bandit samplers | The neighbours of a node | alpha_ij^2 ||h_j||^2 / (k q_ij^2) | EXP3 and EXP3.M | Sum of variance <= 3 x (best fixed) + 10 sqrt(T n^4 ln(n/k) / k^3) |
| Zhang et al. 2021, Thanos | The neighbours of a node | Bounded: 2 z_i . z_mean - ||z_i||^2 | EXP3.M with restarts (Rexp3) | Regret <= C (D ln D)^(1/3) (T sqrt(ln T))^(2/3) against a dynamic oracle |
| Baykal et al., multiplex networks | The L - 1 other layers | Loss: embedding distance between layers | EXP3 with adaptive learning rate | Regret <= 6 sqrt(V_T log n) + 10 M log n |
| Kim and Oh, AdamCB | The n data points, K per batch | Loss that falls when gradient norm rises | EXP3-like semi-bandit with DepRound | O(d sqrt(T) + (sqrt(d) / n^(3/4)) (T/K ln(n/K))^(1/4)) |
| Lisicki et al. 2023, batch selection under label noise | The data points, m per batch | Variance of the predicted correct-class probability | Follow the Perturbed Leader | None |
| Cai and Wang 2018, KBGAN | 20 uniformly drawn candidate negatives | Minus the discriminator distance of the negative (its hardness) | REINFORCE with baseline (not a bandit) | None |
| Zhang et al. 2019, NSCaching | A cache of 50 candidates per positive | No reward. Cache refresh by exp(score) | Not a bandit. Uniform draw from the cache | None |
| Wang et al. 2020, KGPolicy | Paths on an item knowledge graph | Not read in full | Reinforcement learning (not a bandit) | None in the abstract |

The details for each row are in the sections that follow.

## 1. Curriculum bandits (few arms, learning-progress reward)

### Graves et al. 2017, "Automated Curriculum Learning for Neural Networks"

Source: https://arxiv.org/abs/1704.03003 and https://arxiv.org/pdf/1704.03003

- The paper exists. Authors: Graves, Bellemare, Menick, Munos, Kavukcuoglu.
- Arms: one arm for each task. A curriculum of N tasks is an N-armed bandit (Sec. 2.2).
- Rule: EXP3.S of Auer et al. 2002 (Sec. 2.2, Eq. 1). The policy mixes with uniform: pi(i) = (1 - eps) pi_EXP3(i) + eps / N. The weights use a share step with alpha_t = 1/t. Values: eta = 0.001, beta = 0, eps = 0.05.
- Reward: learning progress, divided by the time to process the sample. The signals are (Sec. 3):
  - Prediction gain (PG): L(x, theta) - L(x, theta') on the same sample x.
  - Gradient prediction gain (GPG): squared gradient norm. The paper says that it is biased to high-variance tasks.
  - Self prediction gain (SPG): the same as PG, but on a new sample of the same task.
  - Target prediction gain (TPG) and mean prediction gain (MPG): the gain on the target task or on a random task.
  - Complexity gains (VCG, GVCG, L2G, GL2G): the growth of model complexity.
- Reward scale (Sec. 2.3, Eq. 2): each reward is clipped to the 20th and 80th percentiles of past rewards and mapped to [-1, 1]. The percentiles come from a reservoir sample.
- Bound: the authors prove none. They rely on Auer et al. 2002 and on Herbster and Warmuth 1998.
- Results: PG was "the most consistent signal" for maximum-likelihood training (Sec. 5). GPG, L2G, and GL2G were much worse than uniform on the repeat-copy task (Sec. 4.2). The authors say that uniform sampling "is a surprisingly strong benchmark" (Sec. 5).
- Note: the extraction shows that the text names the same policy "EXP3.S" in one place and "EXP3.P" in another. Check the PDF.

### Matiisen et al. 2017, "Teacher-Student Curriculum Learning"

Source: https://arxiv.org/pdf/1707.00183

- Arms: N subtasks.
- Reward: the change in score of the task. The teacher uses the absolute value, so a task that the student forgets is trained again.
- Rules: an exponentially weighted moving average with epsilon-greedy or Boltzmann choice; a window rule with a regression slope; a rule that samples one past reward from a buffer for each task, in the manner of Thompson sampling.
- Bound: none. The paper reports that adversarial bandit rules (Auer et al. 2002) did not perform well in its experiments.

### Kreutzer et al. 2021, "Bandits Don't Follow Rules"

Source: https://arxiv.org/pdf/2110.06997

- Arms: groups of machine-translation data (2 styles, 5 domains, or up to 99 language pairs).
- Rewards: loss, prediction gain, relative prediction gain, each on the training batch or on a balanced development set. The scale is that of Graves et al.: clip to the 20th and 80th percentiles of the last 5,000 rewards, then map to [-1, 1].
- Rule: EXP3. The authors did not need EXP3.S.
- Bound: none proved. They cite the O(sqrt(T ln d)) bound of Auer et al. 2002.
- Finding: rewards on development batches were better than rewards on training batches, because training loss is confounded by noisy or untypical examples.

## 2. Bandits for importance sampling in SGD (many arms, variance reward)

### Salehi et al. 2017, "Stochastic Optimization with Bandit Sampling" (MABS)

Source: https://arxiv.org/abs/1708.02544 and https://arxiv.org/pdf/1708.02544

- Arms: the n data points (Sec. 2).
- Reward: the derivative of the variance of the gradient estimate, a_i / p_i^2, where a_i comes from the squared gradient norm of point i. The bandit sees an importance-weighted estimate.
- Rule: EXP3. Only the weight of the drawn arm changes: w <- w exp(delta a_i / p_i^3). The probability is p_j = (1 - eta) w_j / W + eta / n, with eta = 0.4.
- Bound (Theorem 2): sum_t V(p_t) <= 3 sum_t V(p*) + 50 sqrt(n^5 T a2 ln n). The factor 3 is (1 - eta) / (1 - 2 eta) at eta = 0.4. The comparator p* is one fixed distribution. Conditions: bounded gradients and a horizon T above a stated threshold.
- The gradient estimate stays unbiased, because each sampled gradient is divided by n p_i (Eq. 3).
- Note: the extraction found two sign conventions for the reward in the paper. Check the PDF.

### Namkoong et al. 2017, "Adaptive Sampling Probabilities for Non-Smooth Optimization"

Source: https://proceedings.mlr.press/v70/namkoong17a/namkoong17a.pdf (ICML 2017)

- Arms: b blocks of coordinates (Sec. 2) or b blocks of examples, for example one block for each class label (Sec. 3). This is the nearest published match to "arms are bands of the pool".
- Loss: squared block gradient norm divided by p_j^2, with a shift that keeps it non-negative (Eq. 5).
- Rule: EXP3-style mirror descent. The probabilities stay in the set {p : p >= p_min}. A KL projection restores the floor (Eq. 7, Algorithm 4).
- Bound (Theorem 3 for coordinate descent, Theorem 7 for SGD): the optimality gap is at most a term with the best fixed p (p >= p_min) in hindsight, plus a bandit term that decays as T^(-3/4) and grows as 1 / p_min.
- The paper states the trade-off: a small p_min makes the comparator better and the bandit term worse. Experiments use p_min from 0.01 / b to 0.5 / b.

### Borsos et al. 2018, "Online Variance Reduction for Stochastic Optimization" (VRB)

Source: https://arxiv.org/pdf/1802.04715

- Arms: the n data points.
- Cost: f_t(p) = sum_i l_t(i)^2 / p(i), where l_t(i) is a loss or a gradient norm.
- Rule: follow-the-regularised-leader, p_t(i) proportional to sqrt(cumulative squared loss + gamma). The result is mixed with uniform, weight theta = (n / T)^(1/3).
- Bound (Theorem 7): pseudo-regret <= 74 L n^(1/3) T^(2/3) against the best fixed distribution, for T >= n and bounded losses. Theorem 8 gives the same order for expected regret and holds for a non-oblivious adversary.
- The paper says that to compete with the best distribution at each step "seems generally hard".

### Liu et al. 2020, "Bandit Samplers for Training Graph Neural Networks"

Source: https://arxiv.org/abs/2006.05806 and https://arxiv.org/pdf/2006.05806

- The paper exists. Authors: Liu, Wu, Zhang, Zhou, Yang, Song, Qi.
- Arms: the neighbours of a node. One draw is one arm (EXP3), or one draw is a set of k arms (EXP3.M with DepRound).
- Reward (Eq. 9): r_ij = alpha_ij^2 ||h_j||^2 / (k q_ij^2). It is the negative derivative of the variance.
- Bound (Theorem 1): sum_t V(Q_t) <= 3 sum_t V(Q*) + 10 sqrt(T n^4 ln(n/k) / k^3), with eta = 0.4 and a horizon above a threshold. Q* is one fixed distribution. Assumptions: embedding norms <= 1, attention weights <= 1, and a probability floor of k eta / n.
- The paper says that the single-play version is "not rigorous" (Sec. 4).
- Author code named in the paper: https://github.com/xavierzw/ogb-geniepath-bs (NOT VERIFIED: the repository page was not opened).

### Zhang et al. 2021, "A Biased Graph Neural Network Sampler with Near-Optimal Regret" (Thanos)

Source: https://arxiv.org/pdf/2103.01089 (NeurIPS 2021)

- Arms: the neighbours of a node.
- Reward (Eq. 3): r_i = 2 z_i . z_mean - ||z_i||^2. It is bounded (Lemma 1). It prefers neighbours near the mean, not neighbours with a large norm.
- The paper reports that the reward of Liu et al. is unstable because of the 1 / p^2 term: mean near 2.5, maximum near 1,800.
- Rule: EXP3.M with a restart of the weights every Delta_T steps (Rexp3 of Besbes et al.).
- Bound (Theorem 3): regret <= C (D ln D)^(1/3) (T sqrt(ln T))^(2/3) against a dynamic oracle (the best set at each step).
- The key step: Lemma 2 bounds the total change of the rewards (the variation budget) by C ln T. It uses a learning rate of 1 / t, Lipschitz activations, bounded weights, and bounded gradients.
- The paper argues that the factor 3 in the bound of Liu et al. allows regret that is linear in T. This is the claim of the Thanos authors. We did not check it.

### Other bandit samplers

- Baykal et al., "Bandit Sampling for Multiplex Networks". Source: https://arxiv.org/pdf/2202.03621. Arms: the other layers of the network. Loss: embedding distance. Rule: EXP3 with an adaptive learning rate. Theorem 1: expected regret <= 6 sqrt(V_T log n) + 10 M log n. The extraction notes that the last line of the proof has other constants.
- Kim and Oh, "Adam Optimization with Adaptive Batch Selection" (AdamCB). Source: https://arxiv.org/html/2512.06795v1. Arms: the data points, K per batch. Rule: EXP3-like semi-bandit with DepRound. Theorem 1: O(d sqrt(T) + (sqrt(d) / n^(3/4)) (T/K ln(n/K))^(1/4)). The paper says that the bound of the earlier "Adam with Bandit Sampling" is incorrect.
- Liu, Wu, Mozafari 2020, "Adam with Bandit Sampling for Deep Learning". Source: https://arxiv.org/abs/2010.12986. Only the abstract was read. It claims a rate of O(sqrt(log n / T)) in some cases. AdamCB disputes the proof. Do not cite this bound.
- Lisicki et al. 2023, "Bandit-Driven Batch Selection for Robust Learning under Label Noise". Source: https://arxiv.org/pdf/2311.00096. Reward: the variance of the predicted probability of the correct class over a history buffer. Rule: Follow the Perturbed Leader. No bound. The authors say that the known guarantees "may no longer apply" after their change.

## 3. Learned negative samplers that are not bandits

### KBGAN (Cai and Wang)

Source: https://arxiv.org/pdf/1711.04071. Code named in the paper: https://github.com/cai-lw/KBGAN (repository page not opened).

- A generator picks one negative from N_s = 20 uniformly drawn candidates.
- Reward: minus the discriminator distance of the picked negative. A harder negative gets a higher reward.
- Rule: one-step REINFORCE with a moving-average baseline. It is reinforcement learning, not a bandit rule with a regret bound.
- Bound: none. The validation curves fluctuate, and the authors could not confirm convergence.
- The authors say that the hardest negatives are likely to be true facts. The small candidate set is one of their guards.

### NSCaching (Zhang, Yao, Shao, Chen)

Source: https://arxiv.org/pdf/1812.06410. Code: https://github.com/AutoML-Research/NSCaching (README says it is the code for the paper, ICDE 2019).

- It is not a bandit and not RL. The ticket list was wrong on this point. The paper contrasts itself with the RL methods IGAN and KBGAN.
- A cache of N1 = 50 high-score candidates is kept for each positive. Negatives are drawn uniformly from the cache.
- The cache refresh adds N2 = 50 random entities and keeps N1 of the N1 + N2 by a draw with probability proportional to exp(score). This was better than keeping the top N1.
- The authors reject "always take the top score" because of exploration and false negatives.
- Bound: none. Convergence theory is listed as future work.

### KGPolicy (Wang et al. 2020)

Source: https://arxiv.org/abs/2003.05753 (abstract only). An RL agent walks an item knowledge graph to find a negative. No bound is in the abstract. Reward and rule were NOT VERIFIED.

### IGAN

NOT VERIFIED. The paper was not opened. NSCaching cites it as a REINFORCE-based method.

## 4. Bandit rules for rewards that change (non-stationary)

### Auer et al. 2002, "The Nonstochastic Multiarmed Bandit Problem" (EXP3, EXP3.S, EXP3.P)

Source: https://homepages.math.uic.edu/~lreyzin/papers/auer02.pdf (SIAM J. Comput. 32(1), 48-77). The copy at cseweb.ucsd.edu could not be fetched.

- Rewards are in [0, 1]. There is no statistical assumption. The bounds hold for any assignment of rewards.
- EXP3: p_i = (1 - gamma) w_i / sum_j w_j + gamma / K. Update: w_j <- w_j exp(gamma xhat_j / K), where xhat_j = x_j / p_j for the drawn arm and 0 for the others.
- Theorem 3.1: G_max - E[G_EXP3] <= (e - 1) gamma G_max + K ln K / gamma.
- Corollary 3.2: with gamma = min{1, sqrt(K ln K / ((e - 1) g))} and g >= G_max, the regret is <= 2.63 sqrt(g K ln K).
- EXP3.S (Sec. 8): the update adds a share term: w_j <- w_j exp(gamma xhat_j / K) + (e alpha / K) sum_i w_i.
- Theorem 8.1: for any sequence of arms j^T with hardness h (number of switches plus one), G_j - E[G_EXP3.S] <= K (h ln(K / alpha) + e alpha T) / gamma + (e - 1) gamma T.
- Corollary 8.3: with alpha = 1 / T and gamma = min{1, sqrt(K (S ln(KT) + e) / ((e - 1) T))}, for every sequence with h <= S: regret <= 2 sqrt(e - 1) sqrt(K T (S ln(KT) + e)).
- EXP3.P (Theorem 6.3): with probability >= 1 - delta, regret <= 4 sqrt(K T ln(KT / delta)) + 453 sqrt(K T ln K) + 8 ln(KT / delta). The tuning text was garbled in the extraction. Check the PDF.
- Adaptive adversary: Sec. 9 says that the results should extend to an adversary that reacts to the past actions of the player. The authors leave the check to the reader. It is not proved in the paper.

### Bubeck and Cesa-Bianchi 2012, survey by a primary author

Source: https://arxiv.org/pdf/1204.5721

- Theorem 3.1: EXP3 has pseudo-regret <= sqrt(2 n K ln K) with a fixed learning rate, and <= 2 sqrt(n K ln K) with eta_t = sqrt(ln K / (t K)). The second form does not need the horizon.
- The survey says that pseudo-regret "has no natural interpretation unless the adversary is oblivious".
- For a non-oblivious adversary, Theorem 3.3 gives E[R_n] <= 5.15 sqrt(n K ln K) + sqrt(n K / ln K) for EXP3.P.
- Sec. 3.4.3: EXP3.P has regret of order sqrt(n K S ln(n K / S)) against a comparator with at most S switches (result of Audibert and Bubeck 2010, not opened).

### Garivier and Moulines 2008, discounted UCB and sliding-window UCB

Source: https://arxiv.org/pdf/0805.3415

- Setting: rewards in [0, B]. The reward distributions are constant between breakpoints and change abruptly. Upsilon_T is the number of breakpoints.
- D-UCB: discounted means and counts with factor gamma. Theorem 1 with gamma = 1 - (4B)^(-1) sqrt(Upsilon_T / T) gives O(sqrt(T Upsilon_T) log T) wrong pulls (Remark 3).
- SW-UCB: means and counts over the last tau steps. Theorem 7 with tau = 2B sqrt(T log T / Upsilon_T) gives O(sqrt(Upsilon_T T log T)) (Remark 9).
- Lower bound (Theorem 13, Corollary 14): no policy has regret below order sqrt(T) when there are breakpoints.
- The paper states the EXP3.S bound for comparison and reports that D-UCB and SW-UCB beat tuned EXP3.S in its simulations.
- Limit for our use: the model assumes abrupt changes of otherwise stationary rewards. Our rewards drift at every step.

### Besbes, Gur, Zeevi, "Non-Stationary Rewards" (Rexp3 and the variation budget)

Source: https://arxiv.org/pdf/1405.3316 (Stochastic Systems 2019)

- Variation budget: V_T >= sum_t sup_k |mu_t^k - mu_(t+1)^k|. It measures the total drift of the mean rewards.
- Rexp3: run EXP3 and reset the weights every Delta steps.
- Theorem 2: with Delta = ceil((K log K)^(1/3) (T / V_T)^(2/3)) and gamma = min{1, sqrt(K log K / ((e - 1) Delta))}, the regret against the dynamic oracle is <= C (K log K)^(1/3) V_T^(1/3) T^(2/3). C is an absolute constant that the extraction did not give.
- Theorem 1: any policy has regret >= C K^(1/3) V_T^(1/3) T^(2/3). The rate is the best possible.
- Assumptions: rewards in [0, 1]; the mean sequences do not depend on the past actions of the policy; V_T is known in advance.

### Raj and Kalyani 2017, discounted Thompson sampling

Source: https://arxiv.org/pdf/1707.09727

- Bernoulli rewards. Each arm keeps Beta counts S and F. All arms are discounted each step: S <- gamma S, F <- gamma F. The played arm adds its reward.
- Bound: none. The authors give a probability expression for the two-armed case and list a regret bound as future work.
- Result for the project: discounted Thompson sampling gives no bound to cite.

### Cesa-Bianchi et al. 2012, fixed share with full information

Source: https://arxiv.org/pdf/1202.3323

- Setting: at each round the learner picks a distribution over d experts and then sees the loss of every expert. Losses are in [0, 1]. The bounds hold for all loss sequences.
- Fixed-share update (Eq. 5): p_(j,t+1) = alpha / d + (1 - alpha) v_(j,t+1), where v is the exponential-weights update.
- Corollary 1: against a comparator sequence with m0 = (switches + 1) and total mass U0, the regret is <= sqrt((U0 m0 / 2) (ln d + ln(e U0 / m0))).
- With U0 = T, this is sqrt((T m0 / 2) (ln d + ln(e T / m0))). This substitution is ours (and of the extraction), not a printed line of the paper.
- NOT VERIFIED: the standard static bound for exponential weights, sqrt((T / 2) ln d). No opened source states it. The primary sources are Freund and Schapire 1997 and Cesa-Bianchi and Lugosi 2006.

## 5. Rewards that push the sampler to the hardest examples, and the guards

### Which rewards cause the drift

- Gradient norm. ANCE argues that the best sampling distribution is proportional to the gradient norm of each instance (Eq. 10, proof deferred to Johnson and Guestrin 2018), and that the top-ranked negatives approximate it. Source: https://arxiv.org/pdf/2007.00808. Katharopoulos and Fleuret state the same optimal distribution. Source: https://arxiv.org/pdf/1803.00942.
- Variance rewards of the form gradient norm / p^2 (Salehi, Liu, Namkoong, Borsos). They raise the probability of high-gradient items by design. For a softmax loss, the gradient of the loss with respect to the score of a negative is its softmax probability. So the high-gradient negatives are the high-score negatives. This last step is our reasoning, not a quote.
- Gradient prediction gain. Graves et al. say that it is biased to high-variance tasks, and it did much worse than uniform on one task. Source: https://arxiv.org/pdf/1704.03003.
- Hardness as reward. KBGAN rewards the generator with minus the discriminator distance. Source: https://arxiv.org/pdf/1711.04071.
- Raw training loss. Mindermann et al. report that selection by loss or by gradient norm picked mostly points with corrupted labels in controlled tests. Source: https://arxiv.org/pdf/2206.07137.
- The limit case. Robinson et al. prove that when the hardness parameter beta goes to infinity, the loss becomes the worst case over all negative distributions (Proposition 3). They report that a larger beta "does not necessarily lead to better representations". Source: https://arxiv.org/pdf/2010.04592.

### Why the hardest negatives hurt

- False negatives. RocketQA checked top-retrieved MS MARCO passages that had no positive label and found that 70% were positives. Hard negatives without a filter gave MRR@10 of 26.03, against 32.39 for in-batch negatives and 36.38 for filtered hard negatives. Source: https://arxiv.org/pdf/2010.08191.
- Gradient noise. SimANS reports that top-ranked negatives have high gradient variance and are likely false negatives, and that bottom-ranked negatives have gradient means near zero. Source: https://arxiv.org/pdf/2210.11773.
- Collapse. FaceNet says that the hardest negatives "can in practice lead to bad local minima early on in training" and to a collapsed model (Sec. 3.2). Source: https://arxiv.org/pdf/1503.03832. Wu et al. explain it: when the negative is very near, noise dominates the direction of the gradient (Sec. 4). Source: https://arxiv.org/pdf/1706.07567.
- Instability. Zhan et al. show that static hard negatives make MRR fluctuate and often fall below random sampling (Sec. 5.1, Sec. 8.2). Source: https://arxiv.org/pdf/2104.08051. ANCE notes that a large learning rate can cause fluctuations (Appendix A.3). Source: https://arxiv.org/pdf/2007.00808.
- These reports match the project result: the top-score sampler was worse than uniform and collapsed at the higher learning rate.

### Table 2. Guards used by the authors

| Guard | Who | What they do | Source |
|---|---|---|---|
| Floor on each probability | Auer; Graves; Salehi; Namkoong | Mix with uniform: gamma / K; eps = 0.05; eta = 0.4; or p >= p_min by projection | auer02.pdf; 1704.03003; 1708.02544; namkoong17a.pdf |
| Clip and rescale the reward | Graves; Kreutzer | Clip to the 20th and 80th percentiles of past rewards, map to [-1, 1] | 1704.03003; 2110.06997 |
| Bounded reward by design | Thanos | Replace the 1 / p^2 reward with a bounded reward | 2103.01089 |
| Reward on other data | Graves (SPG, TPG); Kreutzer (dev rewards) | Measure progress on a new sample or on a dev batch, not on the trained sample | 1704.03003; 2110.06997 |
| Reducible loss | Mindermann et al. (RHO-LOSS) | Score = training loss minus the loss of a model trained on held-out data. Removes noisy, redundant, and outlier points | 2206.07137 |
| Variance bonus | SRNS | False negatives keep a high score with low variance. Pick by score + alpha x std over the last 5 epochs | 2009.03376 |
| Limit the candidate set | KBGAN; NSCaching; ANCE | 20 uniform candidates; uniform draw from a cache of 50; uniform draw of 1 from the top 200 | 1711.04071; 1812.06410; 2007.00808 |
| Centre on the positive | SimANS; FaceNet | p_i proportional to exp(-a (s_i - s_pos - b)^2) over the top 100, a = 0.5, b = 0; semi-hard negatives | 2210.11773; 1503.03832 |
| Clip the hardness weight | Wu et al.; Robinson et al. | Weight = min(lambda, 1 / q(d)); anneal beta to zero | 1706.07567; 2010.04592 |
| Filter | RocketQA | Keep a hard negative only if a cross-encoder scores it below 0.1 | 2010.08191 |
| Add random negatives | STAR | Loss = alpha x (random-negative loss) + (hard-negative loss) | 2104.08051 |
| Importance weights | Salehi; Katharopoulos and Fleuret | Divide each sampled gradient by N p_i, so the gradient estimate stays unbiased | 1708.02544; 1803.00942 |

Notes on the table:

- The importance-weight guard keeps the gradient unbiased. It does not remove false negatives.
- ADORE (Zhan et al.) is the published setting nearest to ours: the document index is fixed, only the query encoder is trained, and hard negatives are retrieved again at each step. Source: https://arxiv.org/pdf/2104.08051.
- The ReSa2 rule of the project (near-score window, then similarity to the correct page, then uniform) is of the same family as SimANS and NSCaching: a band near the positive, then a uniform draw.

## 6. Points that the sources do not settle

- No source bounds the regret of a sampler in terms of retrieval accuracy.
- Our rewards depend on the past choices of the sampler, because the model is trained on them. The adversary is not oblivious. Auer et al. do not prove this case. Borsos et al. (Theorem 8) and Thanos do treat it for their settings. The correct notion would be policy regret, which no opened source covers.
- A training step makes 32 x 7 picks before any reward arrives. No opened source gives the EXP3 or fixed-share bound for such batched feedback.
- The constant C in the Rexp3 bound was not found, so that bound gives a rate and no number.
- Venues were not stated in the fetched text for KBGAN, ANCE, Wu et al., SRNS, Borsos et al., Kreutzer et al., Baykal et al., and AdamCB.
- Active Bias (Chang et al. 2017, https://arxiv.org/abs/1704.07433): only the abstract was read. It weights samples with high prediction variance or near the decision threshold.

## What this means for our decision

The project has K bands as arms, with K near 5 to 10. A round can be one update (T = 1,500) or one mention in one update (T = 48,000).

### Fact that changes the design

The page encoder is frozen. The scores of all 1,000 pool pages are known at each step. So a reward that is a function of the current scores can be computed for every band, not only for the band that was drawn. That is full-information feedback. A reward that needs the result of the update (for example prediction gain) is seen only for the mix that was drawn. That is bandit feedback. The two cases have different rules and different bounds.

### Recommended reward

- Use learning progress, not hardness. Take the prediction gain of Graves et al.: the loss of the mention before the update minus the loss after it. Graves et al. found it the most consistent signal for maximum-likelihood training (https://arxiv.org/pdf/1704.03003, Sec. 5).
- Measure it on a fixed reference set of negatives for the mention (for example a uniform draw from the pool), not on the 7 negatives that were just trained. This follows the self prediction gain of Graves et al. and the dev-batch rewards of Kreutzer et al., who found that rewards on the trained batch are confounded by noisy examples (https://arxiv.org/pdf/2110.06997). The cost is small because the pool scores are cheap.
- Map the reward to [0, 1] with the percentile clip of Graves et al. (20th and 80th percentiles of past rewards, Sec. 2.3). The bounds of Auer et al. need rewards in [0, 1].
- Do not use gradient norm, raw loss, or negative score as the reward. Section 5 shows that each of them drives the sampler to the top band.

### Guard against drift to the hardest pages

- Keep a floor on every band: gamma / K from EXP3 itself (Auer et al.), or a fixed p_min as in Namkoong et al. Namkoong et al. state the cost of the floor in the bound, which makes it easy to analyse.
- Define the bands relative to the correct page, as SimANS and ReSa2 do, so that "the hardest band" is not "the pages that score above the correct page" (https://arxiv.org/pdf/2210.11773).
- Draw uniformly inside a band, as NSCaching and ANCE do (https://arxiv.org/pdf/1812.06410, https://arxiv.org/pdf/2007.00808).
- Option: add the SRNS variance signal as a diagnostic. A page with a high score and low score variance over time is a probable false negative (https://arxiv.org/pdf/2009.03376).

### Recommended rule

- Use EXP3.S (Auer et al., Sec. 8). It is the rule that Graves et al. used for the same problem shape: few arms, learning-progress reward, a model that changes. It has a bound against a comparator that switches arms.
- Do not use discounted Thompson sampling if a bound is required: Raj and Kalyani prove none.
- Sliding-window UCB and discounted UCB have bounds, but only for abrupt changes of stationary rewards (Garivier and Moulines). Our rewards drift at every step and are not i.i.d.
- Rexp3 has the best match of assumptions to slow drift (variation budget). Use it as the second candidate, or as the tool for the analysis: Thanos shows how to bound the variation budget from the learning rate and gradient bounds (https://arxiv.org/pdf/2103.01089, Lemma 2).
- If the final reward is a function of current scores only, use fixed share with full information instead (Cesa-Bianchi et al. 2012). The bound is much tighter.

### The bound that applies

- EXP3.S, Corollary 8.3 of Auer et al.: for every arm sequence with at most S - 1 switches, regret <= 2 sqrt(e - 1) sqrt(K T (S ln(KT) + e)), with rewards in [0, 1], alpha = 1 / T, and the stated gamma. It holds for any reward sequence that is fixed in advance.
- EXP3 against the best single band, Corollary 3.2: regret <= 2.63 sqrt(T K ln K).
- Our arithmetic for K = 8:

| Rounds T | EXP3 (best fixed band) | EXP3.S, S = 2 | EXP3.S, S = 4 | Fixed share, full information, m0 = 3 |
|---|---|---|---|---|
| 1,500 (one round per update) | 414 (27.6% of T) | 1,332 (88.8%) | 1,823 (above T, vacuous) | 145 (9.6%) |
| 48,000 (one round per mention) | 2,343 (4.9%) | 8,663 (18.0%) | 11,955 (24.9%) | 958 (2.0%) |

- Result: with one round per update, the bandit-feedback bounds say almost nothing. The design must make one round per mention (T = 48,000) or use full-information feedback. With K = 5 the EXP3 column improves to 19.2% and 3.4%.
- State the limits in the report: the adversary is not oblivious (Sec. 6), the feedback is batched, and the bound is on reward, not on recall.

### Suggested order of work

1. Fix K bands relative to the correct page and the reference set for the reward.
2. Implement EXP3.S with the Graves reward scale and a uniform floor. Log the band probabilities over time.
3. Compare with uniform, top-score, and ReSa2 at both learning rates. Graves et al. warn that uniform is a strong baseline.
4. For the proof, start from Auer et al. Theorem 8.1. Then try the variation-budget route of Besbes et al. with a drift bound in the style of Thanos Lemma 2.
