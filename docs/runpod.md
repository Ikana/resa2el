# RunPod setup and budget

Budget: $150 for all GPU work (pilot + 12 main runs). Checked against
docs.runpod.io on 2026-10-31 (billing, credentials, manage pods, network volumes).

## How RunPod bills
- Prepaid credits. Charges are per second, and billing runs every 5 minutes.
- When the balance reaches $0, RunPod stops all pods. A pod with a network
  volume keeps its data. A pod without one is deleted and the data is lost.
- A network volume costs money also when no pod runs ($0.07/GB/month).
  If the balance stays at $0, RunPod can delete the volume.
- Credits are not refundable.
- RunPod has no total budget setting. The only spend limit is a default
  of $80 per hour for the account. So the balance itself is our budget cap.

## Setup (Rodrigo does these steps; Claude cannot make accounts or pay)
1. Make an account at runpod.io with your personal email.
2. Billing (console.runpod.io/user/billing):
   - Add credits: **$50 first** (pilot + data download + first month of disk).
     Add more after the pilot, when we know the real hours. Total must stay <= $150.
   - Do **not** turn on auto-pay. If a card is saved, check that auto-pay is off.
   - Notifications: turn on **Low balance alert**, threshold **$15**.
3. Storage: make one network volume, 150 GB, in a data center that has A40 GPUs.
   Cost: about $10.50 per month. Data and checkpoints go in `/workspace`.
4. API key (only if we use scripts): console.runpod.io/user/credentials,
   API Keys, Create API Key. Give it the smallest permission that works.
   Do not put the key in git or in chat.

## Rules for each GPU session
- Start the pod with the network volume attached. Never run without it.
- At the start, set a stop timer in the pod:
  `sleep 4h; runpodctl pod stop $RUNPOD_POD_ID &` (change 4h to the plan).
- Stop the pod when the run ends. Stop, do not terminate: stop keeps `/workspace`.
- Write the GPU hours and the cost of each session in `docs/compute-log.md`.

## Expected cost (A40 48 GB at $0.59/h, price on 2026-10-31)
| Item | Hours | Cost |
|---|---|---|
| Data download + smoke test | ~2 | ~$1 |
| Pilot | 8 | ~$5 |
| 12 main runs | 96 | ~$57 |
| Network volume, 150 GB, ~2 months | – | ~$21 |
| **Total** | | **~$84** (reserve ~$66) |
