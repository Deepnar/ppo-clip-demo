# PPO-Clip Pendulum Demo : 

Small classroom demo of **PPO-Clip (Proximal Policy Optimization)** on Gymnasium's `Pendulum-v1`.

The repository contains:

- `ppo_clip_pendulum_demo.py` — trains and evaluates PPO-Clip
- `ppo_clip_watch.py` — loads the already-trained model and runs it visually
- `ppo_clip_requirements.txt` — Python dependencies
- `ppo_clip_output/ppo_pendulum.zip` — trained PPO model

## Quick setup for presentation

### 1. Clone the repository

```bash
git clone <REPO_URL>
cd ppo-clip-demo
```

### 2. Create a Python 3.11 environment

#### Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r ppo_clip_requirements.txt
```

### 4. Verify the setup

```bash
python -c "import gymnasium, stable_baselines3, torch; print('SETUP_OK')"
```

Expected output:

```text
SETUP_OK
```

## Run the trained model

The model is already trained. No retraining is needed for the presentation.

```bash
python ppo_clip_watch.py
```

By default it runs **3 episodes**.

To run 5 episodes:

```bash
python ppo_clip_watch.py --episodes 5
```

The script loads:

```text
ppo_clip_output/ppo_pendulum.zip
```

and opens a window showing the trained policy controlling the pendulum.

## Optional: retrain the model

This is NOT required for the presentation.

```bash
python ppo_clip_pendulum_demo.py --timesteps 200000
```

Main PPO settings used:

```text
learning rate = 0.0003
rollout steps = 2048
minibatch size = 64
epochs per rollout = 10
gamma = 0.99
GAE lambda = 0.95
clip epsilon = 0.2
```

## Existing training result

```text
Final PPO return: -300.74 +/- 209.10
Improvement over untrained PPO: +983.45
Improvement over random baseline: +922.36
```

For `Pendulum-v1`, rewards are non-positive and **0 is ideal**, so a less-negative return is better.

## If a GPU warning appears

Stable-Baselines3 may print a warning saying PPO with an MLP policy is usually better suited to CPU.

That warning is harmless for the demo. The model will still run.

## Presentation-day checklist

Before class, make sure:

```bash
source .venv/bin/activate
python ppo_clip_watch.py --episodes 3
```

works and opens the Pendulum window.
