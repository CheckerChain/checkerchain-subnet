<div align="center">

# CheckerChain SN87: Next-Gen AI-powered Trustless Crypto Review Platform <!-- omit in toc -->

Check our [Whitepaper](https://docs.checkerchain.com/whitepaper/checkerchain-whitepaper-v2.0) for in-depth study on tRCM protocol of CheckerChain.

[![CheckerChain](https://github.com/CheckerChain/CheckerChain-Asset/blob/4b0c738e6233ae019e772d24212a99800b4e1e84/CheckerChain-Twitter-Cover.png)](https://checkerchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

</div>

---

- [Overview](#overview)
- [Subnet Introduction](#subnet-introduction)
  - [Roadmap](#roadmap)
- [Miner](#miner)
  - [Requirements](#requirements)
  - [Running Miner](#running-miner)
- [Validator](#validator)
  - [Requirements](#requirements)
  - [Running Validator](#running-validator)
- [Auto-Updater Script](#auto-updater-script)
  - [Features](#features)
  - [Usage](#usage)
  - [Integration with PM2](#integration-with-pm2)
  - [Cron Job Setup](#cron-job-setup)
  - [Logging](#logging)
  - [Safety Features](#safety-features)
- [Game Theory on Subnet With tRCM](#game-theory-on-subnet-with-trcm)
- [Resources of CheckerChain](#resources-of-checkerchain)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

CheckerChain is a next-gen AI-powered crypto review platform that powers a trustless review consensus mechanism (tRCM). In tRCM protocol, anyone can participate but the protocol selects the reviewers arbitrarily to review a product. Selected reviewers can only get reward for their work when their review score falls in consensus range. Closer the consensus, more the reward.

Reviewers have a higher probability to make their review closer to consensus only when they are honest. Any dishonest review by any reviewer falls outside of consensus. This generates no or least reward making dishonest reviews highly expensive to perform. This will eventually discourage such attackers from participating in the tRCM protocol.

---

## Subnet Introduction

CheckerChain subnet operates as a decentralized AI-powered prediction layer, continuously refining review scores through machine learning. It is structured into two key components: validators and miners. Validators play a crucial role in distributing product review tasks to miners and aggregating the Ground Truth ratings collected from the main platform. They evaluate miner-generated predictions, benchmarking them against the Ground Truth to ensure accuracy. By maintaining a competitive environment, validators score miners to optimize their models for better precision and efficiency.

Miners, on the other hand, are responsible for running AI models that predict review scores for listed products. These models evolve over time by learning from past predictions and adjusting their algorithms based on discrepancies with the Ground Truth. Through Reinforcement Learning from Human Feedback (RLHF), miners incorporate additional insights from validators and human reviewers, ensuring their models align more closely with real-world assessments. This continuous feedback loop allows the subnet to improve autonomously, reducing biases and increasing reliability in AI-driven ratings.

The subnet follows a decentralized learning and incentive structure, where AI models start with predefined datasets and historical review scores. Over time, miners fine-tune their models by comparing predictions with Ground Truth data, optimizing accuracy through RLHF. Validators play a key role in integrating tRCM-based human feedback into the training process, refining AI predictions iteratively. As a result, miners that consistently produce high-accuracy predictions receive higher benchmarks and greater rewards, creating a self-reinforcing cycle of improvement.

By combining tRCM human-decentralized ratings with AI-driven predictions, CheckerChain’s subnet evolves into a self-learning, decentralized, and transparent review system. The open participation model allows anyone to join as a miner or validator, contributing to an AI-enhanced ecosystem that continuously adapts to real-world opinions. This fusion of human intelligence and AI automation ensures a fair, scalable, and corruption-resistant review platform, setting a new standard for decentralized trust in product evaluations.

### Roadmap

**Phase 1:**

- [x] Subnet launch
- [ ] Leaderboard with scoring methods

**Phase 2:**

- [ ] Integration of Subnet output with CheckerChain dApp
- [ ] Third-party widget release

**Phase 3:**

- [ ] Optimization of subnet logic

---

## Miner

Miner is responsible for LLM-based evaluation of products to generate and submit the prediction of Trust Score for each reviewed product.

### **Steps to Generate Predictions:**

1. **Loop through product IDs** to fetch product details.
2. Use any reasoning AI-models.

- OpenAI Models (including `gpt-4`)
- Anthropic Models (including `claude-3-5-sonnet-20241022`)
  **Construct a ChatGPT prompt** with relevant product parameters.

3. **Generate AI-based scores**, including:
   - **Overall Score (out of 100)**
   - **Breakdown Scores** for key evaluation criteria.
4. **Calculate the final Trust Score** using a weighted aggregation function.
5. **Return the Trust Score** to the Validator for processing.

✅ Customizable Weighting System – Allows different importance levels for various scoring parameters.

### Requirements

Before you proceed with the installation of the subnet, note the following: \
[Running a miner in Checkerchain](docs/running_miner_or_vali.md)

## Validator

Validator is responsible for fetching products from CheckerChain dApp, distributing review tasks to miners, tracking their review status, storing miner predictions, and distributing rewards based on accuracy.

### **Step 1: Fetch Products**

- Retrieves products from the **CheckerChain**.
- If a product has **status = "published"** and **is not in the database**, it is added to the **unmined list** and stored in the database.
- If a product has **status = "reviewed"** and **is already in the database**, it is added to the **reward list** for further processing.

### **Step 2: Store Miner Predictions**

- Fetches predictions from **all miners** for products in the **unmined list**.
- Stores predictions in the **prediction table**, including:
  - **Miner UID**
  - **Product ID**
  - **Predicted Score**

### **Step 3: Scoring Mechanism**

- Implements the scoring mechanism to rank miners and provide weights
- Miners are ranking based on cummulative average of the absolute error between the predicted trust score and the actual trust score at the evaluation time.
- Miner with lowest error ranks best and assigned the best weight.

### **Step 4: Reward Distribution**

- If the **reward list is empty**, initialize an array of zeros (length = number of miners) and update rewards accordingly.
- Otherwise:
  - Select **one product** from the **reward list**.
  - Retrieve **all miner predictions** for that product.
  - Compute **scores based on the difference between predicted scores and the actual trust score**.
  - Update miner rewards accordingly.
  - Remove the product from the database after processing.

✅ Dynamic Database

---

### Requirements

Before you proceed with the installation of the subnet, note the following: \
[Running a Validator in Checkerchain](docs/running_miner_or_vali.md)

---

## Auto-Updater Script

CheckerChain subnet includes an automated update script that helps keep your miner or validator deployment up-to-date with the latest changes from the repository. This is particularly useful for production deployments where manual updates can be time-consuming and error-prone.

### Features

- **Automatic Git Updates**: Pulls the latest changes from the specified branch
- **Dependency Management**: Automatically updates Python dependencies
- **Application Restart**: Optionally restarts your application after updates
- **Safety Checks**: Ensures you're on the correct branch before updating
- **Comprehensive Logging**: Logs all operations to both file and console

### Usage

The autoupdater script is located at `scripts/autoupdater.py` and can be run with various options:

#### Basic Usage

```bash
# Update from current directory using default settings
python scripts/autoupdater.py

# Update with custom repository path
python scripts/autoupdater.py --repo-path /path/to/checkerchain-subnet --venv path/to/venv

# Update from a specific branch
python scripts/autoupdater.py --branch develop
```

#### Advanced Options

```bash
# Full command with all options
python scripts/autoupdater.py \
    --repo-path /path/to/checkerchain-subnet \
    --branch main \
    --requirements requirements.txt \
    --venv /path/to/.venv \
    --restart-command "pm2 restart checkerchain-miner" \
    --force
```

#### Parameters

| Parameter           | Description                                | Default                      |
| ------------------- | ------------------------------------------ | ---------------------------- |
| `--repo-path`       | Path to the repository                     | Current directory            |
| `--branch`          | Branch to pull from                        | `main`                       |
| `--requirements`    | Path to requirements file                  | `requirements.txt`           |
| `--venv`            | Path to virtual environment                | `.venv` in current directory |
| `--restart-command` | Command to restart application             | None                         |
| `--force`           | Force update even if not on default branch | False                        |

### Integration with PM2

For production deployments using PM2, you can set up automatic updates:

```bash
# Example restart commands for different setups
--restart-command "pm2 restart checkerchain-miner"
--restart-command "pm2 restart checkerchain-validator"
--restart-command "systemctl restart checkerchain"
```

### Cron Job Setup

To run automatic updates on a schedule, add a cron job:

```bash
# Edit crontab
crontab -e

# Add entry to run updates every day at 2 AM
0 2 * * * cd /path/to/checkerchain-subnet && python scripts/autoupdater.py --venv /path/to/venv --restart-command "pm2 restart checkerchain-miner" >> /var/log/checkerchain-autoupdate.log 2>&1
```

### Logging

The autoupdater creates detailed logs in `autoupdate.log` in the repository directory. Monitor this file to track update operations:

```bash
# View recent update logs
tail -f autoupdate.log
```

### Safety Features

- **Branch Verification**: Only updates if you're on the specified branch (unless `--force` is used)
- **Git Repository Check**: Verifies the directory is a valid git repository
- **Dependency Validation**: Checks if requirements file exists before attempting updates
- **Error Handling**: Comprehensive error handling with detailed logging

---

## Game Theory on Subnet With tRCM

Subnet works on Reinforcement Learning Through Human Feedback (RLHF) on ground truths of tRCM. tRCM is an acronym for trustless Review Consensus Mechanism. It is the core protocol utilized on CheckerChain to make reviews trustless.

tRCM is based on 2 assumptions for a review to hold any authentic value:

- reviews are performed in zero-knowledge proofs without any control of either the poster or the reviewer.
- honest reviewers in the protocol always establish a majority

In tRCM protocol, anyone can participate but the protocol selects the reviewers arbitrarily to review a product. Selected reviewers can only get reward for their work when their review score falls in consensus range. Closer the consensus, more the reward. Reviewers have a higher probability to make their review closer to consensus only when they are honest. Any dishonest review by any reviewer falls outside of consensus. This generates no or least reward making dishonest reviews highly expensive to perform. This will eventually discourage such attackers from participating in the tRCM protocol.

These scores are vital parameters to derive incentives for each contribution.

- Trust Score: This is an atomic data of a product calculated from reviewer's task. It represents rating of a product in the range of 0 to 100.
- Normalized Trust Score: This is a derived data of a product calculated from Trust score to determine the impact on reward.
- Rating Score: This is a derived data of a product calculated as Trust score out of 5 and processed with Bayesian Average.

When a product gets listed on CheckerChain, tRCM protocol enacts on 30+ parameters of 10 categories to generate these vital scores. And subnet miners need to optimize their AI model to predict the best possible matching trust score.

## Resources of CheckerChain

|             |          Links           |      Notes      |
| :---------: | :----------------------: | :-------------: |
| Application | https://checkerchain.com | Mainnet is Live |
| Leaderboard |                          |                 |

# 🛠️ Troubleshooting: Not Receiving Mine Requests?

If your miner is not receiving mine requests while others are, it's likely an issue on your end. Here's a step-by-step checklist to help you diagnose and resolve it:

### ✅ 1. Check if Your IP and Port Are Publicly Accessible

Your miner must be reachable from the public internet. To verify:

- Open a browser and navigate to your miner’s IP and port:

  ```
  http://<your-ip>:<your-port>
  ```

  For example:

  ```
  http://65.109.76.55:8091
  ```

- If it's working correctly, you should see a response like this:

  ```json
  {
    "message": "Synapse name '' not found. Available synapses ['Synapse', 'CheckerChainSynapse']"
  }
  ```

> 🔒 **Tip:** If you see a "connection refused" or "timeout" error, your port is likely closed. Check your firewall rules and VPS provider’s network configuration.

---

### 🚀 2. Ensure Miner Is Running in Background

If your miner isn't running persistently, it won't handle any incoming requests. Use **PM2**, a production-grade process manager for Node.js and Python apps, to keep your miner alive and restart on failure or reboot.

#### 🔧 PM2 Installation & Usage

```bash
# Install PM2 globally
npm install -g pm2

# Navigate to your miner directory
cd /path/to/miner

# Start the miner with PM2
PYTHONPATH=. pm2 start neurons/miner.py   --interpreter /checkerchain/checkerchain-subnet/.venv/bin/python   --name miner   --   --netuid 87   --wallet.name miner-wallet  --wallet.hotkey default --axon.port 8091   --logging.debug

# Save the process list to start on boot
pm2 save
pm2 startup
```

> 🧠 You can view logs using:
>
> ```bash
> pm2 logs miner
> ```

> 📌 To stop or restart:
>
> ```bash
> pm2 stop miner
> pm2 restart miner
> ```

---

### 🧪 3. Final Checklist

- [ ] Is your `ip:port` reachable from a browser?
- [ ] Is your firewall or VPS provider allowing inbound traffic on that port?
- [ ] Is your miner process running in the background (PM2 recommended)?
- [ ] Are you using the correct netuid and wallet keys?

# 🏁 How can Miner Compete?

In the CheckerChain subnet, miners play a crucial role in evaluating and scoring products based on trust signals. Here's how miners can effectively compete and contribute:

#### ✅ Phase 1 Reward Distribution

- **90% of miners will be rewarded** based on the quality of their responses.
- Emphasis is on **accuracy**.

#### 🧠 Model Strategy

- Miners have two main paths:
  - **Train their own AI models** tailored to predict trust scores.
  - **Leverage robust existing models** (like GPT, Claude, etc.) for reliable outputs.

#### 📝 Prompt Engineering

- Miners can **design their own prompts** to extract high-quality, relevant insights.
- Prompt creativity can lead to predictions that closely match the **actual trust score**.
- Iterative refinement of prompts improves score alignment over time.

#### 🔐 Reliability and Uptime

- Ensure miner nodes are **always available and responsive**.
- Use tools like `pm2` or Docker for stability in production environments.

#### 🧪 Experiment and Iterate

- Regularly test new scoring strategies, datasets, or prompt variations.
- Use the testing environment to **benchmark changes** before deploying them live.

---

> 🚀 The goal is to be consistently close to the validator-approved trust scores. Those who optimize both **model accuracy** and **prompt quality** will thrive in the competition.

## Contributing

Create a PR to this repo. At least 1 reviewer approval required.

## License

This repository is licensed under the MIT License.

```text
# The MIT License (MIT)
# Copyright © 2025 CheckerChain

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
