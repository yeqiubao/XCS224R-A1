# CS 224R Module 1: Key Concepts and Terminology

## Table of Contents
1. [Introduction to Deep Reinforcement Learning](#introduction-to-deep-reinforcement-learning)
2. [Markov Decision Processes](#markov-decision-processes)
3. [Imitation Learning](#imitation-learning)
4. [Policy Gradients](#policy-gradients)
5. [Actor-Critic Methods](#actor-critic-methods)
6. [Q-Learning](#q-learning)
7. [Practical Algorithms](#practical-algorithms)

---

## Introduction to Deep Reinforcement Learning

### Definition
**Deep Reinforcement Learning** addresses:
- **Sequential decision-making problems**: A system needs to make multiple decisions based on a stream of information (observe, take action, observe, take action, ...)
- **Solutions** including:
  - Imitation learning
  - Offline & online RL
  - RL for LLMs
  - Model-free & model-based RL
  - Multi-task & meta RL
  - RL for robots
- **Emphasis**: Solutions that scale to deep neural networks

### Comparison with Supervised Learning

| Supervised Learning | Reinforcement Learning |
|---------------------|------------------------|
| Given labeled data: $\{(x_i, y_i)\}$, learn $f(x) \approx y$ | Learn behavior $\pi(a \mid s)$ |
| Directly told what to output | From experience, indirect feedback |
| Inputs $x$ are independently, identically distributed (i.i.d.) | Data not i.i.d.: actions affect future observations |

### Core Goal
Able to understand and implement existing and emerging methods.

---

## Markov Decision Processes

### Basic Definitions

**State** $s_t$: The state of the "world" at time $t$

**Observation** $o_t$: What the agent observes at time $t$ (only used when missing information)

**Action** $a_t$: The decision taken at time $t$

**Trajectory** $\tau$: Sequence of states/observations and actions
$$\tau = (s_1, a_1, s_2, a_2, \ldots, s_T, a_T)$$
(could be length $T=1$!)

**Reward function** $r(s, a)$: How good is state-action pair $(s, a)$?

**Initial state distribution** $p(s_1)$: Distribution over initial states

**Dynamics** $p(s_{t+1} \mid s_t, a_t)$: Unknown transition probability (Markov property: next state depends only on current state and action, independent of $s_{t-1}$)

### Markov Property
Next state is purely a function of the current state and action (and randomness):
$$p(s_{t+1} \mid s_t, a_t)$$
independent of $s_{t-1}$.

### MDP vs POMDP
- **MDP (Markov Decision Process)**: Fully observable (uses states $s$)
- **POMDP (Partially Observable MDP)**: Partially observable (uses observations $o$)

### Policy Representation

**Policy** $\pi_\theta(a \mid s)$: Behavior, selecting actions based on states or observations
- Can be represented using a neural network
- If only observations available: $\pi_\theta(a_t \mid o_{t-m}, \ldots, o_t)$ (with memory)

**Policy rollout/episode**: Result of running policy: trajectory $s_1, a_1, \ldots, s_T, a_T$

### Trajectory Distribution

The probability of a trajectory under policy $\pi_\theta$:
$$p(s_1, a_1, \ldots, s_T, a_T) = p(s_1) \prod_{t=1}^T \pi_\theta(a_t \mid s_t) p(s_{t+1} \mid s_t, a_t)$$

Denoted as $p_\theta(\tau)$.

### RL Objective

**Goal**: Learn policy $\pi_\theta$ that maximizes expected sum of rewards:

$$\max_\theta \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \sum_{t=1}^T r(s_t, a_t) \right]$$

**Why stochastic policies?**
1. **Exploration**: To learn from experience, must try different things
2. **Modeling stochastic behavior**: Existing data exhibits varying behaviors
3. Can leverage tools from generative modeling (generative model over actions given states/observations)

### Value Functions

**Value function** $V^\pi(s)$: Future expected reward starting at state $s$ and following policy $\pi$

**Q-function** $Q^\pi(s, a)$: Future expected reward starting at state $s$, taking action $a$, then following policy $\pi$

**Advantage function** $A^\pi(s, a)$: How much better it is to take action $a$ than to follow policy $\pi$ at state $s$
$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

**Useful relation**:
$$V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot \mid s)} Q^\pi(s, a)$$

### Types of RL Algorithms

1. **Imitation learning**: Mimic a policy that achieves high reward
2. **Policy gradients**: Directly differentiate the RL objective
3. **Actor-critic**: Estimate value of current policy and use it to make policy better
4. **Value-based**: Estimate value of optimal policy
5. **Model-based**: Learn to model dynamics, use it for planning or policy improvement

### Online vs Offline RL

**Offline**: Using only an existing dataset, no new data from learned policy

**Online**: Using new data from learned policy

**On-policy**: Update uses only data from current policy

**Off-policy**: Update can reuse data from other, past policies

---

## Imitation Learning

### Goal
Given trajectories collected by an expert ("demonstrations"):
$$\mathcal{D} := \{(s_1, a_1, \ldots, s_T)\}$$
(sampled from some unknown policy $\pi_{\text{expert}}$)

**Goal**: Learn a policy $\pi_\theta$ that performs at the level of the expert policy by mimicking it.

### Version 0: Deterministic Policy

1. For deterministic policy, supervised regression to expert's actions:
   $$\min_\theta \frac{1}{|\mathcal{D}|} \sum_{(s,a) \in \mathcal{D}} \|a - \hat{a}\|^2$$
   where $\hat{a} = \pi_\theta(s)$

2. Deploy learned policy

### Version 1: Expressive Policies

1. Train generative model of expert's actions:
   $$\min_\theta -\mathbb{E}_{(s,a) \sim \mathcal{D}}[\log \pi_\theta(a \mid s)]$$
   (maximize the log probability of demo actions under the policy)

2. Deploy learned policy with expressive distribution $\pi(\cdot \mid s)$

### Generative Models for Policies

**Approximating** $p(a \mid s)$:

1. **Mixture of Gaussians**: Output $\mu_1, \sigma_1, w_1, \mu_2, \sigma_2, w_2, \ldots$

2. **Discretize + Autoregressive**: Output $p(a_{t,1}), p(a_{t,2} \mid \hat{a}_{t,1}), p(a_{t,3} \mid \hat{a}_{t,1:2}), \ldots$

3. **Diffusion**: Iteratively denoise
   $$a_t^n = a_t^{n+1} + \sum_{i=1}^n \epsilon_i$$
   where $n = N \ldots 1$

**Important Note**: Neural network expressivity is often distinct from distribution expressivity!

### Compounding Errors

**Problem**: In supervised learning, inputs are independent of predicted labels. In behavior learning, predicted actions affect next state. Errors can lead to drift away from data distribution!

**Covariate shift**: $p_{\text{expert}}(s) \neq p_\pi(s)$
- States visited by expert ≠ states visited by learned policy

**Solutions**:
1. Collect a lot of demo data & hope for the best
2. Collect corrective behavior data

### Addressing Compounding Errors

#### Dataset Aggregation (DAgger)

1. Roll out learned policy $\pi_\theta$: $s'_1, \hat{a}_1, \ldots, s'_T$
2. Query expert action at visited states: $a^* \sim \pi_{\text{expert}}(\cdot \mid s')$
3. Aggregate corrections: $\mathcal{D} \leftarrow \mathcal{D} \cup \{(s', a^*)\}$
4. Update policy: $\min_\theta \mathcal{L}(\pi_\theta, \mathcal{D})$

**Pros**: Data-efficient way to learn from expert  
**Cons**: Can be challenging to query expert when agent has control

#### Human Gated DAgger

1. Start to roll out learned policy $\pi_\theta$: $s'_1, \hat{a}_1, \ldots, s'_t$
2. Expert intervenes at time when policy makes mistake
3. Expert provides (partial) demonstration: $s'_t, a^*_t, \ldots, s'_T$
4. Aggregate new demos: $\mathcal{D} \leftarrow \mathcal{D} \cup \{(s'_i, a^*_i)\}_{i \geq t}$
5. Update policy: $\min_\theta \mathcal{L}(\pi_\theta, \mathcal{D})$

**Pros**: More practical interface for providing corrections  
**Cons**: Can be hard to catch mistakes quickly in some domains

### Summary

**Key idea**: Train expressive policy class via generative modeling on dataset of demonstrations.

**Pros**:
- Algorithm is fully offline
- No need for data from policy (online data can be unsafe, expensive)
- No need to define reward function

**Cons**:
- May need a lot of data for reliable performance
- Cannot outperform demonstrator
- Doesn't allow improvement from practice

---

## Policy Gradients

### Objective

Maximize expected sum of rewards:
$$\max_\theta \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \sum_{t=1}^T r(s_t, a_t) \right]$$

### Gradient Derivation

Using the log-gradient trick:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \nabla_\theta \log p_\theta(\tau) \sum_{t=1}^T r(s_t, a_t) \right]$$

Expanding:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \left( \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right) \left( \sum_{t=1}^T r(s_t, a_t) \right) \right]$$

### REINFORCE Algorithm (Vanilla Policy Gradient)

**Full algorithm**:

1. Sample trajectories $\{\tau_i\}$ from $\pi_\theta(a \mid s)$
2. Compute gradient:
   $$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \left( \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t}) \right) \left( \sum_{t=1}^T r(s_{i,t}, a_{i,t}) \right)$$
3. Update: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

**Intuition**: Imitation gradient, but weighted by reward
- Increase likelihood of actions in high reward trajectories
- Decrease likelihood of actions in negative reward trajectories
- Formalization of "trial-and-error"

**Problem**: Policy gradient is noisy/high-variance

### Improving the Gradient

#### 1. Using Causality

Policy behavior at time $t$ does not affect rewards at time $t' < t$.

**Improved gradient**:
$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t}) \left( \sum_{t'=t}^T r(s_{i,t'}, a_{i,t'}) \right)$$

Use sum of **future rewards** (reward-to-go).

#### 2. Introducing Baselines

Subtract a constant baseline $b$ (e.g., average reward):
$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t}) \left( \sum_{t'=t}^T r(s_{i,t'}, a_{i,t'}) - b \right)$$

**Properties**:
- Subtracting constant baseline is unbiased (change to gradient is 0 in expectation)
- Can reduce variance of gradient
- Average reward is a good baseline

### Implementation

**Efficient computation**: Use single backward pass per trajectory:
$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t}) \hat{R}_{i,t}$$

where $\hat{R}_{i,t} = \sum_{t'=t}^T r(s_{i,t'}, a_{i,t'}) - b$

### Off-Policy Policy Gradient

#### Importance Sampling

Want to use samples from $p(\tau)$ (e.g., previous policy) to estimate expectation under $q(\tau)$ (current policy).

**Identity**:
$$\mathbb{E}_{x \sim p(x)}[f(x)] = \mathbb{E}_{x \sim q(x)} \left[ \frac{p(x)}{q(x)} f(x) \right]$$

**Note**: Important for $q$ to have non-zero support for high probability $p(x)$.

#### Off-Policy Gradient

Say we want to update policy $\pi_{\theta'}$ but use samples from $\pi_\theta$:

$$\nabla_{\theta'} J(\theta') \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \frac{\pi_{\theta'}(a_{i,t} \mid s_{i,t})}{\pi_\theta(a_{i,t} \mid s_{i,t})} \nabla_{\theta'} \log \pi_{\theta'}(a_{i,t} \mid s_{i,t}) \left( \sum_{t'=t}^T r(s_{i,t'}, a_{i,t'}) - b \right)$$

**Problem**: Importance weight $\frac{\pi_{\theta'}(a_{i,t} \mid s_{i,t})}{\pi_\theta(a_{i,t} \mid s_{i,t})}$ can become very small or very large for larger $T$.

**Common final form** (per-timestep, often approximated as 1):
$$\nabla_{\theta'} J(\theta') \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \frac{\pi_{\theta'}(a_{i,t} \mid s_{i,t})}{\pi_\theta(a_{i,t} \mid s_{i,t})} \nabla_{\theta'} \log \pi_{\theta'}(a_{i,t} \mid s_{i,t}) \hat{A}_{i,t}$$

#### KL Constraint

If policy changes a lot before sampling new data, gradient estimate becomes less accurate.

**Solution**: Constrain policy to not stray too far during gradient updates.

**Common choice**: KL divergence constraint
$$\mathbb{E}_{s \sim \pi_\theta} [D_{\text{KL}}(\pi_{\theta'}(\cdot \mid s) \| \pi_\theta(\cdot \mid s))] \leq \delta$$

### Summary

**Key intuition**: Do more high reward stuff, less low reward stuff.

**Characteristics**:
- Gradient still very noisy
- Best with large batch sizes and dense rewards
- Vanilla policy gradient is on-policy (need to recollect data every gradient step)

---

## Actor-Critic Methods

### Motivation

Policy gradients don't make efficient use of data, especially with sparse rewards.

**Idea**: Learn to estimate what is good & bad (value function), then use it for better policy gradient.

### Improving Policy Gradients

Instead of using sampled reward-to-go:
$$\hat{R}_{i,t} = \sum_{t'=t}^T r(s_{i,t'}, a_{i,t'})$$

Use better estimate of expected future rewards:
$$\hat{A}^\pi(s_{i,t}, a_{i,t}) = Q^\pi(s_{i,t}, a_{i,t}) - V^\pi(s_{i,t})$$

**Gradient**:
$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t}) \hat{A}^\pi(s_{i,t}, a_{i,t})$$

### Estimating Value Functions

#### Version 1: Monte Carlo Estimation

**Step 1**: Aggregate dataset of single sample estimates:
$$\mathcal{D} = \{(s_i, \hat{V}_i)\}$$
where $\hat{V}_i = \sum_{t'=t}^T r(s_{i,t'}, a_{i,t'})$ (from roll-out)

**Step 2**: Supervised learning to fit estimated value function:
$$\min_\phi \sum_{(s, \hat{V}) \in \mathcal{D}} \|V_\phi(s) - \hat{V}\|^2$$

#### Version 2: Bootstrapping

**Step 1**: Aggregate dataset of "bootstrapped" estimates:
$$\mathcal{D} = \{(s_i, y_i)\}$$
where $y_i = r(s_i, a_i) + \gamma V_\phi(s'_i)$ (update labels every gradient update!)

**Step 2**: Supervised learning to fit estimated value function:
$$\min_\phi \sum_{(s, y) \in \mathcal{D}} \|V_\phi(s) - y\|^2$$

Also referred to as **temporal difference (TD) learning**.

#### N-Step Returns

**Monte Carlo**:
$$y_{i,t} = \sum_{t'=t}^T r(s_{i,t'}, a_{i,t'})$$

**Bootstrapped**:
$$y_{i,t} = r(s_{i,t}, a_{i,t}) + \gamma V_\phi(s_{i,t+1})$$

**N-step** (hybrid):
$$y_{i,t} = \sum_{t'=t}^{t+n-1} r(s_{i,t'}, a_{i,t'}) + \gamma^n V_\phi(s_{i,t+n})$$

**Trade-off**:
- $n > 1, n < T$ often works best in practice
- Less variance than MC
- Lower bias than 1-step bootstrap

### Discount Factors

For infinite horizon problems, use discount factor $\gamma \in [0, 1)$:

**Value function**:
$$V^\pi(s) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \mid s_0 = s \right]$$

**Q-function**:
$$Q^\pi(s, a) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \mid s_0 = s, a_0 = a \right]$$

**Bellman equation**:
$$Q^\pi(s, a) = r(s, a) + \gamma \mathbb{E}_{s' \sim p(\cdot \mid s,a), a' \sim \pi(\cdot \mid s')} [Q^\pi(s', a')]$$

### Full Actor-Critic Algorithm

1. Sample batch of data $\{(s_{1,i}, a_{1,i}, \ldots, s_{T,i}, a_{T,i})\}$ from $\pi_\theta$
2. Fit $V_\phi^{\pi_\theta}$ to summed rewards in data:
   $$\min_\phi \sum_{i,t} \|V_\phi(s_{i,t}) - y_{i,t}\|^2$$
   where $y_{i,t} = \sum_{t'=t}^{t+n-1} r(s_{i,t'}, a_{i,t'}) + \gamma^n V_\phi(s_{i,t+n})$
3. Evaluate advantage:
   $$\hat{A}^{\pi_\theta}(s_{i,t}, a_{i,t}) = r(s_{i,t}, a_{i,t}) + \gamma V_\phi(s_{i,t+1}) - V_\phi(s_{i,t})$$
4. Evaluate gradient:
   $$\nabla_\theta J(\theta) \approx \sum_{i,t} \nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t}) \hat{A}^{\pi_\theta}(s_{i,t}, a_{i,t})$$
5. Update: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

### Summary

**Actor-critic**: Learn to estimate what is good and bad, then do more of the good stuff.

**Key improvement**: Get better policy gradient by using neural network to estimate value function.

**Policy evaluation methods**:
- **Monte Carlo**: Supervised learning directly on observed sum of future rewards
- **TD/Bootstrap**: Supervised learning on current reward + value estimate of next state
- **N-step**: Hybrid approach

---

## Q-Learning

### From Actor-Critic to Value-Based

**Policy iteration**:
1. **Policy evaluation**: Estimate $Q^\pi$ for current policy $\pi$
2. **Policy improvement**: Update policy to be greedy w.r.t. $Q^\pi$:
   $$\pi'(a \mid s) = \begin{cases} 1 & \text{if } a = \arg\max_a Q^\pi(s, a) \\ 0 & \text{otherwise} \end{cases}$$

### Q-Learning Algorithm

**Key insight**: Can improve policy in the Q-function update itself.

**Algorithm**:
1. Collect data $\{(s_i, a_i, r_i, s'_i)\}$ from some exploration policy
2. Fit Q-function:
   $$\min_\phi \sum_i \|Q_\phi(s_i, a_i) - y_i\|^2$$
   where $y_i = r_i + \gamma \max_{\hat{a}} Q_\phi(s'_i, \hat{a})$

**Terminology**:
- **Bellman equation**: $Q^\pi(s, a) = r(s, a) + \gamma \mathbb{E}_{s', a' \sim \pi} [Q^\pi(s', a')]$
- **Bellman optimality equation**: $Q^*(s, a) = r(s, a) + \gamma \max_{a'} Q^*(s', a')$

**Note**: Q-learning is **off-policy** (can use data from any policy).

### Data Collection

**Exploration strategies**:

1. **$\epsilon$-greedy**: With probability $\epsilon$, take uniformly random action
   - Often start with larger $\epsilon$, decrease over training

2. **Boltzmann exploration**: Take actions with probability proportional to their Q-value
   $$P(a \mid s) \propto \exp(Q(s, a) / \tau)$$

**Goal**: Want coverage for many actions $a$.

### Full Q-Learning Algorithm

1. Collect experience $\{(s_i, a_i, r_i, s'_i)\}$ using exploration policy (e.g., $\epsilon$-greedy)
2. Add to replay buffer $\mathcal{D}$
3. For $K$ iterations:
   - Sample minibatch from $\mathcal{D}$
   - Update Q-function:
     $$\min_\phi \sum_{(s,a,r,s') \in \text{batch}} \|Q_\phi(s, a) - (r + \gamma \max_{\hat{a}} Q_\phi(s', \hat{a}))\|^2$$

### Stabilizing Q-Learning

#### Target Networks

**Problem**: Target $y_i = r_i + \gamma \max_{\hat{a}} Q_\phi(s'_i, \hat{a})$ is a moving target, can lead to unstable optimization.

**Solution**: Freeze parameters used for target Q-values, update periodically.

**Hard target update**:
$$w' \leftarrow w \quad \text{when } \mod(n, N) = 0$$

**Soft target update (Polyak)**:
$$w' \leftarrow \tau \cdot w + (1 - \tau) \cdot w'$$
where $\tau$ is a small constant (e.g., 0.005).

**DQN (Deep Q-Network)**: Q-learning with target networks.

#### Overestimation Problem

**Issue**: When parameterized by function approximator, there is inherent noise in Q-value estimates.

$$Q_{\text{approx}}(s', \hat{a}) = Q_{\text{target}}(s', \hat{a}) + Y_{s',\hat{a}}$$

Even with zero mean noise, $\max$ operation can lead to positive bias:
$$\mathbb{E}[Y_{s',\hat{a}}] = 0 \quad \forall \hat{a} \quad \Longrightarrow \quad \mathbb{E}[Z_s] > 0 \text{ (often)}$$

where $Z_s = \gamma (\max_{\hat{a}} Q_{\text{approx}}(s', \hat{a}) - \max_{\hat{a}} Q_{\text{target}}(s', \hat{a}))$.

#### Double Q-Learning

**Solution**: Use two Q-networks to reduce overestimation.

**Double Q-learning**:
$$y_i = r_i + \gamma Q_{\phi_2}(s'_i, \arg\max_{\hat{a}} Q_{\phi_1}(s'_i, \hat{a}))$$

Or swap roles of $\phi_1$ and $\phi_2$.

**In practice**: Use current network for action selection, target network for value estimation.

#### N-Step Returns

**1-step**:
$$y_i = r_i + \gamma \max_{\hat{a}} Q_\phi(s'_i, \hat{a})$$

**N-step**:
$$y_i = \sum_{t=0}^{n-1} \gamma^t r_{i+t} + \gamma^n \max_{\hat{a}} Q_\phi(s'_{i+n}, \hat{a})$$

**Trade-offs**:
- **Pros**: Less biased target values when Q-values are inaccurate; typically faster learning
- **Cons**: Only actually correct when learning on-policy (not an issue when $N=1$)

**Common practice**: Ignore the problem & still use $N > 1$, or use importance sampling.

### Q-Learning vs Actor-Critic

**Q-Learning**:
- Value-based (no explicit policy)
- Off-policy
- Good for discrete actions or low-dimensional continuous actions

**Actor-Critic**:
- Policy-based with value function
- Can be on-policy or off-policy
- More general, works well for continuous actions

---

## Practical Algorithms

### Proximal Policy Optimization (PPO)

**Off-policy actor-critic with tricks**.

#### Surrogate Objective

**With importance weights**:
$$\tilde{J}(\theta') \approx \sum_{i,t} \frac{\pi_{\theta'}(a_{i,t} \mid s_{i,t})}{\pi_\theta(a_{i,t} \mid s_{i,t})} \hat{A}^{\pi_\theta}(s_{i,t}, a_{i,t})$$

#### Trick #1: Clipping

Clip importance weights to prevent large policy changes:
$$\tilde{J}(\theta') \approx \sum_{i,t} \min\left( \frac{\pi_{\theta'}(a_{i,t} \mid s_{i,t})}{\pi_\theta(a_{i,t} \mid s_{i,t})} \hat{A}^{\pi_\theta}(s_{i,t}, a_{i,t}), \text{clip}\left(\frac{\pi_{\theta'}(a_{i,t} \mid s_{i,t})}{\pi_\theta(a_{i,t} \mid s_{i,t})}, 1-\epsilon, 1+\epsilon\right) \hat{A}^{\pi_\theta}(s_{i,t}, a_{i,t}) \right)$$

#### Trick #2: Minimum

Take minimum w.r.t. original objective:
$$\tilde{J}(\theta') \approx \sum_{i,t} \min\left( \frac{\pi_{\theta'}(a_{i,t} \mid s_{i,t})}{\pi_\theta(a_{i,t} \mid s_{i,t})} \hat{A}, \text{clip}\left(\frac{\pi_{\theta'}(a_{i,t} \mid s_{i,t})}{\pi_\theta(a_{i,t} \mid s_{i,t})}, 1-\epsilon, 1+\epsilon\right) \hat{A} \right)$$

#### Trick #3: Generalized Advantage Estimation (GAE)

Fit $V^\pi$ with Monte Carlo or bootstrapping, then use varying horizon:
$$\hat{A}^{\pi_\theta}(s_{i,t}, a_{i,t}) = \sum_{n=1}^N w_n \hat{A}_n^{\pi_\theta}(s_{i,t}, a_{i,t})$$

where $\hat{A}_n$ uses $n$-step returns.

**Typical hyperparameters**:
- Clipping range $\epsilon = 0.2$
- ~10 epochs when updating policy (~300 gradient steps with batch size 64)
- ~2000 timesteps in batch of data
- ~500 iterations → 1M total timesteps

### Soft Actor-Critic (SAC)

**Fully off-policy actor-critic with replay buffer**.

#### Key Ideas

1. **Maintain replay buffer**: Store all past transitions $(s, a, r, s')$
2. **Fit Q-function**: Use Q-learning style update
   $$y_i = r_i + \gamma Q_\phi(s'_i, \bar{a}')$$
   where $\bar{a}' \sim \pi_\theta(\cdot \mid s'_i)$ is sampled from current policy

3. **Policy update**: Maximize Q-values (with entropy regularization):
   $$\max_\theta \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\theta(\cdot \mid s)} [Q_\phi(s, a) + \alpha \log \pi_\theta(a \mid s)]$$

**Pros**:
- Far more data efficient
- Can reuse all past trial-and-error data

**Cons**:
- Harder to tune hyperparameters
- Less stable than PPO

### Algorithm Comparison

| Algorithm | When to Use |
|-----------|-------------|
| **PPO** | When you care about stability, ease-of-use; when you don't care about data efficiency |
| **DQN** | When you have discrete actions or low-dimensional continuous actions |
| **SAC** | When you care most about data efficiency; when you are okay with tuning hyperparameters, less stability |

### Implementation Details

#### Replay Buffers

- Memory structures that store past experiences
- Break temporal correlations between consecutive training samples
- Prevent recency bias
- Improve sample efficiency

#### Gradient Clipping

Clip gradients to prevent exploding gradients:
$$\text{clip}(\nabla_\theta, -\text{max}, \text{max})$$

#### Huber Loss

Use Huber loss instead of MSE for Q-learning:
$$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta|y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

#### Critic Ensembling

- Use ensemble of Q-networks to measure epistemic uncertainty
- Take minimum over ensemble to reduce overestimation
- Effective in offline RL where distribution shift is a concern

---

## Key Takeaways

1. **RL Objective**: Maximize expected sum of rewards
   $$\max_\theta \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \sum_{t=1}^T r(s_t, a_t) \right]$$

2. **Value Functions**: Estimate expected future rewards
   - $V^\pi(s)$: Value of state under policy $\pi$
   - $Q^\pi(s, a)$: Value of state-action under policy $\pi$
   - $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$: Advantage

3. **Policy Gradients**: Directly optimize policy
   - On-policy: REINFORCE, actor-critic
   - Off-policy: Importance sampling, KL constraints

4. **Value-Based**: Learn Q-function, derive policy
   - Q-learning: Off-policy, good for discrete actions
   - Requires exploration strategies ($\epsilon$-greedy, Boltzmann)

5. **Practical Considerations**:
   - Target networks for stability
   - Replay buffers for data efficiency
   - Clipping/constraints to prevent large policy changes
   - N-step returns for bias-variance trade-off

---

## Mathematical Notation Reference

- $s_t$: State at time $t$
- $o_t$: Observation at time $t$
- $a_t$: Action at time $t$
- $\tau$: Trajectory $(s_1, a_1, \ldots, s_T, a_T)$
- $r(s, a)$: Reward function
- $p(s_{t+1} \mid s_t, a_t)$: Transition dynamics
- $\pi_\theta(a \mid s)$: Policy parameterized by $\theta$
- $p_\theta(\tau)$: Trajectory distribution under policy $\pi_\theta$
- $V^\pi(s)$: Value function
- $Q^\pi(s, a)$: Q-function
- $A^\pi(s, a)$: Advantage function
- $\gamma$: Discount factor
- $\alpha$: Learning rate
- $\epsilon$: Exploration rate (for $\epsilon$-greedy) or clipping parameter (for PPO)
- $\mathcal{D}$: Dataset
- $\mathbb{E}$: Expectation
- $\nabla_\theta$: Gradient with respect to $\theta$
