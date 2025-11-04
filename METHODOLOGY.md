# Nuel-Sim: Methodology & Mathematical Framework

## Table of Contents
1. [Overview](#overview)
2. [Game-Theoretic Formulation](#game-theoretic-formulation)
3. [Information Models](#information-models)
4. [Gameplay Variants](#gameplay-variants)
5. [Learning Framework](#learning-framework)
6. [Experimental Design](#experimental-design)
7. [Analysis Methods](#analysis-methods)

---

## Overview

**Nuel-Sim** is a computational framework for investigating strategic decision-making in N-player standoff games (duels, truels, and beyond). We combine classical game theory with multi-agent reinforcement learning (MARL) to explore:

- **Optimal targeting** under information asymmetry
- **Abstention mechanisms** in competitive scenarios
- **Learning dynamics** in multi-agent systems
- **The role of accuracy heterogeneity** in survival games

### System Architecture Philosophy

```
┌─────────────────────────────────────────────────┐
│           Configuration Layer                    │
│  (Declarative experiment specification)          │
└────────────────┬────────────────────────────────┘
                 │
    ┌────────────┴─────────────┐
    │                          │
    ▼                          ▼
┌─────────────┐        ┌──────────────┐
│ Game Engine │◄──────►│  RL Training │
│  (Physics)  │        │   (Learning) │
└──────┬──────┘        └──────┬───────┘
       │                      │
       └──────┬───────────────┘
              ▼
      ┌───────────────┐
      │   Analytics   │
      │  & Metrics    │
      └───────────────┘
```

**Design Principles**:
- **Modularity**: Observation models, gameplay modes, and strategies are interchangeable
- **Declarative Configuration**: YAML-driven experiment specification
- **Reproducibility**: Seeded randomness and versioned configurations
- **Extensibility**: Abstract base classes for custom components

---

## Game-Theoretic Formulation

### Mathematical Model

An N-player standoff is a dynamic, stochastic game defined by:

**State Space** $\mathcal{S}$:
$$
s_t = (\mathbf{a}, \mathbf{v}, h_t)
$$

where:
- $\mathbf{a} = (a_1, ..., a_N)$ : accuracy vector, $a_i \in [0,1]$ represents player $i$'s hit probability
- $\mathbf{v} = (v_1, ..., v_N)$ : vitality vector, $v_i \in \{0,1\}$ indicates if player $i$ is alive
- $h_t$ : history of shots up to time $t$

**Action Space** $\mathcal{A}_i(s)$ for player $i$:
$$
\mathcal{A}_i(s) = \{j \in \{1,...,N\} : v_j = 1, j \neq i\} \cup \{\text{abstain}\}
$$

**Transition Dynamics**:

For sequential play where player $i$ targets player $j$:

$$
P(s_{t+1} | s_t, a_i = j) = \begin{cases}
a_i & \text{if } v_j^{t+1} = 0 \text{ (hit)} \\
1 - a_i & \text{if } v_j^{t+1} = 1 \text{ (miss)}
\end{cases}
$$

For simultaneous play with action profile $\mathbf{a}_t = (a_1^t, ..., a_N^t)$:

$$
P(s_{t+1} | s_t, \mathbf{a}_t) = \prod_{i: a_i^t \neq \emptyset} P(\text{outcome}_i | a_i^t)
$$

**Utility Function**:

The terminal utility for player $i$ is:

$$
u_i(s_T) = \begin{cases}
\frac{1}{|\{j : v_j^T = 1\}|} & \text{if } v_i^T = 1 \text{ (alive at termination)} \\
0 & \text{otherwise}
\end{cases}
$$

This models a winner-takes-all scenario with shared victory among survivors.

### Game Flow Diagram

```
┌──────────────┐
│  Initialize  │
│   Players    │
│   (Sample    │
│  Accuracies) │
└──────┬───────┘
       │
       ▼
┌──────────────┐      No      ┌─────────────┐
│   Game Over? ├─────────────►│ Select Next │
│   (≤1 alive) │              │   Shooter   │
└──────┬───────┘              └──────┬──────┘
       │ Yes                         │
       │                             ▼
       │                    ┌─────────────────┐
       │                    │ Create          │
       │                    │ Observation     │
       │                    │ (Information    │
       │                    │  Encoding)      │
       │                    └────────┬────────┘
       │                             │
       │                             ▼
       │                    ┌─────────────────┐
       │                    │ Choose Target   │
       │                    │  (Policy π)     │
       │                    └────────┬────────┘
       │                             │
       │                             ▼
       │                    ┌─────────────────┐
       │                    │ Execute Shot    │
       │                    │  (Stochastic)   │
       │                    └────────┬────────┘
       │                             │
       │                             ▼
       │                    ┌─────────────────┐
       │                    │ Update State    │
       │                    │ & Beliefs       │
       │                    └────────┬────────┘
       │                             │
       │                             └────────┐
       │                                      │
       ▼                                      ▼
┌──────────────┐                    (Loop back to
│  Evaluate    │                     Game Over check)
│  Outcome &   │
│   Rewards    │
└──────────────┘
```

---

## Information Models

Information structure fundamentally shapes strategic behavior. We model various degrees of knowledge about opponent capabilities.

### Information Hierarchy

```
┌─────────────────────────────────────┐
│  Perfect Information                │
│  • Know all accuracies exactly      │
│  • Simple observation model         │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Partial Information (Bayesian)     │
│  • Observe shot outcomes only       │
│  • Maintain probability beliefs     │
│  • Update via Bayes' rule           │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Information + Uncertainty          │
│  • Track belief means               │
│  • Track belief variance            │
│  • Reason about confidence          │
└─────────────────────────────────────┘
```

### Observation Model 1: Perfect Information

**Observation Vector**:
$$
o_i = [\underbrace{a_i}_{\text{self}}, \underbrace{a_1, ..., a_{i-1}, a_{i+1}, ..., a_N}_{\text{opponents}}]
$$

**Properties**:
- Fully observable MDP
- Deterministic observation function
- No learning about opponents needed
- Baseline for comparison

### Observation Model 2: Bayesian Belief Tracking

**Prior**: Uniform Beta distribution over each opponent's accuracy
$$
\theta_j \sim \text{Beta}(\alpha_j^{(0)}, \beta_j^{(0)}) = \text{Beta}(1, 1) = \text{Uniform}(0, 1)
$$

**Likelihood**: Bernoulli shots from player $j$
$$
P(\text{hit} | \theta_j) = \theta_j
$$

**Posterior Update**: After observing $k$ hits in $n$ shots from player $j$
$$
\theta_j | \text{data} \sim \text{Beta}(\alpha_j^{(0)} + k, \beta_j^{(0)} + (n-k))
$$

**Point Estimate**: Posterior mean
$$
\hat{\theta}_j = \mathbb{E}[\theta_j | \text{data}] = \frac{\alpha_j + k}{\alpha_j + \beta_j + n}
$$

**Observation Vector**:
$$
o_i = [a_i, \hat{\theta}_i, \hat{\theta}_1, ..., \hat{\theta}_{i-1}, \hat{\theta}_{i+1}, ..., \hat{\theta}_N]
$$

**Diagram: Belief Evolution**

```
Initial (Uniform Prior)          After 5 Shots (3 hits)         After 20 Shots (14 hits)
   
   │                                │                               │         
 1 ├────────────────────           │      ╱╲                      │           ╱╲
   │                               │     ╱  ╲                     │          ╱  ╲
 P │                              │    ╱    ╲                    │         ╱    ╲
 D │                             │   ╱      ╲                   │        ╱      ╲
 F │                            │  ╱        ╲                  │       ╱        ╲
   │                           │ ╱          ╲                 │      ╱          ╲
 0 └─────────────────         └──────────────────           └─────────────────────
   0    0.5    1.0 θ           0    0.5    1.0 θ              0    0.5    1.0 θ
   
   Mean: 0.50                     Mean: 0.57                     Mean: 0.65
   Variance: 0.083                Variance: 0.035                Variance: 0.010
   (High uncertainty)             (Moderate uncertainty)         (Low uncertainty)
```

### Observation Model 3: Bayesian with Uncertainty Quantification

**Extended Observation Vector**:
$$
o_i = [a_i, \hat{\theta}_i, \underbrace{\hat{\theta}_1, \sigma^2(\theta_1)}_{\text{opponent 1}}, ..., \underbrace{\hat{\theta}_N, \sigma^2(\theta_N)}_{\text{opponent N}}]
$$

where the **variance** of the Beta distribution is:
$$
\sigma^2(\theta_j) = \frac{\alpha_j \beta_j}{(\alpha_j + \beta_j)^2 (\alpha_j + \beta_j + 1)}
$$

**Strategic Implications**:
- High variance → uncertain about opponent's true accuracy
- Agents can differentiate between "weak because observed weak" vs. "appears weak due to few observations"
- Enables meta-reasoning about information value

### Abstention and Information Leakage

**Abstention Action**: Shoot at "Ghost" player (dummy target)
- No effect on game state
- **Does not reveal shooter's accuracy**
- Strategic in early game when uncertainty is high

**Information-Theoretic View**:

Let $I(X; Y)$ denote mutual information between random variables $X$ and $Y$.

- Shooting at opponent: $I(\theta_i; \text{outcome}) > 0$ (information leakage)
- Abstaining: $I(\theta_i; \text{abstain}) = 0$ (no information revealed)

### Observation Space Comparison

| Model | Dimension | Information | Complexity | Strategic Depth |
|-------|-----------|-------------|------------|-----------------|
| Perfect | $N$ | Complete | Low | Moderate |
| Bayesian Mean | $N+1$ | Partial | Medium | High |
| Bayesian + Variance | $2N$ | Partial + Uncertainty | High | Very High |

**Trade-off**: Richer observations → more complex learning, but potentially better strategies

---

## Gameplay Variants

The timing and structure of actions fundamentally alter strategic dynamics.

### Sequential Play

**Definition**: Players take turns in fixed order $\{P_1, P_2, ..., P_N\}$

```
Turn Order Visualization:

Round 1:  P₁ → P₂ → P₃ → P₁ → ...
          │    │    │    │
          ▼    ▼    ▼    ▼
         [Choose & Execute Shot]
          │
          ▼
         [Update Beliefs]
          │
          ▼
         [Next Player's Turn]
```

**Properties**:
- **First-mover advantage**: Can eliminate threats before being targeted
- **Perfect observation**: See all previous actions before choosing
- **Deterministic timing**: No uncertainty about turn order
- **Sequential rationality**: Can backwards-induct optimal play

**Strategic Implications**:
- Later players have more information
- Early players must anticipate future actions
- Encourages "strike first" mentality
- Weaker players may never get a turn if eliminated early

### Simultaneous Play

**Definition**: All players choose actions simultaneously, then resolve

```
Simultaneous Resolution:

Time t:   All players observe state s_t
          │
          ▼
          All players choose actions {a₁, a₂, a₃, ...}
          │
          ▼
          Execute ALL shots simultaneously
          │
          ▼
          Resolve outcomes (possible mutual elimination)
          │
          ▼
Time t+1: New state s_{t+1}
```

**Properties**:
- **No turn order bias**: All players act on equal footing
- **Strategic interdependence**: Must reason about others' likely actions
- **Mutual elimination possible**: Multiple players can be eliminated simultaneously
- **Nash equilibrium focus**: Optimal play requires equilibrium concepts

**Key Difference from Sequential**:

| Aspect | Sequential | Simultaneous |
|--------|-----------|--------------|
| Information | Perfect (past) | Imperfect (concurrent) |
| Response time | Immediate | None (simultaneous) |
| Bias | First-mover | None |
| Complexity | Lower | Higher |
| Equilibrium type | Subgame perfect | Nash |

### Random Play

**Definition**: One player selected uniformly at random each turn

```
Random Selection Process:

Each turn:
   alive_players = {P_i : v_i = 1}
   shooter = UniformRandom(alive_players)
   shooter executes action
```

**Properties**:
- **Stochastic timing**: No player knows when they'll act next
- **Expected turn frequency**: $\frac{1}{|\text{alive players}|}$
- **Fairness**: Each alive player has equal probability of acting
- **Unbiased**: No structural advantage

**Strategic Considerations**:
- Cannot plan around specific turn order
- Must maintain robust strategies
- Luck factor in survival
- Tests generalization of learned policies

### Gameplay Comparison Diagram

```
┌─────────────────────────────────────────────────────────┐
│                 SEQUENTIAL GAMEPLAY                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  P₁  →  P₂  →  P₃  →  [P₁ again if alive]             │
│  │      │      │                                        │
│  ▼      ▼      ▼                                        │
│ [Full knowledge of all previous shots]                  │
│                                                          │
│ Advantage: First player                                 │
│ Complexity: Low (perfect info at decision time)         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                SIMULTANEOUS GAMEPLAY                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│       P₁         P₂         P₃                          │
│        ↓          ↓          ↓                          │
│    [Choose]  [Choose]  [Choose]                         │
│        │          │          │                          │
│        └──────────┴──────────┘                          │
│                   ▼                                      │
│            [Execute ALL]                                │
│                                                          │
│ Advantage: None (symmetric)                             │
│ Complexity: High (must predict others)                  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  RANDOM GAMEPLAY                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│            [Random Selection]                            │
│           ╱        │        ╲                           │
│          P₁       P₂       P₃                           │
│         33%       33%       33%                          │
│                                                          │
│ Selected player executes turn                            │
│                                                          │
│ Advantage: Stochastic (no persistent bias)              │
│ Complexity: Medium (uncertain timing)                   │
└─────────────────────────────────────────────────────────┘
```

---

## Learning Framework

We employ multi-agent reinforcement learning (MARL) to discover emergent strategies. The framework uses **centralized training with decentralized execution** (CTDE).

### Markov Game Formulation

A multi-agent standoff is a **stochastic game** (Shapley, 1953):

$$
\mathcal{G} = \langle \mathcal{N}, \mathcal{S}, \{\mathcal{A}_i\}_{i \in \mathcal{N}}, \mathcal{P}, \{r_i\}_{i \in \mathcal{N}}, \gamma \rangle
$$

where:
- $\mathcal{N} = \{1, ..., N\}$ : set of agents
- $\mathcal{S}$ : state space
- $\mathcal{A}_i$ : action space for agent $i$
- $\mathcal{P}: \mathcal{S} \times \mathcal{A}_1 \times ... \times \mathcal{A}_N \rightarrow \Delta(\mathcal{S})$ : transition probability
- $r_i: \mathcal{S} \times \mathcal{A}_1 \times ... \times \mathcal{A}_N \rightarrow \mathbb{R}$ : reward function for agent $i$
- $\gamma \in [0, 1)$ : discount factor

### Policy Representation

Each agent learns a **stochastic policy**:

$$
\pi_\theta: \mathcal{O}_i \rightarrow \Delta(\mathcal{A}_i)
$$

where $\mathcal{O}_i$ is agent $i$'s observation space and $\theta$ are neural network parameters.

**Policy Network Architecture**:

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT LAYER                           │
│              Observation Vector o_i                      │
│         [a_i, θ̂₁, σ²(θ₁), θ̂₂, σ²(θ₂), ...]           │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                 HIDDEN LAYER 1                           │
│              128 neurons + ReLU                          │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                 HIDDEN LAYER 2                           │
│              128 neurons + ReLU                          │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   OUTPUT LAYER                           │
│            Action logits [logit₁, ..., logit_N]         │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                    SOFTMAX                               │
│         π_θ(a|o) = exp(logit_a) / Σ exp(logit_j)       │
└─────────────────────────────────────────────────────────┘
```

### Reward Structure

**Sparse Reward Design** (outcome-focused):

$$
r_i^{(t)} = \begin{cases}
-100 & \text{if agent } i \text{ just eliminated at } t \\
0 & \text{otherwise}
\end{cases}
$$

**Terminal Bonus** (winner-takes-all):
$$
r_i^{(T)} = r_i^{(T)} + 100 \cdot \mathbb{1}[\text{agent } i \text{ alive at termination}]
$$

**Rationale**: 
- Rewards only **survival to the end** (winning) and penalizes **elimination**
- Avoids biasing the learning process toward specific intermediate behaviors
- Lets agents discover optimal strategies without hand-crafted incentives

**Alternative Designs** (not used):

1. **Per-step survival reward**: $+1$ for each turn alive
   - *Issue*: May encourage excessive **abstention** (avoid risk, maximize turns)
   - Could lead to overly passive play

2. **Elimination reward**: Bonus for eliminating opponents
   - *Issue*: **Biased toward stronger players** who are more likely to land hits
   - Weaker players get less training signal despite potentially optimal abstention strategies
   - Conflates accuracy (fixed) with strategic quality

**Expected Return**:
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_i^{(t)} \right]
$$

### PPO Algorithm (Policy Gradient)

**Proximal Policy Optimization** (Schulman et al., 2017) is our primary algorithm.

**Objective Function** (clipped surrogate):

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( \rho_t(\theta) \hat{A}_t, \, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

where:
- $\rho_t(\theta) = \frac{\pi_\theta(a_t | o_t)}{\pi_{\theta_{\text{old}}}(a_t | o_t)}$ : probability ratio
- $\hat{A}_t$ : advantage estimate
- $\epsilon = 0.2$ : clip parameter (prevents large policy updates)

**Advantage Estimation** (GAE):

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error.

**Pseudocode**:

```
Algorithm: PPO for Multi-Agent Standoff

Initialize:
  policy network π_θ (shared across all agents)
  value network V_φ
  replay buffer D = ∅
  
for iteration = 1 to MAX_ITERATIONS:
  // Collect experience
  for episode = 1 to EPISODES_PER_ITER:
    s ← reset_environment()
    for t = 1 to T:
      for each alive agent i:
        o_i ← create_observation(i, s)
        a_i ~ π_θ(·|o_i)
      
      s', r, done ← env.step({a_i})
      store (o_i, a_i, r_i, o_i', done) in D
      
      if done: break
  
  // Policy update
  for epoch = 1 to K_EPOCHS:
    sample minibatch B from D
    compute advantages {Â_t} using GAE
    
    θ ← θ + α ∇_θ L^CLIP(θ)
    φ ← φ + α ∇_φ L^VF(φ)
  
  clear buffer D
```

**Hyperparameters**:
- Learning rate: $\alpha = 10^{-4}$
- Discount factor: $\gamma = 0.98$
- GAE lambda: $\lambda = 0.95$
- Clip parameter: $\epsilon = 0.2$
- Epochs per iteration: $K = 10$
- Batch size: $256$

### Parameter Sharing

**Shared Policy Approach**: All agents use the same policy network $\pi_\theta$

**Motivation**:
1. **Symmetry**: All players face similar strategic problems
2. **Data efficiency**: $N \times$ more training data per iteration
3. **Generalization**: Policy must work for any accuracy level
4. **Simplicity**: Single model to train and evaluate

**Trade-off**: Harder to learn role-specific strategies (e.g., "weak player strategy")

### Training Dynamics Diagram

```
Training Progress Visualization:

Episode Reward                          Policy Entropy
     │                                       │
 100 │         ╱────────────              3.0│╲
     │        ╱                                │ ╲
  50 │       ╱                              2.0│  ╲___
     │    __╱                                  │      ╲___
   0 │___╱                                 1.0│          ────
     └──────────────────► Iteration           └──────────────────► Iteration
     
     Early: Random, high variance             Early: High (exploration)
     Late: Converged, stable                  Late: Low (exploitation)

Abstention Rate                         Win Rate by Accuracy
     │                                       │
 40% │    ╱╲                             60% │          ╱─── (High)
     │   ╱  ╲                                │         ╱
 20% │  ╱    ╲___                        40% │    ____╱
     │ ╱         ╲__                         │___╱         (Medium)
  0% │╱             ───                  20% │ (Low)
     └──────────────────► Iteration           └─────────────────────
     
     Learns when to abstain                   Accuracy paradox emerges
```

### DQN Alternative (Value-Based)

**Deep Q-Network** for discrete action spaces:

**Q-function approximation**:
$$
Q_\theta(o_i, a_i) \approx Q^*(o_i, a_i)
$$

**Bellman optimality**:
$$
Q^*(o, a) = \mathbb{E}_{o'} [r + \gamma \max_{a'} Q^*(o', a')]
$$

**Loss function**:
$$
L(\theta) = \mathbb{E}_{(o,a,r,o') \sim D} \left[ (r + \gamma \max_{a'} Q_{\theta^-}(o', a') - Q_\theta(o, a))^2 \right]
$$

where $\theta^-$ are **target network parameters** (updated slowly for stability).

**Key Differences from PPO**:
- **Off-policy**: Can reuse old experience (replay buffer)
- **Sample efficient**: But less stable in multi-agent settings
- **Deterministic**: Greedy policy $\arg\max_a Q(o, a)$

---

## Experimental Design

### Configuration-Driven Experiments

All experiments are specified declaratively via YAML configuration files.

**Configuration Structure**:

```yaml
# Example: config/bayesian-abstention.yaml

game:
  num_players: 3
  num_rounds: 20
  
gameplay:
  type: "SimultaneousGamePlay"  # Sequential | Simultaneous | Random
  has_ghost: true               # Enable abstention
  
observation:
  model_type: "BayesianAbstentionObservation"
  params:
    num_players: 3
    has_ghost: true
    setup_shots: 0              # Initial belief calibration
    
players:
  marksmanship_range: [0.1, 0.9]  # Accuracy sampling range
  
rllib:
  algorithm: "ppo"
  episodes: 2000
  checkpoint_path: "./checkpoints/"
```

### Experimental Factors

| Factor | Levels | Purpose |
|--------|--------|---------|
| **Observation Model** | Perfect, Bayesian-Mean, Bayesian-Variance | Information asymmetry effects |
| **Gameplay Mode** | Sequential, Simultaneous, Random | Turn structure impact |
| **Abstention** | Enabled, Disabled | Strategic information hiding |
| **N (# players)** | 2, 3, 4, 5+ | Scalability analysis |
| **Accuracy Range** | [0.1, 0.9], [0.3, 0.7], etc. | Heterogeneity effects |

### Experimental Workflow

```
┌─────────────────────┐
│  1. Configuration   │
│  Define experiment  │
│  parameters (YAML)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. Training Phase  │
│  • Initialize RL    │
│  • Collect episodes │
│  • Update policy    │
│  • Save checkpoint  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. Evaluation      │
│  Run N=1000 games   │
│  • Track outcomes   │
│  • Measure targeting│
│  • Compute stats    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  4. Analysis        │
│  • Win rates        │
│  • Strategy patterns│
│  • Statistical tests│
│  • Visualizations   │
└─────────────────────┘
```

### Training Protocol

**Procedure**:
```
for run in {1, 2, 3, 4, 5}:  // Multiple runs for statistical validity
  seed = run
  set_random_seed(seed)
  
  initialize policy π_θ
  
  for iteration in {1, ..., 2000}:
    // Self-play training
    rollouts = []
    for episode in batch:
      accuracies ~ Uniform(marksmanship_range)
      episode_data = simulate_game(π_θ, accuracies)
      rollouts.append(episode_data)
    
    // Policy update
    π_θ ← PPO_update(π_θ, rollouts)
    
    if iteration % 100 == 0:
      evaluate_policy(π_θ)
      log_metrics()
  
  save_checkpoint(π_θ, f"run_{run}/")
```

**Self-Play Curriculum**:
- Agents play against themselves (no fixed opponents)
- Accuracy sampled fresh each episode → generalization
- Shared policy → symmetric competition

### Evaluation Protocol

**Metrics Collection** (1000 episodes):

```
Initialize counters:
  win_counts[player_rank] = 0
  targeting_matrix[shooter_rank][target_rank] = 0
  abstention_rate[player_rank] = 0
  episode_lengths = []

for episode in {1, ..., 1000}:
  // Sample accuracies
  accuracies = sample_from_range()
  rank = argsort(accuracies)  // 0 = weakest, N-1 = strongest
  
  // Run episode
  game = initialize_game(accuracies, trained_policy)
  
  while not game.is_over():
    shooters, targets = game.step()
    
    // Track targeting patterns
    for (shooter, target) in zip(shooters, targets):
      shooter_rank = rank[shooter.id]
      target_rank = rank[target.id] if target != Ghost else "abstain"
      targeting_matrix[shooter_rank][target_rank] += 1
  
  // Record outcomes
  winners = [p for p in players if p.alive]
  for winner in winners:
    win_counts[rank[winner.id]] += 1
  
  episode_lengths.append(game.turn_count)
```

### Baseline Strategies (for comparison)

1. **Random**: Uniform over alive opponents
   $$\pi_{\text{random}}(a|o) = \frac{1}{|\mathcal{A}|}$$

2. **Greedy-Threat**: Always target strongest opponent
   $$\pi_{\text{greedy}}(o) = \arg\max_j \hat{\theta}_j$$

3. **Proportional**: Target with probability ∝ accuracy
   $$\pi_{\text{prop}}(a_j|o) = \frac{\hat{\theta}_j}{\sum_k \hat{\theta}_k}$$

4. **Abstain-Always**: Never shoot (for sanity check)

**Comparison Matrix**:

| Strategy | Information Used | Adaptability | Expected Performance |
|----------|-----------------|--------------|----------------------|
| Random | None | None | Baseline (1/N) |
| Greedy-Threat | Beliefs only | Low | Better than random |
| Proportional | Beliefs only | Medium | Good |
| **RL (PPO)** | **Full observation** | **High** | **Best (hypothesized)** |

---

## Analysis Methods

### Primary Metrics

#### 1. Win Rate by Accuracy Rank

**Definition**: Probability that player with rank $k$ survives to termination

$$
W_k = \frac{\# \text{ episodes where rank-}k \text{ player wins}}{N_{\text{episodes}}}
$$

**Null Hypothesis** (uniform): $H_0: W_0 = W_1 = ... = W_{N-1} = \frac{1}{N}$

**Statistical Test**: Chi-square goodness of fit
$$
\chi^2 = \sum_{k=0}^{N-1} \frac{(O_k - E_k)^2}{E_k}
$$
where $O_k$ = observed wins for rank $k$, $E_k$ = expected wins under $H_0$

**Interpretation**:
- $p < 0.05$ → Win rates differ significantly by accuracy rank
- Look for **accuracy paradox**: Does $W_{\text{weakest}} > W_{\text{strongest}}$?

#### 2. Targeting Matrix

**Definition**: Conditional probability of targeting
$$
T_{ij}^{(m)} = P(\text{target rank } j | \text{shooter rank } i, m \text{ players alive})
$$

**Estimation**:
$$
\hat{T}_{ij}^{(m)} = \frac{\text{count}(i \to j \text{ when } m \text{ alive})}{\text{count}(i \text{ shoots when } m \text{ alive})}
$$

**Visualization**: Heatmap with shooter rank (rows) × target rank (columns)

**Example Output**:
```
When 3 players alive:
                  Target Rank
                0 (Low)  1 (Med)  2 (High)  Ghost
Shooter   0     0.05     0.15     0.65      0.15
Rank      1     0.10     0.05     0.70      0.15
          2     0.10     0.80     0.05      0.05
          
Interpretation:
  • All players strongly target highest accuracy opponent
  • Medium accuracy shows highest abstention rate
  • Lowest accuracy occasionally targets medium (opportunistic)
```

#### 3. Abstention Rate Analysis

**Abstention Rate**:
$$
A_k = \frac{\text{shots at Ghost by rank-}k}{\text{total shots by rank-}k}
$$

**Hypothesis**: $A_{\text{weak}} > A_{\text{strong}}$ (weak players abstain more)

**Test**: Mann-Whitney U test comparing abstention rates of weak vs. strong players

#### 4. Survival Time Distribution

**Metric**: Number of turns survived by each accuracy rank

$$
S_k = \{t_1, t_2, ..., t_n\} \text{ where } t_i = \text{elimination time of rank-}k \text{ in episode } i
$$

**Analysis**: Kaplan-Meier survival curves

```
Survival Probability
    │
1.0 ├────╲                      ← Rank 0 (weakest)
    │     ╲
0.8 │      ╲────╲               ← Rank 1 (medium)
    │            ╲
0.6 │             ╲──╲          ← Rank 2 (strongest)
    │                 ╲
0.4 │                  ╲
    │                   ╲
0.2 │                    ╲
    │                     ╲
0.0 └──────────────────────╲────► Turn Number
    0    5    10   15   20
```

### Comparative Analysis

#### Information Model Comparison

**Research Question**: Does Bayesian belief tracking improve performance vs. perfect information?

**Methodology**:
1. Train separate policies on each observation model
2. Evaluate each in 1000 episodes
3. Compare win rate distributions

**Statistical Test**: Two-sample t-test on mean episode rewards
$$
t = \frac{\bar{R}_{\text{Bayes}} - \bar{R}_{\text{Perfect}}}{\sqrt{s^2_{\text{Bayes}}/n_1 + s^2_{\text{Perfect}}/n_2}}
$$

#### Gameplay Mode Comparison

**Factorial Design**:

| Observation | Sequential | Simultaneous | Random |
|-------------|-----------|--------------|--------|
| Perfect | Exp 1 | Exp 2 | Exp 3 |
| Bayesian-Mean | Exp 4 | Exp 5 | Exp 6 |
| Bayesian-Var | Exp 7 | Exp 8 | Exp 9 |

**ANOVA**: Test for main effects and interactions
$$
Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijk}
$$
where $\alpha$ = observation effect, $\beta$ = gameplay effect

### Visualizations

#### 1. Win Rate Bar Chart with Confidence Intervals

```
Win Rate by Accuracy Rank (Bayesian Abstention, Simultaneous)

     │
0.50 │                    ┌──┐
     │                    │  │
0.40 │          ┌──┐      │  │
     │          │  │      │  │
0.30 │   ┌──┐   │  │      │  │
     │   │  │   │  │      │  │
0.20 │   │  │   │  │      │  │
     │   │  │   │  │      │  │
0.10 │   │  │   │  │      │  │
     │   │  │   │  │      │  │
0.00 └───┴──┴───┴──┴──────┴──┴─────
         Rank 0  Rank 1  Rank 2
        (Weakest)(Medium)(Strongest)
         
         Error bars show 95% Wilson confidence intervals
         Dashed line at 0.33 = uniform expectation
```

#### 2. Targeting Heatmap (Normalized)

```
Targeting Patterns (3 Players Alive)

              Target Rank
           0     1     2   Ghost
        ┌────┬────┬────┬────┐
      0 │0.05│0.20│0.60│0.15│  Weak shooter
        ├────┼────┼────┼────┤
S     1 │0.10│0.05│0.70│0.15│  Medium shooter
h       ├────┼────┼────┼────┤
o     2 │0.15│0.75│0.05│0.05│  Strong shooter
o       └────┴────┴────┴────┘
t
e       Color intensity = probability
r       (Red = high, Yellow = medium, White = low)
```

#### 3. Convergence Analysis

```
Training Convergence: Episode Reward

Reward
  100 │                       ╱────────
      │                      ╱
   50 │                  ___╱
      │               __╱
    0 │           ___╱
      │        __╱
  -50 │    ___╱
      │___╱
 -100 │
      └────────────────────────────────► Iteration
      0   500  1000  1500  2000

Shaded region: ±1 std dev across 5 runs
Shows stable convergence around iteration 1500
```

### Convergence Diagnostics

**Policy Entropy** (measure of exploration):
$$
H(\pi_\theta) = -\sum_a \pi_\theta(a|o) \log \pi_\theta(a|o)
$$

**Gradient Norm** (measure of learning rate):
$$
\|\nabla_\theta J(\theta)\|_2
$$

**Expected**: Both should decrease over training as policy stabilizes

### Reproducibility Checklist

- ✓ Configuration files versioned
- ✓ Random seeds recorded
- ✓ Multiple independent runs (n=5)
- ✓ Statistical significance tests
- ✓ Confidence intervals reported
- ✓ Hyperparameters documented
- ✓ Code and data archived

---

## Theoretical Connections

### Game Theory

- **Nash Equilibrium**: Does learned policy satisfy best-response condition?
- **Subgame Perfection**: For sequential games, analyze backward induction
- **Pareto Efficiency**: Are outcomes socially optimal?
- **Mechanism Design**: Can we design rules to incentivize desired behaviors?

### Information Theory

- **Mutual Information**: $I(\theta_i; \text{observations})$ quantifies learning
- **Entropy**: Measures uncertainty in beliefs
- **Value of Information**: When is gathering info worth forgoing immediate action?

### Bayesian Statistics

- **Conjugate Priors**: Beta-Bernoulli for computational efficiency
- **Posterior Predictive**: $P(\text{future hit} | \text{data})$
- **Credible Intervals**: Quantify uncertainty in ability estimates

### Reinforcement Learning Theory

- **Policy Gradient Theorem**: Theoretical foundation of PPO
- **Convergence Guarantees**: Under what conditions does learning succeed?
- **Sample Complexity**: How much data needed for near-optimal policy?
- **Multi-Agent Learning Dynamics**: Nash-convergence in self-play
