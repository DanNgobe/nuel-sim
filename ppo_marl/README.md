# PPO Multi-Agent Reinforcement Learning (PPO MARL)

This module implements **Proximal Policy Optimization (PPO)** for multi-agent reinforcement learning in the Nuel simulation environment. PPO is a policy gradient method that uses a clipped surrogate objective to prevent large policy updates, making training more stable than vanilla policy gradient methods.

## 🎯 Key Features

- **Actor-Critic Architecture**: Shared network with separate policy and value heads
- **Generalized Advantage Estimation (GAE)**: Reduces variance in advantage estimates
- **Clipped Surrogate Objective**: Prevents destructive policy updates
- **Shared Agent**: Single agent controls all players for efficient learning
- **Experience Batching**: Collects trajectories before updating

## 🏗️ Architecture

### Network Structure
```
Input (Observation) 
    ↓
Shared Layers (Linear + LayerNorm + ReLU + Dropout)
    ↓
┌─────────────────┬─────────────────┐
│   Actor Head    │   Critic Head   │
│   (Policy)      │   (Value)       │
│ Action Logits   │   State Value   │
└─────────────────┴─────────────────┘
```

### PPO Algorithm Components
1. **Policy Network**: Outputs action probabilities
2. **Value Network**: Estimates state values for advantage computation
3. **GAE**: Computes advantages with bias-variance tradeoff
4. **Clipped Loss**: Prevents large policy updates

## 🚀 Quick Start

### Training
```bash
# Basic training (2000 episodes)
python -m ppo_marl.train

# Extended training with plots
python -m ppo_marl.train --episodes 5000 --plot

# Custom episodes
python -m ppo_marl.train --episodes 10000
```

### Evaluation
```bash
# Evaluate trained model
python -m ppo_marl.evaluate --episodes 200

# Evaluate specific model
python -m ppo_marl.evaluate --model-path ppo_marl/models/your_model.pth
```

## ⚙️ Configuration

### Hyperparameters (`ppo_marl/settings.py`)
```python
GAMMA = 0.99              # Discount factor
LEARNING_RATE = 3e-4      # Adam learning rate
CLIP_EPSILON = 0.2        # PPO clipping parameter
VALUE_COEFF = 0.5         # Value loss coefficient
ENTROPY_COEFF = 0.01      # Entropy bonus coefficient
BATCH_SIZE = 64           # Mini-batch size
MEMORY_SIZE = 2048        # Trajectory buffer size
PPO_EPOCHS = 4            # Updates per batch
GAE_LAMBDA = 0.95         # GAE lambda parameter
```

### Key Hyperparameter Explanations

- **CLIP_EPSILON**: Controls how much the policy can change in one update (0.1-0.3 typical)
- **VALUE_COEFF**: Balances policy vs value learning (0.5-1.0 typical)
- **ENTROPY_COEFF**: Encourages exploration (0.01-0.1 typical)
- **GAE_LAMBDA**: Bias-variance tradeoff in advantage estimation (0.9-0.99 typical)

## 🔄 Training Process

1. **Collect Trajectories**: Agents interact with environment
2. **Store Experiences**: States, actions, rewards, log probabilities, values
3. **Compute Advantages**: Use GAE for better advantage estimates
4. **Update Policy**: Multiple epochs with clipped surrogate loss
5. **Update Value Function**: MSE loss between predicted and target values

### PPO Loss Function
```
L_CLIP = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
L_VF = (V_θ - V_target)²
L_S = -β * H(π_θ)

Total Loss = -L_CLIP + c₁ * L_VF + c₂ * L_S
```

Where:
- `r_t` = probability ratio (new_policy / old_policy)
- `A_t` = advantage estimate
- `ε` = clip epsilon
- `H(π_θ)` = policy entropy

## 📊 Training Metrics

The training script tracks and plots:
- **Episode Lengths**: Game duration over time
- **Average Rewards**: Learning progress indicator
- **Win Rates by Position**: Positional bias analysis
- **Win Rates by Accuracy**: Strategy effectiveness

## 🔧 Implementation Details

### Shared Agent Design
- Single neural network controls all players
- Reduces parameter count and training time
- Enables knowledge sharing between positions
- Consistent strategy across all agents

### Memory Management
- Trajectory-based collection (vs step-based in DQN)
- Batch updates every `MEMORY_SIZE` steps
- Clear memory after each update to prevent staleness

### Advantage Computation
```python
def compute_gae(rewards, values, next_values, dones):
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = reward + γ * next_value * (1 - done) - value
        gae = delta + γ * λ * (1 - done) * gae
        advantages.insert(0, gae)
    return advantages
```

## 🆚 PPO vs DQN Comparison

| Aspect | PPO | DQN |
|--------|-----|-----|
| **Type** | Policy Gradient | Value-Based |
| **Output** | Action Probabilities | Q-Values |
| **Exploration** | Stochastic Policy | ε-greedy |
| **Updates** | On-Policy | Off-Policy |
| **Memory** | Trajectory Buffer | Experience Replay |
| **Stability** | Clipped Updates | Target Networks |

### When to Use PPO
- **Continuous/Large Action Spaces**: Natural policy representation
- **Stochastic Environments**: Handles uncertainty well
- **Multi-Agent Settings**: Policy-based coordination
- **Sample Efficiency**: Often more sample efficient than DQN

## 🎮 Integration with Game

### Strategy Interface
```python
class PPOStrategy(BaseStrategy):
    def choose_target(self, me, players, observation=None):
        # Get observation from observation model
        obs = self.observation_model.create_observation(me, players)
        
        # Get action from PPO agent
        action, log_prob, value = self.agent.act(obs, explore=self.explore)
        
        # Convert action to target player
        targets = self.observation_model.get_targets(me, players)
        return targets[action] if action < len(targets) else None
```

### Observation Models
PPO works with the same observation models as DQN:
- `ThreatLevelObservation`: Basic threat ranking
- `BayesianObservationModel`: Belief-based observations
- `TurnAwareThreatObservation`: Turn-order aware
- `SimpleObservation`: Minimal feature set

## 📈 Performance Tips

### Training Optimization
1. **Batch Size**: Larger batches (64-256) for stable gradients
2. **Learning Rate**: Start with 3e-4, adjust based on convergence
3. **Clip Epsilon**: 0.2 is standard, reduce if training unstable
4. **Memory Size**: 2048-4096 steps for good trajectory diversity

### Hyperparameter Tuning
- **High Variance**: Increase `GAE_LAMBDA`, reduce `LEARNING_RATE`
- **Slow Learning**: Increase `LEARNING_RATE`, reduce `CLIP_EPSILON`
- **Poor Exploration**: Increase `ENTROPY_COEFF`
- **Unstable Training**: Reduce `LEARNING_RATE`, increase `PPO_EPOCHS`

## 🐛 Troubleshooting

### Common Issues
1. **Policy Collapse**: Reduce learning rate, increase entropy coefficient
2. **Slow Convergence**: Check reward scaling, increase batch size
3. **High Variance**: Tune GAE lambda, normalize advantages
4. **Memory Issues**: Reduce memory size or batch size

### Import Errors
Ensure virtual environment is activated:
```bash
# Windows
.\venv\Scripts\Activate.ps1

# Run as module
python -m ppo_marl.train --episodes 2000 --plot
```

## 📚 References

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

## 🔮 Future Enhancements

- **Multi-Agent PPO (MAPPO)**: Individual agents with centralized training
- **Curriculum Learning**: Progressive difficulty increase
- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Distributed Training**: Multi-process trajectory collection
- **Advanced Observation Models**: Graph neural networks for player relationships