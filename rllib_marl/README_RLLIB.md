# Ray RLlib Multi-Agent Reinforcement Learning

Fixed RLlib MARL implementation using GameManager as the environment.

## Quick Start

### 1. Train DQN Agent (Default - Recommended)
```bash
python rllib_marl/train.py --algorithm dqn --episodes 2000
```

### 2. Train PPO Agent
```bash
python rllib_marl/train.py --algorithm ppo --episodes 2000
```

### 3. Train with Plotting and Evaluation
```bash
python rllib_marl/train.py --algorithm dqn --episodes 2000 --plot --evaluate
```

### 4. Use Trained Model
```python
from rllib_marl.strategy import RLlibStrategy
from core.observation import ThreatLevelObservation

# Load trained strategy
obs_model = ThreatLevelObservation(3)
strategy = RLlibStrategy(
    checkpoint_path="path/to/checkpoint",
    observation_model=obs_model
)

# Use in game
players[0].strategy = strategy
```

## Key Features

- **GameManager Integration**: Uses existing GameManager as MultiAgentEnv
- **Shared Policy**: All players learn from the same policy
- **Algorithm Support**: DQN (default), PPO - both optimized for discrete action spaces
- **Observation Models**: Compatible with existing observation models
- **Strategy Integration**: Trained models work as game strategies
- **Training Visualization**: Plot episode rewards, lengths, and losses
- **Evaluation Mode**: Run evaluation after training
- **Experience Replay**: DQN uses prioritized replay buffer for better learning

## Architecture

- `environment.py`: Wrapper factory for GameManager
- `config.py`: RLlib algorithm configurations
- `train.py`: Training script with proper Ray setup
- `strategy.py`: Strategy class for using trained models

## Configuration

Edit `rllib_marl/config.py` to modify:
- Environment parameters (players, gameplay, observation model)
- Learning rates and training parameters
- Multi-agent policy mapping
- Resource allocation

## Command Line Arguments

- `--algorithm {dqn,ppo}`: Choose algorithm (default: dqn)
- `--episodes N`: Number of training iterations (default: 2000)
- `--checkpoint-dir PATH`: Custom checkpoint directory
- `--plot`: Generate training statistics plots
- `--evaluate`: Run evaluation after training

## Algorithm Comparison

### DQN (Deep Q-Network) - **Recommended**
- ✅ Designed specifically for discrete action spaces
- ✅ Experience replay for sample efficiency
- ✅ Epsilon-greedy exploration strategy
- ✅ Target network for stable learning
- 🎯 Best for games with discrete choices (like choosing which player to shoot)

### PPO (Proximal Policy Optimization)
- ✅ Works well with discrete actions
- ✅ Stable and reliable training
- ✅ Good for on-policy learning
- 🎯 Alternative if DQN doesn't converge well

## Results Location

- Training checkpoints: `rllib_marl/checkpoints/`
- Training plots: `rllib_marl/checkpoints/training_stats.png` (when using --plot)