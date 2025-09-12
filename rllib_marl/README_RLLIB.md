# Ray RLlib Multi-Agent Reinforcement Learning

Fixed RLlib MARL implementation using GameManager as the environment.

## Quick Start

### 1. Train DQN Agent
```bash
python rllib_marl/train.py --algorithm dqn --episodes 2000
```

### 2. Train PPO Agent
```bash
python rllib_marl/train.py --algorithm ppo --episodes 2000
```

### 3. Use Trained Model
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
- **Algorithm Support**: PPO, DQN with proper configurations
- **Observation Models**: Compatible with existing observation models
- **Strategy Integration**: Trained models work as game strategies

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

## Results Location

Training results saved to: `rllib_marl/results/`