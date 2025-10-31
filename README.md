# Nuel Sim 🎯

A Python simulation framework for **N-player standoffs** (Duels, Truels, Nuels) with Pygame visualization and Multi-Agent Reinforcement Learning (MARL) capabilities.

## Features

- **Multi-Player Games**: 2+ players with various targeting strategies
- **Game Modes**: Sequential, simultaneous, random gameplay
- **ML Integration**: DQN, PPO, and Ray RLlib training
- **Visual Interface**: Real-time Pygame rendering
- **Analytics**: Statistical evaluation and performance plots

## Quick Start

### Installation

```bash
git clone https://github.com/DanNgobe/nuel-sim.git
cd nuel-sim
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux  
source venv/bin/activate

pip install -r requirements.txt
```

### Run Simulation

```bash
# Basic visual simulation
python main.py

# With custom config
python main.py --config config/default.yaml

# With trained model
python main.py --config config/default.yaml --model-path ./my_model

# Statistical evaluation
python -m scripts.evaluate --episodes 1000
```

## Machine Learning

### Train Agents

```bash
# Ray RLlib training
python -m rllib_marl.train --algorithm ppo --episodes 2000 --output-path ./training

# With custom config
python -m rllib_marl.train --config config/default.yaml --output-path ./training
```

### Configuration

Edit `config/default.yaml` or create custom YAML files:
- Number of players
- Game modes (Sequential/Simultaneous)
- Player strategies and accuracies
- Observation models

## Project Structure

```
nuel-sim/
├── core/                  # Game logic
│   ├── gameplay/         # Game modes
│   ├── observation/      # ML observation models
│   └── strategies.py     # Player strategies
├── rllib_marl/           # Ray RLlib integration
├── visual/               # Pygame visualization
├── config/               # Configuration
└── scripts/              # Evaluation tools
```

## Game Modes

- **Sequential**: Players take turns
- **Simultaneous**: All players shoot at once
- **Random**: Random player selection each turn

## Base Player Strategies

- `target_strongest`: Target highest accuracy player
- `target_weakest`: Target lowest accuracy player
- `target_random`: Random target selection
- `target_nearest`: Target closest player

## Observation Models
- **NoInfoObservation**: Minimal information model (only self accuracy)
- **SimpleObservation**: Basic player accuracy information sorted by ID
- **SortedObservation**: Players sorted by accuracy (threat level)
- **BayesianMeanObservation**: Bayesian belief updates using Beta distributions
- **BayesianAbstentionObservation**: Bayesian model with abstention support and shot counting
- **TurnAwareThreatObservation**: Considers turn order distance and accuracy

## Requirements

- Python 3.8+
- PyTorch (for ML training)
- Ray (for distributed training)
- Pygame (for visualization)
- NumPy, Matplotlib, Pandas

**Note**: Virtual environment strongly recommended due to heavy ML dependencies.

## Usage Examples

### Custom Game Setup

```python
from core import Player, Game
from core.gameplay import SimultaneousGamePlay
from core.strategies import target_strongest, target_weakest
from visual import run_game_visual

players = [
    Player(0, "Alice", 0.8, strategy=target_strongest),
    Player(1, "Bob", 0.6, strategy=target_weakest)
]

game = Game(players, SimultaneousGamePlay())
run_game_visual(game)
```

### Batch Evaluation

```python
from scripts.evaluate import run_evaluation

results = run_evaluation(
    episodes=1000,
    strategies=[target_strongest, target_weakest, target_random]
)
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes with tests
4. Submit pull request

## License

MIT License - see LICENSE file for details.