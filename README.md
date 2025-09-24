# Nuel Simul# Machine Learning Integration
- **Deep Q-Network (DQN)**: Multi-agent deep reinforcement learning with experience replay
- **Proximal Policy Optimization (PPO)**: Advanced policy gradient MARL implementation
- **Ray RLlib Support**: Scalable distributed multi-agent reinforcement learning
- **Observation Models**: Advanced Bayesian and abstention-aware observation systems
- **Training Automation**: Built-in evaluation and plot generation after trainingGame 🎯

A comprehensive Python-based simulation framework for **N-player standoffs** (Duels, Truels, Nuels, and Gruels) with real-time **Pygame** visualization, advanced analytics, and **Multi-Agent Reinforcement Learning (MARL)** capabilities. This extensible framework enables researchers, game theorists, and AI enthusiasts to explore strategic interactions in multi-player elimination games using both classical strategies and cutting-edge machine learning approaches.

## 🌟 Key Features

### 🎮 Game Simulation
- **Multi-Player Support**: Simulate any number of players (2+ players)
- **Diverse Game Modes**: Sequential, simultaneous, random, counter-attack, and even-odd gruel gameplay
- **Strategic AI**: Multiple targeting strategies (strongest, weakest, nearest, random, etc.)
- **Accuracy Modeling**: Realistic marksmanship simulation with customizable accuracy ranges

### 🤖 Machine Learning Integration
- **Deep Q-Network (DQN) MARL**: Train intelligent agents using deep reinforcement learning
- **Ray RLlib Support**: Scalable multi-agent reinforcement learning with Ray framework
- **Custom Neural Networks**: Configurable network architectures for different learning scenarios
- **Experience Replay**: Advanced replay buffer implementation for stable learning
- **Training Visualization**: Real-time training metrics and performance plots

### 📊 Advanced Analytics
- **Multiple Observation Models**: ThreatLevel, Bayesian, BayesianAbstention, TurnAware, and SimpleId models
- **Enhanced Evaluation**: Combined heatmaps, accuracy vs win rate analysis, and statistical summaries
- **Automated Visualization**: Save evaluation plots and training metrics to file systems
- **Strategy Comparison**: Single-strategy evaluation mode for isolated performance testing
- **Training Integration**: Automatic post-training evaluation with plot generation

### 🎨 Visual Interface
- **Real-time Visualization**: Live Pygame rendering with player positions and shot trajectories
- **Interactive Display**: Color-coded players, accuracy indicators, and hit/miss feedback
- **Customizable Graphics**: Adjustable screen size, colors, and animation speed

### 🔧 Extensible Architecture
- **Modular Design**: Easy to add new strategies, game modes, and observation models
- **Plugin System**: Extensible framework for custom gameplay mechanics
- **Configuration Management**: Centralized settings for easy experimentation
- **MARL Framework**: Pluggable multi-agent learning algorithms

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Machine Learning Training](#machine-learning-training)
- [Game Modes](#game-modes)
- [Player Strategies](#player-strategies)
- [Observation Models](#observation-models)
- [Configuration](#configuration)
- [Running Evaluations](#running-evaluations)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher (3.11+ recommended for ML features)
- pip package manager
- At least 4GB RAM (8GB+ recommended for large-scale training)

### 1. Clone the Repository
```bash
git clone https://github.com/DanNgobe/nuel-sim.git
cd nuel-sim
```

### 2. Create Virtual Environment (Recommended)

**⚠️ Important**: A virtual environment is **strongly recommended** due to the complex dependencies (PyTorch, Ray, etc.)

#### Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Windows (Command Prompt):
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

#### Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: The installation includes heavy ML dependencies (PyTorch ~2GB, Ray RLlib, etc.). This may take several minutes.

### 4. Verify Installation

#### Basic Simulation:
```bash
python main.py
```

#### ML Training (requires virtual environment):
```bash
# DQN training
python -m dqn_marl.train --episodes 1000 --plot

# Ray RLlib training
python -m rllib_marl.train
```

### Troubleshooting Installation

If you encounter import errors like `ModuleNotFoundError: No module named 'ray'`:
1. Ensure your virtual environment is activated
2. Run commands as modules: `python -m dqn_marl.train` instead of `python dqn_marl/train.py`
3. Verify all dependencies: `pip list | grep -E "(torch|ray|numpy)"`

## ⚡ Quick Start

### Basic Simulation
```python
from core import Player, Game
from core.gameplay import SimultaneousGamePlay
from visual import run_game_visual
import config

# Create players with different accuracies
players = [
    Player(id=0, name="Alice", accuracy=0.8, strategy=target_strongest),
    Player(id=1, name="Bob", accuracy=0.6, strategy=target_weakest),
    Player(id=2, name="Charlie", accuracy=0.7, strategy=target_random)
]

# Create and run game
game = Game(players, gameplay=SimultaneousGamePlay())
run_game_visual(game)
```

### Command Line Usage
```bash
# Run basic simulation
python main.py

# Run statistical evaluation
python scripts/evaluate.py --episodes 1000

# Run with custom configuration
python main.py  # Edit config/settings.py first
```

## 🤖 Machine Learning Training

### DQN Multi-Agent Training

Train intelligent agents using Deep Q-Networks:

```bash
# Basic training (2000 episodes)
python -m dqn_marl.train

# Training with plots and evaluation
python -m dqn_marl.train --episodes 5000 --plot --evaluate

# Extended training with all features
python -m dqn_marl.train --episodes 20000 --plot --evaluate
```

#### Training Features:
- **Experience Replay**: Stores and samples past experiences for stable learning
- **Target Networks**: Separate target networks for improved stability
- **Epsilon-Greedy Exploration**: Balances exploration vs exploitation
- **Training Metrics**: Episode lengths, rewards, win rates, and learning curves
- **Automatic Evaluation**: Post-training performance analysis with plot generation
- **Model Management**: Automatic saving to organized directory structure

### PPO Multi-Agent Training

Train using Proximal Policy Optimization:

```bash
# Basic PPO training
python -m ppo_marl.train

# PPO with plots and evaluation
python -m ppo_marl.train --episodes 5000 --plot --evaluate

# Extended PPO training
python -m ppo_marl.train --episodes 20000 --plot --evaluate
```

### Ray RLlib Training

Leverage Ray's distributed training capabilities:

```bash
# Start RLlib training
python -m rllib_marl.train

# Monitor training with TensorBoard
tensorboard --logdir ~/ray_results
```

#### RLlib Features:
- **Scalable Training**: Distributed across multiple cores/machines
- **Algorithm Variety**: PPO, DQN, A3C, and more
- **Hyperparameter Tuning**: Built-in optimization
- **Checkpoint Management**: Save and resume training

### Training Configuration

Customize training in `dqn_marl/settings.py`:

```python
# Network Architecture
HIDDEN_LAYERS = [128, 128, 64]  # Neural network layers
LEARNING_RATE = 0.001           # Adam optimizer learning rate

# Training Parameters
BATCH_SIZE = 32                 # Mini-batch size
MEMORY_SIZE = 10000            # Replay buffer capacity
EPSILON_DECAY = 0.995          # Exploration decay rate
TARGET_UPDATE_FREQ = 100       # Target network update frequency

# Game Environment
MAX_GAME_LENGTH = 100          # Maximum rounds per episode
REWARD_WIN = 100               # Reward for winning
REWARD_ELIMINATION = -50       # Penalty for being eliminated
```

### Model Management

Trained models are automatically saved with organized structure:

```bash
# DQN Models saved to:
dqn_marl/models/{num_players}/{gameplay_type}/{observation_model_name}.pth

# PPO Models saved to:
ppo_marl/models/{num_players}/{gameplay_type}/{observation_model_name}.pth

# Evaluation results saved to:
{model_directory}/evaluation/heatmaps.png
{model_directory}/evaluation/statistics.png

# Example structure:
dqn_marl/models/4/SimultaneousGamePlay/BayesianAbstentionObservation_with_abstention_setup_10.pth
dqn_marl/models/4/SimultaneousGamePlay/evaluation/heatmaps.png
dqn_marl/models/4/SimultaneousGamePlay/evaluation/statistics.png
```

### Important Notes

**Virtual Environment Required**: Due to the complex ML dependencies (PyTorch, Ray, NumPy, etc.), you **must** use a virtual environment and run training scripts as modules:

✅ **Correct**:
```bash
# Activate virtual environment first
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source venv/bin/activate     # Mac/Linux

# Then run as module with new evaluation features
python -m dqn_marl.train --episodes 5000 --plot --evaluate
python -m ppo_marl.train --episodes 5000 --plot --evaluate
python -m scripts.evaluate --episodes 1000 --strategy DQNStrategy --output ./results
```

❌ **Incorrect** (will cause import errors):
```bash
python dqn_marl/train.py --episodes 5000 --plot --evaluate
python scripts/evaluate.py --episodes 1000
```

The module approach (`-m`) ensures Python can properly resolve all package imports and dependencies.

## 🎲 Game Modes

### Sequential Gameplay
Players take turns in a fixed order, one shot per turn.
```python
from core.gameplay import SequentialGamePlay
gameplay = SequentialGamePlay()
```

### Simultaneous Gameplay
All alive players shoot at the same time, then resolve hits.
```python
from core.gameplay import SimultaneousGamePlay
gameplay = SimultaneousGamePlay()
```

### Random Gameplay
One randomly selected player shoots each turn.
```python
from core.gameplay import RandomGamePlay
gameplay = RandomGamePlay()
```

### Counter-Attack Gameplay
If a target survives being shot, they immediately counter-attack.
```python
from core.gameplay import CounterAttackRandomGamePlay
gameplay = CounterAttackRandomGamePlay()
```

### Even-Odd Gruel
Players can only target those with opposite parity (even vs odd positions).
```python
from core.gameplay import EvenOddGruelGamePlay
gameplay = EvenOddGruelGamePlay()
```

## 🧠 Player Strategies

### Built-in Strategies

#### Threat-Based Targeting
```python
from core.strategies import (
    target_strongest,     # Target highest accuracy player
    target_weakest,       # Target lowest accuracy player
    target_stronger,      # Target player stronger than self
)
```

#### Positional Targeting
```python
from core.strategies import (
    target_nearest,       # Target closest player (by position)
    target_random,        # Random target selection
)
```

#### Advanced Strategies
```python
from core.strategies import target_stronger_or_strongest
# Xu's strategy: Target second strongest if strong enough, else strongest
```

### Custom Strategy Example
```python
def target_by_threat_level(me, players):
    """Target based on threat level (accuracy × survival probability)"""
    alive = [p for p in players if p != me and p.alive]
    if not alive:
        return None
    
    # Calculate threat scores
    threat_scores = []
    for p in alive:
        threat = p.accuracy * estimate_survival_probability(p, alive)
        threat_scores.append((threat, p))
    
    # Return highest threat
    return max(threat_scores, key=lambda x: x[0])[1]
```

## 👁️ Observation Models

### Threat Level Observation
Basic observation model that sorts players by accuracy.
```python
from core.observation import ThreatLevelObservation
obs_model = ThreatLevelObservation(num_players=3)
```

### Bayesian Mean Observation
Maintains belief distributions about player accuracies using Beta distributions.
```python
from core.observation import BayesianMeanObservation
obs_model = BayesianMeanObservation(
    num_players=3,
    setup_shots=10     # Initial belief calibration
)
```

### Bayesian Abstention Observation
Advanced Bayesian model supporting abstention (shooting ghost) without belief updates.
```python
from core.observation import BayesianAbstentionObservation
obs_model = BayesianAbstentionObservation(
    num_players=3,
    has_ghost=True,    # Enable abstention mechanics
    setup_shots=10     # Initial calibration shots
)
```

### Turn-Aware Threat Observation
Considers turn order and distance in threat assessment.
```python
from core.observation import TurnAwareThreatObservation
obs_model = TurnAwareThreatObservation(num_players=3)
```

### Simple ID Observation
Deterministic observation model that sorts targets by player ID.
```python
from core.observation import SimpleIdObservation
obs_model = SimpleIdObservation(num_players=3, has_ghost=False)
```

## ⚙️ Configuration

### Basic Configuration (`config/settings.py`)

```python
# Game Settings
NUM_PLAYERS = 4
NUM_ROUNDS = None  # Unlimited rounds
HAS_GHOST = False  # Ghost player for abstention modeling

# Gameplay and Observation
GAME_PLAY = SimultaneousGamePlay()
OBSERVATION_MODEL = ThreatLevelObservation(NUM_PLAYERS)
DEFAULT_STRATEGY = target_strongest

# Player Settings
MARKSMANSHIP_RANGE = (0.3, 0.9)  # Min/max accuracy range

# Visual Settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 1  # Simulation speed
PLAYER_RADIUS = 30

# Custom Player Assignments
ASSIGNED_DEFAULT_STRATEGIES = [target_strongest, target_weakest, target_random]
ASSIGNED_DEFAULT_ACCURACIES = [0.8, 0.6, 0.7]
```

### Advanced Configuration Options

#### Color Customization
```python
# Colors (R, G, B)
COLOR_BACKGROUND = (30, 30, 30)      # Dark gray
COLOR_PLAYER_ALIVE = (0, 150, 255)   # Blue
COLOR_PLAYER_DEAD = (100, 100, 100)  # Gray
COLOR_HIT = (0, 255, 0)              # Green arrows
COLOR_MISS = (255, 0, 0)             # Red arrows
```

#### Performance Tuning
```python
# For large-scale simulations
FPS = 10  # Faster simulation
SCREEN_WIDTH = 1200  # Larger display
PLAYER_RADIUS = 20   # Smaller players for more space
```

## 📈 Running Evaluations

### Enhanced Statistical Analysis
```bash
# Basic evaluation with plot display
python -m scripts.evaluate --episodes 1000

# Single strategy evaluation
python -m scripts.evaluate --episodes 1000 --strategy TargetRandom

# Save evaluation plots to directory
python -m scripts.evaluate --episodes 1000 --strategy DQNStrategy --output ./results

# Compare different strategies
python -m scripts.evaluate --episodes 2000 --strategy PPOStrategy --output ./ppo_results
```

### Evaluation Features
- **Combined Visualizations**: All heatmaps and statistics saved in organized image files
- **Strategy Isolation**: Test single strategies across all players for pure performance analysis
- **Accuracy vs Win Rate**: Scatter plots showing correlation between skill and success
- **Automated Saving**: Optional output directory for batch analysis and archiving
- **Training Integration**: Automatic evaluation after DQN/PPO training completion

### Sample Output
```
=== Survivor Counts ===
S1: 847 episodes
S2: 153 episodes

Win counts after 1000 episodes:
P1: 245 wins
P2: 302 wins
P3: 298 wins
P4: 155 wins

Win counts by accuracy rank (worst to best):
A0: 123 wins
A1: 234 wins
A2: 334 wins
A3: 309 wins

=== Accuracy Rank Shot Matrix by Number of Alive Players ===
Alive players: 4
        A0      A1      A2      A3
A0      0       45      23      12
A1      34      0       56      28
A2      28      42      0       67
A3      31      29      45      0

Heatmaps saved to: ./results/heatmaps.png
Statistics saved to: ./results/statistics.png
```

## 📚 API Reference

### Core Classes

#### Player
```python
class Player:
    def __init__(self, id, name, accuracy, x=0, y=0, strategy=None, alive=True):
        """
        Args:
            id: Unique player identifier
            name: Display name
            accuracy: Hit probability (0.0 to 1.0)
            x, y: Screen position coordinates
            strategy: Targeting strategy function
            alive: Current life status
        """
```

#### Game
```python
class Game:
    def __init__(self, players, gameplay, observation_model=None, max_rounds=None):
        """
        Args:
            players: List of Player objects
            gameplay: GamePlay strategy object
            observation_model: Observation model for ML integration
            max_rounds: Maximum rounds before forced termination
        """
```

### Strategy Function Interface
```python
def strategy_function(me: Player, players: List[Player]) -> Player:
    """
    Args:
        me: The player making the decision
        players: All players in the game
    
    Returns:
        Target player to shoot at, or None if no valid target
    """
```

## 💡 Examples

### Example 1: Tournament Simulation
```python
import random
from core import Player, Game
from core.gameplay import SimultaneousGamePlay
from core.strategies import *

def run_tournament(strategies, num_games=100):
    """Run a tournament between different strategies"""
    results = {strategy.__name__: 0 for strategy in strategies}
    
    for _ in range(num_games):
        players = []
        for i, strategy in enumerate(strategies):
            accuracy = random.uniform(0.5, 0.9)
            players.append(Player(i, f"Player_{strategy.__name__}", accuracy, strategy=strategy))
        
        game = Game(players, SimultaneousGamePlay())
        while not game.is_over():
            game.run_turn()
        
        winners = game.get_alive_players()
        if winners:
            results[winners[0].strategy.__name__] += 1
    
    return results

# Run tournament
strategies = [target_strongest, target_weakest, target_random, target_nearest]
results = run_tournament(strategies)
print(results)
```

### Example 2: Custom Observation Model
```python
from core.observation import ObservationModel

class CustomObservationModel(ObservationModel):
    def __init__(self, num_players):
        self.num_players = num_players
        self.shot_history = []
    
    @property
    def name(self):
        return "CustomObservationModel"
    
    def create_observation(self, player, players):
        # Create custom observation vector
        obs = [player.accuracy]  # Self accuracy
        
        # Add relative threat levels
        for p in players:
            if p != player:
                threat = p.accuracy if p.alive else 0.0
                obs.append(threat)
        
        return obs
    
    def update(self, shots, remaining_rounds=None):
        self.shot_history.extend(shots)
    
    # ... implement other required methods
```

### Example 3: Batch Analysis
```python
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_strategy_performance():
    """Analyze how different strategies perform against each other"""
    strategies = [target_strongest, target_weakest, target_random]
    results = defaultdict(list)
    
    for _ in range(1000):
        players = [Player(i, f"P{i}", random.uniform(0.4, 0.9), strategy=s) 
                  for i, s in enumerate(strategies)]
        
        game = Game(players, SimultaneousGamePlay())
        while not game.is_over():
            game.run_turn()
        
        winner = game.get_alive_players()[0] if game.get_alive_players() else None
        if winner:
            results[winner.strategy.__name__].append(winner.accuracy)
    
    # Plot results
    for strategy, accuracies in results.items():
        plt.hist(accuracies, alpha=0.7, label=strategy)
    
    plt.xlabel('Winner Accuracy')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Strategy Performance vs Accuracy')
    plt.show()
```

## 🛠️ Development

### Project Structure
```
nuel-sim/
├── core/                           # Core game logic
│   ├── gameplay/                  # Game mode implementations
│   │   ├── base.py               # Base gameplay classes
│   │   ├── random_mode.py        # Random gameplay mode
│   │   ├── sequential.py         # Sequential gameplay
│   │   └── simultaneous.py       # Simultaneous gameplay
│   ├── observation/               # Observation models for ML
│   │   ├── bayesian_abstention_observation.py    # Abstention-aware Bayesian model
│   │   ├── bayesian_mean_observation.py          # Standard Bayesian model
│   │   ├── observation_model.py                  # Base observation interface
│   │   ├── simple_id_observation.py              # Simple deterministic model
│   │   ├── threat_level_observation.py           # Basic threat-based model
│   │   └── turn_aware_threat_observation.py      # Turn-aware model
│   ├── game.py                    # Main game engine
│   ├── game_manager.py            # Game state management
│   ├── player.py                  # Player class
│   ├── player_manager.py          # Player state management
│   └── strategies.py              # Targeting strategies
├── dqn_marl/                      # Deep Q-Network MARL implementation
│   ├── models/                    # Trained model storage
│   ├── agent.py                   # DQN agent logic
│   ├── network.py                 # Neural network architectures
│   ├── replay_buffer.py           # Experience replay buffer
│   ├── settings.py                # DQN training configuration
│   ├── strategy.py                # ML-based strategy integration
│   └── train.py                   # Training script with evaluation
├── ppo_marl/                      # Proximal Policy Optimization MARL
│   ├── models/                    # Trained model storage
│   ├── agent.py                   # PPO agent implementation
│   ├── evaluate.py                # PPO evaluation utilities
│   ├── memory.py                  # PPO memory buffer
│   ├── network.py                 # PPO network architectures
│   ├── settings.py                # PPO configuration
│   ├── strategy.py                # PPO strategy integration
│   └── train.py                   # PPO training with evaluation
├── rllib_marl/                    # Ray RLlib integration
│   ├── config.py                  # RLlib configuration
│   ├── environment.py             # Multi-agent environment wrapper
│   ├── strategy.py                # RLlib strategy integration
│   └── train.py                   # RLlib training script
├── visual/                        # Visualization components
│   └── pygame_visual.py           # Pygame rendering
├── config/                        # Configuration management
│   ├── factories.py               # Object creation factories
│   └── settings.py                # Game settings
├── scripts/                       # Utility scripts
│   └── evaluate.py                # Enhanced statistical evaluation
├── suppress_warnings.py           # Warning suppression utilities
├── main.py                       # Entry point
├── requirements.txt              # Dependencies
└── venv/                         # Virtual environment (after setup)
```

### Adding New Features

#### New Game Mode
1. Create a new class inheriting from `GamePlay` in `core/gameplay/`
2. Implement required methods: `choose_shooters()`, optionally override others
3. Add import to `core/gameplay/__init__.py`
4. Use in configuration

#### New Strategy
1. Add function to `core/strategies.py` following the interface pattern
2. Import and use in configuration or player creation

#### New Observation Model
1. Create class inheriting from `ObservationModel` in `core/observation/`
2. Implement all abstract methods
3. Add to `__init__.py` and use in configuration

### Running Tests
```bash
# Currently no test suite - contributions welcome!
# Recommended: pytest framework
pip install pytest
pytest tests/
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes** with proper documentation
4. **Add tests** for new functionality
5. **Submit a pull request** with a clear description

### Contribution Areas
- 🧪 **Testing**: Add unit tests and integration tests
- 📊 **Analytics**: Additional statistical analysis and visualization methods
- 🎮 **Game Modes**: New gameplay mechanics and rule variations
- 🧠 **Observation Models**: Novel observation and belief systems
- 🤖 **MARL Algorithms**: Advanced multi-agent learning implementations
- 🎨 **Visualization**: Enhanced graphics and interactive interfaces
- 📚 **Documentation**: Code documentation and research tutorials
- ⚡ **Performance**: Optimization for large-scale distributed training

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by classical game theory problems (duels, truels, etc.)
- Built with Python ecosystem tools (Pygame, NumPy, Matplotlib)
- Community contributions and feedback

## 📞 Support

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/DanNgobe/nuel-sim/issues)
- **Discussions**: Join conversations in [GitHub Discussions](https://github.com/DanNgobe/nuel-sim/discussions)
- **Documentation**: Check this README and code comments

---

**Happy Simulating! 🎯**