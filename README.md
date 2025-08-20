# Nuel Simulation Game 🎯

A comprehensive Python-based simulation framework for **N-player standoffs** (Duels, Truels, Nuels, and Gruels) with real-time **Pygame** visualization and advanced analytics. This extensible framework enables researchers, game theorists, and enthusiasts to explore strategic interactions in multi-player elimination games.

## 🌟 Key Features

### 🎮 Game Simulation
- **Multi-Player Support**: Simulate any number of players (2+ players)
- **Diverse Game Modes**: Sequential, simultaneous, random, counter-attack, and even-odd gruel gameplay
- **Strategic AI**: Multiple targeting strategies (strongest, weakest, nearest, random, etc.)
- **Accuracy Modeling**: Realistic marksmanship simulation with customizable accuracy ranges

### 📊 Advanced Analytics
- **Bayesian Observation Models**: Dynamic belief updating based on observed player performance
- **Statistical Evaluation**: Win rate analysis, survivor distribution, and strategic effectiveness metrics
- **Data Visualization**: Comprehensive charts and heatmaps for strategy analysis
- **Export Capabilities**: Save simulation results for external analysis

### 🎨 Visual Interface
- **Real-time Visualization**: Live Pygame rendering with player positions and shot trajectories
- **Interactive Display**: Color-coded players, accuracy indicators, and hit/miss feedback
- **Customizable Graphics**: Adjustable screen size, colors, and animation speed

### 🔧 Extensible Architecture
- **Modular Design**: Easy to add new strategies, game modes, and observation models
- **Plugin System**: Extensible framework for custom gameplay mechanics
- **Configuration Management**: Centralized settings for easy experimentation

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
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
- Python 3.8 or higher
- pip package manager

### 1. Clone the Repository
```bash
git clone https://github.com/DanNgobe/nuel-sim.git
cd nuel-sim
```

### 2. Create Virtual Environment

#### Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
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

### 4. Verify Installation
```bash
python main.py
```

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

### Bayesian Observation Model
Advanced model that maintains belief distributions about player accuracies.
```python
from core.observation import BayesianObservationModel
obs_model = BayesianObservationModel(
    num_players=3,
    setup_shots=10,     # Initial belief calibration
    ucb_scale=1.0       # Exploration parameter
)
```

### Turn-Aware Threat Observation
Considers turn order and distance in threat assessment.
```python
from core.observation import TurnAwareThreatObservation
obs_model = TurnAwareThreatObservation(num_players=3)
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

### Statistical Analysis
```bash
# Run 1000 episodes with detailed analysis
python scripts/evaluate.py --episodes 1000
```

### Evaluation Features
- **Win Rate Analysis**: Per-player and per-accuracy-rank statistics
- **Survivor Distribution**: How many players typically survive
- **Shot Matrix Analysis**: Who shoots whom, organized by accuracy rankings
- **Heatmap Visualization**: Visual representation of targeting patterns
- **Statistical Plots**: Automated generation of analysis charts

### Sample Output
```
=== Survivor Counts ===
S1: 847 episodes
S2: 153 episodes

Win counts after 1000 episodes:
P0: 445 wins
P1: 402 wins
P2: 153 wins

Win counts by accuracy rank (worst to best):
A0: 123 wins
A1: 334 wins
A2: 543 wins
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
├── core/                    # Core game logic
│   ├── gameplay/           # Game mode implementations
│   ├── observation/        # Observation models for ML
│   ├── game.py            # Main game engine
│   ├── player.py          # Player class
│   ├── player_manager.py  # Player state management
│   └── strategies.py      # Targeting strategies
├── visual/                 # Visualization components
│   └── pygame_visual.py   # Pygame rendering
├── config/                 # Configuration management
│   └── settings.py        # Game settings
├── scripts/               # Utility scripts
│   └── evaluate.py        # Statistical evaluation
├── main.py               # Entry point
└── requirements.txt      # Dependencies
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
- 📊 **Analytics**: Enhanced statistical analysis and visualization
- 🎮 **Game Modes**: New gameplay mechanics and rule sets
- 🧠 **AI Strategies**: Advanced targeting algorithms
- 🎨 **Visualization**: Improved graphics and user interface
- 📚 **Documentation**: Code documentation and tutorials

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