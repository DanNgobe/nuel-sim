# Configuration Usage Guide

## YAML Configuration System

Nuel Sim uses YAML files for all configuration.

## Configuration Files

- **Default**: `config/default.yaml`
- **Example**: `config/training_example.yaml`

## Configuration Structure

```yaml
game:
  num_players: 3
  num_rounds: 20  # null for infinite
  run_mode: "visualize"  # "visualize" or "single"
  
gameplay:
  type: "RandomGamePlay"  # "SequentialGamePlay", "RandomGamePlay", "SimultaneousGamePlay"
  has_ghost: true

observation:
  model_type: "BayesianAbstentionObservation"
  params:
    num_players: 3
    has_ghost: true
    setup_shots: 0

players:
  marksmanship_range: [0.1, 0.9]
  strategy_types: ["TargetRandom", "TargetRandom", "TargetRandom"]
  accuracies: []  # Empty for random generation

rllib:
  algorithm: "ppo"  # "ppo" or "dqn"
  checkpoint_base_path: "rllib_marl/checkpoints"

visual:
  screen_width: 800
  screen_height: 600
  # ... other visual settings
```

## Usage

### Main Simulation
```bash
# Default config
python main.py

# Custom config
python main.py --config config/training_example.yaml

# Custom config with trained model
python main.py --config config/training_example.yaml --model-path ./my_model
```

### Training
```bash
# Default config
python -m rllib_marl.train --algorithm ppo --episodes 2000 --output-path ./training

# Custom config
python -m rllib_marl.train --config config/training_example.yaml --output-path ./training
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--algorithm` | Training algorithm (ppo, dqn) | dqn |
| `--episodes` | Number of training iterations | 2000 |
| `--config` | Path to YAML config file | config/default.yaml |
| `--output-path` | Directory for training outputs | From config |
| `--plot` | Generate training plots | False |
| `--evaluate` | Run evaluation after training | False |

## Output Structure

When you specify an output path, the training will create:
```
your_output_path/
├── checkpoint_000001/     # Ray RLlib checkpoints
├── checkpoint_000500/
├── checkpoint_001000/
├── training_stats.png     # Training plots (if --plot used)
└── evaluation/            # Evaluation results (if --evaluate used)
    └── [observation_model_name]/
```

## Creating Custom Configurations

1. Copy `config/default.yaml` to a new file
2. Modify the settings for your experiment
3. Use the new config file with training commands

Example for a 4-player simultaneous game:
```yaml
game:
  num_players: 4
  
gameplay:
  type: "SimultaneousGamePlay"
  
players:
  strategy_types: ["TargetStrongest", "TargetWeakest", "TargetRandom", "TargetNearest"]
```

