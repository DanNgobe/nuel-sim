# Nuel Simulation Game

A Python-based simulation of **Duels, Truels, Nuels, and Gruels** using **Pygame** for visualization. This framework allows you to simulate and visualize various N-player standoffs with different strategies (simultaneous, sequential, random) and game mechanics.

## Features
- **Nuel Simulation**: Simulate duels, truels, and other N-player games.
- **Pygame Visualization**: Visualizes players, shots, and eliminations as arrows.
- **Flexible Game Modes**: Supports simultaneous, sequential, and random turn-based gameplay.
- **Extensible Framework**: Easily extendable to support new strategies or game rules.

## Installation

### 1. Clone the Repository
Clone this repository to your local machine:

```bash
git clone https://github.com/DanNgobe/nuel-sim.git
cd nuel-sim
```

### 2. Create and Activate a Virtual Environment
Create a virtual environment and activate it:

#### On **Mac/Linux**:
```bash
python3 -m venv venv
source venv/bin/activate
```
#### On **Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
Install the required dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Run the Simulation
To run the game simulation with Pygame visualization:

```bash
python main.py
```

### 5. Configuration

The game settings can be customized in the `config.py` file. This file allows you to adjust parameters such as the number of players, game mode, player attributes, and visual settings. 


## How to Play

The simulation will start, and the players will take turns based on the configured game mode (sequential, simultaneous, or random). You can observe the interactions as arrows are drawn on the screen showing the shots from the shooters to their targets.

### Game Modes
- **Sequential**: Players take turns in order.
- **Simultaneous**: All players shoot at the same time.
- **Random**: Players take random turns.

## Customization

- Modify the `main.py` file to change the number of players, game mode, and other settings.
- Extend the `Player` class to implement different strategies for targeting and shooting.
- Customize the visuals by editing `visual/pygame_visual.py` for unique styles or animations.