import torch
import seaborn as sns
import matplotlib.pyplot as plt
from marl.strategy import create_agent, create_observation
from core import Player
import config
import random
import numpy as np

# Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_players = config.NUM_PLAYERS
model_path = config.get_model_path(num_players, game_play=config.GAME_PLAY)

# Create agent
agent = create_agent(num_players, model_path=model_path, is_evaluation=True)

# Simulate a game state
players = []
for i in range(num_players):
    accuracy = random.uniform(*config.MARKSMANSHIP_RANGE)
    player = Player(f"P{i}", accuracy=accuracy)
    player.alive = random.choice([True, True, True, False])  # Mostly alive, some dead
    players.append(player)

# Plot setup
fig, axs = plt.subplots(nrows=num_players, figsize=(10, 1.5 * num_players), squeeze=False)

for i, player in enumerate(players):
    obs = create_observation(player, players)
    obs_tensor = torch.tensor([obs], dtype=torch.float32).to(device)

    with torch.no_grad():
        q_values = agent.policy_net(obs_tensor).cpu().numpy().flatten()

    sns.heatmap(
        [q_values],
        annot=True,
        cmap="coolwarm",
        cbar=False,
        xticklabels=[f"P{j}" for j in range(num_players) if j != i],
        ax=axs[i][0]
    )

    axs[i][0].set_title(f"Q-Values for {player.name} (Accuracy={player.accuracy:.2f}, Alive={player.alive})")
    axs[i][0].set_ylabel("")
    axs[i][0].set_yticks([])

plt.tight_layout()
plt.suptitle("Per-Player Q-Value Heatmaps", fontsize=14, y=1.02)
plt.subplots_adjust(top=0.95)
plt.show()
# Optional: save to file
# plt.savefig("q_values_all_players_unmasked.png", dpi=200)
