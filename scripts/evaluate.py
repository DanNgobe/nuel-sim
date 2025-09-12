# Import warning suppression first
import suppress_warnings
suppress_warnings.suppress_ray_warnings()

import config.settings as config
from config.factories import create_game_objects
from core.game_manager import GameManager
from core.player import Player
import random
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np

def evaluate(num_episodes=100):
    # Create game objects using factories to avoid circular imports
    game_objects = create_game_objects()
    
    # Create GameManager with configuration
    game_manager = GameManager(
        num_players=config.NUM_PLAYERS,
        gameplay=game_objects['gameplay'],
        observation_model=game_objects['observation_model'],
        max_rounds=config.NUM_ROUNDS,
        marksmanship_range=config.MARKSMANSHIP_RANGE,
        strategies=game_objects['strategies'],
        assigned_accuracies=config.ASSIGNED_ACCURACIES,
        has_ghost=config.HAS_GHOST
    )
    
    # Initialize counters - will be updated after first game creation
    win_counts = {}
    win_counts_accuracy = {}
    survivor_counts = {}

    # Stats: alive_count -> shooter_rank -> target_rank -> count
    round_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for episode in range(num_episodes):
        # Create a new game for this episode
        game = game_manager.reset_game()
        players = game.players
        
        # Initialize win_counts with actual player names from the first game
        if episode == 0:
            win_counts = {player.name: 0 for player in players}
            win_counts_accuracy = {f"A{i}": 0 for i in range(len(players))}
            survivor_counts = {f"S{i}": 0 for i in range(len(players) + 1)}
        
        # Sort players by accuracy for ranking
        sorted_accuracy = sorted(players, key=lambda p: p.accuracy if p.accuracy > 0 else 999)

        while not game.is_over():
            alive = game.get_alive_players()
            alive_count = len(alive)

            game.run_auto_turn()
            last_round = game.history[-1]
            for shooter, target, hit in last_round:
                if shooter and target:
                    shooter_rank = sorted_accuracy.index(shooter)
                    target_rank = sorted_accuracy.index(target)
                    round_stats[alive_count][shooter_rank][target_rank] += 1

        # Track win info
        alive_players = [p for p in players if p.alive]
        for player in alive_players:
            if player.name == "Ghost":
                continue
            player_index = sorted_accuracy.index(player)
            win_counts[player.name] += 1
            win_counts_accuracy[f"A{player_index}"] += 1

        survivor_counts[f"S{len(alive_players)}"] += 1

    # Get the number of players for matrix calculations
    num_players = len(players)

    # Display survivor counts
    print("\n=== Survivor Counts ===")
    for survivors, count in survivor_counts.items():
        print(f"{survivors}: {count} episodes")
        

    # Win summaries
    print("\nWin counts after {} episodes:".format(num_episodes))
    for name, count in win_counts.items():
        print(f"{name}: {count} wins")

    print("\nWin counts by accuracy rank (worst to best):")
    for name, count in win_counts_accuracy.items():
        print(f"{name}: {count} wins")

    # Display round-by-round shooting matrix
    print("\n=== Accuracy Rank Shot Matrix by Number of Alive Players ===")
    for alive_count in sorted(round_stats.keys(), reverse=True):
        print(f"\nAlive players: {alive_count}")
        header = "\t" + "\t".join([f"A{j}" for j in range(num_players)])
        print(header)
        matrix = np.zeros((num_players, num_players), dtype=int)
        for i in range(num_players):
            row = [f"A{i}"]
            for j in range(num_players):
                count = round_stats[alive_count][i].get(j, 0)
                matrix[i, j] = count
                row.append(str(count))
            print("\t".join(row))

        # Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu", cbar=False,
                    xticklabels=[f"A{j}" for j in range(num_players)],
                    yticklabels=[f"A{i}" for i in range(num_players)])
        plt.title(f"Heatmap of Shots (Alive Players: {alive_count})")
        plt.xlabel("Target Accuracy Rank")
        plt.ylabel("Shooter Accuracy Rank")
        plt.tight_layout()
        plt.show()

    # Plot survivor counts
    plt.figure(figsize=(6, 4))
    survivor_labels = list(survivor_counts.keys())
    survivor_values = list(survivor_counts.values())
    plt.bar(survivor_labels, survivor_values, color='mediumseagreen')
    plt.title(f"Survivor Counts per Episode\n({num_episodes} Episodes)")
    plt.xlabel("Number of Survivors")
    plt.ylabel("Episodes")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # Summary Win Charts
    plt.figure(figsize=(12, 5))


    plt.subplot(1, 2, 1)
    names = list(win_counts.keys())
    wins = list(win_counts.values())
    plt.bar(names, wins, color='skyblue')
    plt.title(f"Win Counts by Player Position\n({num_episodes} Episodes)")
    plt.xlabel("Player (Position)")
    plt.ylabel("Wins")
    plt.grid(axis='y')

    plt.subplot(1, 2, 2)
    accuracy_names = list(win_counts_accuracy.keys())
    accuracy_wins = list(win_counts_accuracy.values())
    plt.bar(accuracy_names, accuracy_wins, color='salmon')
    plt.title(f"Win Counts by Accuracy Rank\n({num_episodes} Episodes)")
    plt.xlabel("Accuracy Rank (A0 = Worst)")
    plt.ylabel("Wins")
    plt.grid(axis='y')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate game performance.")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes to run (default: 2000)")
    args = parser.parse_args()

    evaluate(num_episodes=args.episodes)
