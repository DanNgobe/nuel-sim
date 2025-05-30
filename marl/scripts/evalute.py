import config as config
from core.game import Game
from core.player import Player
import random
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np

def evaluate(num_episodes=100):
    num_players = config.NUM_PLAYERS
    win_counts = {f"P{i}": 0 for i in range(num_players)}
    win_counts_accuracy = {f"A{i}": 0 for i in range(num_players)}

    # Stats: alive_count -> shooter_rank -> target_rank -> count
    round_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for episode in range(num_episodes):
        players = []
        for i in range(num_players):
            accuracy = config.ASSIGNED_DEFAULT_ACCURACIES[i] if i < len(config.ASSIGNED_DEFAULT_ACCURACIES) else random.uniform(*config.MARKSMANSHIP_RANGE)
            strategy = config.ASSIGNED_DEFAULT_STRATEGIES[i] if i < len(config.ASSIGNED_DEFAULT_STRATEGIES) else config.DEFAULT_STRATEGY
            players.append(Player(id=i, name=f"P{i}", accuracy=accuracy, strategy=strategy))

        sorted_accuracy = sorted(players, key=lambda p: p.accuracy)
        game = Game(players, gameplay=config.GAME_PLAY, observation_model=config.OBSERVATION_MODEL, max_rounds=config.NUM_ROUNDS)

        while not game.is_over():
            alive = game.get_alive_players()
            alive_count = len(alive)

            game.run_turn()
            if alive_count == 2: continue
            last_round = game.history[-1]
            for shooter, target, hit in last_round:
                if shooter and target:
                    shooter_rank = sorted_accuracy.index(shooter)
                    target_rank = sorted_accuracy.index(target)
                    round_stats[alive_count][shooter_rank][target_rank] += 1

        # Track win info
        alive_players = [p for p in players if p.alive]
        if len(alive_players) == 1:
            winner = alive_players[0]
            win_counts[winner.name] += 1
            acc_index = sorted_accuracy.index(winner)
            win_counts_accuracy[f"A{acc_index}"] += 1

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
