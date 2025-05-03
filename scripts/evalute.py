# marl/evaluate.py
import config as config
from core.game import Game
from core.player import Player
import random
import argparse
import matplotlib.pyplot as plt

def evaluate(num_episodes=100):
    num_players = config.NUM_PLAYERS
    win_counts = {f"P{i}": 0 for i in range(num_players)}
    win_counts_accuracy = {f"A{i}": 0 for i in range(num_players)}

    for episode in range(num_episodes):
        players = []
        for i in range(num_players):
            accuracy = random.uniform(0.3, 0.9)
            players.append(Player(f"P{i}", accuracy=accuracy, strategy=config.DEFAULT_STRATEGY))
        sorted_accuracy = sorted(players, key=lambda p: p.accuracy, reverse=False) # from worst to best
        
        game = Game(players, gameplay=config.GAME_PLAY)

        while not game.is_over():
            game.run_turn()

        alive_players = [p for p in players if p.alive]
        if len(alive_players) == 1:
            winner = alive_players[0]
            win_counts[winner.name] += 1
            acc_index = sorted_accuracy.index(winner)
            win_counts_accuracy[f"A{acc_index}"] += 1

    # Final win rates
    print("\nWin counts after {} episodes:".format(num_episodes))
    for name, count in win_counts.items():
        print(f"{name}: {count} wins")

    print("\nWin counts by accuracy rank (worst to best):")
    for name, count in win_counts_accuracy.items():
        print(f"{name}: {count} wins")

    # Plotting
    plt.figure(figsize=(12, 5))

    # Plot 1: Wins by player name
    plt.subplot(1, 2, 1)
    names = list(win_counts.keys())
    wins = list(win_counts.values())
    plt.bar(names, wins, color='skyblue')
    plt.title(f"Win Counts by Player Position\n({num_episodes} Episodes)")
    plt.xlabel("Player (Position)")
    plt.ylabel("Number of Wins")
    plt.grid(axis='y')

    # Plot 2: Wins by player accuracy rank
    plt.subplot(1, 2, 2)
    accuracy_names = list(win_counts_accuracy.keys())
    accuracy_wins = list(win_counts_accuracy.values())
    plt.bar(accuracy_names, accuracy_wins, color='salmon')
    plt.title(f"Win Counts by Accuracy Rank\n({num_episodes} Episodes)")
    plt.xlabel("Accuracy Rank (A0=Worst Accuracy)")
    plt.ylabel("Number of Wins")
    plt.grid(axis='y')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate game performance.")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes to run (default: 2000)")
    args = parser.parse_args()

    evaluate(num_episodes=args.episodes)
