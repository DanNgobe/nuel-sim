# Import warning suppression first
import suppress_warnings
suppress_warnings.suppress_ray_warnings()

import config.config_loader as config
from config.factories import create_game_objects, create_strategy
from core.game_manager import GameManager
from core.player import Player
import random
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np
import os
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportion_confint

def evaluate(num_episodes=100, single_strategy=None, output_path=None):
    # Create game objects using factories to avoid circular imports
    game_objects = create_game_objects()
    
    cfg = config.get_config()
    
    # Handle single strategy mode
    if single_strategy:
        # Create the same strategy for all players
        strategies = [create_strategy(single_strategy, game_objects['observation_model']) 
                     for _ in range(cfg['game']['num_players'])]
    else:
        strategies = game_objects['strategies']
    
    # Create GameManager with configuration
    game_manager = GameManager(
        num_players=cfg['game']['num_players'],
        gameplay=game_objects['gameplay'],
        observation_model=game_objects['observation_model'],
        max_rounds=cfg['game']['num_rounds'],
        marksmanship_range=tuple(cfg['players']['marksmanship_range']),
        strategies=strategies,
        assigned_accuracies=cfg['players']['accuracies'],
        has_ghost=cfg['gameplay']['has_ghost']
    )
    
    # Initialize counters - will be updated after first game creation
    win_counts = {}
    win_counts_accuracy = {}
    survivor_counts = {}

    # Stats: alive_count -> shooter_rank -> target_rank -> count
    round_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    # Abstention analysis data
    abstention_by_rank = defaultdict(int)  # rank -> abstention_count
    total_shots_by_rank = defaultdict(int)  # rank -> total_shots
    abstention_by_alive_count = defaultdict(lambda: defaultdict(int))  # alive_count -> rank -> abstentions
    total_by_alive_count = defaultdict(lambda: defaultdict(int))  # alive_count -> rank -> total_opportunities
    
    # Abstention by actual accuracy values
    abstention_by_accuracy = defaultdict(int)  # accuracy -> abstention_count
    total_shots_by_accuracy = defaultdict(int)  # accuracy -> total_shots
    
    # Collect accuracy vs win rate data
    accuracy_win_data = []  # List of (accuracy, won) tuples for scatter plot

    for episode in range(num_episodes):
        # Create a new game for this episode
        game = game_manager.reset_game()
        players = game.players
        
        # Initialize win_counts with actual player names from the first game
        if episode == 0:
            win_counts = {player.name: 0 for player in players}
            win_counts_accuracy = {f"A{i}": 0 for i in range(len(players))}
            survivor_counts = {f"S{i}": 0 for i in range(len(players) + 1)}
            # Get the number of players for matrix calculations
            num_players = len(players)
        
        # Sort players by accuracy for ranking
        sorted_accuracy = sorted(players, key=lambda p: p.accuracy if p.accuracy > 0 else 999)

        while not game.is_over():
            alive = game.get_alive_players()
            alive_count = len(alive)

            game.run_auto_turn()
            last_round = game.history[-1]
            for shooter, target, hit in last_round:
                if shooter:  # Player took a turn
                    shooter_rank = sorted_accuracy.index(shooter)
                    shooter_accuracy = shooter.accuracy
                    total_shots_by_rank[shooter_rank] += 1
                    total_shots_by_accuracy[shooter_accuracy] += 1
                    total_by_alive_count[alive_count][shooter_rank] += 1
                    
                    if target and target.name != "Ghost":  # Shot at a real player
                        target_rank = sorted_accuracy.index(target)
                        round_stats[alive_count][shooter_rank][target_rank] += 1
                    else:  # Abstained (target is None or Ghost)
                        abstention_by_rank[shooter_rank] += 1
                        abstention_by_accuracy[shooter_accuracy] += 1
                        abstention_by_alive_count[alive_count][shooter_rank] += 1
                        # For matrix display, abstention goes to last column (ghost index)
                        if cfg['gameplay']['has_ghost']:
                            ghost_index = num_players - 1
                            round_stats[alive_count][shooter_rank][ghost_index] += 1

        # Track win info and accuracy data for all players
        alive_players = [p for p in players if p.alive]
        
        # Collect accuracy vs win data for all players in this episode
        for player in players:
            if player.name != "Ghost":
                is_winner = player in alive_players
                accuracy_win_data.append((player.accuracy, is_winner))
        
        # Track wins for survivors
        for player in alive_players:
            if player.name == "Ghost":
                continue
            player_index = sorted_accuracy.index(player)
            win_counts[player.name] += 1
            win_counts_accuracy[f"A{player_index}"] += 1

        survivor_counts[f"S{len(alive_players)}"] += 1

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
    
    # Statistical analysis of win rates
    print("\n=== Statistical Analysis ===")
    
    # Chi-square test for win rate differences
    win_values = list(win_counts_accuracy.values())
    if len(win_values) > 1 and sum(win_values) > 0:
        expected = [sum(win_values) / len(win_values)] * len(win_values)
        chi2, p_value = chi2_contingency([win_values, expected])[:2]
        print(f"Chi-square test for equal win rates: p = {p_value:.4f}")
        if p_value < 0.05:
            print("✓ Win rates are significantly different")
        else:
            print("~ No significant difference in win rates")
    
    # Confidence intervals for win rates
    print("\nWin Rate Confidence Intervals (95%):")
    for name, count in win_counts_accuracy.items():
        if num_episodes > 0:
            rate = count / num_episodes
            ci_low, ci_high = proportion_confint(count, num_episodes, alpha=0.05, method='wilson')
            print(f"{name}: {rate:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
    
    # Display abstention analysis
    print("\n=== Abstention Analysis ===")
    print("\nAbstention Rate by Accuracy Rank:")
    for rank in sorted(abstention_by_rank.keys()):
        total = total_shots_by_rank[rank]
        abstentions = abstention_by_rank[rank]
        rate = (abstentions / total * 100) if total > 0 else 0
        print(f"A{rank}: {abstentions}/{total} ({rate:.1f}%)")
    
    print("\nAbstention Rate by Accuracy Value:")
    for accuracy in sorted(abstention_by_accuracy.keys()):
        total = total_shots_by_accuracy[accuracy]
        abstentions = abstention_by_accuracy[accuracy]
        rate = (abstentions / total * 100) if total > 0 else 0
        print(f"{accuracy:.1f}: {abstentions}/{total} ({rate:.1f}%)")
    
    print("\nAbstention Rate by Alive Players and Accuracy Rank:")
    for alive_count in sorted(abstention_by_alive_count.keys(), reverse=True):
        print(f"\nAlive players: {alive_count}")
        for rank in sorted(abstention_by_alive_count[alive_count].keys()):
            total = total_by_alive_count[alive_count][rank]
            abstentions = abstention_by_alive_count[alive_count][rank]
            rate = (abstentions / total * 100) if total > 0 else 0
            print(f"  A{rank}: {abstentions}/{total} ({rate:.1f}%)")

    # Display round-by-round shooting matrix
    print("\n=== Accuracy Rank Shot Matrix by Number of Alive Players ===")
    
    # Collect all heatmap data
    heatmap_data = []
    for alive_count in sorted(round_stats.keys(), reverse=True):
        print(f"\nAlive players: {alive_count}")
        # Create column labels - replace last with "Abstain" if ghost exists
        col_labels = [f"A{j}" for j in range(num_players)]
        if cfg['gameplay']['has_ghost']:
            col_labels[-1] = "Abstain"
        header = "\t" + "\t".join(col_labels)
        print(header)
        
        # Create matrix - exclude ghost row if it exists
        matrix_size = num_players - 1 if cfg['gameplay']['has_ghost'] else num_players
        matrix = np.zeros((matrix_size, num_players), dtype=int)
        
        for i in range(matrix_size):
            row = [f"A{i}"]
            for j in range(num_players):
                count = round_stats[alive_count][i].get(j, 0)
                matrix[i, j] = count
                row.append(str(count))
            print("\t".join(row))
        heatmap_data.append((alive_count, matrix))

    # Create combined heatmap figure
    num_heatmaps = len(heatmap_data)
    if num_heatmaps > 0:
        cols = min(2, num_heatmaps)  # Max 2 columns
        rows = (num_heatmaps + cols - 1) // cols  # Calculate rows needed
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if num_heatmaps == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for idx, (alive_count, matrix) in enumerate(heatmap_data):
            ax = axes[idx] if num_heatmaps > 1 else axes[0]
            
            # Create labels - replace last column with "Abstain" if ghost exists
            x_labels = [f"A{j}" for j in range(num_players)]
            if cfg['gameplay']['has_ghost']:
                x_labels[-1] = "Abstain"
            
            # Y labels exclude ghost row if it exists
            matrix_rows = matrix.shape[0]
            y_labels = [f"A{i}" for i in range(matrix_rows)]
            
            sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu", cbar=False,
                       xticklabels=x_labels, yticklabels=y_labels, ax=ax)
            ax.set_title(f"Shots (Alive: {alive_count})")
            ax.set_xlabel("Target Accuracy Rank")
            ax.set_ylabel("Shooter Accuracy Rank")
        
        # Hide empty subplots
        for idx in range(num_heatmaps, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f"Round-by-Round Shooting Matrices ({num_episodes} Episodes)")
        plt.tight_layout()
        
        if output_path:
            heatmap_path = os.path.join(output_path, "heatmaps.png")
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            print(f"Heatmaps saved to: {heatmap_path}")
        else:
            plt.show()

    # Create abstention analysis plots
    if cfg['gameplay']['has_ghost'] and (abstention_by_rank or any(abstention_by_alive_count.values())):
        fig_abs, axes_abs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Abstention rate by accuracy rank (top left)
        ranks = sorted(abstention_by_rank.keys())
        rates = [(abstention_by_rank[rank] / total_shots_by_rank[rank] * 100) if total_shots_by_rank[rank] > 0 else 0 for rank in ranks]
        rank_labels = [f"A{rank}" for rank in ranks]
        
        axes_abs[0, 0].bar(rank_labels, rates, color='orange', alpha=0.7)
        axes_abs[0, 0].set_title("Abstention Rate by Accuracy Rank")
        axes_abs[0, 0].set_xlabel("Accuracy Rank (A0 = Worst)")
        axes_abs[0, 0].set_ylabel("Abstention Rate (%)")
        axes_abs[0, 0].grid(axis='y', alpha=0.3)
        
        # Abstention rate by alive count (heatmap)
        alive_counts = sorted(abstention_by_alive_count.keys(), reverse=True)
        all_ranks = sorted(set().union(*[abstention_by_alive_count[ac].keys() for ac in alive_counts]))
        
        if alive_counts and all_ranks:
            heatmap_matrix = np.zeros((len(alive_counts), len(all_ranks)))
            for i, alive_count in enumerate(alive_counts):
                for j, rank in enumerate(all_ranks):
                    total = total_by_alive_count[alive_count][rank]
                    abstentions = abstention_by_alive_count[alive_count][rank]
                    rate = (abstentions / total * 100) if total > 0 else 0
                    heatmap_matrix[i, j] = rate
            
            sns.heatmap(heatmap_matrix, annot=True, fmt=".1f", cmap="Reds", 
                       xticklabels=[f"A{r}" for r in all_ranks],
                       yticklabels=[f"{ac} alive" for ac in alive_counts],
                       ax=axes_abs[0, 1], cbar_kws={'label': 'Abstention Rate (%)'})
            axes_abs[0, 1].set_title("Abstention Rate by Alive Players vs Accuracy Rank")
            axes_abs[0, 1].set_xlabel("Accuracy Rank")
            axes_abs[0, 1].set_ylabel("Number of Alive Players")
        
        # Abstention rate by accuracy value (bottom, spanning both columns)
        accuracies = sorted(abstention_by_accuracy.keys())
        acc_rates = [(abstention_by_accuracy[acc] / total_shots_by_accuracy[acc] * 100) if total_shots_by_accuracy[acc] > 0 else 0 for acc in accuracies]
        acc_labels = [f"{acc:.1f}" for acc in accuracies]
        
        # Use subplot2grid to span both bottom columns
        plt.subplot2grid((2, 2), (1, 0), colspan=2, fig=fig_abs)
        plt.bar(acc_labels, acc_rates, color='green', alpha=0.7)
        plt.title("Abstention Rate by Accuracy Value")
        plt.xlabel("Accuracy Value")
        plt.ylabel("Abstention Rate (%)")
        plt.grid(axis='y', alpha=0.3)
        
        # Hide the unused bottom subplots
        axes_abs[1, 0].set_visible(False)
        axes_abs[1, 1].set_visible(False)
        
        plt.suptitle(f"Abstention Analysis ({num_episodes} Episodes)")
        plt.tight_layout()
        
        if output_path:
            abstention_path = os.path.join(output_path, "abstention_analysis.png")
            plt.savefig(abstention_path, dpi=300, bbox_inches='tight')
            print(f"Abstention analysis saved to: {abstention_path}")
        else:
            plt.show()

    # Combined survivor counts and win charts
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Win counts by player position
    names = list(win_counts.keys())
    wins = list(win_counts.values())
    axes[0, 0].bar(names, wins, color='skyblue')
    axes[0, 0].set_title(f"Win Counts by Player Position")
    axes[0, 0].set_xlabel("Player (Position)")
    axes[0, 0].set_ylabel("Wins")
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Add statistical test for position
    if len(wins) > 1 and sum(wins) > 0:
        expected = [sum(wins) / len(wins)] * len(wins)
        chi2, p_value = chi2_contingency([wins, expected])[:2]
        sig_text = f"p = {p_value:.3f}" + (" *" if p_value < 0.05 else "")
        axes[0, 0].text(0.02, 0.98, sig_text, transform=axes[0, 0].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Win counts by accuracy rank
    accuracy_names = list(win_counts_accuracy.keys())
    accuracy_wins = list(win_counts_accuracy.values())
    axes[1, 0].bar(accuracy_names, accuracy_wins, color='salmon')
    axes[1, 0].set_title(f"Win Counts by Accuracy Rank")
    axes[1, 0].set_xlabel("Accuracy Rank (A0 = Worst)")
    axes[1, 0].set_ylabel("Wins")
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Add statistical significance annotation
    if len(accuracy_wins) > 1 and sum(accuracy_wins) > 0:
        expected = [sum(accuracy_wins) / len(accuracy_wins)] * len(accuracy_wins)
        chi2, p_value = chi2_contingency([accuracy_wins, expected])[:2]
        sig_text = f"p = {p_value:.3f}" + (" *" if p_value < 0.05 else "")
        axes[1, 0].text(0.02, 0.98, sig_text, transform=axes[1, 0].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

     # Survivor counts
    survivor_labels = list(survivor_counts.keys())
    survivor_values = list(survivor_counts.values())
    axes[0, 1].bar(survivor_labels, survivor_values, color='mediumseagreen')
    axes[0, 1].set_title(f"Survivor Counts per Episode")
    axes[0, 1].set_xlabel("Number of Survivors")
    axes[0, 1].set_ylabel("Episodes")
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Add statistical test for survivor distribution
    if len(survivor_values) > 1 and sum(survivor_values) > 0:
        expected = [sum(survivor_values) / len(survivor_values)] * len(survivor_values)
        chi2, p_value = chi2_contingency([survivor_values, expected])[:2]
        sig_text = f"p = {p_value:.3f}" + (" *" if p_value < 0.05 else "")
        axes[0, 1].text(0.02, 0.98, sig_text, transform=axes[0, 1].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Accuracy vs Win Rate scatter plot
    if len(accuracy_win_data) > 0:
        # Process collected accuracy vs win data
        accuracy_dict = defaultdict(list)
        
        # Group by accuracy level
        for accuracy, won in accuracy_win_data:
            accuracy_dict[accuracy].append(won)
        
        # Calculate win rates for each accuracy level
        player_accuracies = []
        player_win_rates = []
        
        for accuracy, wins_list in accuracy_dict.items():
            win_rate = (sum(wins_list) / len(wins_list)) * 100
            player_accuracies.append(accuracy)
            player_win_rates.append(win_rate)
        
        # Create scatter plot
        axes[1, 1].scatter(player_accuracies, player_win_rates, color='purple', s=100, alpha=0.7)
        axes[1, 1].set_title(f"Accuracy vs Win Rate")
        axes[1, 1].set_xlabel("Player Accuracy")
        axes[1, 1].set_ylabel("Win Rate (%)")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 100)
        
        # Add trend line if we have enough data points
        if len(player_accuracies) > 1:
            z = np.polyfit(player_accuracies, player_win_rates, 1)
            p = np.poly1d(z)
            axes[1, 1].plot(player_accuracies, p(player_accuracies), "r--", alpha=0.8, linewidth=2)
            
            # Add correlation coefficient
            correlation = np.corrcoef(player_accuracies, player_win_rates)[0, 1]
            axes[1, 1].text(0.05, 0.95, f'r = {correlation:.3f}', 
                           transform=axes[1, 1].transAxes, fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.suptitle(f"Game Statistics Summary ({num_episodes} Episodes)")
    plt.tight_layout()
    
    if output_path:
        stats_path = os.path.join(output_path, "statistics.png")
        plt.savefig(stats_path, dpi=300, bbox_inches='tight')
        print(f"Statistics saved to: {stats_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate game performance.")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes to run (default: 2000)")
    parser.add_argument("--strategy", type=str, help="Single strategy to use for all players (e.g., 'TargetRandom', 'DQNStrategy')")
    parser.add_argument("--output", type=str, help="Output directory to save plots (if not specified, plots are shown)")
    args = parser.parse_args()

    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    evaluate(num_episodes=args.episodes, single_strategy=args.strategy, output_path=args.output)
