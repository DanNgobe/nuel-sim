import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import os
import config.config_loader as config
from config.factories import create_game_objects, create_strategy
from core.game_manager import GameManager
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from statsmodels.stats.multitest import multipletests
from itertools import combinations

def compute_payoff_matrix(strategies, episodes=1000, model_path=None):
    """Compute payoff matrix for given strategies via simulation"""
    cfg = config.get_config()
    n = len(strategies)
    n_combinations = n ** cfg['game']['num_players']
    payoff_matrix = np.zeros((cfg['game']['num_players'], n, n_combinations))
    
    # Create game objects using the updated factory system
    game_objects = create_game_objects(model_path=model_path)
    
    # For N-player game, iterate through all strategy combinations
    for i, strat_combo in enumerate(product(range(n), repeat=cfg['game']['num_players'])):
        print(f"Testing combination {i+1}/{n_combinations}: {[strategies[s] for s in strat_combo]}")
        
        # Create strategies for this combination
        strategy_objects = []
        for s in strat_combo:
            strategy_objects.append(create_strategy(strategies[s], game_objects['observation_model'], model_path))
        
        # Run simulation
        game_manager = GameManager(
            num_players=cfg['game']['num_players'],
            gameplay=game_objects['gameplay'],
            observation_model=game_objects['observation_model'],
            max_rounds=cfg['game']['num_rounds'],
            strategies=strategy_objects,
            has_ghost=cfg['gameplay']['has_ghost']
        )
        
        wins = [0] * cfg['game']['num_players']
        for _ in range(episodes):
            game = game_manager.reset_game()
            while not game.is_over():
                game.run_auto_turn()
            
            # Count wins
            alive = game.get_alive_players()
            for player in alive:
                if player.name != "Ghost":
                    wins[player.id] += 1
        
        # Store payoffs (win rates)
        for player_id in range(cfg['game']['num_players']):
            payoff_matrix[player_id][strat_combo[player_id]][i] = wins[player_id] / episodes
    
    return payoff_matrix

def plot_pairwise_comparisons(strategies, payoffs, episodes=1000):
    """Plot pairwise comparison p-values as heatmap"""
    cfg = config.get_config()
    
    fig, axes = plt.subplots(1, cfg['game']['num_players'], figsize=(4*cfg['game']['num_players'], 3))
    if cfg['game']['num_players'] == 1:
        axes = [axes]
    
    for player in range(cfg['game']['num_players']):
        avg_payoffs = np.mean(payoffs[player], axis=1)
        win_counts = (avg_payoffs * episodes).astype(int)
        
        # Create p-value matrix
        n = len(strategies)
        p_matrix = np.ones((n, n))
        sig_matrix = np.zeros((n, n))
        
        if len(win_counts) > 2:
            p_values = []
            pairs = []
            
            for i, j in combinations(range(n), 2):
                count = np.array([win_counts[i], win_counts[j]])
                nobs = np.array([episodes, episodes])
                _, p_val = proportions_ztest(count, nobs)
                p_values.append(p_val)
                pairs.append((i, j))
            
            # Apply BH correction
            rejected, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
            
            # Fill matrix
            for (i, j), p_adj, is_sig in zip(pairs, p_corrected, rejected):
                p_matrix[i, j] = p_adj
                p_matrix[j, i] = p_adj
                sig_matrix[i, j] = is_sig
                sig_matrix[j, i] = is_sig
        
        # Create annotations
        annot = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                if i == j:
                    annot[i, j] = "-"
                else:
                    star = "*" if sig_matrix[i, j] else ""
                    annot[i, j] = f"{p_matrix[i, j]:.3f}{star}"
        
        sns.heatmap(p_matrix, 
                   annot=annot, 
                   fmt='',
                   xticklabels=strategies,
                   yticklabels=strategies,
                   cmap='RdYlBu',
                   vmin=0, vmax=0.1,
                   ax=axes[player])
        axes[player].set_title(f'Player {player+1} P-values (BH corrected)')
    
    plt.tight_layout()
    
    obs_model = create_game_objects()['observation_model']
    filename = f"pairwise_comparisons_{cfg['game']['num_players']}p_{obs_model.name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved pairwise comparisons as {filename}")
    plt.show()

def plot_payoff_matrix(strategies, episodes=1000, model_path=None):
    """Plot payoff matrices for each player"""
    print("\n=== PAYOFF MATRIX INTERPRETATION ===")
    print("Higher values = better performance (higher win rate)")
    print("Each cell shows win rate (0.0 to 1.0) when using that strategy")
    print("Values averaged across all possible opponent combinations\n")
    
    payoffs = compute_payoff_matrix(strategies, episodes, model_path)
    n_strategies = len(strategies)
    
    cfg = config.get_config()
    fig, axes = plt.subplots(1, cfg['game']['num_players'], figsize=(5*cfg['game']['num_players'], 4))
    if cfg['game']['num_players'] == 1:
        axes = [axes]
    
    for player in range(cfg['game']['num_players']):
        # Average payoff across all opponent combinations
        avg_payoffs = np.mean(payoffs[player], axis=1)
        
        # Create heatmap data
        heatmap_data = avg_payoffs.reshape(1, -1)
        
        sns.heatmap(heatmap_data, 
                   annot=True, 
                   fmt='.3f',
                   xticklabels=strategies,
                   yticklabels=[f'P{player+1}'],
                   cmap='RdYlBu_r',
                   ax=axes[player])
        axes[player].set_title(f'Player {player+1} Win Rate by Strategy')
        axes[player].set_xlabel('Strategy')
        
        # Statistical analysis with proportions z-test + BH correction
        win_counts = (avg_payoffs * episodes).astype(int)
        if len(win_counts) > 2:  # Need at least 3 strategies for pairwise comparisons
            p_values = []
            comparisons = []
            
            # Pairwise comparisons
            for i, j in combinations(range(len(strategies)), 2):
                count = np.array([win_counts[i], win_counts[j]])
                nobs = np.array([episodes, episodes])
                _, p_val = proportions_ztest(count, nobs)
                p_values.append(p_val)
                comparisons.append(f"{strategies[i]} vs {strategies[j]}")
            
            # Apply Benjamini-Hochberg correction
            if p_values:
                rejected, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
                
                # Print detailed results
                print(f"\nPlayer {player+1} Pairwise Comparisons (BH corrected):")
                for comp, p_orig, p_adj, is_sig in zip(comparisons, p_values, p_corrected, rejected):
                    print(f"  {comp}: p = {p_orig:.4f}, p_adj = {p_adj:.4f} {'*' if is_sig else ''}")
        
        # Print interpretation
        best_strategy = strategies[np.argmax(avg_payoffs)]
        print(f"Player {player+1}: Best strategy is {best_strategy} (win rate: {np.max(avg_payoffs):.3f})")
        
        # Print confidence intervals
        print(f"Player {player+1} Strategy Confidence Intervals (95%):")
        for i, strategy in enumerate(strategies):
            wins = int(avg_payoffs[i] * episodes)
            ci_low, ci_high = proportion_confint(wins, episodes, alpha=0.05, method='wilson')
            print(f"  {strategy}: [{ci_low:.3f}, {ci_high:.3f}]")
    
    plt.tight_layout()
    
    # Save the figure with observation model type and number of players
    obs_model = create_game_objects()['observation_model']
    filename = f"payoff_matrix_{cfg['game']['num_players']}p_{obs_model.name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot as {filename}")
    plt.show()
    
    # Also plot pairwise comparisons
    plot_pairwise_comparisons(strategies, payoffs, episodes)

def plot_strategy_heatmap(strategies, episodes=1000, model_path=None):
    """Plot strategy vs strategy heatmap for 2-player case"""
    cfg = config.get_config()
    if cfg['game']['num_players'] != 2:
        print("Strategy heatmap only available for 2-player games")
        return
    
    n = len(strategies)
    payoff_matrix = np.zeros((n, n))
    
    for i, strat1 in enumerate(strategies):
        for j, strat2 in enumerate(strategies):
            print(f"Testing {strat1} vs {strat2}")
            
            # Create game objects using the updated factory system
            game_objects = create_game_objects(model_path=model_path)
            
            strategy_objects = []
            for strat in [strat1, strat2]:
                strategy_objects.append(create_strategy(strat, game_objects['observation_model'], model_path))
            
            game_manager = GameManager(
                num_players=2,
                gameplay=game_objects['gameplay'],
                observation_model=game_objects['observation_model'],
                strategies=strategy_objects,
                has_ghost=cfg['gameplay']['has_ghost']
            )
            
            p1_wins = 0
            for _ in range(episodes):
                game = game_manager.reset_game()
                while not game.is_over():
                    game.run_auto_turn()
                
                alive = game.get_alive_players()
                if any(p.id == 0 and p.alive for p in alive):
                    p1_wins += 1
            
            payoff_matrix[i][j] = p1_wins / episodes
    
    # Statistical test with proportions z-test + BH correction
    p_values = []
    comparisons = []
    
    # Pairwise strategy comparisons
    for i, j in combinations(range(len(strategies)), 2):
        # Compare strategy i vs j across all matchups
        i_wins = np.sum((payoff_matrix[0, i, :] * episodes).astype(int))
        j_wins = np.sum((payoff_matrix[0, j, :] * episodes).astype(int))
        total_games = episodes * len(strategies)  # Total games per strategy
        
        count = np.array([i_wins, j_wins])
        nobs = np.array([total_games, total_games])
        _, p_val = proportions_ztest(count, nobs)
        p_values.append(p_val)
        comparisons.append(f"{strategies[i]} vs {strategies[j]}")
    
    # Apply BH correction
    if p_values:
        rejected, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
        
        print(f"\nPairwise Strategy Comparisons (BH corrected):")
        any_significant = False
        for comp, p_orig, p_adj, is_sig in zip(comparisons, p_values, p_corrected, rejected):
            print(f"  {comp}: p = {p_orig:.4f}, p_adj = {p_adj:.4f} {'*' if is_sig else ''}")
            if is_sig:
                any_significant = True
        
        if any_significant:
            print("✓ Some strategies perform significantly differently")
        else:
            print("~ No significant differences between strategies")
        
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(payoff_matrix, 
               annot=True, 
               fmt='.3f',
               xticklabels=strategies,
               yticklabels=strategies,
               cmap='RdYlBu_r')
    plt.title('Player 1 Win Rate Matrix')
    plt.xlabel('Player 2 Strategy')
    plt.ylabel('Player 1 Strategy')
    
    # Save the figure with observation model type
    obs_model = create_game_objects()['observation_model']
    filename = f"strategy_heatmap_2p_{obs_model.name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot as {filename}")
    plt.show()

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Compute and plot payoff matrices.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes per combination (default: 1000)")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--model-path", type=str, help="Path to trained model checkpoint")
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config.load_config(args.config)
    else:
        config.load_config()
    
    # Define strategies to test
    test_strategies = [
        "RLStrategy",
        # "TargetStrongest",
        "TargetBelievedStrongest",
        "TargetRandom",
        # "TargetWeaker",
    ]
    
    cfg = config.get_config()
    print(f"Computing payoff matrix for {cfg['game']['num_players']} players with {args.episodes} episodes per combination...")
    
    if cfg['game']['num_players'] == 2:
        plot_strategy_heatmap(test_strategies, episodes=args.episodes, model_path=args.model_path)
    else:
        plot_payoff_matrix(test_strategies, episodes=args.episodes, model_path=args.model_path)