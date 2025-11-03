import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import config.config_loader as config
from config.factories import create_game_objects, create_strategy
from core.game_manager import GameManager
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportion_confint

def compute_payoff_matrix(strategies, episodes=1000):
    """Compute payoff matrix for given strategies via simulation"""
    cfg = config.get_config()
    n = len(strategies)
    n_combinations = n ** cfg['game']['num_players']
    payoff_matrix = np.zeros((cfg['game']['num_players'], n, n_combinations))
    
    # For N-player game, iterate through all strategy combinations
    for i, strat_combo in enumerate(product(range(n), repeat=cfg['game']['num_players'])):
        print(f"Testing combination {i+1}/{n_combinations}: {[strategies[s] for s in strat_combo]}")
        
        # Create strategies for this combination
        strategy_objects = [create_strategy(strategies[s], create_game_objects()['observation_model']) 
                          for s in strat_combo]
        
        # Run simulation
        game_manager = GameManager(
            num_players=cfg['game']['num_players'],
            gameplay=create_game_objects()['gameplay'],
            observation_model=create_game_objects()['observation_model'],
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

def plot_payoff_matrix(strategies, episodes=1000):
    """Plot payoff matrices for each player"""
    print("\n=== PAYOFF MATRIX INTERPRETATION ===")
    print("Higher values = better performance (higher win rate)")
    print("Each cell shows win rate (0.0 to 1.0) when using that strategy")
    print("Values averaged across all possible opponent combinations\n")
    
    payoffs = compute_payoff_matrix(strategies, episodes)
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
        
        # Statistical analysis
        win_counts = (avg_payoffs * episodes).astype(int)
        if len(win_counts) > 1 and sum(win_counts) > 0:
            expected = [sum(win_counts) / len(win_counts)] * len(win_counts)
            chi2, p_value = chi2_contingency([win_counts, expected])[:2]
            
            # Add p-value to plot
            sig_text = f"p = {p_value:.3f}" + (" *" if p_value < 0.05 else "")
            axes[player].text(0.02, 0.98, sig_text, transform=axes[player].transAxes, 
                            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
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

def plot_strategy_heatmap(strategies, episodes=1000):
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
            
            strategy_objects = [
                create_strategy(strat1, create_game_objects()['observation_model']),
                create_strategy(strat2, create_game_objects()['observation_model'])
            ]
            
            game_manager = GameManager(
                num_players=2,
                gameplay=create_game_objects()['gameplay'],
                observation_model=create_game_objects()['observation_model'],
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
    
    # Statistical test for strategy differences
    win_counts = (payoff_matrix * episodes).astype(int)
    if np.sum(win_counts) > 0:
        flat_wins = win_counts.flatten()
        expected = [np.mean(flat_wins)] * len(flat_wins)
        chi2, p_value = chi2_contingency([flat_wins, expected])[:2]
        print(f"\nStatistical test for strategy differences: p = {p_value:.4f}")
        if p_value < 0.05:
            print("✓ Strategies perform significantly differently")
        else:
            print("~ No significant difference between strategies")
    
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
    
    # Add statistical significance to plot
    if np.sum(win_counts) > 0:
        sig_text = f"p = {p_value:.3f}" + (" *" if p_value < 0.05 else "")
        plt.text(0.02, 0.98, sig_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Save the figure with observation model type
    obs_model = create_game_objects()['observation_model']
    filename = f"strategy_heatmap_2p_{obs_model.name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot as {filename}")
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute and plot payoff matrices.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes per combination (default: 1000)")
    args = parser.parse_args()
    
    # Define strategies to test
    test_strategies = [
        # "TargetStrongest",
        "TargetBelievedStrongest",
        "TargetRandom",
        # "RLlibStrategy"
        "TargetWeaker",
    ]
    
    cfg = config.get_config()
    print(f"Computing payoff matrix for {cfg['game']['num_players']} players with {args.episodes} episodes per combination...")
    
    if cfg['game']['num_players'] == 2:
        plot_strategy_heatmap(test_strategies, episodes=args.episodes)
    else:
        plot_payoff_matrix(test_strategies, episodes=args.episodes)