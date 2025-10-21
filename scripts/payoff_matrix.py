import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import config.settings as config
from config.factories import create_game_objects, create_strategy
from core.game_manager import GameManager

def compute_payoff_matrix(strategies, episodes=1000):
    """Compute payoff matrix for given strategies via simulation"""
    n = len(strategies)
    n_combinations = n ** config.NUM_PLAYERS
    payoff_matrix = np.zeros((config.NUM_PLAYERS, n, n_combinations))
    
    # For N-player game, iterate through all strategy combinations
    for i, strat_combo in enumerate(product(range(n), repeat=config.NUM_PLAYERS)):
        print(f"Testing combination {i+1}/{n_combinations}: {[strategies[s] for s in strat_combo]}")
        
        # Create strategies for this combination
        strategy_objects = [create_strategy(strategies[s], create_game_objects()['observation_model']) 
                          for s in strat_combo]
        
        # Run simulation
        game_manager = GameManager(
            num_players=config.NUM_PLAYERS,
            gameplay=create_game_objects()['gameplay'],
            observation_model=create_game_objects()['observation_model'],
            max_rounds=config.NUM_ROUNDS,
            strategies=strategy_objects,
            has_ghost=config.HAS_GHOST
        )
        
        wins = [0] * config.NUM_PLAYERS
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
        for player_id in range(config.NUM_PLAYERS):
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
    
    fig, axes = plt.subplots(1, config.NUM_PLAYERS, figsize=(5*config.NUM_PLAYERS, 4))
    if config.NUM_PLAYERS == 1:
        axes = [axes]
    
    for player in range(config.NUM_PLAYERS):
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
        
        # Print interpretation
        best_strategy = strategies[np.argmax(avg_payoffs)]
        print(f"Player {player+1}: Best strategy is {best_strategy} (win rate: {np.max(avg_payoffs):.3f})")
    
    plt.tight_layout()
    
    # Save the figure with observation model type and number of players
    obs_model = create_game_objects()['observation_model']
    filename = f"payoff_matrix_{config.NUM_PLAYERS}p_{obs_model.name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot as {filename}")
    plt.show()

def plot_strategy_heatmap(strategies, episodes=1000):
    """Plot strategy vs strategy heatmap for 2-player case"""
    if config.NUM_PLAYERS != 2:
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
                has_ghost=config.HAS_GHOST
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
    
    parser = argparse.ArgumentParser(description="Compute and plot payoff matrices.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes per combination (default: 1000)")
    args = parser.parse_args()
    
    # Define strategies to test
    test_strategies = [
        # "TargetStrongest",
        "TargetBelievedStrongest",
        "TargetRandom",
        "RLlibStrategy"
    ]
    
    print(f"Computing payoff matrix for {config.NUM_PLAYERS} players with {args.episodes} episodes per combination...")
    
    if config.NUM_PLAYERS == 2:
        plot_strategy_heatmap(test_strategies, episodes=args.episodes)
    else:
        plot_payoff_matrix(test_strategies, episodes=args.episodes)