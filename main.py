import gym

from classes.agent import Agent
from utils.plot import plot_learning_curve
import utils.run as run
   
def cartpole():
    env = gym.make('CartPole-v0')
    batch_size = 5
    n_epochs = 4
    alpha = 0.01
    agent = Agent(n_actions = env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=env.observation_space.shape)
    
    N = 20
    n_games = 300
    indices, score_history = run.run_loop(env, agent, n_games, N)
    
    figure_file = 'plots/cartpole-v1.png'
    plot_learning_curve(indices, score_history, figure_file)

def frozenlake():
    env = gym.make('FrozenLake-v1')
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions = env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=(1,), gae_lambda=0.95, policy_clip=0.2)
    
    N = 3
    n_games = 1000
    indices, score_history = run.run_loop(env, agent, n_games, N)
    
    figure_file = 'plots/frozenlake-v1.png'
    plot_learning_curve(indices, score_history, figure_file)
    
def frozenlake_ppo_llm():
    env = gym.make('FrozenLake-v1')
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    
    input_dims = None # TODO: Init models with correct dimensions
    
    agent = Agent(n_actions = env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=input_dims, gae_lambda=0.95, policy_clip=0.2)
    
    N = 3
    n_games = 1000
    indices, score_history = run.run_ppo_llm_loop(env, agent, n_games, N)
    
    figure_file = 'plots/frozenlake-v1-ppollm.png'
    plot_learning_curve(indices, score_history, figure_file)
    
def main():
    cartpole()
    # frozenlake()
    # frozenlake_ppo_llm()

if __name__ == '__main__':
    main()
    
            
            
             