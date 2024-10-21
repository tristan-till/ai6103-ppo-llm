from agent import ContinuousAgent, DiscreteAgent
from functools import partial

def get_agent_class(env_is_discrete, rpo_alpha):
    return (DiscreteAgent
        if env_is_discrete
        else partial(ContinuousAgent, rpo_alpha=rpo_alpha)
    )
    
def get_agent(envs, env_is_discrete, rpo_alpha, device):
    agent_class = get_agent_class(env_is_discrete, rpo_alpha)
    agent = agent_class(envs).to(device)
    return agent