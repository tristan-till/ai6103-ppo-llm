### READ THIS ###
    
The current code will still throw an error, as the gym version uses an outdated numpy library.

Go to ".../venv/Lib/site-packages/gym/utils/passive_env_checker.py" and swap:

np.bool8 -> np.bool

in lines: 225, 233, 237

If anyone can be bothered to research a compatible numpy version, which doesnt crash the rest of
the code, please do. Before submitting the project this should be fixed!!


### Install Ollama on cluster ###

Install Ollama:
`curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
sudo tar -C /usr -xzf ollama-linux-amd64.tgz`



### Current Status ###

This codebase is designed to work on the CartPole environment (you can test that as well), so the PPO algorithm is "correct" in theory. There are still some issues with convergence using the FrozenLake environment, i'm assuming this is due to the 1-dim input + discriminate output and/or resulting issues with exploration or gradient calculation, so don't worry if the algorithm does not converge for you in the current setting, it didn't for me either.

our main focus should be on the ppo-llm portion, where i have proposed a general structure we could follow, but feel free to put your own spin on it as well. suggestions are made purely on the basis of them seeming the easiest/most effective way to do things after some quick research.

* Identify a sensible fixed-length input, which can be fed to the model's actor and critic networks. Maybe the paper proposes a solution to this already!
* Enhance the current PPO. The original paper references a really interesting/well-written blog post, which we can follow!
* Get LLAVA and LLAMA up and running as local API-endpoints, which should be way quicker and cheaper than using the actual APIs. Ollama seems very convenient, but whoever decides to implement this feel free to choose whatever you feel most comfortable with
* The original paper links a study to sentence transformers, which should be publicly available as well. If we want consistency in the image-to-text pipeline we could launch this as a separate API-endpoint as well, seems like a waste though. Also, i'm 99% certain there are pretrained sentence transformers, which should do the job very well, finetuning our own could be an interesting research project afterwards as well


### Developer guidelines ###

To make this semi-coordinated, please stick to the following as best as possible:

* snake_case for code, UpperCamelCase for classes, kebab-case for branches
* Try be as clean as possible (or do so before a commit/rebase). Avoid nonsense variable names, try keep complexity of individual code-snippets as easily maintainable as possible
* Create an issue for whatever you are working on, reference the issue number in your branch-name (1-my-branch) and commit message "[\#1] Add initial files"
* Please work in separate branches and **rebase** regularly. I know this is always very unpopular (and if all of you have a strong opposing opinion, then I won't insist on this) but if possible, **avoid merging** your branches and use rebase instead. Can explain why this is beneficial for the git-Tree if you want. (git checkout my-branch)
* Add whatever else you want in here as well!