### READ THIS ###
    
Run 

conda env create -p [SomePath]\.conda -f environment/environment.yaml

then conda activate [SomePath]\.conda

replacing [SomePath]

It should install everything needed. This assumes Cuda12.4. Change as needed - or remove pytorch gpu altogether if needed.  

Tensorboard is installed as well to monitor training.

to run:
tensorboard --logdir [SomePath]\runs

### Install Ollama on cluster ###

Install Ollama:
`curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
sudo tar -C /usr -xzf ollama-linux-amd64.tgz`

### Current Status ###

PPO Algo taken from https://github.com/Chris-hughes10/simple-ppo, with a tutorial explaining everything very nicely by the same guy here https://medium.com/@chris.p.hughes10/understanding-ppo-a-game-changer-in-ai-decision-making-explained-for-rl-newcomers-913a0bc98d2b

Did few small changes to make it work for FrozenLake. Works out of box for cartpole and Cheetah. 

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