# Continual learning agent from scratch

The   goal   of   the   project   is   to   explore   and   understand   the   challenge   that
continual learning means to modern artificial learning agents. 

Humans are great at learning online, and utilize the previously acquired knowledge in
novel   tasks,   which   is   not   the   case   for   artificial   agents.   
In   recent   years many   works   focused   on   understanding   the   nature   of   
catastrophic forgetting () or continual task learning ().   
Our goal is to understand the architectural principles that can support solving continual learning tasks.
Thus, we define a convolutional neural network from scratch and train it for image classification.  
Then we split the training data to different tasks, and train the model on the tasks 
according to different training regimes (e.g. blocked and interleaved are
the two extremes and intermediate solutions also possible – blocked by 20– 40 – 60 sample batches of 1 task). 

The project aims to explore different techniques offered by the literature,
to   increase   performance   during   continual   learning.   
The   engineering principles that might lead to better performance could be freezing weights,
introducing memory, introducing replay (via a recurrent layer), introducing
inductive biases, applying meta-learning, and so on.
