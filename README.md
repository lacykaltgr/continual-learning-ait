# Online continual learning agent

The   goal   of   the   project   is   to   explore   and   understand   the   challenge   that
continual learning means to modern artificial learning agents. 

Humans are great at learning online, and utilize the previously acquired knowledge in novel   tasks,   which   is   not   the   case   for   artificial  agents.   
In   recent   years many   works   focused   on   understanding   the   nature   of   catastrophic forgetting or continual task learning.   
Our goal is to understand the architectural principles that can support solving continual learning tasks.

* Thus, we define a convolutional neural network from scratch and train it for general image classification.  
* Then we split the training data to different tasks, 
* and train the model on the tasks according to different training regimes. 

The project aims to explore different techniques offered by the literature,
to   increase   performance   during   continual   learning.   
The   engineering principles that might lead to better performance could be freezing weights,
introducing memory, introducing replay (via a recurrent layer), introducing
inductive biases, applying meta-learning, and so on.

Among these, we investigated generative replay using a retrieval algorithm called Maximally Interfered Retrieval (MIR).
We implemented generative replay using a guided diffusion model.

Due to time and resource constraints, the model does not fully meet the requirements of complete online continual learning, as we utilized a partially pretrained generative architecture. However, we propose that with further advancements, it could be made fully online through just the expansion of the model.

# Instructions to use

The execution of the program is fairly intuitive; one simply needs to run all the cells until the Training section. Unfortunately, this part cannot be executed within a reasonable time frame without Google Colab Pro, as it consumes very much GPU memory. (Albeit at a significantly slower pace, the training can be run by setting the device to CPU within the params section.) Nevertheless, the results are available. Subsequently, an evaluation of the model can be performed, and the training process is analyzed using plots. 
There is also a section where image generation is possible and can be executed. 


This repository contains the following files:

- models/                # trained classifier model used for development
- guided_diffusion/      # Guided Diffusion package by OpenAI
- notebooks/             # notebooks used in development
- torch_utils/           # addition utils for loading pretrained model       
- classifier.py          # Classifier model
- data_preparation.py    # Function, classes to load datasets and split them into tasks
- generator.py           # Generator model
- utils.py               # Utility functions for training
- main.ipynb             # Main file for the experiment
      

# The architecture
The architecture utilizes the Generative Replay based MIR(Maximally Interferred Retrival) idea and a modified Stable Diffusion model.
![img.png](img.png)


# The team
Székely Anna,
Aryan Prabhudesai,
Freund László

