- Train DNN ( Deep neural network ) to play snake;
- 30x30 board with 1-pixel border;
- Board image will be 32x32 RGB image;

//Objective
- Provide hands-on experience in RL;
- Experiment different scenarios, approaches and what helps the agents learning and how deep reinforcement learning works;
- Even though the scenario shown is a 30x30 board with one food, no grass and one-tile border, we can then try other scenarios ( Provavelmente da mais nota )

//Game and scenarios
- Reset() -> Starts a new game;
- step(action) -> To make a move;
- action can be:
	- -1 -> snake goes left;
	- 0 -> snake goes the same direction;
	- 1 -> snake goes right.

//Parameters chosen here:

 def __init__(self,
      width,               # width of free region of the board
      height,              # height of the free region of the board
      food_amount=1,       # number of food items present in the board
      border = 0,          # gray border added outside the board
      grass_growth = 0,    # how much the grass grows per step in the board
      max_grass = 0        # maximum amount of grass per location

- When the snake enters a cell, it receives a reward equal to the amount of grass present in that cell. To encourage exploration, you can set the amount of grass to 0 after it is eaten, using a small amount such as 0.05 of the maximum grass;
- To grow grass slowly, use grass_growth parameter.

//Simpler scenarios
- By adjusting the border and size of the free region of the board, you can create simpler scenarios. For instance:
-> game = SnakeGame(width=14, height = 14, border = 1) : The game will have a board with 14x14 free locations surrounded by a gray border of 1 cell, resulting in a 16x16 image. This size, being a power of two, is practical for convolution and pooling

- Since training the agent can take some time (minutes or hours), you may opt for smaller boards or variants of the game. For example, here is an agent trained to play the game with 0.05 of maximum grass.

The `step(action)` method returns the following four values:

- The state of the board after the action is taken, represented by a matrix with the image of the board.
- The reward for the last action and state transition.
- A flag with a value of True if the state reached is a terminal state (the snake died), or False otherwise.
- A dictionary with the total score so far.

The `reset()` method returns the same values, but the board state is the initial state and the reward is zero.

//Better examples

Playing the game randomly may not result in many good examples for the agent to learn. To improve this, you can use a set of experiences taken from games played by other agents. For instance, gathering many games played by humans could enrich the training data for your agent.

However, it may not be practical to gather so many games played by humans. Instead, you can write your own code to play the game following some heuristic. For example, the video below shows some games being played by a heuristic that tries to go straight for the food. Although this is often suicide, it's better than playing randomly.

If you want to enhance your pool of examples with some heuristic, you can use the `get_state()` method of the snake game object to cheat and get the coordinates of the snake, its direction, and the coordinates of the food items. The values returned by `get_state()` are:

- The current total score.
- A list with the coordinates of the food sources (the list has only one item for the default of only one piece of food at a time).
- The coordinates of the head of the snake.
- A list with the coordinates for the tail segments of the snake.
- The current direction of movement, an integer from 0 through 3 indicating, in order, N, E, S, W.

Note that this information should not be used by your agent. **The agent must learn solely from the images depicting the board state**. However, you can use this information to generate examples to start the training.

//Deep Q-Network
-> receives a state s as input and outputs the Q-value Q(s, a) for every action a. This allows the network to act as a regular table, where you can look up the Q-value using the state as the input key, and observe the produced values as outputs for every action. Typically, DQN is trained using experience replay and a target network.

Implement the deep Q learning agent using Pytorch, and perform an ablation study. Consider the architecture of your network: do you need convolutions? How many layers? How many nodes per hidden layer? You may want to experiment with various architectures and tune hyper-parameters for exploration, and report on these results to explain your final choice.

//Exploration Strategies
- tradeoff between exploration and exploitation has been extensively studied in the literature.  Many strategies exist. We have discussed e-Greedy where the chosen action 'a' is the maximizer of que Q-value function with probability 1-e or a random action taken uniformly from the set 'A' of possible actions:
	-> a(s) = argMax Qpi(a,s) with probability 1-e
	-> a(s) = randomUniform(A) with probability e
-> The probability of exploration e is, in general, decreased according to a decay pattern we saw in class. But there are other possibilities. For example, you could investigate other relevant distributions, namely depending on the current state s, or the states already seen so far. Or you could try a constant e throughout training.  You should implement, test and analyze two exploration strategies.

//Experience Replay
- Maintaining a replay buffer allows us to reuse collected data multiple times. Additionally, sampling batches randomly from the buffer breaks the correlation between consecutive data, which can make training more stable. Implementing the replay buffer enables experience replaying. Implement a replay buffer for your agent.

//Target Network
- During the update process, we attempt to push the Q values towards the target Q values, which are the immediate reward plus the bootstrapped Q values. Since a single update affects the entire network, it causes the non-stationarity of the target Q values. To make the target Q values stationary, we can use another network to provide the target Q values. We can periodically update this network using the weights of the original network. This is known as the target network, and you will need to implement it.

//Summary
Overall, for this assignment, you are required to:

- Implement a DQN agent with the following features:
– Different exploration strategies (at least two)
– Experience replay (training with a replay buffer)
– Target network (using another network to provide target updates).
- Tune hyper-parameters:
– Network architecture (number of layers, number of neurons)
– Learning rate
– Exploration factor 
– Other parameters you want to experiment with
- - Ablation Study: Compare different models in terms of learning speed, performance, and stability. In each comparison, a component of the model is removed (either experience replay (ER), target network (TN), or both). The following comparisons will be made:
    - DQN vs. DQN-ER
    - DQN vs. DQN-TN
    - DQN vs. DQN-ER-TN

It is important to keep other parameters fixed when performing the ablation study.

//BONUS POINTS
If you have successfully completed the tasks above and understand the methods and processes, you will do well on the assignment. If you wish to go beyond the expected requirements, there are a couple of options for you:

- Learn a model to predict the reward and use it to dream about future states in the game, as an exploration strategy. Can this help safety?
- Learn a model that can predict the next state of the board and use the embeddings as extra features.
- You may come across other methods such as the Double DQN or the Dueling DQN. Try one or both and compare their performance in terms of reward and training stability with the Vanilla DQN. Describe the method and your analysis of the results

//COMPUTE
- Use alakazam clusters
