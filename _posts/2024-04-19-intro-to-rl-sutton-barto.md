---
title: 'Introduction to Reinforcment Learning'
date: 2024-04-22
permalink: /posts/2024/04/intro-to-rl/intro
tags:
  - AI
  - Reinforcement Learning
---
*Conspectus of Sutton & Barto "Reinforcement Learning: An Introduction"*

## 1.1 Reinforcement Learning
Reinforcement learning is learning what to do -- how to map situations to actions -- so as to maximise a numerical reward signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them. In the most interesting and challenging cases, actions may affect not only the immediate reward but also the next situation and, through that, all subsequent rewards. These two characteristics -- trial-and-error search and **delayed reward** -- are the two most important distinguishing features of reinforcement learning.

Sutton & Barto formalise the problem of RL using ideas from dynamcal systems theory, specifically, as the optimal control of incompletely-known Markov decision process.

How reinforcement learning is different from other types of learing:
- vs supervising learning (SL): in RL, we do not know the desired behaviour in advance, while in SL the desired outputs are given in training data.
- vs unsupervised learning (UL): RL is focused on maximising a reward signal instead of finding hidden structure, which is the goal of unsupervised learning.

One of the challenges that arise in reinforcement learning, and not in other kinds of learning, is the trade-off between exploration and exploitation. To obtain a lot of reward, an RL agent must prefer actions that it has tried in the past and found to be effective in producing reward. But to discover such actions, it has to try actions tht it has not selected before. The agent has to **exploit** what it has already experienced in order to obtain reward, but it also has to **explore** in order to make better action selections in the future. The dilemma is that neither exploration nor exploitation can be pursued exclusively without failing at the task.

Notably, the ability of some reinforcement learning methods to learn with parameterised approximators addresses the classical "curse of dimensionality" in operations research and control theory.


## 1.2 Examples
- A master chess player makes a move.
- An adaptive controller adjusts parameters of a petroleum refinery's operation in real time.
- A gazelle calf struggles to its feet minutes after being born. Half an hour later, it is running at 20 miles per hour.
- A mobile robot decides whether it should enter a new room in search of more trash to collect or start trying to find its way back to its battery recharging station.
- Phil prepares his breakfast.

These examples share features that are so basic that they are so easy to overlook. All involve interaction between an active decision-making agent and its environment, within which the agent seeks to achieve a goal despite uncertainty about its environment. The agent's actions affect the future state of the environment, thereby affecting the actions and opportunities availavble to the agent at later times. Correct choice requires taking into account indirect, delayed consequences of actions, and thus may require foresight or planning.

At the same time, in all of these examples the effects of actions cannot be fully predicted. Thus, the agent must monitor its environment feequently and react appropriately.

In all of these examples, the agent can use its experience to improve its performance over time. The knowledge the agent brings to the task at the start -- either from previous experience with related tasks or built into it by design or evolution -- influences what is useful or easy to learn, but interaction with the environment is essential for adjusting behaviour to exploit specific features of the task.

## 1.3 Elements of Reinforcement Learning
Beyond the agent and the environment, one can indetify 4 main subelements of a reinforcement learning system:
1) Policy $\pi$
2) Reward signal
3) Value function
4) Model of the environment (optional)

A **policy** $\pi$ defines the learning agent's way of behaving at a given time. Roughly speaking, a policy is a mapping from perceived states of the environment to actions to be taken when in those states. In some cases the policy may be a simple function or lookup table, whereas in others it may involve extensive computation such as a search process. The policy is the core of a reinforcement learning agent in the sense that it alone is sufficient to determine behaviour. In general, policies may be stochastic, specifying probabilities for each action.

A **reward signal** defines the goal of a reinforcement learning problem. On each time step, the environment sends to the RL agent a single number called the (instant) **reward** $r$. The agent's sole objective is to maximise the total reward it receives over the long run. In general, reward signals may be stochastic functions of the state of the environment and the actions taken.

Whereas the reward signal indicates what is good in an immediate sense, a **value function** specifies what is good in the long run. Roughly speaking, the value of a state is the total amount of reward an agent can expect to accumulate over the future, starting from that state. Values indicate the long-term desirability of states after taking into account the states that are likely to follow and the rewards available in those states. For example, **a state might always yield a low immediate reward but still have a high value because it is regularly followed by other states that yield high rewards**. Or the reverse could be true.

**It is values with which we are most concerned when making and evaluating decisions**. Action choices are made based on value judgements. **We seek actions that bring about states of highest value, not highest reward, because these actions obtain the greatest amoutn of reward for us over the long run**. Unfortunately, it is much harder to determine values than it is to determine rewards. Rewards are basically given directly by the environment, but values must be estimated and re-estimated fro the sequences of observations an agent makes over its entire lifetime. In fact, **the most important component of almost all reinforcement learning algorithms we consider is a method for efficiently estimating values**.

The final element of some reinfrocement learning systems is a model of the environment. This is something that allows inferences to be made about how the environment will behave. For example, given a state and action, the model might predict the resultant next state and next reward. Models are used for **planning**, by which we mean any way of deciding on a course of action by considering possible future situations before they are actually experienced. Methods for solving RL problems that use models and planning are called **model-based methods**, as opposed to simpler **model-free methods** that are explicitly trial-and-error learniners -- viewed as almost the opposite of planning. 

## 1.4 Limitations and Scope
RL relies heavily on the concept of state -- as input to the policy and value function, and as both input to and output from the model. Informally, we can think of the state as a signal conveying to the agent some sense of "how the environment is" at a particular time. The concern of this book is not with constructing, designing, changing or learning the state signal, but wtih deciding what action to take as a function of whatever state signal is available.

Most of the RL methods we consider in this book are structured around estimating value functions, but it is not strictly necessary to do this to solve RL problems. For example, solution methods such as **genetic algorithms, genetic programming, simulated annealing**, and other optimisation methods never estimate value functions. These methods apply multiple static policies, each interacting over an extended period of time with a separate instance of the environment. The policies that obtain the most reward, and random variations of them, are carried over to the next generation of policies, and the process repeats. We call these **evolutionary methods** because their operation is analogous to the way biological evolution produces organisms with skilled behaviour even if they do not learn during their individual lifetimes. If the space of policies is sufficiently small, or can be structured so that good policies are common or easy to find -- or if a lot of time is available for the search -- then evolutionary methods can be effective. Besides, evolutionary methods have advantages on problems in which the learning agent cannot sense the complete state of its environment. 

The focus of this book is on RL methods that learn while interacting with the environment, which evolutionary methods do not do. Methods able to take advantage of the details of individual behavioral interactions can be much more efficient than evolutionary methods in many cases. Evolutionary methods ignore much of the useful structure of the RL problem; they 
-  do not use the fact that the policy they are searching for is a function from states to actions;
- do not notice which state the agent passes through during its lifetime;
do not notice which acitons the agent selects.

Athough evolution and learning share many features and naturally work together, we do not consider eveolutionary methods by themselves to be especially well-suited to RL problems and, accordingly, we do not cover them in this book.

## 1.5 An Extended Example: Tic-Tac-Toe
The minimax approach from game theory assumes a particular way of playing by the opponent, usually optimal. Therefore, a minimax player would never reach a game state from which it could lose, even if in fact it always won from that state because of incorrect play by the opponent. Classical optimisation methods for sequential decision problems, such as dynamic programming, can compute an optimal solution for any opponent, but require as input a complete specification of that opponent, including the probabilities with which the opponent makes each move in each board state. 

Let us assume that the opponent information is not available a priori for this problem, as it is not for the vast majority of problems of practical interest. On the other hand, such informatino can be estimated from experience, in this case by playing many games against the opponent. About the best one can do on this problem is to 
1) Learn a model of the opponent's behaviour, up to some level of confidence, 
2) And then apply dynamic programming to compute an optimal solution given the approximate opponent model.
In the end, this is not that different from some of the RL methods we examine later in this book!

An evolutionary method applied to this problem would directly search the space of possible policies for one with a high probability of winning against the opponent. Here, a policy is a rule that tells the player what move to make for every state of the game -- every possible confiuration of Xs and Os o nthe three-by-three board. For each policy considered, an estimate of its winning probability would be obtained by playing some number of games against the opponent. This evaluation would then direct which policy or policies were considered next. A typical evolutionary method would hill-climb in policy space, successively generating and evaluating policies in an attempt to obtain incremental improvements. Or, perhaps, a genetic-style algorithm could be  used that would maintain and evaluate a population of policies. Literlly hundreds of different optimisation methods could be applied. 

Here is how the tic-tac-toe problem would be approached with a method making use of a value function:
1. Set up a table of numbers, one for each possible state of the game. Each number will be the latest estimate of the probability of our winning from that state. We treat this estimate as the state's value, and the whole table is the learned value function. Assuming we always play Xs, for all states with three Xs in a row the probability of winning is 1, because we have already won. Similarly, for all states with three Os in a row, or that are filled up, the correct probability is 0, as we cannot win from them. We set the inital values of all the other states to 0.5, representing a guess that we have a 50% chance of winning.
2. Play many games against the opponent. To select our moves, we examine the states that would result from each of our possible moves and look up  their current values in the table. Most of the time we move **greedily**, selecting the move that leads to the state with greatest value, that is, with the highest estimated probability of winning (**exploitation**). Ocassionally, however, we select randomly from among the other moves instead. These are called **exploratory** moves because they cause us to experience states that we might otherwise never see. 
3. While we are playing, we change the values of the states in which we find ourselves during the game. We attempt to make them more accurate estimates of the probabilities of winning. To do this, we "back up" the value of the state after each greedy move to the state before the move -- the current value of the earlier state is updated to be closer to the value of the later state. This can be done by moving the earlier state's value a fraction of the way towards the value of the later state. Let $S_t$ denote the state before the greedy move, and $S_{t+1}$ the state after the move. Then the update to the estimated value of $S_t$, denoted $V(S_t)$, can be written as
$$V(S_t) \leftarrow V(S_t) + \alpha [V(S_{t+1}) - V(S_t)],$$
where $\alpha$ is a small positive fraction called the **step-size** parameter or **learning rate**. This update rule is an example of a **temporal-difference learning** method, so called ecause its changes are based on a difference between estimates at two succesive times $V(S_{t+1}) - V(S_t)$.

If the step-size parameter is reduced properly over time, this method converges, for any fixed opponent, to the true probabilities of winning from each state given optimal play by our player. Furthermore, the greedy moves (not exploratory moves!) are the optimal moves against this (potentially imperfect) opponent. If $\alpha$ does not converge to 0 over time, this player also plays well against opponents that slowly change their way of playing.

### Evolutionary vs value function methods
1. Evolutionary methods hold the policy fixed and play many games against the opponent, or simulate many games using a model of the opponent. The frequency of wins gives an unbiased estimate of the probability of winning with that policy, and can be used to direct the next policy selection. However, each policy change is made only after many games, and only the final outcome of each game is used. That is, **what happens during traning is ignored**. For example, if the player wins, then all of its behaviour in the game is given credit, independently of how specific moves might have been critical to the win. Credit is even given to moves that never occured (?). 
2. Value function methods allow indicidual states to be evaluated.

Both methods search the space of policies, but learning a value function takes advantage of information available during the course of play.

This example illustrates some of the key features of RL methods:
- There is the emphasis on learning while interacting with an environment (in this case with an opponent player).
- There is a clear goal, and correct behaviour requires planning or foresight that takes into account delayed effects of one's choices. E.g. the simple RL player would learn to set up multi-move traps for a shortsighted opponent.

The general principles of RL apply to continuous-time problems as well, although the theory gets more complicated and we omit it from this introductory treatment.

The artificial neural network (ANN) provides the program with the ability to generalise from its experience, so that in new states it selects moves based on information saved from similar states faces in the past, as determined by its network. How well an RL system can work in problems with large state sets is tied to how appropriately it can generalise from past experience. It is in this role that we have the greatest need for supervised learning methods with RL. ANNs and deep learning are not the only, or necessarily the best, way to do this.

### Model-free vs model based RL
Because models of the environment have to be reasonably accurate to be useful, model free methods can have advantages over more complex methods when the real bottleneck in solving a problem is the difficulty of constructing a sufficiently accurate environment model. Model-free methods are also important building blocks for model-based methods. 

### *Exercises*

*Exercise 1.1: Self-Play* Suppose, instead of playing against a random opponent, the reinforcement learning algorithm described above played against itself, with both sides learning. What do you think would happen in this case? Would it learn a different policy for selecting moves?

*Exercise 1.2: Symmetries* Many tic-tac-toe positions appear di↵erent but are really the same because of symmetries. How might we amend the learning process described above to take advantage of this? In what ways would this change improve the learning process? Now think again. Suppose the opponent did not take advantage of symmetries. In that case, should we? Is it true, then, that symmetrically equivalent positions should necessarily have the same value?

*Exercise 1.3: Greedy Play* Suppose the reinforcement learning player was greedy, that is, it always played the move that brought it to the position that it rated the best. Might it learn to play better, or worse, than a nongreedy player? What problems might occur?


*Exercise 1.4: Learning from Exploration* Suppose learning updates occurred after all moves, including exploratory moves. If the step-size parameter is appropriately reduced over time (but not the tendency to explore), then the state values would converge to a di↵erent set of probabilities. What (conceptually) are the two sets of probabilities computed when we do, and when we do not, learn from exploratory moves? Assuming that we do continue to make exploratory moves, which set of probabilities might be better to learn? Which would result in more wins?

*Exercise 1.5: Other Improvements* Can you think of other ways to improve the reinforce- ment learning player? Can you think of any better way to solve the tic-tac-toe problem as posed?



## 1.6 Summary
Reinforcement learning is a computational approach to understanding and automating goal-directed learning and decision making. It is distinguished from other computational approaches by its emphasis on learning by an agent from direct interaction with its environment, without requiring exemplary supervision or complete models of the environment.

RL uses the formal framework of Markov decision processes to define the interaction between a learning agent and its environment in terms of states, actions, and rewards. This framework is intended to be a simple way of representing essential features of the AI problem. including **cause and effect**, a sense of uncertainty and nondeterminism, and the existence of explicit goals.

The concepts of value and value function are key to the most RL methods considered in this book. The authors take the position that value functions are important for efficient search in the space of policies. The use of value functions distinguishes RL methods from evolutionary methods that search directly in policy space guided by evaluations of entire policies.


## 1.7 Histroy of Reinforcement Learning
Two main threads that were pursued independently before interwining in modern RL:
1. Learning by trial and error, originating in the psychology of animal learning
2. Optimal control, using value functions and dynamic programming (mostly no relation to learning!)

Third, less distinct thread: temporal-difference methods. All three threads came together in the late 1980s to produce the modern field of RL as it is presented in this book.

