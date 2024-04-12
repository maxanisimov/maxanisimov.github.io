---
title: 'Introduction to Reinforcment Learning: Markov Process'
date: 2023-11-05
permalink: /posts/2023/11/intro-to-rl/markov-process
tags:
  - AI
  - Reinforcement Learning
  - Markov Process
---

One of the core concepts of basic reinforcement learning are a Markov Process and a Markov Reward Process.

**Definition 1**: A Markov process is a tuple ($S$, $P$), where $S$ is a state space and $P$ is a transition probability matrix.
Put it simply, a Markov process is a system which transitions from a state $s$ to a state $s'$ according to a conditional probability distribution $P(S_{t+1}=s' | S_t = s)$.

```python
class MarkovProcess:
    def __init__(self, state_space: list, transition_probability_matrix: numpy.ndarray) -> None:
        '''Markov Proces is a tuple (S, P), where S is a state space and P is a transition probability matrix.'''
        self.state_space = state_space
        self.transition_probability_matrix = transition_probability_matrix

        self.validate_inputs()

    def validate_inputs(self):

        ### Dimensions
        assert len(self.state_space) == len(set(self.state_space)), 'Some state names are not unique!'
        self.num_states = len(self.state_space)

        # Transition prob shape
        assert self.transition_probability_matrix.shape[0] == self.num_states
        assert self.transition_probability_matrix.shape[0] == self.transition_probability_matrix.shape[1]

        # Transition prob entries
        assert numpy.all(self.transition_probability_matrix <= 1)
        assert numpy.all(self.transition_probability_matrix >= 0)
        probs_sum = numpy.sum(self.transition_probability_matrix, axis=1)
        assert numpy.all(numpy.isclose(probs_sum, 1) + numpy.isclose(probs_sum, 0)), 'Sum of probabilities must be 0 (for terminal states) or 1 in each row!'

### Example
state_space = [f's{i}' for i in range(4)]
transition_probability_matrix = numpy.array(
    [
        [5/6, 1/6, 0,     0],
        [0 ,  1/6, 4/6, 1/6],
        [0 ,  0,   3/4, 1/4],
        [1,   0,   0,     0],
    ]
)
markov_process = MarkovProcess(state_space=state_space, transition_probability_matrix=transition_probability_matrix)

```

**Definition 2**: A Markov Reward Process (MRP) is a tuple ($S$, $P$, $R$, $\gamma$), where $S$ is a state space, $P$ is a transition probability matrix, R is reward vector denoting a reward of leaving each state and $\gamma$ is a discount factor that discounts rewards over time. 

In other terms, an MRP is a Markov Process augmented with the reward vector $R$ and the discount factor $\gamma$.
```python
class MarkovRewardProcess:
    def __init__(
        self, reward_vector: numpy.ndarray, discount_factor: numbers.Real,
        markov_process: MarkovProcess = None, state_space: list = None, transition_probability_matrix: numpy.ndarray = None, 
    ) -> None:
        '''
        A Markov Reward Process (MRP) is a tuple (S, P, R, gamma). Note: there is no action space!
        '''
        
        assert (markov_process is not None) or (state_space is not None and transition_probability_matrix is not None)
        if markov_process is None:
            self._markov_process = MarkovProcess(
                state_space=state_space, transition_probability_matrix=transition_probability_matrix
            ) # NOTE: this validates state space and TPM
        self.state_space = self._markov_process.state_space
        self.num_states = self._markov_process.num_states
        self.transition_probability_matrix = self._markov_process.transition_probability_matrix
        self.reward_vector = reward_vector
        self.discount_factor = discount_factor

        self.validate_inputs()

        # Results
        self.value_function = None

    def validate_inputs(self):

        # Reward vector shape
        if len(self.reward_vector.shape) == 2:
            assert self.reward_vector.shape[1] == 1
            self.reward_vector = self.reward_vector.reshape(-1,)
        assert len(self.reward_vector) == self.num_states

        ### Discount factor
        assert 0 <= self.discount_factor <= 1

    def calculate_value_function(self, convert_to_series: bool = False):
        identity_matrix = numpy.diag(numpy.ones(self.num_states))
        inv_matrix = numpy.linalg.inv(identity_matrix - self.discount_factor * self.transition_probability_matrix)
        self.value_function = inv_matrix @ self.reward_vector
        if convert_to_series:
            self.value_function = pandas.Series(data=self.value_function, index=self.state_space)
        return self.value_function                   

### Example
reward_vector = numpy.array(
    [70, 50, -20, -100]
)
discount_factor = 0.6
mrp = MarkovRewardProcess(
    state_space=state_space, transition_probability_matrix=transition_probability_matrix, reward_vector=reward_vector, discount_factor=discount_factor
)
mrp_value_function = mrp.calculate_value_function(convert_to_series=True)

```
