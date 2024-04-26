---
title: 'Multi-armed Bandits: Code'
date: 2024-04-26
permalink: /posts/2024/04/intro-to-rl/multi-armed-bandits-code
tags:
  - AI
  - Reinforcement Learning
  - Multi-armed Bandits
---

# Chapter 2: Multi-armed Bandits


```python
import numpy, pandas
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')

from typing import Union
import numbers
import collections
import plotly.express as px

%load_ext autoreload
%autoreload 2
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload


## 2.1. A $k$-armed Bandit Problem

### Theory
The value of an arbitrary action $a$, denoted as $q_{*}(a)$, is the expected reward given that $a$ is selected:
$$q_{*}(a) = E[R_t|A_t=a]$$
The value above is the theoretical (ground truth) value. We denote the estimated value of action $a$ at time step $t$ as $Q_t(a)$.

## 2.2. Action-value methods

### Exercises
*Exercise 2.1. In $\epsilon$-greedy action selection, for the case of two actions and $\epsilon$=0.5, what is the probability that the greedy action is selected?*

From the book:<br>
"... simple alternative is to behave greedily most of the time, but every once in a while, say with small probability $\epsilon$, select randomly from among all the actions with equal probability, independently of the action-value estimates."

$$P(\text{greedy action is selected}) = P(\text{greedy action is selected} \cap \text{greedy selection}) + P(\text{greedy action is selected} \cap \text{random selection}) = $$
$$ = P(\text{greedy action is selected} | \text{greedy selection}) * P(\text{greedy selection}) + P(\text{greedy action is selected} | \text{random selection}) * P(\text{random selection}) = $$
$$ = 1 * (1 - \epsilon) + 0.5 * \epsilon $$

Therefore:
$$P(\text{greedy action is selected}) = 1 - 0.5 \epsilon = 1 - 0.5 * 0.5 = 0.75$$

## 2.3. The 10-armed Testbed

### Theory

"To roughly assess the relative effectiveness of the greedy and $\epsilon$-greedy action-value methods, we compared them numerically on a suite of test problems. This was a set of 2000 randomly generated $k$-armed bandit problems with $k=10$. For each bandit problem, such as the one shown in Figure 2.1, the action values, $q_{*}(a)$, $a = 1, . . . , 10$, were selected according to a normal distribution with mean $0$ and variance $1$."


```python
numpy.random.seed(2024)
```


```python
q_ground_truths = numpy.random.normal(loc=0.0, scale=1.0, size=10)
q_ground_truths
```




    array([ 1.66804732,  0.73734773, -0.20153776, -0.15091195,  0.91605181,
            1.16032964, -2.619962  , -1.32529457,  0.45998862,  0.10205165])



#### *Figure 2.1*
"Then, when a learning method applied to that problem selected action $A_t$ at time step $t$, the actual reward, $R_t$, was selected from a normal distribution with mean $q_{*}(A_t)$ and variance $1$."


```python
num_sim = 2_000

actual_rewards = []
for qgt in q_ground_truths:
    cur_rewards = numpy.random.normal(loc=qgt, scale=1.0, size=num_sim)
    actual_rewards.append(cur_rewards)
```


```python
sns.violinplot(pandas.DataFrame(actual_rewards).T)
plt.axhline(0.0, color='black', ls='--')
plt.xlabel('Action #')
plt.title('Reward distribution')

plt.tight_layout();
```


    
![png](chapter2-multi-armed-bandits/output_12_0.png)
    


#### *Full Simulation (Figure 2.2)*


```python
def argmax_multiple(array: numpy.array) -> Union[numpy.array, int]:
    '''Return indices of all maximum values in the array.'''
    max_indices = numpy.argwhere(array == numpy.max(array)).flatten()
    if len(max_indices) == 1:
        return max_indices[0]
    return max_indices

def greedy_policy(Q_values: numpy.ndarray) -> int:
    max_idx = argmax_multiple(array=Q_values)
    if isinstance(max_idx, numpy.ndarray):
        max_idx = numpy.random.choice(max_idx)
    return max_idx

def epsilon_policy(num_states: int) -> int:
    return numpy.random.choice(num_states)

def epsilon_greedy_policy(Q_values, epsilon: numbers.Real):
    use_epsilon_policy = numpy.random.choice([True, False], p=[epsilon, 1-epsilon])
    if use_epsilon_policy:
        num_states = len(Q_values)
        return epsilon_policy(num_states=num_states)
    else:
        return greedy_policy(Q_values=Q_values)
```


```python
class GaussianBandit:
    def __init__(self, reward_distribution_params: list[dict], q_initial_values: numpy.ndarray = None, random_seed: int = 2024) -> None:
        '''A simple bandit algorithm with Gaussian coditional reward distribution (see "A simple bandit algorithm" table in 2.4).'''
        self.num_actions = len(reward_distribution_params)
        self.reward_distribution_params = reward_distribution_params

        self.Q = None
        self.N = None
        self.actions = None
        self.rewards = None

        self.q_initial_values = q_initial_values
        
        numpy.random.seed(random_seed)

    def _initialise(self) -> None:
        '''Initialise estimates of Q-value functions and numbers of observations per action.'''
        if self.q_initial_values is None:
            self.Q = numpy.zeros(self.num_actions)
        else:
            self.Q = self.q_initial_values
        self.N = numpy.zeros(self.num_actions)
        self.actions = []
        self.rewards = []
        
    def _call_bandit(self, action_num: int) -> numbers.Real:
        '''Sample a reward from its conditional distribution reward|action=a.'''
        assert 0 <= action_num <= self.num_actions - 1
        cur_param_dct = self.reward_distribution_params[action_num]
        return numpy.random.normal(size=1, **cur_param_dct)[0]

    def learn(self, epsilon: numbers.Real, max_iter: int = 1_000, learning_rate: numbers.Real = None) -> numpy.ndarray:
        if self.Q is None or self.N is None or self.actions is None or self.rewards is None:
            self._initialise()
        
        if learning_rate is None:
            lr_type = '1/N'
        else:
            lr_type = 'custom'
        
        iter_num = 0
        while iter_num <= max_iter:
            iter_num += 1
            cur_action = epsilon_greedy_policy(Q_values=self.Q, epsilon=epsilon)
            self.actions.append(cur_action)
            cur_reward = self._call_bandit(action_num=cur_action)
            self.rewards.append(cur_reward)

            # Update
            self.N[cur_action] = self.N[cur_action] + 1
            if lr_type == '1/N':
                learning_rate = 1 / self.N[cur_action]
            # else: use what the user passed as a learning rate
            self.Q[cur_action] = self.Q[cur_action] + learning_rate * (cur_reward - self.Q[cur_action])

        return self.Q
```


```python
num_bandits = 2_000
num_time_steps = 1_000
epsilon = 0.1
num_actions = 10
epsilon_values = [0, 0.01, 0.1]

q_ground_truths = []
Q_estimates = {ev: [] for ev in epsilon_values}
actions = {ev: [] for ev in epsilon_values}
rewards = {ev: [] for ev in epsilon_values}
optimal_action_flags = {ev: [] for ev in epsilon_values}

for i in range(num_bandits): # create independent bandits
    if i%100 == 0:
        print(f'Bandit #{i}')
    # Generate ground truths for the value function
    numpy.random.seed(i)
    cur_q_ground_truths = numpy.random.normal(loc=0.0, scale=1.0, size=num_actions)
    cur_reward_distribution_params = [
        {'loc': mean, 'scale': 1}
            for mean in cur_q_ground_truths
    ]
    q_ground_truths.append(cur_q_ground_truths)

    for epsilon in epsilon_values:
        if i%100 == 0:
            print(f'epsilon={epsilon}') 
        
        # Run the bandit
        gaussian_bandit = GaussianBandit(
            reward_distribution_params=cur_reward_distribution_params, 
            random_seed=i # NOTE: it is important to change seed to make sure that bandits are different!
        )
        cur_Q_est = gaussian_bandit.learn(epsilon=epsilon, max_iter=num_time_steps)

        # Store results
        Q_estimates[epsilon].append(cur_Q_est)
        actions[epsilon].append(gaussian_bandit.actions)
        rewards[epsilon].append(gaussian_bandit.rewards)
        
        # Calculate the % of optimal actions and store it
        cur_best_action = numpy.argmax(cur_q_ground_truths)
        optimal_action_flags[epsilon].append([a == cur_best_action for a in gaussian_bandit.actions])
```

    Bandit #0
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #100
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #200
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #300
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #400
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #500
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #600
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #700
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #800
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #900
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #1000
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #1100
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #1200
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #1300
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #1400
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #1500
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #1600
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #1700
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #1800
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #1900
    epsilon=0
    epsilon=0.01
    epsilon=0.1



```python
bandit_num = 0

pandas.DataFrame(
    {
        'True Q-value': q_ground_truths[0],
        'Estimated Q-value ($\epsilon$=0)': Q_estimates[0][bandit_num],
        'Estimated Q-value ($\epsilon$=0.01)': Q_estimates[0.01][bandit_num],
        'Estimated Q-value ($\epsilon$=0.1)': Q_estimates[0.1][bandit_num]
    }
)
```

    <>:6: SyntaxWarning: invalid escape sequence '\e'
    <>:7: SyntaxWarning: invalid escape sequence '\e'
    <>:8: SyntaxWarning: invalid escape sequence '\e'
    <>:6: SyntaxWarning: invalid escape sequence '\e'
    <>:7: SyntaxWarning: invalid escape sequence '\e'
    <>:8: SyntaxWarning: invalid escape sequence '\e'
    /var/folders/c3/4d4hvn_n173cnw7lk14cms2m0000gn/T/ipykernel_72002/941066188.py:6: SyntaxWarning: invalid escape sequence '\e'
      'Estimated Q-value ($\epsilon$=0)': Q_estimates[0][bandit_num],
    /var/folders/c3/4d4hvn_n173cnw7lk14cms2m0000gn/T/ipykernel_72002/941066188.py:7: SyntaxWarning: invalid escape sequence '\e'
      'Estimated Q-value ($\epsilon$=0.01)': Q_estimates[0.01][bandit_num],
    /var/folders/c3/4d4hvn_n173cnw7lk14cms2m0000gn/T/ipykernel_72002/941066188.py:8: SyntaxWarning: invalid escape sequence '\e'
      'Estimated Q-value ($\epsilon$=0.1)': Q_estimates[0.1][bandit_num]





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>True Q-value</th>
      <th>Estimated Q-value ($\epsilon$=0)</th>
      <th>Estimated Q-value ($\epsilon$=0.01)</th>
      <th>Estimated Q-value ($\epsilon$=0.1)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.764052</td>
      <td>0.000000</td>
      <td>1.829986</td>
      <td>1.305294</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.400157</td>
      <td>0.000000</td>
      <td>2.091589</td>
      <td>1.004728</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.978738</td>
      <td>0.000000</td>
      <td>2.064829</td>
      <td>1.083612</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.240893</td>
      <td>2.231516</td>
      <td>2.220248</td>
      <td>2.234172</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.867558</td>
      <td>0.000000</td>
      <td>1.424770</td>
      <td>1.966103</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.977278</td>
      <td>-0.858781</td>
      <td>-1.011384</td>
      <td>-1.099267</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.950088</td>
      <td>0.000000</td>
      <td>1.408015</td>
      <td>0.180388</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.151357</td>
      <td>0.000000</td>
      <td>-0.030234</td>
      <td>-0.060630</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.103219</td>
      <td>0.000000</td>
      <td>0.086194</td>
      <td>0.270738</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.410599</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.209698</td>
    </tr>
  </tbody>
</table>
</div>




```python
avg_rewards = pandas.DataFrame(
    {eps: pandas.DataFrame(rewards[eps]).mean(axis=0) for eps in rewards}  
)
```


```python
avg_rewards.plot(figsize=(20,5), color={0: 'tab:green', 0.01: 'tab:red', 0.1: 'tab:blue'})
plt.legend(title='$\epsilon$ value', loc='upper left')
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.tight_layout();
```

    <>:2: SyntaxWarning: invalid escape sequence '\e'
    <>:2: SyntaxWarning: invalid escape sequence '\e'
    /var/folders/c3/4d4hvn_n173cnw7lk14cms2m0000gn/T/ipykernel_72002/1870126046.py:2: SyntaxWarning: invalid escape sequence '\e'
      plt.legend(title='$\epsilon$ value', loc='upper left')



    
![png](chapter2-multi-armed-bandits/output_19_1.png)
    



```python
px.line(avg_rewards, template='plotly_white').update_layout(
    legend=dict(
        title="epsilon value",
        y=0.99,
        x=0.01,
        orientation="h"
    ),
    xaxis=dict(title='Steps'),
    yaxis=dict(title='Average reward')
)
```




```python
optimal_action_props = pandas.DataFrame(
    {eps: pandas.DataFrame(optimal_action_flags[eps]).mean(axis=0) for eps in optimal_action_flags}  
)
```


```python
(100 * optimal_action_props).plot(figsize=(20,5), color={0: 'tab:green', 0.01: 'tab:red', 0.1: 'tab:blue'})
plt.legend(title='$\epsilon$ value', loc='upper left')
plt.xlabel('Steps')
plt.ylabel('Optimal action, %')
plt.tight_layout();
```

    <>:2: SyntaxWarning:
    
    invalid escape sequence '\e'
    
    <>:2: SyntaxWarning:
    
    invalid escape sequence '\e'
    
    /var/folders/c3/4d4hvn_n173cnw7lk14cms2m0000gn/T/ipykernel_72002/206306060.py:2: SyntaxWarning:
    
    invalid escape sequence '\e'
    



    
![png](chapter2-multi-armed-bandits/output_22_1.png)
    



```python
px.line(optimal_action_props, template='plotly_white').update_layout(
    legend=dict(
        title="epsilon value",
        y=0.99,
        x=0.01,
        orientation="h"
    ),
    xaxis=dict(title='Steps'),
    yaxis=dict(title='Optimal action', tickformat = ',.0%')
)
```



### Exercises

*Exercise 2.2. Bandit example Consider a $k$-armed bandit problem with $k=4$ actions, denoted $1$, $2$, $3$, and $4$. Consider applying to this problem a bandit algorithm using $\epsilon$-greedy action selection, sample-average action-value estimates, and initial estimates of $Q_1(a) = 0 \: \forall a$.<br>
Suppose the initial sequence of actions and rewards is $A_1=1$, $R_1=1$, $A_2=2$, $R_2=1$, $A_3=2$, $R_3=2$, $A_4=2$, $R_4=2$, $A_5=3$, $R_5=0$.*<br>

Let us write the value estimates for each action at each timestemp using the sample-average action-value estimation:<br>
1. $Q_1(\mathbf{a}) = (0, 0, 0, 0)$: any action is the best
2. $Q_2(\mathbf{a}) = (1, 0, 0, 0)$: Action $1$ is the best action
3. $Q_3(\mathbf{a}) = (1, 1, 0, 0)$: Actions $1$ and $2$ are the best actions
4. $Q_4(\mathbf{a}) = (1, 1.5, 0, 0)$: Action $2$ is the best action
5. $Q_5(\mathbf{a}) = (1, 5/3, 0, 0)$: Action $2$ is the best action
6. $Q_6(\mathbf{a}) = (1, 5/3, 0, 0)$: Action $2$ is the best action


*(a) On some of these time steps the $\epsilon$ case may have occurred, causing an action to be selected at random. On which time steps did this definitely occur?*
- Step 2: Action $1$ was the best (greedy) action, but action $2$ was taken
- Step 5 $Q(a_2)$ was the largest, but $a_3$ was taken

*(b) On which time steps could this possibly have occurred?*<br>
$\epsilon$ case may have occured on all the other steps (1, 3 and 4) since we do not know the value of $\epsilon$ and there is no contradition to the $\epsilon$ case on any of those steps:
- Step 1: it is a tie among all of the actions, so this could be both greedy and $\epsilon$ case
- Step 3: there is a tie between $a_1$ and $a_2$, so selecting either of them is in line with greedy policy. But it is also possible that action 2 was taken using an $\epsilon$ approach since the probability of it being selected given in the exploratory regime is 0.25 > 0
- Step 4: $a_2$ could have been chosen using greed policy (as its value is the highest) or using $\epsilon$ policy (with probability 0.25)


*Exercise 2.3 In the comparison shown in Figure 2.2, which method will perform best in the long run in terms of cumulative reward and probability of selecting the best action? How much better will it be? Express your answer quantitatively.*

It looks like the $\epsilon=0.01$ method should be the best in long run as it has not reached the plateau in the cumulative reward and best action selection probability graphs. Notably, reward for taking the action 2 is stochastic as it takes values of $1$ and $2$ in the initial trace. When reward distributions are stochastic, exploration is helpful to estimate the ground truth of value functions accurately.<br>
<span style="color:red">TODO: how much better will it be (quantitatively)?</span>.

## 2.4. Incremental Implementation 

### Theory
$Q_n = \dfrac{R_{1} + ... + R_{n-1}}{n-1}$<br><br>
$Q_{n+1} = \dfrac{R_{1} + ... + R_{n}}{n} = \dfrac{R_{1} + ... + R_{n-1}}{n-1} * \dfrac{n-1}{n} + \dfrac{R_n}{n} = Q_n * \dfrac{n-1}{n} + \dfrac{R_n}{n} =$<br><br><br>
$= Q_n - \dfrac{1}{n} Q_n + \dfrac{R_n}{n} =$<br><br>
$= Q_n + \dfrac{1}{n} [R_n - Q_n]$

This leads to a more general icremental update rule:
$$Q_{n+1} = Q_n + \alpha_n[\text{Target}_{n} - Q_n]$$
where $\alpha_n$ is a step-size (more known as learning rate in deep learning).

## 2.5. Tracking a Nonstationary Problem

### Theory
We often encouter RL problems that are non-stationary, that is, their reward probabilities change over time. In such cases, it makes more sense to give more weight to recent rewards rather than to long-past rewards. One of the most popular ways of doing this is to use a constant step-size parameter:
$$Q_{n+1} \equiv Q_n + \alpha[R_n - Q_n], \quad \text{(2.5)}$$

where $\alpha \in (0,1]$ is constant. This results in $Q_{n+1} being a weighted average of past rewards and the initial estimate $Q_1$:<br><br>
$Q_{n+1} = Q_n + \alpha[R_n - Q_n] = \alpha R_n + (1-\alpha)Q_n = \alpha R_n + (1-\alpha) [Q_{n-1} + \alpha[R_{n-1} - Q_{n-1}]] = $<br><br>
$= \alpha R_n + (1-\alpha)\alpha R_{n-1} + (1-\alpha)^2 Q_{n-1} = \alpha R_n + (1-\alpha)\alpha R_{n-1} + (1-\alpha)^2 [Q_{n-2} + \alpha[R_{n-2} - Q_{n-2}]] = $<br><br>
$ = \alpha R_n + (1-\alpha)\alpha R_{n-1} + (1-\alpha)^2\alpha R_{n-2} + (1-\alpha)^3 Q_{n-2} = ...$<br><br>
$ ...= (1-\alpha)^n Q_{1} + \sum_{i=1}^{n}\alpha(1-\alpha)^{n-i}R_i \quad \text{(2.6)}$

Note that $(1-\alpha)^n + \sum_{i=1}^{n}\alpha(1-\alpha)^{n-i} = 1$ <span style="color:red">TODO</span>: prove
<br><br>

Sometimes it is convenient to vary the step size: let us denote $\alpha_n(a)$ the step-size parameter used to process the reward recevied after the $n^{\text{th}}$ selection of action $a$. A stochastic approximation theory tells us that the following conditions for $\alpha_n(a)$ are required to assure convergence of the Q-value to its ground truth with probability 1:<br><br>
- $\sum_{n=1}^{\infty}\alpha_n(a) = \infty$ (the steps are large enough to overcome any initial conditions or random fluctuations) <br><br>
-  $\sum_{n=1}^{\infty}\alpha_n^{2}(a) < \infty$ (the steps become small enough to assure convergence)

However, sequences of step-size parameters that meet these convergence criteria are often used only in theoretical work, and they are seldom used in applications and empirical research.


```python

```

### Exercises
*Exercise 2.4 If the step-size parameters, $\alpha_n$, are not constant, then the estimate $Q_n$ is a weighted average of previously received rewards with a weighting different from that given by (2.6). What is the weighting on each prior reward for the general case, analogous to (2.6), in terms of the sequence of step-size parameters?*

<span style="color:red">TODO</span>: Move the answer from my notebook

*Exercise 2.5 (programming). Design and conduct an experiment to demonstrate the difficulties that sample-average methods have for nonstationary problems. Use a modified version of the 10-armed testbed in which all the $q_{*}(a)$ start out equal and then take independent random walks (say by adding a normally distributed increment with mean zero and standard deviation 0.01 to all the $q_{*}(a)$ on each step). Prepare plots like Figure 2.2 for an action-value method using sample averages, incrementally computed, and another action-value method using a constant step-size parameter, $\epsilon=0.1$. Use $\epsilon=0.1$ and longer runs, say of 10,000 steps.*

<span style="color:red">TODO</span>


```python
class NonStationaryGaussianBandit(GaussianBandit):
    def __init__(self, reward_distribution_params: list[dict], random_seed: int = 2024):
        super().__init__(reward_distribution_params, random_seed)

    def _initialise(self) -> None:
        '''Initialise estimates of Q-value functions and numbers of observations per action.'''
        self.Q = numpy.zeros(self.num_actions)
        self.N = numpy.zeros(self.num_actions)
        self.actions = []
        self.rewards = []
        
    def _call_bandit(self, action_num: int, prev_reward_distribution: dict[str: numbers.Real]) -> numbers.Real:
        '''Sample a reward from its conditional distribution reward|action=a.'''
        assert 0 <= action_num <= self.num_actions - 1

        # update the reward distribution
        random_increment = numpy.random.normal(size=1, loc=0, scale=0.01)[0]
        self.reward_distribution_params[action_num] = {'loc': prev_reward_distribution['loc'] + random_increment, 'scale': prev_reward_distribution['scale']}

        cur_param_dct = self.reward_distribution_params[action_num]
        return numpy.random.normal(size=1, **cur_param_dct)[0]

    def learn(self, epsilon: numbers.Real, max_iter: int = 1_000, learning_rate: numbers.Real = None) -> numpy.ndarray:
        if self.Q is None or self.N is None or self.actions is None or self.rewards is None:
            self._initialise()
        
        if learning_rate is None:
            lr_type = '1/N'
        else:
            lr_type = 'custom'
        
        iter_num = 0
        while iter_num <= max_iter:
            iter_num += 1
            cur_action = epsilon_greedy_policy(Q_values=self.Q, epsilon=epsilon)
            self.actions.append(cur_action)
            cur_reward = self._call_bandit(action_num=cur_action, prev_reward_distribution=self.reward_distribution_params[cur_action])
            self.rewards.append(cur_reward)

            # Update
            self.N[cur_action] = self.N[cur_action] + 1
            if lr_type == '1/N':
                learning_rate = 1 / self.N[cur_action]
            # else: use what the user passed as a learning rate
            self.Q[cur_action] = self.Q[cur_action] + learning_rate * (cur_reward - self.Q[cur_action])

        return self.Q
```


```python
num_bandits = 2_000
num_time_steps = 1_000
num_actions = 10
epsilon_values = [0, 0.01, 0.1]

nonstationary_bandit_q_ground_truths = []
nonstationary_bandit_Q_estimates = {ev: [] for ev in epsilon_values}
nonstationary_bandit_actions = {ev: [] for ev in epsilon_values}
nonstationary_bandit_rewards = {ev: [] for ev in epsilon_values}
nonstationary_bandit_optimal_action_flags = {ev: [] for ev in epsilon_values}

for i in range(num_bandits): # create independent bandits
    if i%100 == 0:
        print(f'Bandit #{i}')
    # Generate ground truths for the value function
    numpy.random.seed(i)
    cur_q_ground_truths = numpy.random.normal(loc=0.0, scale=1.0, size=num_actions)
    cur_reward_distribution_params = [
        {'loc': mean, 'scale': 1}
            for mean in cur_q_ground_truths
    ]
    nonstationary_bandit_q_ground_truths.append(cur_q_ground_truths)

    for epsilon in epsilon_values:
        if i%100 == 0:
            print(f'epsilon={epsilon}') 
        
        # Run the bandit
        gaussian_bandit = NonStationaryGaussianBandit(
            reward_distribution_params=cur_reward_distribution_params, 
            random_seed=i # NOTE: it is important to change seed to make sure that bandits are different!
        )
        cur_Q_est = gaussian_bandit.learn(epsilon=epsilon, max_iter=num_time_steps)

        # Store results
        nonstationary_bandit_Q_estimates[epsilon].append(cur_Q_est)
        nonstationary_bandit_actions[epsilon].append(gaussian_bandit.actions)
        nonstationary_bandit_rewards[epsilon].append(gaussian_bandit.rewards)
        
        # Calculate the % of optimal actions and store it
        cur_best_action = numpy.argmax(cur_q_ground_truths)
        nonstationary_bandit_optimal_action_flags[epsilon].append([a == cur_best_action for a in gaussian_bandit.actions])
```

    Bandit #0
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #100
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #200
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #300
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #400
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #500
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #600
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #700
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #800
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #900
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #1000
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #1100
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #1200
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #1300
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #1400
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #1500
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #1600
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #1700
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #1800
    epsilon=0
    epsilon=0.01
    epsilon=0.1
    Bandit #1900
    epsilon=0
    epsilon=0.01
    epsilon=0.1



```python
nonstationary_bandit_avg_rewards = pandas.DataFrame(
    {eps: pandas.DataFrame(nonstationary_bandit_rewards[eps]).mean(axis=0) for eps in nonstationary_bandit_rewards}  
)
pandas.DataFrame(
    {eps: nonstationary_bandit_rewards[eps][0] for eps in nonstationary_bandit_rewards}
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.00</th>
      <th>0.01</th>
      <th>0.10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.898426</td>
      <td>3.962310</td>
      <td>6.026194</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.759176</td>
      <td>2.823060</td>
      <td>4.886944</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.761252</td>
      <td>2.825136</td>
      <td>4.889020</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.763328</td>
      <td>2.827212</td>
      <td>4.891096</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.765404</td>
      <td>2.829288</td>
      <td>4.893172</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>996</th>
      <td>2.824657</td>
      <td>4.888541</td>
      <td>6.952425</td>
    </tr>
    <tr>
      <th>997</th>
      <td>2.826733</td>
      <td>4.890617</td>
      <td>6.954501</td>
    </tr>
    <tr>
      <th>998</th>
      <td>2.828808</td>
      <td>4.892693</td>
      <td>6.956577</td>
    </tr>
    <tr>
      <th>999</th>
      <td>2.830884</td>
      <td>4.894768</td>
      <td>6.958652</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>2.832960</td>
      <td>4.896844</td>
      <td>6.960728</td>
    </tr>
  </tbody>
</table>
<p>1001 rows × 3 columns</p>
</div>




```python
nonstationary_bandit_avg_rewards.plot(figsize=(20,5), color={0: 'tab:green', 0.01: 'tab:red', 0.1: 'tab:blue'})
plt.legend(title='$\epsilon$ value', loc='upper left')
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.title('(Non-stationary bandit)')
plt.tight_layout();
```

    <>:2: SyntaxWarning:
    
    invalid escape sequence '\e'
    
    <>:2: SyntaxWarning:
    
    invalid escape sequence '\e'
    
    /var/folders/c3/4d4hvn_n173cnw7lk14cms2m0000gn/T/ipykernel_72002/3713408095.py:2: SyntaxWarning:
    
    invalid escape sequence '\e'
    



    
![png](chapter2-multi-armed-bandits/output_36_1.png)
    



```python
nonstationary_bandit_optimal_action_props = pandas.DataFrame(
    {eps: pandas.DataFrame(nonstationary_bandit_optimal_action_flags[eps]).mean(axis=0) for eps in nonstationary_bandit_optimal_action_flags}  
)

(100 * nonstationary_bandit_optimal_action_props).plot(figsize=(20,5), color={0: 'tab:green', 0.01: 'tab:red', 0.1: 'tab:blue'})
plt.legend(title='$\epsilon$ value', loc='upper left')
plt.xlabel('Steps')
plt.ylabel('Optimal action, %')
plt.tight_layout();
```

    <>:6: SyntaxWarning:
    
    invalid escape sequence '\e'
    
    <>:6: SyntaxWarning:
    
    invalid escape sequence '\e'
    
    /var/folders/c3/4d4hvn_n173cnw7lk14cms2m0000gn/T/ipykernel_72002/2358998182.py:6: SyntaxWarning:
    
    invalid escape sequence '\e'
    



    
![png](chapter2-multi-armed-bandits/output_37_1.png)
    



```python
num_bandits = 2_000
num_time_steps = 10_000 # use longer runs with a constant step size!
num_actions = 10
epsilon = 0.1
step_size = 0.1

nsbandit_const_alpha_q_ground_truths = []
nsbandit_const_alpha_Q_estimates = []
nsbandit_const_alpha_actions = []
nsbandit_const_alpha_rewards = []
nsbandit_const_alpha_optimal_action_flags = []

for i in range(num_bandits): # create independent bandits
    if i%100 == 0:
        print(f'Bandit #{i}')
    # Generate ground truths for the value function
    numpy.random.seed(i)
    cur_q_ground_truths = numpy.random.normal(loc=0.0, scale=1.0, size=num_actions)
    cur_reward_distribution_params = [
        {'loc': mean, 'scale': 1}
            for mean in cur_q_ground_truths
    ]
    nsbandit_const_alpha_q_ground_truths.append(cur_q_ground_truths)
    
    # Run the bandit
    gaussian_bandit = NonStationaryGaussianBandit(
        reward_distribution_params=cur_reward_distribution_params, 
        random_seed=i # NOTE: it is important to change seed to make sure that bandits are different!
    )
    cur_Q_est = gaussian_bandit.learn(epsilon=epsilon, max_iter=num_time_steps, learning_rate=step_size)

    # Store results
    nsbandit_const_alpha_Q_estimates.append(cur_Q_est)
    nsbandit_const_alpha_actions.append(gaussian_bandit.actions)
    nsbandit_const_alpha_rewards.append(gaussian_bandit.rewards)
    
    # Calculate the % of optimal actions and store it
    cur_best_action = numpy.argmax(cur_q_ground_truths)
    nsbandit_const_alpha_optimal_action_flags.append([a == cur_best_action for a in gaussian_bandit.actions])
```

    Bandit #0
    Bandit #100
    Bandit #200
    Bandit #300
    Bandit #400
    Bandit #500
    Bandit #600
    Bandit #700
    Bandit #800
    Bandit #900
    Bandit #1000
    Bandit #1100
    Bandit #1200
    Bandit #1300
    Bandit #1400
    Bandit #1500
    Bandit #1600
    Bandit #1700
    Bandit #1800
    Bandit #1900



```python
nsbandit_const_alpha_avg_rewards = pandas.DataFrame(nsbandit_const_alpha_rewards).mean(axis=0)
```


```python
nsbandit_const_alpha_avg_rewards.plot(figsize=(20,5))
plt.legend(title='$\epsilon$ value', loc='upper left')
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.title('(Non-stationary bandit)')
plt.tight_layout();
```

    <>:2: SyntaxWarning:
    
    invalid escape sequence '\e'
    
    <>:2: SyntaxWarning:
    
    invalid escape sequence '\e'
    
    /var/folders/c3/4d4hvn_n173cnw7lk14cms2m0000gn/T/ipykernel_72002/2136081910.py:2: SyntaxWarning:
    
    invalid escape sequence '\e'
    
    No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.



    
![png](chapter2-multi-armed-bandits/output_40_1.png)
    



```python
nsbandit_const_alpha_optimal_action_props = pandas.DataFrame(nsbandit_const_alpha_optimal_action_flags).mean(axis=0)
(100 * nsbandit_const_alpha_optimal_action_props).plot(figsize=(20,5))
plt.legend(title='$\epsilon$ value', loc='upper left')
plt.xlabel('Steps')
plt.ylabel('Optimal action, %')
plt.tight_layout();
```

    <>:3: SyntaxWarning:
    
    invalid escape sequence '\e'
    
    <>:3: SyntaxWarning:
    
    invalid escape sequence '\e'
    
    /var/folders/c3/4d4hvn_n173cnw7lk14cms2m0000gn/T/ipykernel_72002/2900906807.py:3: SyntaxWarning:
    
    invalid escape sequence '\e'
    
    No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.



    
![png](chapter2-multi-armed-bandits/output_41_1.png)
    



```python

```

## 2.6. Optimistic Initial Values

### Theory


### *Figure 2.3*


```python

```

### Exercises
*Exercise 2.6: Mysterious Spikes. The results shown in Figure 2.3 should be quite reliable because they are averages over 2000 individual, randomly chosen 10-armed bandit tasks. Why, then, are there oscillations and spikes in the early part of the curve for the optimistic method? In other words, what might make this method perform particularly better or worse, on average, on particular early steps?*

Even if you choose an optimal action at the beginning, its value estimate will be revises downwards, which will make it suboptimal at the next step. However, if that action is in fact optimal, the agent will have to take all other actions and revise their values as well before it figures out what the ground truth optimum is.<br>
Also, the agent may sometimes ignore the very best action until the very end, and once it is taken, it may stick to it for a long time until its value estimate is smaller than that of the others (due to more pessimistic value estimates).

<span style="color:red">TODO</span> is this a good answer?

*Exercise 2.7: Unbiased Constant-Step-Size Trick. In most of this chapter we have used sample averages to estimate action values because sample averages do not produce the initial bias that constant step sizes do (see the analysis leading to (2.6)). However, sample averages are not a completely satisfactory solution because they may perform poorly on nonstationary problems. Is it possible to avoid the bias of constant step sizes while retaining their advantages on nonstationary problems?<br>
One way is to use a step size of $\beta_n = \alpha / \bar{o}_n$ to process the $n^{\text{th}}$ reward for a particular action, where $\alpha > 0$ is a conventional constant step size, and $\bar{o}_n$ is a trace of one that starts at 0:*
$$\bar{o}_n = \bar{o}_{n-1} + \alpha (1 - \bar{o}_{n-1})$$
*for $n > 0$ with  $\bar{o}_0 = 0$*.<br>
*Carry out an analysis like that in (2.6) to show that $Q_n$ is an exponential recency-weighted average without initial bias.*

## 2.7 Upper-Confidence-Bound Action Selection



## 2.8 Gradient Bandit Algorithms


```python

```

*Exercise 2.9* Show that in the case of two actions, the soft-max distribution is the same as that given by the logistic, or sigmoid, function often used in statistics and artificial neural networks.

SEE MY NOTES


```python

```

## 2.9. Associative Search (Contextual Bandits)



## 2.10. Summary


