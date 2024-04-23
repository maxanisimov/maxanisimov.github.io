---
title: 'Multi-armed Bandits'
date: 2024-04-24
permalink: /posts/2024/04/intro-to-rl/multi-armed-bandits
tags:
  - AI
  - Reinforcement Learning
  - Multi-armed Bandits
---
*Conspectus of Sutton & Barto "Reinforcement Learning: An Introduction"*

# *Tabular Solution Methods*
In this part of the book, authors discuss cases in which the state and action sapces are small enough for the approximate value functions to be represented as arrays (or tables). In this case, the methods can often find exactly the optimal value and the optimal policy.

Bandit problem is an RL problem in which there is only a single state. Markov decision process (MDP) is a generalisation of the bandit problem, so that an MDP has more than one state. 

Three fundamental classes of methods for solving finite Markov decision problems:
- Dynamic programming
  - Pros: well-developed mathematically
  - Cons: require a complete and accurate model of the environment
- Monte Carlo methods
  - Pros: do not require a model and are conceptually simple
  - Cons: Not well-suited for step-by-step incremental computation
- Temporal-difference learning
  - Pros: require no model and are fully incremental
  - Cons: more complex to analyse

The methods allso differ in several ways with respect to their efficiency and speed of convergence. They can also be combined to obtain the best features of each of them. E.g., the stengths of Monte Carlo methods can be combined with strengths of temporal-difference methods via multi-step bootstrapping methods.

# Chapter 2: Multi-armed Bandits
The most important feature distinguishing RL from other types of learning is that it uses training information that *evaluates* the actions taken rather than *instructs* by giving correct actions. This is what creates the need for active exploration.

#### Evaluative vs instructive feedback
- **Evaluative feedback** indicates how good the action taken was, but not whether it was the best or the worst action possible. 
- **Instructive feedback** indicates the correct action to take, independently of the  action actually taken. This kind of feedback is the basis of supervised learning.

Associative problem is the problem in which actions are taken in more than one situation (state). 

## 2.1 A k-armed Bandit Problem
Imagine you are faced repeatedly with a choice among $k$ different actions. After taking each action, you receive a numerical reward from a startionary probability distribution that depends on the action you selected. Your objective is to maximise the expected total reward over some time period, for example, over 1000 action selections, or **time steps**. This i the form of the **k-armed bandit problem**, so named by analogy to a slot machine ("one-armed bandit"), except that it has $k$ levers instead of one. Through repeated action selections you are to maximise your winnings by concentrating your actions on the best levers.

In our k-armed bandit problem, each of the $k$ actions has an expected or mean reward given that that action is selected. Let us call this the **value** of that action. Denot action selected on time step $t$ as $A_t$ and the corresponding reward as $R_t$. The value of an arbitraty action $a$, denoted as $q_{*}(a)$, is the expected reward given that the action $a$ is taken:

$$q_{*}(a) \equiv E[R_t|A_t=a]$$

If you knew the value of each action in the action space, you would just have to always select the action with the highest value to solve the k-armed bandit problem. We assume you do not know the action values with certainty, although you may have estimates. We denote the estimated value of action $a$ at time step $t$ as $Q_t(a)$. We would like the estimated value $Q_t(a)$ to be close to the ground truth value $q_{*}(a)$.

If you choose an action with the highest estimated value, we call this action **greedy**. In this case, you are **exploiting** your current knowledge of the values of the actions. If instead you select of the non-greedy actions, you are **exploring**, because this enables you to improve your estimate of the non-greedy action's value.

When having imprecise value estimates, reward is lower in the short run during exploration, but higher in the long run because after you have discovered the better actions, you can exploit them many times. Because it is not possible both to explore and to exploit with any single action selection, one often refers to the "conflict" between exploration and exploitation.

## 2.2 Action-value methods
Let us discuss methods for estimating the values of actions and for using the estimates to make action selection decisions, or so-called **action-value methods**.

### Sample-average method
One natural way to estimate the true value of an action is by averaging received rewards:

$$Q_{t}(a) = \dfrac{\text{Sum of rewards when $a$ is taken prior to $t$}}{\text{Number of times $a$ is taken prior to $t$}} =$$
$$ = \dfrac{\sum_{i=1}^{t-1} R_{i} * \mathbf{1}_{A_i=a}}{\sum_{i=1}^{t-1} \mathbf{1}_{A_i=a}} \quad \text{(2.1)}$$ 

If the denominator "Number of times $a$ is taken prior to $t$" is zero, we define $Q_{t}(a)$ as some default value, such as $0$. As the denominator goes to infinity, $Q_{t}(a)$ converges to the ground truth value $q_{*}(a)$ by the law of large numbers (LLN).

The presented method is called a sample-average method for estimating action values.

### Using value estimates to optimise actions
The simplest action selection rule is to select one of the actions with the highest estimated value. If there is more than one greedy action, a selection is made among them in some arbitraty way (e.g. randomly). Therefore, the greedy action selection is defined as
$$A_t \equiv \argmax_{a} Q_t(a) \quad \text{(2.2)}$$
with ties broken arbitrarily.

### $\epsilon$-greedy methods
Idea: behaving greedily most of the time, but every once in a while, e.g. with small probability $\epsilon$, instead select randomly from among all the actions with equal probability.

Advantage: in the limit, as the number of steps increases, every action will be sampled an infinite number of times, thus ensuring that all the $Q_t(a)$ converge to $q_{*}(a)$. However, these are asymptotic guarantees, and say little about the practical effectiveness of the $\epsilon$-greedy methods.

*Exercise 2.1* In $\epsilon$-greedy action selection, for the case of two actions and $\epsilon=0.5$, what is the probability that the greedy aciton is selected?

Answer: $P[\argmax_{a} Q_{t}(a) \text{ is selected}] = (1 - \epsilon) + \epsilon/2 = 1 - \epsilon/2$ 


## 2.3 The 10-armed Testbed

## 2.4

## 2.5

## 2.6

## 2.7

## 2.8

## 2.9

## 2.10