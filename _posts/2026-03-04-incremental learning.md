# Incremental learning in the context of RL

From https://www.nature.com/articles/s42256-022-00568-3 (which is for supervised learning)

When deep leaarning systems are trained on new data steams, they tend to forget their previously acquired capabilities (at least in pre-LLM ere). 
Continual learning (a.k.a. lifelong learning) is the aimed at resolving this problem in incremental learning.

There are 3 fundamental types of *supervised* continual learning, which we consider in the context of RL:
- task-incremental learning (TIL): incrementally learning clearly distinguishable tasks
- domain-incremental learning (DIL): learning the same kind of problem but in different contexts
- class-incremental learning (CIL): incrementally learning to distinguish between a growing number of objects or classes

## TIL
It is always clear which task must be performed, for example, via task label. The challenge is not always the catastrophic forgetting, but the efficient sharing of the knowledge from one task to another.

Real-life examples: learning to play different sports or musical instruments, because it is usually clear which sport or instrument you are playing!

In the context of RL: providing a task label as an observation feature.

## DIL
The structure of the problem is the same for any task, but the context or *input distribution* changes. We still learn a set of tasks, but now
it is more convenient to think about them as of *domains*.

Crucial difference to TIL: at test time, the agent does not know from which task (MDP) the observation is sampled from.

In the context of RL: *obvervation distribution* is different from task to task all else being equal (i.e. the policy is the same); this is equivalent 
to the transition dynamics distribution shift. Example: learning to drive in different weather conditions.

Preventing forgetting *by design* (as in TIL) is not possible in DIL.


## CIL
An algorithm must learn to discriminate between a growing number of objects or classes.

This is not applicable to RL since this is a classification problem!

*To be continued...*
