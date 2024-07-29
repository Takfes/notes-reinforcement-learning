## Introduction to Reinforcement Learning

- [Lecture Video](https://www.youtube.com/watch?v=2pWv7GOvuf0&t=3338)

---

- **Reinforcement Learning Paradigm**
- Reinforcement learning is a framework for solving control tasks (also called decision problems) by building agents that learn from the environment by interacting with it through trial and error and receiving rewards (positive or negative) as unique feedback.
- Reinforcement Learning is a computational approach to learning from actions. We build an agent that learns from the environment by interacting with it through trial and error and receiving rewards (negative or positive) as feedback.
- The RL process is a loop that outputs a sequence of state, action, reward, and next state.
- there is no supervisor, only reward signal
- feedback may not be instantaneous
- time really matters (sequential non i.i.d data)
- agent's actions affect the subsequent data it receives

---

- **Observation, Action, Reward**
- at each step t the agent
- executes action At
- receives observation Ot
- receives reward Rt
- agent influences the environment through the action it makes
- history $Ht=A_1,O_1,R_1..A_t,O_t,R_t$
- history determines what happens next

---

- **Action Space**
- Discrete space: the number of possible actions is finite.
- Continuous space: the number of possible actions is infinite.

---

- **State**
- State s: is a complete description of the state of the world (there is no hidden information). In a fully observed environment.
- Observation o: is a partial description of the state. In a partially observed environment.
- state is the information used to determine what happens next
- state is a function of history $S_t=f(H_t)$
- different definitions of state : observation that we see, agent state, environment state
- State and Observations may be used interchangebly to denote full and partial environment information respectively

---

- **Rewards**
- reward Rt is a scalar feedback signal
- indicates how well agent is doing at step t
- agent's job is to maximise cumulative reward
- The reward is fundamental in RL because it’s the only feedback for the agent. Thanks to it, our agent knows if the action taken was good or not.
- **Reward Hypothesis** all goals can be described by the maximisaion of expected cumulative reward
- ultimately we are after picking actions, thus, there must always exist a conversion between different objectives
- Because RL is based on the reward hypothesis, which is that all goals can be described as the maximization of the expected return (expected cumulative reward).

---

- **Return**
- in Reinforcement Learning, we aim to learn to take actions that maximize the expected cumulative reward.
- The cumulative reward is called **Return**.
- $R(τ) = r_{t+1} + r_{t+2} + r_{t+3} + r_{t+4}  ...$
- However, in reality, we can’t just add them like that.
- The reward near the distant time steps, will be more discounted since we’re not really sure we’ll be able to get it.
- To discount the rewards, we define a discount rate called **gamma**. It must be between 0 and 1. Most of the time between 0.95 and 0.99.
- The larger the gamma, the smaller the discount. This means our agent cares more about the long-term reward.
- On the other hand, the smaller the gamma, the bigger the discount. This means our agent cares more about the short term reward.
- $R(τ) = r_{t+1} + \gamma * r_{t+2} + \gamma^2 * r_{t+3} + \gamma ^ 3 * r_{t+4}...$

---

- **Markov Property**
- $P[S_t+_1 | S] = P[S_t+_1 | S_1,...S_t]$
- _the future is independent of the past given the present_
- state is sufficient statistic of the future
- ultimately we are after making decision about actions for the future
- the state representation defines what happens next - our job is to build a state representation that is actually useful for predicting what is about to happen next

---

- **Task Taxonomy**
- Episodic tasks : when we have a starting point and an ending point (a terminal state). This creates an episode: a list of States, Actions, Rewards, and new States.
- For instance, think about Super Mario Bros: an episode begin at the launch of a new Mario Level and ends when you’re killed or you reached the end of the level.
- Continuing tasks : These are tasks that continue forever (no terminal state). In this case, the agent must learn how to choose the best actions and simultaneously interact with the environment.
- For instance, an agent that does automated stock trading. For this task, there is no starting point and terminal state. The agent keeps running until we decide to stop it.

---

- **Environment Observability**
- in Fully Observable Environments agent directly observes environment state
- agent = environment = information state $O_t=S_t^a=S_t^e$
- this is **Markov Decision Process (MDP)**
- Partial observability where agent indirectly observes environment
- this is **Partially observable Markov Decision Process (POMDP)**
- agent must construct its own state representation
- state can be constructed based on fully history or based on beliefs (bayesian approach)

---

**Reward vs Return vs Value Function**

- Reward: A reward is a numerical signal provided by the environment to an agent at each step of an MDP. It represents the immediate desirability or quality of a particular state-action pair. In other words, it indicates how much the agent values the outcome of taking a specific action in a specific state. The reward is typically denoted by the symbol "R" and can be positive, negative, or zero.

- Return: The return is the **cumulative total of rewards** obtained from a specific state onward until the end of an episode (or the end of the task). It represents the notion of long-term desirability or value of being in a particular state and taking a particular action. The return is often denoted by the symbol "G" and can be defined in different ways, such as the sum of discounted rewards or the sum of rewards until a terminal state is reached.

- Value function: A value function estimates the expected return or value of being in a particular state (or state-action pair) under a given policy. It quantifies the desirability or quality of states based on the expected future rewards that can be obtained from those states. There are two main types of value functions in MDPs:

- State value function (V(s)): It represents the expected return from being in a particular state "s" under a given policy. It quantifies the long-term value or desirability of a state.

- Action value function (Q(s, a)): It represents the expected return from taking a particular action "a" in a particular state "s" under a given policy. It quantifies the long-term value or desirability of taking a specific action in a specific state.

---

- **Sequential Decision Making**
- Reinforcement Learning : environment is _initially unknown_, agent interacts with the environment and improves its policy
- RL is like a trial and error approach, the agent shoud discover a good policy in an unknown environment, without losing too much reward along the way
- this is where the **exploration vs exploitation** dilemma comes into play
- Exploration is exploring the environment by trying random actions in order to find more information about the environment.
- for instance think of the choice of picking a restaurant:
- Exploitation is exploiting known information to maximize the reward.
- Exploitation: You go to the same one that you know is good every day and take the risk to miss another better restaurant.
- Exploration: Try restaurants you never went to before, with the risk of having a bad experience but the probable opportunity of a fantastic experience.

---

- **Planning, Prediction & Control**
- **Planning** : a model of the environment is _known_ , the agent performs computations with its model and improves its policy
- in planning, the agent has perfect understanding of the environment and can essentially "look-ahead"
- Planning in RL refers to the process of constructing a policy or decision-making strategy without direct interaction with the environment.
- Planning algorithms use models or representations of the environment dynamics to simulate potential trajectories and optimize the policy based on anticipated outcomes.
- The goal of planning is to find an optimal or near-optimal policy that maximizes the expected cumulative reward.

<br>

- **Prediction** (in rl lingo) evaluate the future given a policy
- also known as value estimation or value prediction, focuses on estimating the expected return or value of being in a given state under a given policy.
- It involves learning a value function that assigns a value to each state or state-action pair based on the expected future rewards.
- Prediction is typically performed using methods like dynamic programming, Monte Carlo methods, or Temporal Difference (TD) learning.
- By accurately predicting values, an RL agent can assess the desirability of different states or state-action pairs and make informed decisions.

<br>

- **Control** (in rl lingo) optimise the future by finding the best policy
- Control refers to the process of taking actions in the environment to maximize the expected cumulative reward.
- It involves learning an optimal or near-optimal policy by iteratively improving the estimated value function through exploration and exploitation.
- oftentimes, we first need to solve the prediction problem (i.e. evaluate potential policies) before moving to control (i.e.) selecting the optimal one
- Control algorithms use value function estimation methods (such as TD learning) and policy improvement techniques (such as policy iteration or policy gradient methods) to find the best actions to take in each state.
- The objective of control is to discover an optimal policy that maximizes the long-term expected reward by iteratively updating the value function and the policy.

<br>

- **Overview**
- Markov Process
- Markov Reward Process = Markov Process + Rewards
- Markov Decision Process = Markov Reward Process + Decision
- Markov Property : Future is independent of the past given the future
- **Sampling** a Markov Chain = sampling **state transitions sequences** produced from the underlying chain

---

- **Markov Reward Process (MRP)** adding Rewards and Discount Factor to Markov Chain
- $R$ is the reward we are getting from a single state
- Ultimately we want to optimize the cumulative sum of such rewards $R$
- $G$ is the total discounted reward

---

- **Markov Decision Process (MDP)**
- adding decision making on top of MRP and we get MDPs.
- Note the difference between Bellman Expectation and Optimality Equations.

