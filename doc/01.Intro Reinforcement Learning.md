## Introduction to Reinforcement Learning [Lecture Video](https://www.youtube.com/watch?v=2pWv7GOvuf0&t=3338)

___
* **Reinforcement Learning Paradigm**
* there is no supervisor, only reward signal
* feedback may not be instantaneous
* time really matters (sequential non i.i.d data)
* agent's actions affect the subsequent data it receives
___
* **Rewards**
* reward Rt is a scalar feedback signal
* indicates how well agent is doing at step t
* agent's job is to maximise cumulative reward
* **Reward Hypothesis** all goals can be described by the maximisaion of expected cumulative reward
* ultimately we are after picking actions, thus, there must always exist a conversion between different objectives
___
* **Observation, Action, Reward**
* at each step t the agent
* executes action At
* receives observation Ot
* receives reward Rt
* agent influences the environment through the action it makes
* history $Ht=A_1,O_1,R_1..A_t,O_t,R_t$
* history determines what happens next
___
* **State**
* state is the information used to determine what happens next
* state is a function of history $S_t=f(H_t)$
* different definitions of state : observation that we see, agent state, environment state
___
* **Markov Property**
* $ P[S_t+_1 | S] = P[S_t+_1 | S_1,...S_t] $
* the future is independent of the past given the present
* state is sufficient statistic of the future
* ultimately we are after making decision about actions for the future
* the state representation defines what happens next - our job is to build a state representation that is actually useful for predicting what is about to happen next
___
* **Environment Observability**
* in Fully Observable Environments agent directly observes environment state 
* agent = environment = information state $O_t=S_t^a=S_t^e$
* this is **Markov Decision Process (MDP)**
* Partial observability where agent indirectly observes environment
* this is **Partially observable Markov Decision Process (POMDP)**
* agent must construct its own state representation
* state can be constructed based on fully history or based on beliefs (bayesian approach)
___
* **RL Agent components**
* policy : agent's behaviour function
* value function : how good is each state and action
* model : agenet's representation of the environment
* this is the superset of components that may be used by an agent

<br>

* **policy** is a map from state to action
* deterministic policy $a=\pi(s)$
* stochastic policy $\pi(a|s)=P[A=a|S=s]$

<br>

* **value function** is a prediction of future reward
* used to evaluate the goodness/badness of states
* $v_\pi(s)=E_\pi[R_t+\gamma R_t+_1 + \gamma^2 R_t+_2 + ... |S=s]$
* value function depends on the way the agents is behaving, i.e. on the policy that follows - which explains the $\pi$ subscript
* the horizon of the reward is determined by the discount factor $\gamma$
* risk is already accounted for since we are calculating the expected reward

<br>

* **model** predicts what the environment will do next
* _transitions_ a model that is used to predict 
the next state
* _rewards_ a model that is used to predict 
* state transition model : $P_{ss'}^a=P[S'=s'|S=s,A=a]$
the next immediate reward
* reward transition model : $R_s^a=E[R|S=s,A=a]$
* model is optional in several cases and this is what we call model free methods
___
* **Agents Taxonomy**
* we taxonomize the rl agents based on which of the key components are present
* value based - if contains a value function - in such cases, policy is implicit (i.e. always pick the best value function) - agent stores the value
* policy based - data structure to maintain the optimal policy without ever explicitly calculate the value function - agent stores the policy
* actor critic - is the compination of the above two, i.e. an agent that stores and operates based on both
* **Agents Taxonomy based on model**
* model free - agent does not explicitly try to represent the environment, instead its decisions are based on value function or policy, without explicitly try to model how the environment works
* model based - we build a model to understand the environment and use this in conjuction with policy and value function
___
* **Sequential Decision Making**
* Reinforcement Learning : environment is _initially unknown_, agents interacts with the environment and improves its policy
* RL is like a trial and error approach, the agent shoud discover a good policy in an unknown environment, without losing too much reward along the way
* this is where the **exploration vs exploitation** dilemma comes into play
* Planning : a model of the environment is _known_ , the agent performs computations with its model and improves its policy
* in planning, the agent has perfect understanding of the environment and can essentially "look-ahead"
* prediction (in rl lingo) evaluate the future given a policy
* control (in rl lingo) optimise the future by finding the best policy
* oftentimes, we first need to solve the prediction problem (i.e. evaluate potential policies) before selecting the optimal one
