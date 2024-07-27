## Policy vs Value Based Methods

- [Hugging Face Resources](https://huggingface.co/learn/deep-rl-course/unit1/two-methods?fw=pt)

---

- **The Policy** π is the brain of our Agent, it’s the function that tells us what action to take given the state we are in
- So it defines the agent’s behavior at a given time
- This Policy is the function we want to learn, our goal is to find the **optimal policy π\***, the policy that maximizes expected return when the agent acts according to it. We find this π\* through training
- **There are two approaches** to train our agent to find this optimal policy π\*:
- Directly, by teaching the agent to learn **which action to take, given the current state: Policy-Based Methods.**
- Indirectly, teach the agent to learn **which state is more valuable and then take the action that leads to the more valuable states: Value-Based Methods.**

<br>

- Policy-based methods: Directly train the policy to select what action to take given a state (or a probability distribution over actions at that state). In this case, we don’t have a value function.
- Value-based methods: Indirectly, by training a value function that outputs the value of a state or a state-action pair. Given this value function, our policy will take an action.
- Consequently, whatever method you use to solve your problem, you will have a policy. In the case of value-based methods, you don’t train the policy: your policy is just a simple pre-specified function (for instance, the Greedy Policy) that uses the values given by the value-function to select its actions. So the difference is:
- In policy-based training, the optimal policy (denoted π\*) is found by training the policy directly.
- In value-based training, finding an optimal value function (denoted Q* or V*, we’ll study the difference below) leads to having an optimal policy.

---

- **Policy Based Methods** which define a mapping from each state to the best corresponding action.
- Alternatively, it could define a probability distribution over the set of possible actions at that state.
- **Deterministic**: a policy at a given state will always return the same action : $a=\pi(s)$
- **Stochastic**: outputs a probability distribution over actions : $\pi(a|s)=P[A=a|S=s]$

---

- **Value Based Methods** instead of learning a policy function, we learn a value function that maps a state to the expected value of being at that state.
- The value of a state is the expected discounted return the agent can get if it starts in that state, and then acts according to our policy.
- $v_\pi(s)=E_\pi[R_t+\gamma R_t+_1 + \gamma^2 R_t+_2 + ... |S=s]$

<br>

- value function depends on the way the agent is behaving, i.e. on the policy that follows - which explains the $\pi$ subscript
- “Act according to our policy” just means that our policy is “going to the state with the highest value”.
- $v_\pi(s)$ is **the value function**; a prediction of future reward

<br>

- $E*\pi[R_t+\gamma R*{t+1} + \gamma^2 R\_{t+2} + ...$ is the Expected Discounted Reward
- the horizon of the reward is determined by the discount factor $\gamma$
- risk is already accounted for since we are calculating the expected reward

---

- **Agent Components**
- policy : agent's behaviour function
- value function : how good is each state and action
- model : agent's representation of the environment
- this is the superset of components that may be used by an agent

---

- **Model** predicts what the environment will do next
- _state transition_ model used to predict the next state : $P_{ss'}^a=P[S'=s'|S=s,A=a]$
- _rewards_ transition model : $R_s^a=E[R|S=s,A=a]$
- model is optional in several cases and this is what we call model free methods

---

- **Agents Taxonomy**
- we taxonomize the rl agents based on which of the key components are present
- value based - if contains a value function - in such cases, policy is implicit (i.e. always pick the best value function) - agent stores the value
- policy based - data structure to maintain the optimal policy without ever explicitly calculate the value function - agent stores the policy
- actor critic - is the combination of the above two, i.e. an agent that stores and operates based on both

<br>

- **Agents Taxonomy based on model**
- model free - agent does not explicitly try to represent the environment, instead its decisions are based on value function or policy, without explicitly try to model how the environment works
- model based - we build a model to understand the environment and use this in conjuction with policy and value function
