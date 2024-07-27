## Bellman Equations & Value Functions

- [Hugging Face Resources](https://huggingface.co/learn/deep-rl-course/unit2/two-types-value-based-methods?fw=pt) the two Value Functions
- [Hugging Face Resources](https://huggingface.co/learn/deep-rl-course/unit2/bellman-equation?fw=pt) the Bellman Equation
- [Youtube Video](https://www.youtube.com/watch?v=lfHX2hHRMVQ&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=2) Lecture 2: Markov Decision Process

---

### The two Value Functions

- **State Value Function** : $V_π(s) = E_π[G_t|S_t=s]$
- For each state, the state-value function $V_π(s)$
- outputs the expected return $E_π[G_t...]$
- if the agent starts at that state $|S_t=s]$ and then
- follows the policy forever afterward $_π$ (for all future timesteps, if you prefer).

<br>

- **Action Value Function** : $Q_π(s,α) = E_π[G_t|S_t=s, Α_t=α]$
- Also know as State, Action Value Function
- In the action-value function, for each state and action pair, the action-value function outputs the expected return if the agent starts in that state, takes that action, and then follows the policy forever after.

<br>

- For the state-value function, we calculate the value of a state
- For the action-value function, we calculate the value of the state-action pair $(S_t,A_t)$ hence the value of taking that action at that state
- In either case, whichever value function we choose (state-value or action-value function), the returned value is the expected return.
- However, the problem is that to calculate EACH value of a state or a state-action pair, we need to sum all the rewards an agent can get if it starts at that state.
- This can be a computationally expensive process, and that’s where the Bellman equation comes in to help us.

---

### The Bellman Equations

- The Bellman equation is a recursive equation that works like this: instead of starting for each state from the beginning and calculating the return, we can consider the value of any state as
- the idea of the Bellman equation is that instead of calculating each value as the sum of the expected return, which is a long process, we calculate the value as the sum of **immediate reward + the discounted value of the state that follows.**
- The immediate reward $R_{t+1}$ + the discounted value of the state that follows $\gamma * V(S_{t+1})$

---

- The **Bellman Expectation Equations** :

<br>

- **State Value Function**
- $V_π(s) = E_π[G_t|S_t=s]$
- $V_π(s) = E_π[R_{t+1} + \gamma * V_π(S_{t+1})|S_t=s]$

<br>

- **Action Value Function**
- $Q_π(s,α) = E_π[G_t|S_t=s, Α_t=α]$
- $Q_π(s,α) = E_π[R_{t+1} + \gamma * Q_π(S_{t+1},A_{t+1})|S_t=s, Α_t=α]$

---

- The **Bellman Optimality Equations** :

<br>

- **State Value Function**
- $V_*(s) = maxR_s^α + \gamma * \Sigma P_{ss'}^α V_*(s')$, which translates into :
- $maxR_s^α$ : look at the possible actions and select the one with the max return
- $\Sigma P_{ss'}^α V_*(s')$ : After that, assuming a stochastic transistion process, we may end up in different stages based on the environment chances (dice rolling). To account for that, we take the summation of the discounted expectation of the states we may end up into

<br>

- **Action Value Function**
- $Q_*(s,α) = R_s^α + \gamma * \Sigma P_{ss'}^α max Q_*(s',a')$, which translates into :
- $R_s^α$ account for the reward that corresponds to the selected action
- After that, the environment dice roll comes into play. That is, depending on the selected action, we may end up to few different states (that we don't control). Where we actually end up though, is not a matter of our decision but is the outcome of the environment's stochastic process.
- For the sake of the example, let's assume that after the said dice roll, we may end up in state A or state B. No matter, where we end up, we will get to pick our next action. That is, in either situations, we control the outcome, i.e. which action we will take from that point onwards. Therefore, in either case (A or B) we can select the best action (max value) (depending on the case).
- Thus, this $\gamma * \Sigma P_{ss'}^α max Q_*(s',a')$ tells us that we take the expectation between Amax and Bmax.
