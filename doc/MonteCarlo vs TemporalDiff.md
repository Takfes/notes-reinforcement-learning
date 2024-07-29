
**Resources**
- [Hugging Face Resources](https://huggingface.co/learn/deep-rl-course/unit2/mc-vs-td?fw=pt) Monte Carlo vs Temporal Difference Learning
- [Youtube Video](https://www.youtube.com/watch?v=PnHCvfgC_ZA&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=4) RL Course by David Silver - Lecture 4: Model-Free Prediction
- [Youtube Video](https://www.youtube.com/watch?v=AJiG3ykOxmY&list=PLzvYlJMoZ02Dxtwe-MmH4nOB5jYlMGBjr&index=4) Temporal Difference Learning (including Q-Learning) | Reinforcement Learning Part 4

---

## Introduction
- So far we have seen methods to solve MRP | MDP provided full knowledge over the dynamic governing our system.
- **Model Free prediciton** is about solving the MRP without prior knowledge of its dynamics
- Remember that an RL agent learns by interacting with its environment. The idea is that given the experience and the received reward, the agent will update its value function or policy.
- **Monte Carlo** and **Temporal Difference Learning** are two different strategies on how to train our value function or our policy function. Both of them use experience to solve the RL problem.
- With Monte Carlo, we update the value function from a complete episode, and so we use the actual accurate discounted return of this episode.
- With TD Learning, we update the value function from a step, and we replace $G_t$, which we don't know, with an estimated return called the TD target.

---

## Monte Carlo (MC)
- Monte Carlo methods estimate value functions based on complete episodes of interaction with the environment.
- These methods sample sequences of states, actions, and rewards until the end of an episode, and then use the obtained returns (cumulative rewards) to update value estimates.

<br>

- Monte Carlo waits until the end of the episode, calculates $G_t$ (return) and uses it as a target for updating $V(S_t)$. So it requires a complete episode of interaction before updating our value function.
- $V(S_t) := V(S_t) + α * [G_t - V(S_t)]$
- which translates to :
- The new value of state t updates as, former estimation of value of state t (expected return starting at state t), plus learning rate $\alpha$, times the difference,returns at timestep t (or "the target") minus former estimation of value of state t (expected return starting at state t)

<br>

- There are two different implementations for of the Monte Carlo Policy Evaluation, namely, **"First Visit Monte Carlo"** and **"Every Visit Monte Carlo"**.

<br>

- Monte Carlo methods converge to the solution with the minimum mean-squared error (MSE), meaning they converge to the value function that minimizes the average squared difference between the estimated and true values.
- By averaging over multiple episodes, Monte Carlo methods reduce the effects of initial conditions and fluctuations, leading to more accurate value estimates as the number of episodes increases.

The Monte Carlo (MC) approach in the context of Reinforcement Learning (RL) refers to methods that use sampled sequences of states, actions, and rewards (i.e., episodes) to estimate the value of a policy. This approach relies on the law of large numbers to estimate the expected returns from states or state-action pairs.

**MC Implementation**

1. **Generate Episodes** : Interact with the environment using the policy π to generate episodes. Each episode consists of a sequence of state-action-reward (SAR) tuples and ends in a terminal state.

Example trajectory : (s0,a0,r0),(s1,a1,r1)...(sT,aT,rT)

2. **Calculate Returns** : For each episode, calculate the return 𝐺𝑡 _from each time step 𝑡 to the end of the episode_. The return is the sum of the discounted rewards from that time step onwards:

$$ G*{t} = ∑*{k=0}^{T-t} γ^{k}r\_{t+k}$$

3. **Estimate Value Function** : Update the value estimates for each state based on the returns observed. The value of a state is the average of the returns observed when visiting that state.


---

## Temporal Difference (TD):
- Temporal Difference methods update value estimates incrementally after each time step, based on the current state, immediate reward, and the estimated value of the next state.
- TD methods combine ideas from both Monte Carlo and dynamic programming approaches, allowing for online learning and updating value estimates without requiring complete episodes.
- But because we didn’t experience an entire episode, we don’t have $G_t$ (expected return). Instead, we estimate $G_t$ by adding $R_{t+1}$ and the discounted value of the next state.
- This is called **bootstrapping**. It’s called this because TD bases its update in part on an existing estimate $V(S_{t+1}) and not a complete sample $G_t$.

<br>

- Temporal Difference, on the other hand, waits for only one interaction (one step) $S_{t+1}$ to form a TD target and update based on the
- $V(S_t) = V(S_t) + α * [R_{t+1} + \gamma * V(S_{t+1}) - V(S_{t})]$
- which translates to:
- The new value of state t updates as, the former estimation of value of state t, plus learning rate α times, the difference between, the sum of reward plus the discounted value of the next state (also known as TD target : $R_{t+1} + \gamma * V(S_{t+1})$), minus the former estimation of value of state t.
- After the update, we continue to interact with this environment with our updated value function.
- This method is called TD(0) or one-step TD (update the value function after any individual step).

<br>

**TD Implementation**
The TD approach estimates value functions using bootstrapping, meaning it updates estimates based on other learned estimates rather than waiting for the final outcome (as in Monte Carlo methods). The key idea is to update the value of a state based on the value of the next state and the reward received, providing a more immediate update.

TD(0) Algorithm
The simplest form of the TD method is TD(0), which updates the value of a state V(s) based on the observed reward and the estimated value of the next state.

The update rule for TD(0) is

$$ V*{(S*{t})} ← V*{(S*{t})} + \alpha [r_{t+1} + \gamma V_{(S_{t+1})} - V_{(S_{t})}] $$

where assuming :

- TD Target : $ r*{t+1} + \gamma V*{(S\_{t+1})} $
- TD Error : TD Target - $ V*{(S*{t})} $

the above update rule turns out to be :

$ V*{(S*{t})} ← V*{(S*{t})} + \alpha $ \* TD Error


<br>

- Unlike Monte Carlo methods, TD methods converge to the solution of a maximum likelihood Markov model, which means they converge to the value function that is closest to the true underlying Markov model of the environment.
- TD methods make use of bootstrapping, where the value of the next state is estimated based on the current value function, incorporating predictions of future states into the update process.

---

- **TD Lambda**
- **n-step TD** : in the TD process, instead of looking one step ahead and update that value to our original state, we can look ahead n steps ahead and update our original state. As this n increases we tend to approach Monte Carlo.
- **TD(λ) (TD lambda)** : is an extension of the basic TD learning method that allows it to consider multiple steps into the future when updating value estimates, rather than just the immediate reward and next state's value estimate. It does this by introducing a parameter λ, which determines the degree to which future steps influence the current value estimate.

<br>

- Here's a high-level overview of how TD(λ) works:
- **Initialization**: Initialize the value function estimate and eligibility trace. The eligibility trace is a memory that records the recent visitation frequency of states or state-action pairs, depending on whether you're using TD(λ) for value function or action-value function estimation.
- **Interaction**: As the agent interacts with the environment, it maintains this eligibility trace, updating it at each step to both 'decay' past state(-action)s and 'boost' the current state(-action).
- **Value Update**: When the agent receives a reward, it uses the TD error (the difference between the observed reward plus next state's value estimate, and the current state's value estimate) to update the value estimates of not just the current state(-action), but all past state(-action)s, with the update being weighted by the eligibility trace.

<br>

- The λ parameter determines the balance between short-term and long-term considerations. When λ = 0, TD(λ) reduces to standard one-step TD learning: the agent only considers the immediate reward and next state's value. When λ = 1, TD(λ) takes into account the entire sequence of future rewards, making it equivalent to Monte Carlo methods.
- TD(λ) offers a flexible way to trade off between bias and variance in reinforcement learning. Smaller values of λ lead to higher bias but lower variance, making learning more stable but potentially less accurate. Larger values of λ lead to lower bias but higher variance, making learning potentially more accurate but also more unstable.

### Differences Between MC and TD

**Update Frequency**

Monte Carlo: Updates the value estimates only at the end of an episode. This requires waiting until the episode is complete to calculate the total return 𝐺𝑡​ for each state visited.

Temporal Difference: Updates the value estimates after each time step, making it more efficient and allowing for updates during the episode.

**Variance and Bias**

Monte Carlo: Estimates are unbiased because they are based on actual returns observed. However, they can have high variance due to the complete episode's variability.

Temporal Difference: Estimates are biased because they use the current value estimates (bootstrapping). However, they typically have lower variance compared to Monte Carlo methods.

---

- **MC & TD Convergence**:
- Monte Carlo methods converge to the solution with the minimum mean-squared error (MSE), which means they converge to the value function that minimizes the average squared difference between the estimated and true values.
- The MSE is a measure of how close the estimated values are to the true values. By minimizing the MSE, MC methods aim to find the value function that provides the most accurate representation of the true underlying value function.
- Convergence to the minimum MSE occurs as the number of episodes increases. By averaging over multiple episodes, MC methods reduce the effects of initial conditions and random fluctuations, leading to more accurate value estimates.

<br>

- Temporal Difference methods converge to the solution of a maximum likelihood Markov model. This means they converge to the value function that is closest to the true underlying Markov model of the environment.
- The maximum likelihood Markov model represents the most likely transition dynamics and rewards in the environment. TD methods aim to approximate this model by iteratively updating value estimates based on observed state transitions and immediate rewards.
- By incorporating predictions of future states into the update process, TD methods gradually refine the value estimates, aligning them with the expected behavior of the environment according to the maximum likelihood model.

