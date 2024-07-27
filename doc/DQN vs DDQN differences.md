## DQN & DDQN implementation

### How does DDQN differ from DQN?

Fixed Q-Target: In order to calculate the Q-Target we need to estimate the discounted optimal Q-value of the next state by using Bellman equation. The problem is that the same network weights are used to calculate the Q-Target and the Q-value. This means that everytime we are modifying the Q-value, the Q-Target also moves with it. To avoid this issue, a separate network with fixed parameters is used for estimating the Temporal Difference Target. The target network is updated by copying parameters from our Deep Q-Network after certain C steps.

Double DQN: Method to handle overestimation of Q-Values. This solution uses two networks to decouple the action selection from the target Value generation:

DQN Network to select the best action to take for the next state (the action with the highest Q-Value)
Target Network to calculate the target Q-Value of taking that action at the next state. This approach reduces the Q-Values overestimation, it helps to train faster and have more stable learning.

- [source](https://huggingface.co/learn/deep-rl-course/unit3/glossary)

More sources :

- [Blog by David R. Pugh](https://davidrpugh.github.io/stochastic-expatriate-descent/categories/#deep-reinforcement-learning)
- [Hugging Face](https://huggingface.co/learn/deep-rl-course/unit3/deep-q-algorithm)
- [Thomas Simonini](https://www.freecodecamp.org/news/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682/)
- [Johny Code](https://www.youtube.com/watch?v=EUrWGTCGzlA&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte) + [github code implementation](https://github.com/johnnycode8/gym_solutions)
- [CIS 522 - Deep Learning](https://www.youtube.com/watch?v=Lb5ADHnRQV8)
- [ai.stackexchange](https://ai.stackexchange.com/questions/22776/what-exactly-is-the-advantage-of-double-dqn-over-dqn)
- [youtube - Deep Reinforcement Learning with Double Q-learning](https://www.youtube.com/watch?v=FTfkpCCaORI&list=PLCip3d1iHEMWlWV9fGh4eDTUs_otqosZU&index=5)
- [Saasha Nair](https://www.youtube.com/watch?v=fnVIgAGhA08)
