## Vectorize Environments

### make_vec_env

The **make_vec_env** function is a utility for easily creating a vectorized environment where multiple instances of an environment can be run in parallel. This is particularly useful for algorithms that benefit from collecting diverse experiences from multiple sources simultaneously, such as A2C or PPO.

**Key Features:**

- Multiple Instances: It allows for the creation of multiple instances of an environment, which can be run in parallel. This helps in collecting more samples in a shorter period, thereby accelerating the training process.
- Automatic Setup: It handles the setup of these environments in a way that they can be used directly with the learning algorithms provided by Stable-Baselines3.
- Custom Configuration: Users can specify the number of environments to create, and optionally pass environment-specific parameters such as custom wrappers or configurations.

### DummyVecEnv

**DummyVecEnv** is a specific kind of vectorized environment wrapper that runs multiple environments sequentially in a single process. While it does not offer the true parallelism of processes or threads, it still provides an easy way to manage and step through multiple environments simultaneously, making it easier to integrate with algorithms expecting vectorized inputs.

**Key Features:**

- Simplicity: It’s a simpler alternative to more complex parallel execution environments and is often used for testing or development when the overhead of true parallelism is unnecessary.
- Batching Actions: Despite running sequentially, DummyVecEnv takes actions for all environments at once and steps through them, which can be more efficient than stepping through each environment individually in a loop.
- Compatibility: It ensures compatibility with other parts of the Stable-Baselines3 library, which expect inputs from multiple environments at once.
