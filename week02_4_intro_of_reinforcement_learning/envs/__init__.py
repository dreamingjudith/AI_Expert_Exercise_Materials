from gym.envs.registration import register

register(
    id='Maze-v0',
    entry_point='envs.maze_env:MazeEnvSample5x5',
    max_episode_steps=2000,
)

register(
    id='MazeSample5x5-v0',
    entry_point='envs.maze_env:MazeEnvSample5x5',
    max_episode_steps=2000,
)

register(
    id='MazeRandom5x5-v0',
    entry_point='envs.maze_env:MazeEnvRandom5x5',
    max_episode_steps=2000,
    nondeterministic=True,
)

register(
    id='MazeSample10x10-v0',
    entry_point='envs.maze_env:MazeEnvSample10x10',
    max_episode_steps=10000,
)

register(
    id='MazeRandom10x10-v0',
    entry_point='envs.maze_env:MazeEnvRandom10x10',
    max_episode_steps=10000,
    nondeterministic=True,
)

register(
    id='MazeSample3x3-v0',
    entry_point='envs.maze_env:MazeEnvSample3x3',
    max_episode_steps=1000,
)

register(
    id='MazeRandom3x3-v0',
    entry_point='envs.maze_env:MazeEnvRandom3x3',
    max_episode_steps=1000,
    nondeterministic=True,
)


register(
    id='MazeSample100x100-v0',
    entry_point='envs.maze_env:MazeEnvSample100x100',
    max_episode_steps=1000000,
)

register(
    id='MazeRandom100x100-v0',
    entry_point='envs.maze_env:MazeEnvRandom100x100',
    max_episode_steps=1000000,
    nondeterministic=True,
)

register(
    id='MazeRandom10x10-plus-v0',
    entry_point='envs.maze_env:MazeEnvRandom10x10Plus',
    max_episode_steps=1000000,
    nondeterministic=True,
)

register(
    id='MazeRandom20x20-plus-v0',
    entry_point='envs.maze_env:MazeEnvRandom20x20Plus',
    max_episode_steps=1000000,
    nondeterministic=True,
)

register(
    id='MazeRandom30x30-plus-v0',
    entry_point='envs.maze_env:MazeEnvRandom30x30Plus',
    max_episode_steps=1000000,
    nondeterministic=True,
)


register(
    id='MyMountainCar-v0',
    entry_point='envs.my_mountaincar:MyMountainCarEnv',
    max_episode_steps=2000,
    reward_threshold=-110.0
)

register(
    id='MyCartPole-v0',
    entry_point='envs.my_cartpole:MyCartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)


register(
    id='MyPendulum-v0',
    entry_point='envs.my_pendulum:MyPendulumEnv',
    max_episode_steps=200
)


register(
    id='MyCliffWalking-v0',
    entry_point='envs.my_cliff_walk:CliffWalkingEnv',
    max_episode_steps=200,
)