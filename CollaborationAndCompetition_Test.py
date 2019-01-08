import numpy as np
import torch
from collections import deque
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
from agent import Agent

env = UnityEnvironment(file_name="Tennis.app")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# Number of agents in the environment
num_agents = len(env_info.agents)
print('num_agents = ', num_agents)

# Number of actions in the environment
action_size = brain.vector_action_space_size
print('action_size = ', action_size)

# Examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('state_size = ', state_size)


# Create the ddpg agents
agent_0 = Agent(state_size=state_size, action_size=action_size, num_agents=num_agents)
agent_1 = Agent(state_size=state_size, action_size=action_size, num_agents=num_agents)

agent_0.actor_local.load_state_dict(torch.load('checkpoint_actor_agent_0.pth'))
agent_1.actor_local.load_state_dict(torch.load('checkpoint_actor_agent_1.pth'))

def maddpg(n_episodes=5, max_t=1000):                                            # Play for a max of 5 episodes
    scores_deque = deque(maxlen=100)
    scores = []

    for i_episode in range(1, n_episodes+1):
        score = 0                                                                # Set the score to 0 before the episode begins
        env_info = env.reset(train_mode=False)[brain_name]                       # Reset the environment
        states = np.reshape(env_info.vector_observations, (1, 48))

        agent_0.reset()
        agent_1.reset()

        for t in range(max_t):
            actions = np.zeros([num_agents, action_size])
            actions[0] = agent_0.act(states)[0][:2]                             # Select an action agent_0
            actions[1] = agent_1.act(states)[0][2:]                             # Select an action agent_1

            env_info = env.step(actions)[brain_name]                            # Take selected actions on the environment

            next_states = np.reshape(env_info.vector_observations, (1, 48))     # Get next state from the environment
            rewards = env_info.rewards                                          # Get reward for taking selected actions
            dones = env_info.local_done                                         # Check to see if the episode has terminated or completed
            states = next_states                                                # Set the state as the new_state of the env
            score += np.max(rewards)                                            # Update the scores based on rewards

            if np.any(dones):                                                   # Break the loop after the episode is done
                break

        scores_deque.append(score)
        scores.append(score)

        print('\rEpisode {}\tAverage Score: {:.4f}'.format(i_episode, np.mean(scores_deque)))

    return scores

scores = maddpg()

fig = plt.figure()                              # Plotting the graph showing the increase in Average scores
plt.xlabel('Episodes')
plt.ylabel('Scores')
plt.plot(np.arange(len(scores)), scores)
plt.show()

env.close()