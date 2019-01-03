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

# Number of actions in the environment
action_size = brain.vector_action_space_size

# Examine the state space
states = env_info.vector_observations
state_size = states.shape[1]

# agent = []        ##### TRY THIS LATER. PUT ALL AGENTS IN AN ARRAY.

# Create the ddpg agents
# for i in range(num_agents):
agent_0 = Agent(num_agents=num_agents, state_size=len(states.flatten()), action_size=action_size, random_seed=3)
agent_1 = Agent(num_agents=num_agents, state_size=len(states.flatten()), action_size=action_size, random_seed=3)

# for i in range(1):                                         # play game for 5 episodes
#     env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
#     states = env_info.vector_observations                  # get the current state (for each agent)
#     scores = np.zeros(num_agents)                          # initialize the score (for each agent)
#     while True:
#         actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
#         actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
#         env_info = env.step(actions)[brain_name]           # send all actions to tne environment
#         next_states = env_info.vector_observations         # get next state (for each agent)
#         rewards = env_info.rewards                         # get reward (for each agent)
#         dones = env_info.local_done                        # see if episode finished
#
#         # print('states: \n', states)
#         # print('actions: \n', actions)
#
#         scores += env_info.rewards                         # update the score (for each agent)
#         states = next_states                               # roll over states to next time step
#         if np.any(dones):                                  # exit loop if episode finished
#             break
#     print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


def maddpg(n_episodes=2000, max_t=1000):
    """Deep Deterministic Policy Gradient

    Params:
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of time-steps per episode"""

    scores_deque = deque(maxlen=100)
    scores = []
    best_score = -np.inf

    for i_episode in range(1, n_episodes+1):
        score = np.zeros(num_agents)                       # Set the score to 0 before the episode begins
        env_info = env.reset(train_mode=True)[brain_name]  # Reset the environment
        states = np.reshape(env_info.vector_observations, (1, 48))  # Get states for all agents

        agent_0.reset()
        agent_1.reset()

        for t in range(max_t):
            actions = np.zeros([num_agents, action_size])
            actions[0] = agent_0.act(states)                 # Select an action agent_0
            actions[1] = agent_1.act(states)                 # Select an action agent_1
            actions = actions.flatten()
            env_info = env.step(actions)[brain_name]        # Take selected actions on the environment
            next_states = env_info.vector_observations      # Get next state from the environment
            rewards = env_info.rewards                      # Get reward for taking selected actions
            dones = env_info.local_done                     # Check to see if the episode has terminated or completed

            agent_0.step(states, actions, rewards[0], next_states, dones, t, 0) # Learning from sampled set of experiences
            agent_1.step(states, actions, rewards[1], next_states, dones, t, 1) # Learning from sampled set of experiences

            states = next_states                            # Set the state as the new_state of the env
            score += np.max(rewards)                        # Update the scores based on rewards

            if np.any(dones):                               # Break the loop after the episode is done
                break

        ep_best_score = np.max(score)
        scores_deque.append(ep_best_score)
        scores.append(ep_best_score)

        if ep_best_score > best_score:
            best_score = ep_best_score

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if i_episode % 10 == 0:
            torch.save(agent_0.actor_local.state_dict(), 'checkpoint_actor_agent_0.pth')
            torch.save(agent_0.critic_local.state_dict(), 'checkpoint_critic_agent_0.pth')
            torch.save(agent_1.actor_local.state_dict(), 'checkpoint_actor_agent_1.pth')
            torch.save(agent_1.critic_local.state_dict(), 'checkpoint_critic_agent_1.pth')
        if np.mean(scores_deque) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            torch.save(agent_0.actor_local.state_dict(), 'checkpoint_actor_agent_0.pth')
            torch.save(agent_0.critic_local.state_dict(), 'checkpoint_critic_agent_0.pth')
            torch.save(agent_1.actor_local.state_dict(), 'checkpoint_actor_agent_1.pth')
            torch.save(agent_1.critic_local.state_dict(), 'checkpoint_critic_agent_1.pth')
            break

    return scores

scores = maddpg()

fig = plt.figure()                              # Plotting the graph showing the increase in Average scores
plt.xlabel('Episodes')
plt.ylabel('Scores')
plt.plot(np.arange(len(scores)), scores)
plt.show()

env.close()