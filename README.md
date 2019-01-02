# DeepRL-Nanodegree-Project3 (CollaborationAndCompetition)

* Set-up: Two-player game where agents control rackets to bounce ball over a net.
* Goal: The agents must bounce ball between one another while not dropping or sending ball out of bounds.
* Agents: The environment contains two agent linked to a single Brain named TennisBrain. After training you can attach another Brain named MyBrain to one of the agent to play against your trained model.
* Agent Reward Function (independent):
  1) +0.1 To agent when hitting ball over net.
  1) -0.1 To agent who let ball hit their ground, or hit ball out of bounds.
* Brains: One Brain with the following observation/action space.
* Vector Observation space: 8 variables corresponding to position and velocity of ball and racket.
* Vector Action space: (Continuous) Size of 2, corresponding to movement toward net or away from net, and jumping.
* Visual Observations: None.
* Reset Parameters: One, corresponding to size of ball.
* Benchmark Mean Reward: 2.5
