# Project 2: Continuous Control

The project consists of training an agent to control an arm to reach a goal position

The agent will operate a robotic arm with two joints within the reacher Unity environment. The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with 4 numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

A positive reward is given when the agent reach a moving goal, so the aim of the agent is to reach the goal and keep the goal within reach for as much time as possible

The project is part of the Udacity deep learning course

## Table of contents


## Getting Started

These instructions will help with setting up the project

### Prerequisites
Create a virtual environment with conda:
```
conda env create -f environment.yml
conda activate drl
```

This will take care of installing all the dependencies needed

### Installing

The following steps allows to setting up the project correctly

Say what the step will be

```
move the 'Reacher_Linux' folder inside 'environment/'
```

## Running the code

The project is divided in two sections: training & evaluation

### Training

To start training the agent run the following:

```
python Reacher-DDPG.py
```
The code will generate Tensorboard stats for visualisation. You can see them by running:
```tensorboard --logdir=runs``` from the ```drl_continuous_control``` folder

### Evaluation
During each run a snapshot of the agent is taken at regular intervals.

To look at the agent in action using the previously saved model run the following:

```
python Reacher-DDPG-evaluate.py
```

## Results
Here is a video of the agent in action:

https://youtu.be/2kSd00ENSbI

Here is a graph of the progression of the score from Tensorboard (up) and the average of the last 100 scores (down)
![alt text](images/tensorboard_continuous.png)
The agent successfully reaches an average of 30 points around episode 325

Here is the log from console:
![alt text](images/log_continuous.png)
