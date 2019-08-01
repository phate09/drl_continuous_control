from unityagents import UnityEnvironment


class Reacher_wrapper:
    def __init__(self, path, seed=0):
        env = UnityEnvironment(file_name=path, seed=seed)
        # get the default brain
        brain_name = env.brain_names[0]
        self.brain = env.brains[brain_name]

        # reset the environment
        self.env_info = env.reset(train_mode=True)[brain_name]

        # number of agents in the environment
        print('Number of agents:', len(self.env_info.agents))

        # number of actions
        self.action_size = self.brain.vector_action_space_size
        self.state_size = self.brain.vector_action_space_size
        self.action_type = self.brain.vector_action_space_type


if __name__ == '__main__':
    reache = Reacher_wrapper(path="Reacher_Linux_NoVis/Reacher.x86_64")
