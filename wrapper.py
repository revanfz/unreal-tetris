import gymnasium as gym

class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        for i in range(self.repeat):
            if not i:
                obs, reward, done, _, info = self.env.step(action)
            else:
                obs, reward, done, _, info = self.env.step(0)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, False, info