import gymnasium as gym

class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        done = False
        info = {}
        total_reward = 0.0
        for _ in range(self.skip):
            obs, reward, done, _, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, False, info