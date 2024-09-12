import gymnasium as gym

class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        done = False
        info = {}
        obs, total_reward, done, _, info = self.env.step(action)
        for _ in range(self.skip):
            if done:
                break
            obs, reward, done, _, info = self.env.step(0)
            total_reward += reward
        return obs, total_reward, done, False, info