import gymnasium as gym

class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action, last_info = None):
        done = False
        info = {}
        total_reward = 0.0
        for _ in range(self.skip):
            obs, reward, done, _, info = self.env.step(action)
            if last_info:
                if info["number_of_lines"] > last_info["number_of_lines"]:
                    reward += 10 * (info["number_of_lines"] - last_info["number_of_lines"])
            total_reward += reward
            last_info = info
            if done:
                total_reward -= 5
                break
        return obs, total_reward, done, False, info