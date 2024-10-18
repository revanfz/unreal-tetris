import os
import gymnasium as gym

from gymnasium import error
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.video.fx.resize import resize


class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
        self.env = env

    def step(self, action, last_info = None):
        done = False
        info = {}
        total_reward = 0.0
        for _ in range(self.skip):
            obs, reward, done, truncated, info = self.env.step(action)
            if last_info:
                if info["number_of_lines"] > last_info["number_of_lines"]:
                    reward += 10 * (info["number_of_lines"] - last_info["number_of_lines"])
            total_reward += reward
            last_info = info
            if done:
                total_reward -= 5
                break
        return obs, total_reward, done, truncated, info
    

class RecordVideo(gym.Wrapper):
    def __init__(self, env, path: str, format: str):
        super().__init__(env)
        self.env = env
        self.path = path
        self.format = format
        self.episode = 1
        self.frame_captured = []

        if not os.path.isdir(self.path):
            os.makedirs(self.path)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.frame_captured.append(self.env.render().copy())
        if done:
            self.close()
            self.episode += 1
        return obs, reward, done, truncated, info
    
    def close(self):
        if len(self.frame_captured) > 0:
            if self.format in ["mp4", "avi", "webm", "ogv", "gif"]:
                filename = "{}/{}.{}".format(self.path, self.episode, self.format)
                clip = ImageSequenceClip(self.frame_captured, fps=self.env.metadata.get("fps", 60)).fx(resize, width=480)
                if self.format == "gif":
                    clip.write_gif(filename)
                else:
                    clip.write_videofile(filename, threads=2)
            else:
                raise error.Error(
                    f"Invalid recording format. Supported are mp4, avi, webm, ogv, gif"
                )
        self.frame_captured.clear()