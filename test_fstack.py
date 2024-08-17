import gym_tetris
import torchvision.transforms as T

from PIL import Image
from numpy import ndarray
from utils import preprocess_frame_stack
from gym_tetris.actions import MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack, GrayScaleObservation

def save_frame(frame: ndarray, path: str) -> None:
     # Konversi frame ke PIL Image
    # img = Image.fromarray(frame)
    img = T.ToPILImage()(frame)
    # Simpan gambar
    img.save(path)



if __name__ == "__main__":
    env = gym_tetris.make("TetrisA-v3")
    env = JoypadSpace(env, MOVEMENT)
    env = GrayScaleObservation(env)
    env = FrameStack(env, 4)

    done = True

    for step in range(25):
        if done:
            state, info = env.reset()

        state, reward, done, truncated, info = env.step(env.action_space.sample())
        print(info)
        env.render()

    env.close()