import torch
import numpy as np
import gymnasium as gym
import torch.multiprocessing as mp
import torchvision.transforms as T

from actor_critic import ActorCritic

def transformImage(array: np.ndarray) -> torch.Tensor:
    """
    Transform numpy array to tensor

    Args:
        array: numpy array

    Return:
        tensor: tensor
    """
    transformed_array = T.transforms.Compose([
        T.transforms.ToTensor(),
        T.transforms.Grayscale(num_output_channels=1),
        T.transforms.Resize((84, 84))
    ])
    return transformed_array(array).numpy()

class Worker(mp.Process):
    global_constant_max_episode_across_all_workers = 10000
    global_constant_step_before_sync = 5

    def __init__(
        self,
        global_actor_critic,
        optimizer,
        gamma,
        beta,
        lr,
        name,
        global_eps_idx,
        env_id,
    ):
        super(Worker, self).__init__()
        self.name = "Worker-%i" % name
        self.episode_idx = global_eps_idx
        self.env = gym.make(env_id, render_mode="human")
        self.wrapper = gym.wrappers.RecordEpisodeStatistics(
            self.env,
            deque_size=1 * Worker.global_constant_max_episode_across_all_workers,
        )
        self.global_actor_critic = global_actor_critic
        # input_dims = [sum(obs_space.shape[0] for obs_space in self.wrapper.observation_space.values())]
        self.local_actor_critic = ActorCritic(
            # input_dims, 
            self.wrapper.action_space.n, gamma, beta
        )
        self.optimizer = optimizer

    def run(self):
        t_step = 1
        while (
            self.episode_idx.value
            < Worker.global_constant_max_episode_across_all_workers
        ):
            done = False
            observation, _ = self.wrapper.reset(seed=42)
            score = 0
            self.local_actor_critic.memory.clear()
            while not done:
                action = self.local_actor_critic.choose_action(
                    transformImage(observation["matrix_image"])
                )
                observation_, reward, done, _, info = self.wrapper.step(action)
                score += reward
                self.local_actor_critic.memory.store(
                    transformImage(observation_["matrix_image"]), action, reward
                )
                if t_step % Worker.global_constant_step_before_sync == 0 or done:
                    loss = self.local_actor_critic.calculate_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for local_param, global_param in zip(
                        self.local_actor_critic.parameters(),
                        self.global_actor_critic.parameters(),
                    ):
                        global_param._grad = local_param.grad
                    self.optimizer.step()
                    self.local_actor_critic.load_state_dict(
                        self.global_actor_critic.state_dict()
                    )
                    self.local_actor_critic.memory.clear()
                t_step += 1
                observation = observation_
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
            print(self.name, "episode", self.episode_idx.value, "reward %.1f" % score)
        print(np.array(self.wrapper.return_queue).flatten())

if __name__ == "__main__":
    raise NotImplementedError("Please implement the main function.")