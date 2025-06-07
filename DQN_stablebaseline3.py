import os
import gymnasium
import ale_py
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from gymnasium.wrappers import RecordVideo
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList

# Create preprocessed Skiing environment
env_id = "ALE/Skiing-v5"

# env = gymnasium.make(env_id,render_mode="rgb_array")
# env = GrayscaleObservation(env, keep_dim=True)  # Output shape (H,W)
# env = ResizeObservation(env, (84, 84))      # Resize to 84x84
# env = FrameStackObservation(env, 4)        # Stack 4 frames -> (84,84, 4)
# env = RecordVideo(
#     env,
#     video_folder="dqn videos/",
#     episode_trigger=lambda episode_id: episode_id % 20 == 0,
#     name_prefix="dqn_skiing_eval"
# )

env = make_atari_env(env_id, n_envs=1, seed=42)  # Parallel environments
env = VecFrameStack(env, n_stack=4)  # Frame stacking
env = VecVideoRecorder(
    env,
    video_folder="train_videos/",
    record_video_trigger=lambda step: step % 10000 == 0,  
    video_length=1000, 
    name_prefix="ppo_training"
)

eval_callback = EvalCallback(
    env,
    best_model_save_path='./logs/best_model',
    log_path='./logs/results',
    eval_freq=10000,
    deterministic=True,
    render=False
)

# Define DQN policy network (CNN)
policy_kwargs = dict(
    net_arch=[512, 256]
)

# Initialize A2C agent
model = DQN(
    "CnnPolicy",
    env,
    learning_rate=2.5e-3,
    buffer_size=100_000,
    learning_starts=10_000,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1_000,
    policy_kwargs=policy_kwargs,
    tensorboard_log="./dqn_skiing_tensorboard/",
    verbose=1,
)

class RewardLossLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLossLogger, self).__init__(verbose)
        self.episode_rewards = []
        self.losses = []

    def _on_step(self) -> bool:
        # Log episode reward when an episode is done.
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        # Record loss if available (this depends on the algorithm implementation)
        if "loss" in self.locals:
            self.losses.append(self.locals["loss"])
        return True

    def _on_training_end(self) -> None:

        with open("ppo_rewards.txt", "w") as f:
            for r in self.episode_rewards:
                f.write(f"{r}\n")

        if self.losses:
            with open("ppo_losses.txt", "w") as f:
                for l in self.losses:
                    f.write(f"{l}\n")

        # Plot reward history and loss history at the end of training.
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(self.episode_rewards)
        ax[0].set_title("Reward History")
        ax[0].set_xlabel("Episode")
        ax[0].set_ylabel("Reward")

        if self.losses:  # Plot loss only if data was recorded
            ax[1].plot(self.losses)
            ax[1].set_title("Loss History")
            ax[1].set_xlabel("Training Step")
            ax[1].set_ylabel("Loss")
        else:
            ax[1].text(0.5, 0.5, "Loss data not available", ha="center", va="center")
            ax[1].set_title("Loss History")

        plt.tight_layout()
        plt.savefig("dqn_training_plots.png")
        plt.show()

# Train the agent
model.learn(
    total_timesteps=1_000_000,
    callback=eval_callback
)

# Save and evaluate
model.save("dqn_skiing")
env.close()
