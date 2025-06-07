import os
import gymnasium
import ale_py
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback

# Create preprocessed Skiing environment
env_id = "ALE/Skiing-v5"

env = make_atari_env(env_id, n_envs=8, seed=42)  # Parallel environments
env = VecFrameStack(env, n_stack=4)  # Frame stacking
env = VecVideoRecorder(
    env,
    video_folder="train_videos/",
    record_video_trigger=lambda step: step % 10000 == 0,  
    video_length=1000, 
    name_prefix="ppo_training"
)

# Define PPO policy network (CNN)
policy_kwargs = dict(
    net_arch=[dict(pi=[512, 256], vf=[512, 256])]  # Actor/Critic MLP layers after CNN
)

# Initialize PPO agent
model = PPO(
    "CnnPolicy",
    env,
    learning_rate=2.5e-3,
    n_steps=128,
    batch_size = 256,
    gamma=0.99,
    ent_coef=0.01,  # Encourage exploration
    policy_kwargs=policy_kwargs,
    tensorboard_log="./ppo_skiing_tensorboard/",
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
        plt.savefig("ppo_training_plots.png")
        plt.show()

reward_loss_logger = RewardLossLogger()

obs = env.reset()
print("Initial observation shape:", obs.shape)

# Train the agent
model.learn(
    total_timesteps=500_000,
    callback=reward_loss_logger
)

# Save and evaluate
model.save("ppo_skiing")

env.close()
