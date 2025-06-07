from math import e
import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from collections import deque, defaultdict
import random
import matplotlib.pyplot as plt
import gymnasium as gym
import ale_py
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

gym.register_envs(ale_py)

# Environment setup with proper dimension handling
env = gym.make('ALE/Skiing-v5', render_mode = "rgb_array")
env = GrayscaleObservation(env, keep_dim=False)  # Output shape (H,W)
env = ResizeObservation(env, (84, 84))      # Resize to 84x84
env = FrameStackObservation(env, 4)        # Stack 4 frames -> (4,84,84)
# Record the episodes every 10 episodes
env = gym.wrappers.RecordVideo(
    env,
    episode_trigger=lambda num: num % 50 == 0,
    video_folder="saved-video-folder",
    name_prefix="video-",
)
env = RecordEpisodeStatistics(env)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("pl_training_log.txt"),
        logging.StreamHandler()
    ]
)
state_shape = env.observation_space.shape  # Should be (4, 84, 84)
action_space = env.action_space.n

#Hyperparameters
learning_rate = 0.5
gamma = 0.98
episodes = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # choose GPU if available
print(f"Using device: {device}")

# Model Construction
class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._conv_output_size(input_shape), 1024),  # increase hidden layer
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
     )
        self.to(device)  # Move model to device (GPU or CPU)

    def _conv_output_size(self, shape):
        with torch.no_grad():
            return self.conv(torch.zeros(1, *shape)).view(1, -1).size(1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class SkiingPolicyGradient:
    def __init__(self):
        self.input_shape = env.observation_space.shape
        self.num_actions = env.action_space.n
        
        self.policy = PolicyNetwork(self.input_shape, self.num_actions).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.gamma = gamma
        self.saved_log_probs = []
        self.rewards = []
        self.reward_scale = 0.01  # New hyperparameter
        self.entropy_coef = 0.02  # For exploration

        # Episode policy and reward history
        self.policy_history = []
        self.reward_episode = []
        self.reward_history = []  # Track episode rewards
        self.loss_history = []    # Track policy losses

        # self.last_action = None  # Track previous action
        # self.last_log_prob = None  # Track previous log probability

        self.best_reward = -float('inf') # track best policy
        self.best_model_path = "best_policy.pth"

    def _preprocess_state(self, state):
        # Convert state to NumPy array if needed
        if not isinstance(state, np.ndarray):
            state = np.asarray(state)
        # Ensure channels-first format
        if state.ndim == 3 and state.shape[-1] == 4:
            state = state.transpose(2, 0, 1)
        return torch.FloatTensor(state).unsqueeze(0).to(device)
    def _get_shaped_reward(self, state, original_reward):
        """Custom reward"""
        # Extract position/velocity estimates from state
        # y_position = state[0].mean()  # Proxy for vertical progress
        # x_center = abs(state[1].mean() - 0.5)  # Center alignment

        # Initialize velocity calculation
        velocity = 0.0
        
        # Only calculate velocity if we have previous state
        if self.prev_state is not None:
            try:
                # Convert states to numpy arrays if needed
                current_state = np.array(state) if not isinstance(state, np.ndarray) else state
                prev_state = np.array(self.prev_state) if not isinstance(self.prev_state, np.ndarray) else self.prev_state
                
                # Calculate velocity as frame difference
                velocity = np.abs(current_state - prev_state).mean()
            except Exception as e:
                print(f"Velocity calculation error: {str(e)}")
                velocity = 0.0
        
        shaped_reward = (
            original_reward +
            # 0.1 * y_position +
            # 1.0 / (x_center + 1e-6) +
            0.5 * velocity
        )

        # Store current state for next calculation
        self.prev_state = state.copy() if isinstance(state, np.ndarray) else np.array(state)
        
        return shaped_reward * self.reward_scale

    def select_action(self, state, episode = 0):

        state = self._preprocess_state(state)
        logits = self.policy(state)
        # print("logits shape:", logits.shape)
        # action = env.action_space.sample() #V1

        dist = Categorical(logits=logits)
        # action = dist.sample() # V2
        
        if episode < 20:
            action = torch.tensor([env.action_space.sample()], dtype=torch.long, device = device)
        else:
            # Value gradient 
            action = dist.sample()
        # self.last_log_prob = log_prob # Use stored log probability
        # Store for policy update

        # add log probability of chosen action to history
        # if len(self.policy.policy_history) > 0:
        #     self.policy.policy_history = torch.cat[self.policy_history, dist.log_prob(action).reshape(1)]
        # else:
        #     self.policy.policy_history = dist.log_prob(action).reshape(1)

        log_prob = dist.log_prob(action)
        self.saved_log_probs.append(log_prob)
        # print("Selected action:", action)
        return action

    def update_policy(self):
        R = 0
        rewards = []

        # Discount future rewards back to the present using gamma

        # Calculate discounted returns
        for r in self.rewards[::-1]:
            # assert(r ==1.0)
            R = r + self.gamma * R
            rewards.insert(0, R)

        # Scale rewards
        if len(rewards) == 0 or len(self.saved_log_probs) == 0:
            return
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + float(np.finfo(np.float32).eps))

        # Check for empty buffers
        if not self.saved_log_probs:
            return

        # Verify tensor types
        assert all(isinstance(lp, torch.Tensor) for lp in self.saved_log_probs), \
        "Log probs must be tensors"

        # Calculate loss
        policy_loss = []
        entropy = []
        for log_prob,R in zip(self.saved_log_probs, rewards):
            log_prob = log_prob.view(1)
            policy_loss.append(-log_prob * R)
            entropy.append(-log_prob.exp() * log_prob) # entropy = -sum(p * logp)

        policy_loss = torch.stack(policy_loss).sum()
        entropy = torch.stack(entropy).sum()
        # policy_loss = torch.stack([
        #     -log_prob * R for log_prob, R in zip(self.saved_log_probs, rewards)
        # ]).sum()

        # Final loss calculation
        loss = policy_loss - self.entropy_coef * entropy

        # Update network weights   
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5) # Gradient clipping, originally 0.5
        self.optimizer.step()

        # Store training metrics in AGENT (not policy)
        self.loss_history.append(loss.item())
        self.reward_history.append(sum(self.rewards))        
        #Save and intialize episode history counters
        self.policy_history.append(torch.cat(self.saved_log_probs).sum().item())  # Save policy log probabilities

        # Clear buffers
        self.rewards.clear()
        self.saved_log_probs.clear()

        # Reset action history after update
        # self.last_action = None
        # self.last_log_prob = None

    def save_checkpoint(self, episode):
        torch.save({
            'episode': episode,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'rewards': self.rewards,
        }, f'skiing_pg_checkpoint_ep{episode}.pth')

    def load_checkpoint(filename, model, optimizer=None, device='cpu'):
        """Load saved training state"""
        checkpoint = torch.load(filename, map_location=device,weights_only=False)
    
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
    
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
        return {
            'episode': checkpoint['episode'],
            'rewards': checkpoint['rewards'],
        }

    if os.path.exists("checkpoint_ep400.pth"):
    # Resume training
         checkpoint = load_checkpoint("checkpoint_ep400.pth", policy, optimizer)
         start_ep = checkpoint['episode'] + 1
         episode_rewards = checkpoint['rewards']
    else:
         # Start new training
         start_ep = 0
         episode_rewards = []

class TrainingVisualizer:
    def __init__(self, window_size=100):
        # Initialize data containers
        self.episode_rewards = []
        # self.td_errors = []
        # self.bellman_errors = []
        self.window_size = window_size
        
        # Create figure with subplots
        self.fig, (self.ax1) = plt.subplots(
            1, 1, 
            figsize=(8, 8),
            constrained_layout = True 
            )
        # plt.ion()  # Enable interactive mode for real-time updates

    def update_metrics(self, episode, reward):
        """Store and visualize metrics in real-time"""
        # Store metrics
        self.episode_rewards.append(reward)
        # if td_error is not None:
        #     self.td_errors.extend(td_error)
        # if bellman_error is not None:
        #     self.bellman_errors.append(bellman_error)

        # Update plots every 10 episodes
        if episode % 10 == 0:
            self._update_plots()
            # plt.pause(0.001)  # Brief pause to update display

    def _update_plots(self):
        """Internal method to refresh all plots"""
        # Clear previous drawings
        self.ax1.clear()
        # self.ax2.clear()
        # self.ax3.clear()

        # Plot 1: Episode Rewards
        self.ax1.plot(self.episode_rewards, label='Raw')
        self.ax1.plot(self._moving_average(self.episode_rewards), 'r-', label='Smoothed')
        self.ax1.set_title('Episode Rewards')
        self.ax1.set_ylabel('Total Reward')
        # self.ax1.legend() #V1
        self.ax1.legend(loc='upper left') #V2

        # Plot 2: TD-Errors
        # if self.td_errors:
        #     self.ax2.plot(self.td_errors, alpha=0.3, label='Raw')
        #     self.ax2.plot(self._moving_average(self.td_errors), 'r-', label='Smoothed')
        #     self.ax2.set_title('TD-Errors')
        #     self.ax2.set_ylabel('Error Magnitude')
        #     # self.ax2.legend()
        #     self.ax1.legend(loc='upper right') #V2

        # # Plot 3: Bellman Errors
        # if self.bellman_errors:
        #     self.ax3.plot(self.bellman_errors[-1], 'g-', label='Batch Average')
        #     self.ax3.set_title('Bellman Errors (Loss)')
        #     self.ax3.set_ylabel('Loss')
        #     self.ax3.set_xlabel('Training Batches')
        #     # self.ax3.legend()
        #     self.ax1.legend(loc='upper right') #V2

        # plt.tight_layout()

    def _moving_average(self, data):
        """Calculate moving average for smoothing"""
        return np.convolve(data, np.ones(self.window_size)/self.window_size, mode='valid')

    def save_plots(self, filename='PL_results.png'):
        """Save final plots to file"""
        plt.ioff()
        self._update_plots()
        plt.savefig(filename, bbox_inches='tight') #V2
        plt.close()

# Training loop
agent = SkiingPolicyGradient()
visualizer = TrainingVisualizer()

try: 
    # running_reward = 10
    for episode in range(episodes):
        state, _ = env.reset() # Reset environment and record the starting state
        # agent.last_action = None  # Reset at episode start
        # agent.last_log_prob = None
        agent.prev_state = None  # Reset reward shaping tracker
        ep_raw_reward = 0
        ep_shaped_reward = 0
        best_episode = 0 # track best policy
        done = False

        while not done:
            action = agent.select_action(state)
            # Store current action for potential repetition
            # agent.last_action = action
            # Step through environment using chosen action
            next_state, raw_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Apply reward shaping
            shaped_reward = agent._get_shaped_reward(next_state, raw_reward)
            # agent.rewards.append(shaped_reward)
            agent.rewards.append(raw_reward*0.0001) # resize reward to avoid exploding gradients

            state = next_state
            ep_raw_reward += raw_reward
            ep_shaped_reward += shaped_reward

            if done:
                break

        # Used to determine when the environment is solved.
        # running_reward = (running_reward * 0.99) + (time * 0.01)

        agent.update_policy()

        agent.policy.load_state_dict(agent.policy.state_dict())

        # After each episode update:
        visualizer.update_metrics(
            episode=episode,
            reward=ep_shaped_reward, 
         )
        
        # save checkpoint
        if episode % 100 == 0:
            agent.save_checkpoint(episode)
    
        # State monitoring
        # if episode % 20 == 0:
        #     # preprocessed_state = agent._preprocess_state(state)
        #     logging.info(f"""
        #     Episode {episode} Diagnostics:
        #     - Avg Q-value: {torch.mean(agent.policy(state)):.2f}
        #     - Max Q-value: {torch.max(agent.policy(state)):.2f}
        #     - Min Q-value: {torch.min(agent.policy(state)):.2f}
        #     """)

        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, time))
        #     break

        # Clear CUDA cache periodically
        if episode % 100 == 0:
            torch.cuda.empty_cache()

        logging.info(f"Episode {episode}, Raw Rewards: {ep_raw_reward:.1f}, Shaped Rewards: {ep_shaped_reward:.2f}", info["episode"])
        
        # Save best policy
        if ep_shaped_reward > agent.best_reward:
            agent.best_reward = ep_shaped_reward
            best_episode = episode
            torch.save(agent.policy.state_dict(), agent.best_model_path)
            print(f"New best policy saved with reward: {agent.best_reward:.2f}")

            # temp_env = RecordVideo(
            #     env,
            #     video_folder="best_episodes",
            #     episode_trigger=lambda x: True,  # Record all temp episodes
            #     name_prefix=f"best_{episode}"
            # )
    logging.info(f"Best policy: Episode:{best_episode}, Shaped rewards: {agent.best_reward:.2f}")
    env.close()

except Exception as e:
    print(f"Crash detected: {str(e)}")
    print("Saving emergency checkpoint...")
    current_episode = locals().get('episode', 0)
    torch.save(agent.policy.state_dict(), 'emergency.pth')
    visualizer.save_plots() 
    raise

# Save results
torch.save(agent.policy.state_dict(), "policy_gradients.pth")

# Plot results
visualizer.save_plots() 

window = int(episodes/20)

fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9,9]);
rolling_mean = pd.Series(agent.reward_history).rolling(window).mean()
std = pd.Series(agent.reward_history).rolling(window).std()
ax1.plot(rolling_mean)
ax1.fill_between(range(len(agent.reward_history)),rolling_mean-std, rolling_mean+std, color='orange', alpha=0.2)
ax1.set_title('Episode Length Moving Average ({}-episode window)'.format(window))
ax1.set_xlabel('Episode'); ax1.set_ylabel('Episode Length')

ax2.plot(agent.reward_history)
ax2.set_title('Reward History')
ax2.set_xlabel('Episode'); ax2.set_ylabel('Episode Length')

fig.tight_layout(pad=2)
    
plt.show()
fig.savefig('results.png')