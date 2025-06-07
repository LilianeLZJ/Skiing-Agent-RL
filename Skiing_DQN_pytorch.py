class my_class(object):
    pass

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
import random
import matplotlib.pyplot as plt
import gymnasium as gym
import ale_py
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
from gymnasium.wrappers import FrameStackObservation, RecordEpisodeStatistics, RecordVideo

gym.register_envs(ale_py)

# Environment setup with proper dimension handling
env = gym.make('ALE/Skiing-v5', render_mode = "rgb_array")
env = GrayscaleObservation(env, keep_dim=False)  # Output shape (H,W)
env = ResizeObservation(env, (84, 84))           # Resize to 84x84
env = FrameStackObservation(env, 4)              # Stack 4 frames -> (4,84,84)
env = gym.wrappers.RecordVideo(
    env,
    episode_trigger=lambda num: num % 20 == 0,
    video_folder="saved-video-folder",
    name_prefix="video-",
)
state_shape = env.observation_space.shape  # Should be (4, 84, 84)
action_space = env.action_space.n

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, stride=4),  # Input channels=4
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),  # Calculated for 84x84 input
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
         # Convert input to tensor if it's a numpy array
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Add batch dimension if missing (N, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 3136)
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor(np.array([x[0] for x in batch])).to(device)
        actions = torch.LongTensor([x[1] for x in batch]).to(device)
        rewards = torch.FloatTensor([x[2] for x in batch]).to(device)
        next_states = torch.FloatTensor(np.array([x[3] for x in batch])).to(device)
        dones = torch.FloatTensor([x[4] for x in batch]).to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

def save_checkpoint(episode, model, optimizer, rewards, losses, hyperparams, filename="checkpoint.pth"):
    """Save full training state"""
    torch.save({
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rewards': rewards,
        'losses': losses,
        'hyperparams': hyperparams,
        'epsilon': epsilon  # If using epsilon-greedy
    }, filename)

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
        'losses': checkpoint['losses'],
        'hyperparams': checkpoint['hyperparams'],
        'epsilon': checkpoint.get('epsilon', 0.01)
    }

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("training_log.txt"),
        logging.StreamHandler()
    ]
)

# Hyperparameters optimized for time-based objectives
EPISODES = 500
BATCH_SIZE = 64
GAMMA = 0.98 #V2 0.96
EPSILON_START = 1.0
EPSILON_END = 0.01
# EPSILON_DECAY = 0.998 # V1
# EPSILON_DECAY = 0.985 # V2
EPSILON_DECAY = 0.99 # V4
TARGET_UPDATE = 50
LEARNING_RATE = 0.003 #0.005


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # choose GPU if available
print(f"Using device: {device}")

# Initialize components
# score_tracker = ScoreTracker()
policy_net = DQN(state_shape, action_space).to(device)
target_net = DQN(state_shape, action_space).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
replay_buffer = ReplayBuffer(100000)

td_errors = []
bellman_errors = []
prev_state = None # Track previous state for reward shaping

def _get_shaped_reward(state, original_reward):
    """Custom reward"""
    # Extract position/velocity estimates from state
    # x_center = abs(state[1].mean() - 0.5)  # Center alignment
    global prev_state
    # Initialize velocity calculation
    velocity = 0.0
        
    # Only calculate velocity if we have previous state
    if prev_state is not None:
        try:
            # Convert states to numpy arrays if needed
            current_state = np.array(state) if not isinstance(state, np.ndarray) else state
            prev_state = np.array(prev_state) if not isinstance(prev_state, np.ndarray) else prev_state
                
            # Calculate velocity as frame difference
            velocity = np.abs(current_state - prev_state).mean()
        except Exception as e:
            print(f"Velocity calculation error: {str(e)}")
            velocity = 0.0
        
    shaped_reward = (original_reward + 0.8 * velocity)

    # Store current state for next calculation
    prev_state = state.copy() if isinstance(state, np.ndarray) else np.array(state)
        
    return shaped_reward

def select_action(state, epsilon):
    """Epsilon-greedy action selection without explicit epsilon parameter"""

    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            return policy_net(state_tensor).argmax().item()

def update_model():
    global td_errors, bellman_errors

    td_errors = deque(maxlen=10000)
    rewards = []

    if len(replay_buffer) < BATCH_SIZE:
        return
    
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
    
    current_q = policy_net(states).gather(1, actions.unsqueeze(1))
    next_q = target_net(next_states).max(1)[0].detach()
    target_q = rewards + (1 - dones) * GAMMA * next_q
    
    loss = nn.SmoothL1Loss()(current_q.squeeze(), target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

     # Calculate TD errors for this batch
    td_error_batch = (target_q - current_q.squeeze()).tolist()
    bellman_error = loss.item()
    
    # Store errors
    td_errors.extend(td_error_batch)
    bellman_errors.append(bellman_error)
    
    return td_error_batch, bellman_error  # Explicit return statement

class TrainingVisualizer:
    def __init__(self, window_size=100):
        # Initialize data containers
        self.episode_rewards = []
        self.td_errors = []
        self.bellman_errors = []
        self.window_size = window_size
        
        self.td_errors = deque(maxlen=10000)

        # Create figure with subplots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(
            3, 1, 
            figsize=(14, 16),
            constrained_layout = True 
            )
        # plt.ion()  # Enable interactive mode for real-time updates

    def update_metrics(self, episode, reward, td_error=None, bellman_error=None):
        """Store and visualize metrics in real-time"""
        # Store metrics
        self.episode_rewards.append(reward)
        if td_error is not None:
            if isinstance(td_error, float):
                td_error = [td_error]
            self.td_errors.extend(td_error)
        if bellman_error is not None:
            self.bellman_errors.append(bellman_error)

        # Update plots every 10 episodes
        if episode % 10 == 0:
            self._update_plots()
            # plt.pause(0.001)  # Brief pause to update display

    def _update_plots(self):
        """Internal method to refresh all plots"""
        # Clear previous drawings
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # Plot 1: Episode Rewards
        self.ax1.plot(self.episode_rewards, label='Raw')
        self.ax1.plot(self._moving_average(self.episode_rewards), 'r-', label='Smoothed')
        self.ax1.set_title('Episode Rewards')
        self.ax1.set_ylabel('Total Reward')
        # self.ax1.legend() #V1
        self.ax1.legend(loc='upper left') #V2

        # Plot 2: TD-Errors
        if self.td_errors:
            self.ax2.plot(self.td_errors, alpha=0.3, label='Raw')
            self.ax2.plot(self._moving_average(self.td_errors), 'r-', label='Smoothed')
            self.ax2.set_title('TD-Errors')
            self.ax2.set_ylabel('Error Magnitude')
            # self.ax2.legend()
            self.ax1.legend(loc='upper right') #V2

        # Plot 3: Bellman Errors
        if self.bellman_errors:
            self.ax3.plot(self.bellman_errors, 'g-', label='Batch Average')
            self.ax3.set_title('Bellman Errors (Loss)')
            self.ax3.set_ylabel('Loss')
            self.ax3.set_xlabel('Training Batches')
            # self.ax3.legend()
            self.ax1.legend(loc='upper right') #V2

        # plt.tight_layout()

    def _moving_average(self, data):
        """Calculate moving average for smoothing"""
        return np.convolve(data, np.ones(self.window_size)/self.window_size, mode='valid')

    def save_plots(self, filename='training_metrics.png'):
        """Save final plots to file"""
        plt.ioff()
        self._update_plots()
        plt.savefig(filename, bbox_inches='tight') #V2
        plt.close()

if os.path.exists("checkpoint_ep400.pth"):
# Resume training
     checkpoint = load_checkpoint("checkpoint_ep400.pth", policy_net, optimizer)
     start_ep = checkpoint['episode'] + 1
     episode_rewards = checkpoint['rewards']
else:
     # Start new training
     start_ep = 0
     episode_rewards = []


# Training loop
try:

    episode_rewards = deque(maxlen=1000)
    # Integration with Training Loop
    visualizer = TrainingVisualizer()
    epsilon = EPSILON_START
    rewards = []

    for episode in range(start_ep, EPISODES):
        state, _ = env.reset()  # Now returns (4,84,84) array
        total_reward = 0
        raw_reward = 0
        state_shape = 0
        # time_penalty = 0
        # gate_passed = 0
        # crash_penalty = 0
        done = False
    
        while not done:
            # No need for transpose with correct wrapper setup
            action = select_action(state, epsilon)       
            next_state, raw_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # update score
            # score_tracker.update()

            # calculate reward
            # reward = last_y - current_y #Version 1
            # reward = score_tracker.current_episode["total_score"] * (-1) #Version 2 

            # Gymnasium default reward
            # reward == -1: time penalty
            # reward == +5 : gate passed
            # reward == -5 : crash
        
            # print(f"Reward:{reward}")

            # if reward == -1:
            #     time_penalty += 1
            
            # if reward == +5:
            #     gate_passed += 1
            
            # if reward == -5:
            #     crash_penalty += 1 

            # if terminated:
            #     reward -= 5
            
            # Apply reward shaping
            shaped_reward = _get_shaped_reward(next_state, raw_reward)
            rewards.append(shaped_reward)
            
            replay_buffer.push(state, action, raw_reward, next_state, terminated or truncated)
            state = next_state
            total_reward += shaped_reward*0.1
        
            update_model()
                
            if terminated or truncated:
                break
    
         # delay epsilon 
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # update target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # After each episode update:
        visualizer.update_metrics(
            episode=episode,
            reward=total_reward,
            td_error=td_errors[-1],
            bellman_error=bellman_errors[-1] 
         )

        if episode % 50 == 0:
            save_checkpoint(
                episode=episode,
                model=policy_net,
                optimizer=optimizer,
                rewards=episode_rewards,
                losses=bellman_errors,
                hyperparams={
                    'gamma': GAMMA,
                    'lr': LEARNING_RATE,
                    'batch_size': BATCH_SIZE
                },
                filename=f"checkpoint_ep{episode}.pth"
            )
    
        # State monitoring
        if episode % 10 == 0:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            logging.info(f"""
            Episode {episode} Diagnostics:
            - Avg Q-value: {torch.mean(q_values):.2f}
            - Max Q-value: {torch.max(q_values):.2f}
            - Min Q-value: {torch.min(q_values):.2f}
            - Gradient Norm: {torch.norm(torch.cat([p.grad.flatten() for p in policy_net.parameters() if p.grad is not None])).item():.2f}
            """)

        # Visual verification of agent behavior
        if episode % 100 == 0:
            env.render()

        # Clear CUDA cache periodically
        if episode % 100 == 0:
            torch.cuda.empty_cache()

        # logging.info(f"Episode {episode+1}, Gate Passes:{gate_passed},Time Penalty:{time_penalty},Crash Penalty{crash_penalty},Reward: {total_reward:.1f}, Epsilon: {epsilon:.3f}")
        logging.info(f"Episode {episode+1},Reward: {total_reward:.1f}, Epsilon: {epsilon:.3f}, Loss: {bellman_errors[-1]:.3f}")


except Exception as e:
    print(f"Crash detected: {str(e)}")
    print("Saving emergency checkpoint...")
    torch.save(policy_net.state_dict(), "emergency.pth")
    raise

# save errors
# np.save("training_errors.npy", agent.training_error)

# Save results
torch.save(policy_net.state_dict(), "skiing_dqn_final.pth")

# After training completes
# Visualize training process
visualizer.save_plots()