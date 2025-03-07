# evaluation.py

import torch
import numpy as np

from models import ImitationNet
import config
from config import DEMO_DATA_PATH, NUM_DEMO_EPISODES, MAX_EPISODE_STEPS, LAYOUT_NAME, NUM_PLAYERS
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld



def evaluate_model(num_episodes=10, max_steps=200):
    # Initialize the Overcooked environment.
    mdp = OvercookedGridworld.from_layout_name(
    layout_name=LAYOUT_NAME,
    num_players=NUM_PLAYERS
    )
    env = OvercookedEnv(mdp)

    
    # Infer state dimensions by resetting the environment.
    state = env.reset()
    input_dim = state.flatten().shape[0]
    output_dim = config.NUM_ACTIONS
    
    model = ImitationNet(input_dim, output_dim)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    model.eval()
    
    total_rewards = []
    
    for ep in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        for t in range(max_steps):
            # Prepare state for the model (flatten if needed).
            state_tensor = torch.tensor(state, dtype=torch.float32).view(1, -1)
            with torch.no_grad():
                logits = model(state_tensor)
            action = torch.argmax(logits, dim=1).item()
            
            # For agent 1, use a random action.
            random_action = env.action_space.sample()
            actions = [action, random_action]
            state, reward, done, info = env.step(actions)
            episode_reward += reward
            if done:
                break
        total_rewards.append(episode_reward)
        print(f"Episode {ep + 1}/{num_episodes}, Reward: {episode_reward}")
    
    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {num_episodes} episodes: {avg_reward}")

if __name__ == "__main__":
    evaluate_model()
