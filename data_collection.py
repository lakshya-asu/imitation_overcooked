import random
import pickle
from config import DEMO_DATA_PATH, NUM_DEMO_EPISODES, MAX_EPISODE_STEPS, LAYOUT_NAME, NUM_PLAYERS, ACTIONS
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

def collect_demonstrations(num_episodes=NUM_DEMO_EPISODES, max_steps=MAX_EPISODE_STEPS, save_path=DEMO_DATA_PATH):
    demonstrations = []

    # Generator function that creates a new MDP instance.
    def mdp_fn(outside_info):
        return OvercookedGridworld.from_layout_name(LAYOUT_NAME, num_players=NUM_PLAYERS)

    # Create the environment using the generator function.
    env = OvercookedEnv(mdp_fn, horizon=400)

    for ep in range(num_episodes):
        state = env.reset()
        episode_data = []
        for t in range(max_steps):
            # Sample random actions for each player from the ACTIONS list.
            actions = [random.choice(ACTIONS) for _ in range(NUM_PLAYERS)]
            episode_data.append((state, actions))
            state, reward, done, info = env.step(actions)
            if done:
                break
        demonstrations.append(episode_data)
        print(f"Collected episode {ep + 1}/{num_episodes} with {len(episode_data)} steps.")

    # Save the collected demonstrations to disk.
    with open(save_path, 'wb') as f:
        pickle.dump(demonstrations, f)
    print(f"Saved expert demonstrations to {save_path}")

if __name__ == '__main__':
    collect_demonstrations()
