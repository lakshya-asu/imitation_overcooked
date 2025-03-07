# config.py

from overcooked_ai.src.overcooked_ai_py.mdp.actions import Action

# Hyperparameters for training
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
BATCH_SIZE = 32

# Data and model file paths
DEMO_DATA_PATH = 'data/expert_demos.pkl'
MODEL_SAVE_PATH = 'models/imitation_model.pth'

# Environment and training settings
NUM_DEMO_EPISODES = 100
MAX_EPISODE_STEPS = 200

# Overcooked environment setup
LAYOUT_NAME = "cramped_room"   # Example layout name in Overcooked
NUM_PLAYERS = 2

# Define available actions using the Action class.
# Action.ALL_ACTIONS is defined in your actions file and is a list of valid actions.
ACTIONS = Action.ALL_ACTIONS  
NUM_ACTIONS = len(ACTIONS)
