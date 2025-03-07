# Imitation Learning for Overcooked AI

This repository contains an imitation learning project that uses behavioral cloning (BC) to train an agent in the Overcooked AI environment.

## Setup

1. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
2. Ensure the Overcooked AI Environment is Installed:
Follow the instructions in the Overcooked AI repository (or your local installation).

3. Create Necessary Folders:
Make sure the folders data/ and models/ exist for saving demonstration data and model checkpoints.


## Project Structure
- config.py: Hyperparameters and file paths.
- data_collection.py: Script for collecting expert demonstration data.
- models.py: Neural network model definition.
- training.py: Script for training the imitation learning model.
- evaluation.py: Script for evaluating the trained model.
- main.py: Command-line interface for running the project.


---

### Final Notes

- **Expert Policy:** The provided expert agent in `data_collection.py` is a placeholder that samples random actions. Replace it with a heuristic or scripted expert to gather higher-quality demonstrations.
- **State Representation:** Adjust the way states are processed (e.g., flattening) if the Overcooked environment provides structured observations.
- **Multi-Agent Coordination:** In this code, we train only for one agent while the other acts randomly. For full cooperative learning, consider extending the code to train both agents or use joint training techniques.

This complete codebase should serve as a strong starting point for your imitation learning project on Overcooked AI. Feel free to modify and extend it based on your experiment needs.
