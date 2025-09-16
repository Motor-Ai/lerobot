import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mai.utils.dataset import preprocess_dataset


# repo_id = 'yaak-ai/L2D'
repo_id = 'yaak-ai/lerobot-driving-school'
dataset = LeRobotDataset(repo_id)

# dataset = preprocess_dataset(dataset)

for data in dataset:
    print(data.keys())
    break

