import os
from typing import TypeVar, Any, Optional

import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

ModelType = TypeVar('ModelType', bound=BaseAlgorithm)


class Trainer:

    def __init__(self, model: ModelType, path: Optional[str] = None):
        self.model = model
        self.path = path

        self.target_training_step = 15_000
        self.log_frequency = 5_000
        save_frequency = 5_000

        self.checkpoint_callback = CheckpointCallback(
            save_freq=save_frequency,
            save_path=self.path,
            name_prefix="model",
        )
        self.callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=path, model=self.model)

    def train(self):
        self.model.learn(
            total_timesteps=int(self.target_training_step),
            callback=self.callback,
            log_interval=5,
        )
        self.model.save(self.path)
        self.model.save_replay_buffer(self.path)


class SaveOnBestTrainingRewardCallback(BaseCallback):

    def __init__(self, check_freq: int, log_dir: str, verbose=1, model=None):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.model = model
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(f"Best Mean Reward: {evaluate_policy(self.model, self.model.env, n_eval_episodes=10)}")
        return True
