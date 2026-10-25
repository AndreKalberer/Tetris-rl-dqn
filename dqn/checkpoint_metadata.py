"""Explicit environment settings beside each SB3 checkpoint."""
from importlib.metadata import version
import json
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback

ENVIRONMENT_VERSION = 2


def read_settings(model_path):
    path = Path(model_path).with_suffix('.json')
    if not path.exists():
        return {}
    metadata = json.loads(path.read_text())
    if metadata.get('environment_version') != ENVIRONMENT_VERSION:
        raise ValueError(f'Unsupported environment version in {path}')
    return metadata['settings']


def write_metadata(model_path, settings):
    Path(model_path).with_suffix('.json').write_text(json.dumps(dict(
        environment_version=ENVIRONMENT_VERSION, settings=settings,
        packages={name: version(name) for name in ['tetris-gymnasium', 'gymnasium',
                  'numpy', 'torch', 'stable-baselines3']}), indent=2)+'\n')


class ExperimentCheckpoint(CheckpointCallback):
    def __init__(self, *, settings, **kwargs):
        super().__init__(**kwargs)
        self.settings = settings

    def _on_step(self):
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            write_metadata(self._checkpoint_path(extension='zip'), self.settings)
        return result
