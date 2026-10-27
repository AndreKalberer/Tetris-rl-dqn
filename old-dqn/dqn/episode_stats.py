"""Episode accounting shared by bounded diagnostic rollouts."""
from dataclasses import dataclass


@dataclass
class EpisodeStats:
    completed: int = 0
    completed_lines: int = 0
    completed_steps: int = 0
    partial_lines: int = 0
    partial_steps: int = 0
    best_lines: int = 0
    best_steps: int = 0

    def add(self, lines: int, done: bool):
        self.partial_lines += int(lines)
        self.partial_steps += 1
        if done:
            self.completed += 1
            self.completed_lines += self.partial_lines
            self.completed_steps += self.partial_steps
            self.best_lines = max(self.best_lines, self.partial_lines)
            self.best_steps = max(self.best_steps, self.partial_steps)
            self.partial_lines = self.partial_steps = 0

    @property
    def mean_lines(self):
        return self.completed_lines / self.completed if self.completed else None

    @property
    def mean_steps(self):
        return self.completed_steps / self.completed if self.completed else None
