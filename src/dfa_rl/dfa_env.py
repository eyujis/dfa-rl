import random
from typing import Tuple, Dict, Any, Optional

class DFAEnv:
    """
    Environment:
      - action space: integers 0..K-1 for symbols in fsa.alphabet
      - observation: 0 at reset; then (last_action + 1) after each step
      - reward: +1 on accepting state, otherwise step_penalty (negative)
      - done: True on accept or after max_steps
    """
    def __init__(self, fsa, max_steps: Optional[int] = None, seed: Optional[int] = None,
                 step_penalty: float = -0.01, reward_goal: float = 1.0):
        self.fsa = fsa
        self.rng = random.Random(seed)

        self.alphabet = list(self.fsa.alphabet)
        self.idx_to_action = {i: a for i, a in enumerate(self.alphabet)}
        self.action_space_n = len(self.alphabet)

        self.non_accepting = [s for s in self.fsa.states if s not in self.fsa.accept_states]
        if not self.non_accepting:
            raise ValueError("FSA has no non-accepting states to start from.")

        self.max_steps = int(max_steps) if max_steps is not None else 50
        self.step_penalty = float(step_penalty)     # negative per step
        self.reward_goal = float(reward_goal)       # positive on accept

        self.state: Optional[str] = None
        self.steps = 0
        self._last_action_idx: Optional[int] = None
        self.done = False

    def reset(self, seed: Optional[int] = None) -> int:
        if seed is not None:
            self.rng.seed(seed)
        self.state = self.rng.choice(self.non_accepting)
        self.steps = 0
        self._last_action_idx = None
        self.done = False
        return 0  # observation token ("no previous action yet")

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode finished. Call reset().")
        if not (0 <= action < self.action_space_n):
            raise ValueError(f"Action must be in [0,{self.action_space_n-1}]")

        a_sym = self.idx_to_action[action]
        self._last_action_idx = action
        self.steps += 1

        # transition
        next_state = self.fsa.next_state(self.state, a_sym)
        if next_state is None:
            next_state = self.state
        self.state = next_state

        # reward & done: +1 only if accepting, else negative step penalty
        if self.fsa.is_accepting(self.state):
            reward = self.reward_goal
            self.done = True
        else:
            reward = self.step_penalty
            self.done = (self.steps >= self.max_steps)

        obs = action + 1
        info = {"state": self.state, "action_symbol": a_sym, "steps": self.steps}
        return obs, reward, self.done, info

    def sample_action(self) -> int:
        return self.rng.randrange(self.action_space_n)
