from collections import deque
import numpy as np
import torch
import torch.optim as optim
import random
from .dfa_agent import DFAAgent
from .dfa_env import DFAEnv
from .qnet import QNet
from torch import nn


class ReplayBuffer:
    def __init__(self, capacity=20_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, sp, done):
        self.buf.append((s, a, r, sp, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, sp, d = zip(*batch)
        return (torch.stack(s),
                torch.tensor(a, dtype=torch.long),
                torch.tensor(r, dtype=torch.float32),
                torch.stack(sp),
                torch.tensor(d, dtype=torch.float32))

    def __len__(self): return len(self.buf)


##
# dqn_train - this is the main driving loop for the agent
#
def dqn_train(
        env,  # dfa environment
        n_hist: int = 3,                        # number of past steps it considers for each action
        episodes: int = 500,                    # number of episodes in the epmem
        gamma: float = 0.99,                    # discount factor
        lr: float = 1e-3,                       # learning rate 0.001 is typical
        batch_size: int = 64,                   # ?size of sample to be drawn from replay buffer
        buffer_capacity: int = 20_000,          # ?size of replay buffer for passing rewards back to past Q-values
        start_learning_after: int = 500,        # how many random actions to take before learning
        target_update_every: int = 200,         # ?how often to retrain the network?
        eps_start: float = 1.0,                 # starting value for episilon
        eps_end: float = 0.05,                  # bottom value for epsilon
        eps_decay_steps: int = 5000,            # epsilon drops over this many steps
        seed: int = 0,                          # rng seed lets you get repeatable results
        device: str = "cpu",                    # whether to use cpu or gpu
):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)

    K = len(env.alphabet)  # number of actions
    agent = DFAAgent(n_hist=n_hist, K=K, seed=seed)
    obs_dim = n_hist * (K + 1)
    n_actions = K

    q = QNet(obs_dim, n_actions).to(device)
    qt = QNet(obs_dim, n_actions).to(device)
    qt.load_state_dict(q.state_dict());
    qt.eval()
    opt = optim.Adam(q.parameters(), lr=lr)
    buf = ReplayBuffer(buffer_capacity)

    def epsilon(step):
        if eps_decay_steps <= 0: return eps_end
        t = min(1.0, step / eps_decay_steps)
        return eps_start + (eps_end - eps_start) * t

    global_step, grad_steps = 0, 0
    success_log, length_log = [], []

    episode_rewards = []

    for ep in range(episodes):
        obs = env.reset()  # int token from env (0 on reset)
        agent.reset()
        agent.observe(obs)
        s = agent.encode().to(device)

        done, total_r, steps = False, 0.0, 0
        while not done and steps < env.max_steps:
            steps += 1;
            global_step += 1

            # ε-greedy over Q(s, ·)
            if random.random() < epsilon(global_step):
                a = env.sample_action()
            else:
                with torch.no_grad():
                    qvals = q(s.unsqueeze(0))
                    a = int(torch.argmax(qvals, dim=1).item())

            obs_next, r, done, info = env.step(a)  # env still returns a token
            agent.observe(obs_next)
            sp = agent.encode().to(device)

            buf.push(s, a, r, sp, float(done))
            s = sp;
            total_r += r

            if len(buf) >= start_learning_after:
                S, A, R, SP, D = buf.sample(batch_size)
                S, A, R, SP, D = S.to(device), A.to(device), R.to(device), SP.to(device), D.to(device)

                q_sa = q(S).gather(1, A.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    target = R + (1.0 - D) * gamma * qt(SP).max(1).values
                loss = nn.functional.mse_loss(q_sa, target)

                opt.zero_grad();
                loss.backward();
                opt.step()
                grad_steps += 1
                if grad_steps % target_update_every == 0:
                    qt.load_state_dict(q.state_dict())

        success_log.append(1 if total_r > 0 else 0)
        length_log.append(steps)
        episode_rewards.append(total_r)

        if (ep + 1) % max(1, episodes // 10) == 0:
            wr = np.mean(success_log[-50:]) if success_log else 0.0
            print(f"Episode {ep + 1:4d}/{episodes} | recent win-rate(50)={wr:.2f} "
                  f"| steps={steps} total_r={total_r:.1f}")

    return q, {"success": success_log, "lengths": length_log, "rewards": episode_rewards}
