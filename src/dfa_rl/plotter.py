import matplotlib.pyplot as plt
import numpy as np

def plot_training(logs, window: int = 0):
    rewards = np.array(logs["rewards"])
    print(logs["rewards"])
    lengths = np.array(logs["lengths"])
    print(logs["lengths"])
    episodes = np.arange(1, len(rewards)+1)

    def rolling_mean(x, w):
        if w <= 1:
            return x
        # pad with NaN at start so arrays align with episodes
        cumsum = np.cumsum(np.insert(x, 0, 0))
        smooth = (cumsum[w:] - cumsum[:-w]) / float(w)
        return np.concatenate([np.full(w-1, np.nan), smooth])

    rewards_smooth = rolling_mean(rewards, window)
    lengths_smooth = rolling_mean(lengths, window)

    fig, axes = plt.subplots(2, 1, figsize=(10,6), sharex=True)

    # Reward curve
    axes[0].plot(episodes, rewards, color="tab:blue", alpha=0.3, label="Episode reward")
    if window > 1:
        axes[0].plot(episodes, rewards_smooth, color="tab:blue", label=f"Rolling mean (w={window})")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Reward per episode")
    axes[0].legend(loc="lower left")

    # Steps curve
    axes[1].plot(episodes, lengths, color="tab:orange", alpha=0.3, label="Episode length")
    if window > 1:
        axes[1].plot(episodes, lengths_smooth, color="tab:orange", label=f"Rolling mean (w={window})")
    axes[1].set_ylabel("Steps")
    axes[1].set_xlabel("Episode")
    axes[1].set_title("Steps per episode")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    plt.show()
