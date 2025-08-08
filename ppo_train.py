import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordEpisodeStatistics
from torch.utils.tensorboard import SummaryWriter
import ale_py

# PPO网络结构不变，省略，直接用你原来的PPOActorCritic

class PPOActorCritic(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        c, h, w = obs_shape
        assert h == 84 and w == 84, "输入必须是 84x84"

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            n_flatten = self.feature_extractor(torch.zeros(1, *obs_shape)).shape[1]

        self.policy_net = nn.Sequential(
            nn.Linear(n_flatten, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.value_net = nn.Sequential(
            nn.Linear(n_flatten, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x / 255.0)  # 归一化
        return self.policy_net(features), self.value_net(features)

    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value.squeeze(-1)


def train():
    env_id = "ALE/Pong-v5"
    env = gym.make("ALE/Pong-v5", frameskip=1)
    env = AtariPreprocessing(env, grayscale_obs=True, frame_skip = 1, scale_obs=False)
    env = FrameStackObservation(env, 4)
    env = RecordEpisodeStatistics(env)
    total_timesteps = 5_000_000
    rollout_steps = 128
    update_epochs = 4
    minibatch_size = 64  # 这里调小点，因为单进程
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.1
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    learning_rate = 2.5e-4
    device = torch.device("cuda")

    writer = SummaryWriter(f"runs/PPO_ALE_single_{int(time.time())}")

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    agent = PPOActorCritic(obs_shape, n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # 存储变量，单环境，rollout_steps长度
    obs = torch.zeros((rollout_steps,) + obs_shape).to(device)
    actions = torch.zeros(rollout_steps).to(device)
    logprobs = torch.zeros(rollout_steps).to(device)
    rewards = torch.zeros(rollout_steps).to(device)
    dones = torch.zeros(rollout_steps).to(device)
    values = torch.zeros(rollout_steps).to(device)

    global_step = 0
    next_obs, _ = env.reset(seed=42)
    next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
    next_done = 0

    num_updates = total_timesteps // rollout_steps

    for update in range(1, num_updates + 1):
        for step in range(rollout_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze(0))
            action = action.item()
            logprob = logprob.item()
            value = value.item()

            actions[step] = action
            logprobs[step] = logprob
            values[step] = value

            next_obs_np, reward, done, trunc, info = env.step(action)
            rewards[step] = reward
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device)
            next_done = float(done)

            if "episode" in info:
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            if done or trunc:
                next_obs, _ = env.reset()
                next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
                next_done = 0

        # 计算GAE优势
        with torch.no_grad():
            _, _, _, next_value = agent.get_action_and_value(next_obs.unsqueeze(0))
            next_value = next_value.item()

        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(rollout_steps)):
            if t == rollout_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values

        # 扁平化处理，单环境已是1D，直接用
        b_obs = obs
        b_logprobs = logprobs
        b_actions = actions.long()
        b_returns = returns
        b_advantages = advantages
        b_values = values

        batch_size = rollout_steps
        inds = np.arange(batch_size)
        for epoch in range(update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = F.mse_loss(newvalue, b_returns[mb_inds])
                entropy_loss = entropy.mean()

                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)

        if update % 50 == 0:
            torch.save(agent.state_dict(), f"ppo_{env_id}_update{update}.pth")

    env.close()
    writer.close()

if __name__ == "__main__":
    train()