"""Visual world model, differentiable latent imagination, and actor-critic."""

from contextlib import contextmanager, nullcontext
from copy import deepcopy
import torch
from torch import nn

from .config import upstream_config
from .rssm import make_rssm, flatten_state, detach_state, stack_states
from .vendor import networks


@contextmanager
def frozen(module):
    parameters = list(module.parameters())
    flags = [p.requires_grad for p in parameters]
    for p in parameters:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)


def lambda_returns(rewards, discounts, next_values, lambda_):
    """Inputs [H, N, 1]; next_values[h] is V(s_{h+1}), including bootstrap."""
    if rewards.shape != discounts.shape or rewards.shape != next_values.shape:
        raise ValueError("Lambda-return inputs must have identical [H,N,1] shapes")
    result, carry = [], next_values[-1]
    for h in reversed(range(rewards.shape[0])):
        carry = rewards[h] + discounts[h] * (
            (1 - lambda_) * next_values[h] + lambda_ * carry)
        result.append(carry)
    return torch.stack(result[::-1])


def clip_gradients(parameters, maximum):
    """Global L2 clipping, with a float64 fallback for finite but very large gradients."""
    gradients = [p.grad for p in parameters if p.grad is not None]
    norm = torch.stack(torch._foreach_norm(gradients, 2)).norm()
    if not torch.isfinite(norm):
        # Squaring finite float32 gradients can overflow during long SNN BPTT.
        # This is not a NaN replacement: genuinely non-finite gradients still fail.
        norm = torch.stack([g.double().norm() for g in gradients]).norm()
    if not torch.isfinite(norm):
        raise FloatingPointError("Non-finite training gradients")
    coefficient = (maximum / (norm + 1e-6)).clamp_max(1).float()
    torch._foreach_mul_(gradients, coefficient)
    return norm.detach()


def head(c, feature_size, shape, layers=2, dist="symlog_disc", scale=1.0):
    u = upstream_config(c)
    return networks.SpikeMLP(
        feature_size, shape, layers, c.hidden, "LIFNode", u.LIFNode,
        "PopNorm", u.PopNorm, dist=dist, outscale=scale,
        device=c.device, spike_times=c.io_steps)


class WorldModel(nn.Module):
    def __init__(self, c, actions):
        super().__init__()
        self.c = c
        u = upstream_config(c)
        shapes = {"image": (64, 64, 3)}
        self.encoder = networks.MultiEncoder(
            shapes, u, mlp_keys="$^", cnn_keys="image", act="LIFNode", norm="PopNorm",
            cnn_depth=c.cnn_depth, kernel_size=4, minres=4, mlp_layers=2,
            mlp_units=c.hidden, symlog_inputs=True)
        self.dynamics = make_rssm(c, actions, self.encoder.outdim)
        features = self.dynamics.feature_size
        self.decoder = networks.MultiDecoder(
            u, features, shapes, mlp_keys="$^", cnn_keys="image",
            act="LIFNode", norm="PopNorm", cnn_depth=c.cnn_depth,
            kernel_size=4, minres=4, mlp_layers=2, mlp_units=c.hidden,
            cnn_sigmoid=False, image_dist="mse", vector_dist="symlog_mse")
        self.reward = head(c, features, (255,), scale=0.0)
        self.cont = head(c, features, (), dist="binary")

    def preprocess(self, batch):
        data = {k: torch.as_tensor(v, device=self.c.device) for k, v in batch.items()}
        data["image"] = data["image"].float() / 255 - 0.5
        for k in ("action", "reward", "is_first", "is_terminal"):
            data[k] = data[k].float()
        return data

    def infer(self, data):
        embed = self.encoder(data)
        state = None
        b = self.c.burn_in
        if b:
            with torch.no_grad():
                warm, _ = self.dynamics.observe(
                    embed[:, :, :b].detach(), data["action"][:, :b], data["is_first"][:, :b])
                state = {k: v[:, -1].detach() for k, v in warm.items()}
        return self.dynamics.observe(embed[:, :, b:], data["action"][:, b:],
                                     data["is_first"][:, b:], state)

    def loss(self, batch):
        data = self.preprocess(batch)
        post, prior = self.infer(data)
        feat = self.dynamics.get_feat(post)
        b = self.c.burn_in
        image_loss = -self.decoder(feat)["image"].log_prob(data["image"][:, b:]).mean()
        reward_loss = -self.reward(feat).log_prob(data["reward"][:, b:, None]).mean()
        cont_loss = -self.cont(feat).log_prob(1 - data["is_terminal"][:, b:, None]).mean()
        kl, dyn, rep = self.dynamics.kl_loss(post, prior)
        activity = post["deter"].mean()
        regularizer = self.c.spike_reg * activity if self.c.method != "gru" else activity * 0
        total = image_loss + reward_loss + cont_loss + kl + regularizer
        metrics = dict(model_loss=total.detach(), image_loss=image_loss.detach(),
                       reward_loss=reward_loss.detach(), cont_loss=cont_loss.detach(),
                       dynamics_kl=dyn.mean().detach(), representation_kl=rep.mean().detach(),
                       deter_activity=activity.detach(),
                       prior_entropy=self.dynamics.get_dist(prior).entropy().mean().detach())
        if "steps" in prior:
            metrics["effective_steps"] = prior["steps"].mean().detach()
        else:
            metrics["effective_steps"] = total.new_tensor(self.c.core_steps)
        return total, detach_state(post), data["is_terminal"][:, b:], metrics


class Agent(nn.Module):
    def __init__(self, c, actions):
        super().__init__()
        self.c, self.actions = c, actions
        self.wm = WorldModel(c, actions)
        u = upstream_config(c)
        features = self.wm.dynamics.feature_size
        self.actor = networks.ActionHead(
            features, actions, 2, c.hidden, "LIFNode", u.LIFNode, "PopNorm",
            u.PopNorm, dist="normal", min_std=0.1, max_std=1.0,
            spike_times=c.io_steps)
        self.value = head(c, features, (255,), scale=0.0)
        self.slow_value = deepcopy(self.value).requires_grad_(False)
        self.register_buffer("return_quantiles", torch.zeros(2))
        self.to(c.device)
        # Optimizers own each network exactly once. Slow value is a frozen EMA.
        def optimizer(module, lr):
            return torch.optim.Adam(
                [p for p in module.parameters() if p.requires_grad], lr=lr, eps=c.opt_eps)
        self.model_opt = optimizer(self.wm, c.model_lr)
        self.actor_opt = optimizer(self.actor, c.actor_lr)
        self.value_opt = optimizer(self.value, c.value_lr)
        self.updates = 0

    def autocast(self):
        return (torch.autocast("cuda", dtype=torch.bfloat16)
                if self.c.precision == "bf16" else nullcontext())

    def update_parameters(self, loss, optimizer, parameters):
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = clip_gradients(parameters, self.c.grad_clip)
        optimizer.step()
        return norm.detach()

    def imagine(self, start, horizon=None, sample=True):
        state = start
        states, features, actions, entropies = [start], [], [], []
        for _ in range(horizon or self.c.imag_horizon):
            feature = self.wm.dynamics.get_feat(state)
            policy = self.actor(feature.detach())
            action = policy.sample() if sample else policy.mode()
            state = self.wm.dynamics.img_step(state, action, sample=sample)
            features.append(feature)
            actions.append(action)
            entropies.append(policy.entropy())
            states.append(state)
        features.append(self.wm.dynamics.get_feat(state))
        return (torch.stack(features, 1), stack_states(states, 0),
                torch.stack(actions), torch.stack(entropies))

    def behavior_loss(self, post, terminal):
        start = flatten_state(post)
        with frozen(self.wm), frozen(self.value):
            feats, states, actions, entropy = self.imagine(start)
            next_feat = feats[:, 1:]
            rewards = self.wm.reward(next_feat).mode()
            discounts = self.c.discount * self.wm.cont(next_feat).mean
            values = self.value(feats).mode()
            returns = lambda_returns(rewards, discounts, values[1:], self.c.lambda_)
            with torch.no_grad():
                q = torch.quantile(returns.detach().float(), returns.new_tensor([0.05, 0.95]).float())
                self.return_quantiles.lerp_(q, 0.01)
                scale = (self.return_quantiles[1] - self.return_quantiles[0]).clamp_min(1)
                valid = (1 - terminal.flatten()).reshape(1, -1, 1)
                weights = valid * torch.cumprod(
                    torch.cat([torch.ones_like(discounts[:1]), discounts[:-1]], 0), 0)
            advantage = (returns - values[:-1].detach()) / scale
            actor_loss = -(weights * (
                advantage + self.c.actor_entropy * entropy[..., None])).mean()
        detached = feats[:, :-1].detach()
        value_dist = self.value(detached)
        with torch.no_grad():
            slow = self.slow_value(detached).mode()
        value_loss = -(weights.squeeze(-1) * (
            value_dist.log_prob(returns.detach()) + value_dist.log_prob(slow))).mean()
        metrics = dict(actor_loss=actor_loss.detach(), value_loss=value_loss.detach(),
                       actor_entropy=entropy.mean().detach(), imagined_return=returns.mean().detach(),
                       imagined_reward=rewards.mean().detach())
        if "steps" in states:
            metrics["imag_effective_steps"] = states["steps"][1:].mean().detach()
        return actor_loss, value_loss, metrics

    def train_batch(self, batch):
        with self.autocast():
            model_loss, post, terminal, metrics = self.wm.loss(batch)
        metrics["model_grad_norm"] = self.update_parameters(
            model_loss, self.model_opt, self.wm.parameters())
        with self.autocast():
            actor_loss, value_loss, behavior_metrics = self.behavior_loss(post, terminal)
        metrics.update(behavior_metrics)
        metrics["actor_grad_norm"] = self.update_parameters(
            actor_loss, self.actor_opt, self.actor.parameters())
        metrics["value_grad_norm"] = self.update_parameters(
            value_loss, self.value_opt, self.value.parameters())
        with torch.no_grad():
            for target, source in zip(self.slow_value.parameters(), self.value.parameters()):
                target.lerp_(source, self.c.slow_fraction)
        self.updates += 1
        return {k: float(v.detach().float()) for k, v in metrics.items()}

    @torch.no_grad()
    def act(self, obs, state=None, previous_action=None, evaluation=False):
        image = torch.as_tensor(obs["image"], device=self.c.device).float() / 255 - 0.5
        first = torch.as_tensor(obs["is_first"], device=self.c.device).bool()
        if state is None:
            state = self.wm.dynamics.initial(image.shape[0])
        if previous_action is None:
            previous_action = image.new_zeros(image.shape[0], self.actions)
        with self.autocast():
            embed = self.wm.encoder({"image": image})
            post, _ = self.wm.dynamics.obs_step(
                state, previous_action, embed, first, sample=not evaluation)
            distribution = self.actor(self.wm.dynamics.get_feat(post))
            action = distribution.mode() if evaluation else distribution.sample()
        return action.float().clamp(-1, 1), detach_state(post)

    def training_state(self):
        return dict(model=self.state_dict(), model_opt=self.model_opt.state_dict(),
                    actor_opt=self.actor_opt.state_dict(), value_opt=self.value_opt.state_dict(),
                    updates=self.updates)

    def load_training_state(self, state):
        self.load_state_dict(state["model"])
        for name in ("model_opt", "actor_opt", "value_opt"):
            # Adam's CPU step tensors can alias the supplied dictionary; each
            # restored optimizer must own its state, not mutate the snapshot.
            getattr(self, name).load_state_dict(deepcopy(state[name]))
        self.updates = state["updates"]
