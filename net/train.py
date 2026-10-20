"""
Training loop for the dual-head network.

Loss = (z - v)² - π^T log p + c||θ||²
     = MSE(value) + CrossEntropy(policy) + L2 regularization

Key details:
- Masked policy loss: -inf before softmax, not zero after (§3.1)
- Adam optimizer with multi-step LR schedule
- Gradient clipping
- Training diagnostics: policy/value loss split, entropy, calibration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Optional
import time
import os

from .model import DualHeadNet
from .replay import ReplayBuffer
from utils.device import get_device


class Trainer:
    """
    Handles network training from the replay buffer.
    """
    
    def __init__(
        self,
        net: DualHeadNet,
        lr: float = 2e-3,
        l2: float = 1e-4,
        lr_milestones: list[int] = None,
        lr_gamma: float = 0.1,
        grad_clip: float = 1.0,
        device=None,
        optimizer: str = 'adamw',
        weight_decay: float = 1e-2,
    ):
        """
        Args:
            optimizer: 'adamw' (default) — AdamW with DECOUPLED weight decay
                `weight_decay`, applied to conv/linear weights only (not
                BatchNorm or biases). Plain Adam's weight_decay (and an L2 term
                in the loss, which is equivalent) gets rescaled by Adam's
                per-parameter step sizes, so heavily-updated weights are barely
                regularised; AdamW avoids that (needtofix L3).
                'adam' — the plan's literal loss  (z-v)^2 - pi.log p + l2*||w||^2,
                with the L2 term over conv/linear weights added explicitly and
                logged as 'l2_loss'.
            weight_decay: AdamW decay (per step it shrinks weights by lr * weight_decay).
            l2: coefficient c of the explicit L2 term for optimizer='adam'.
        """
        device = get_device(device)
        self.net = net.to(device)
        self.device = device
        self.lr = lr
        self.l2 = l2
        self.grad_clip = grad_clip
        self.optimizer_name = optimizer.lower()

        # Regularise conv/linear weights; never BatchNorm parameters or biases
        self._decay_params = [p for n, p in net.named_parameters()
                              if p.ndim > 1 and p.requires_grad]
        decay_ids = {id(p) for p in self._decay_params}
        no_decay = [p for p in net.parameters() if id(p) not in decay_ids and p.requires_grad]

        if self.optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                [{'params': self._decay_params, 'weight_decay': weight_decay},
                 {'params': no_decay, 'weight_decay': 0.0}],
                lr=lr,
            )
        elif self.optimizer_name == 'adam':
            self.optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0)
        else:
            raise ValueError(f"unknown optimizer {optimizer!r}; use 'adamw' or 'adam'")
        
        if lr_milestones:
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=lr_milestones,
                gamma=lr_gamma,
            )
        else:
            self.scheduler = None
        
        # Training stats
        self.total_steps = 0
        self.history = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'policy_entropy': [],
            'lr': [],
        }
    
    def train_step(
        self,
        states: np.ndarray,
        policy_targets: np.ndarray,
        value_targets: np.ndarray,
        legal_masks: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Single training step.

        Args:
            states: (B, C, H, W)
            policy_targets: (B, action_size) — visit count distribution π
            value_targets: (B, 1) — game outcome z
            legal_masks: (B, action_size) bool. If given, illegal logits are set
                to -inf BEFORE the softmax (§3.1), so probability is only spread
                over legal moves.
        
        Returns:
            dict with 'policy_loss', 'value_loss', 'total_loss', 'entropy'
        """
        self.net.train()
        
        s = torch.tensor(states, dtype=torch.float32, device=self.device)
        pi = torch.tensor(policy_targets, dtype=torch.float32, device=self.device)
        z = torch.tensor(value_targets, dtype=torch.float32, device=self.device)
        
        mask = None
        if legal_masks is not None:
            mask = torch.as_tensor(legal_masks, dtype=torch.bool, device=self.device)

        # Forward (the net applies masked_fill(-inf) when given a mask)
        policy_logits, value = self.net(s, legal_mask=mask)

        # Policy loss: cross-entropy = -π^T log p
        # π is already normalized visit distribution
        log_probs = F.log_softmax(policy_logits, dim=1)
        if mask is not None:
            # Illegal entries are -inf; π is 0 there, and 0 * -inf would be NaN
            log_probs_safe = log_probs.masked_fill(~mask, 0.0)
        else:
            log_probs_safe = log_probs
        policy_loss = -torch.sum(pi * log_probs_safe, dim=1).mean()
        
        # Value loss: MSE = (z - v)²
        value_loss = F.mse_loss(value, z)
        # Kept for the per-epoch value-head calibration diagnostic
        self._last_value_pairs = (value.detach().float().cpu().numpy().ravel(),
                                  z.detach().float().cpu().numpy().ravel())
        
        # Total loss. With AdamW the weight decay is applied by the optimizer;
        # with plain Adam the plan's c*||w||^2 term is added here.
        total_loss = policy_loss + value_loss
        l2_loss = None
        if self.optimizer_name == 'adam' and self.l2 > 0:
            l2_loss = self.l2 * sum((p * p).sum() for p in self._decay_params)
            total_loss = total_loss + l2_loss
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        
        self.optimizer.step()
        self.total_steps += 1
        
        # Compute entropy for diagnostics
        with torch.no_grad():
            probs = log_probs.exp()   # exactly 0 on masked moves
            entropy = -torch.sum(probs * log_probs_safe, dim=1).mean()
        
        stats = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item(),
            'entropy': entropy.item(),
        }
        if l2_loss is not None:
            stats['l2_loss'] = l2_loss.item()
        
        return stats
    
    def train_epoch(
        self,
        buffer: ReplayBuffer,
        steps: int = 2000,
        batch_size: int = 512,
    ) -> dict:
        """
        Train for a fixed number of gradient steps from the buffer.
        
        Returns aggregate statistics.
        """
        if len(buffer) < batch_size:
            print(f"Buffer too small ({len(buffer)} < {batch_size}), skipping training")
            return {}
        
        epoch_stats = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'entropy': [],
        }
        if self.optimizer_name == 'adam' and self.l2 > 0:
            epoch_stats['l2_loss'] = []
        
        start_time = time.time()
        value_preds, value_targets = [], []

        for step in range(steps):
            states, policies, values, masks = buffer.sample(batch_size, return_masks=True)
            stats = self.train_step(states, policies, values, legal_masks=masks)
            
            for k, v in stats.items():
                epoch_stats[k].append(v)
            v_pred, v_true = self._last_value_pairs
            value_preds.append(v_pred)
            value_targets.append(v_true)
            
            if (step + 1) % max(100, steps // 4) == 0 or (step + 1) == steps:
                print(f"  [Train] Step {step+1}/{steps} | ploss={stats['policy_loss']:.4f} | vloss={stats['value_loss']:.4f} | ent={stats['entropy']:.4f}", flush=True)
        
        elapsed = time.time() - start_time
        
        # Aggregate
        agg = {}
        for k, v in epoch_stats.items():
            agg[f'mean_{k}'] = float(np.mean(v))
            agg[f'final_{k}'] = v[-1]
        agg['elapsed_seconds'] = elapsed
        agg['steps'] = steps
        agg.update(value_calibration(np.concatenate(value_preds), np.concatenate(value_targets)))
        
        # Store history
        self.history['policy_loss'].append(agg['mean_policy_loss'])
        self.history['value_loss'].append(agg['mean_value_loss'])
        self.history['total_loss'].append(agg['mean_total_loss'])
        self.history['policy_entropy'].append(agg['mean_entropy'])
        self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
        
        return agg
    
    def step_scheduler(self):
        """Step the learning rate scheduler (call once per iteration)."""
        if self.scheduler is not None:
            self.scheduler.step()
    
    def save_checkpoint(self, path: str, iteration: int = 0, extra: dict = None):
        """Save model and optimizer state."""
        checkpoint = {
            'model_state_dict': self.net.state_dict(),
            'net_config': getattr(self.net, 'config', None),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'iteration': iteration,
            'history': self.history,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if extra:
            checkpoint.update(extra)
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        # Atomic: a kill mid-write leaves the previous file intact, never a truncated one
        tmp = path + '.tmp'
        torch.save(checkpoint, tmp)
        os.replace(tmp, path)
    
    def load_checkpoint(self, path: str) -> dict:
        """Load model and optimizer state. Returns the checkpoint dict."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.history = checkpoint.get('history', self.history)
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint


def value_calibration(pred: np.ndarray, target: np.ndarray, n_bins: int = 10) -> dict:
    """
    Value-head calibration over the batches seen in a training epoch.

    - value_calibration_error: predictions are bucketed into `n_bins` equal
      bins over [-1, 1]; the error is the position-weighted mean of
      |mean outcome - mean prediction| per bin (0 = perfectly calibrated).
    - value_sign_accuracy: on decisive targets (z = ±1), how often the
      predicted value has the right sign.
    - value_mean_abs: mean |v|; stuck near 0 means the head outputs "draw"
      for everything (the §2.2 collapse symptom).
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.size == 0:
        return {}
    bins = np.clip(((pred + 1.0) / 2.0 * n_bins).astype(int), 0, n_bins - 1)
    error = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            error += in_bin.sum() * abs(target[in_bin].mean() - pred[in_bin].mean())
    decisive = np.abs(target) > 0.5
    sign_acc = float((np.sign(pred[decisive]) == np.sign(target[decisive])).mean()) if decisive.any() else float('nan')
    return {
        'value_calibration_error': float(error / pred.size),
        'value_sign_accuracy': sign_acc,
        'value_mean_abs': float(np.abs(pred).mean()),
    }


def create_evaluator(net: DualHeadNet, device=None, action_size: int = 4096):
    """
    Create an evaluate callable for MCTS from a network.

    The evaluate callable takes a state and returns (policy, value).
    This is the §3.4 indirection that lets the batched inference server
    replace this function without MCTS changing.
    """
    device = get_device(device)
    net.to(device)

    def evaluate(state):
        net.eval()
        with torch.no_grad():
            # Encode state
            encoded = state.encode()
            x = torch.tensor(encoded, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Forward pass
            policy_logits, value = net(x)
            
            # Convert to numpy
            policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            v = value.squeeze().item()

        # Network output is in the canonical frame; MCTS works in absolute actions
        if hasattr(state, 'orient_policy'):
            policy = state.orient_policy(policy)
        
        return policy, v
    
    return evaluate
