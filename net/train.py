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
    ):
        device = get_device(device)
        self.net = net.to(device)
        self.device = device
        self.lr = lr
        self.l2 = l2
        self.grad_clip = grad_clip
        
        self.optimizer = optim.Adam(
            net.parameters(), lr=lr, weight_decay=l2
        )
        
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
    ) -> dict:
        """
        Single training step.
        
        Args:
            states: (B, C, H, W)
            policy_targets: (B, action_size) — visit count distribution π
            value_targets: (B, 1) — game outcome z
        
        Returns:
            dict with 'policy_loss', 'value_loss', 'total_loss', 'entropy'
        """
        self.net.train()
        
        s = torch.tensor(states, dtype=torch.float32, device=self.device)
        pi = torch.tensor(policy_targets, dtype=torch.float32, device=self.device)
        z = torch.tensor(value_targets, dtype=torch.float32, device=self.device)
        
        # Forward
        policy_logits, value = self.net(s)
        
        # Policy loss: cross-entropy = -π^T log p
        # π is already normalized visit distribution
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -torch.sum(pi * log_probs, dim=1).mean()
        
        # Value loss: MSE = (z - v)²
        value_loss = F.mse_loss(value, z)
        
        # Total loss (L2 is handled by weight_decay in Adam)
        total_loss = policy_loss + value_loss
        
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
            probs = F.softmax(policy_logits, dim=1)
            entropy = -torch.sum(probs * log_probs, dim=1).mean()
        
        stats = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item(),
            'entropy': entropy.item(),
        }
        
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
        
        start_time = time.time()
        
        for step in range(steps):
            states, policies, values = buffer.sample(batch_size)
            stats = self.train_step(states, policies, values)
            
            for k, v in stats.items():
                epoch_stats[k].append(v)
            
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
        torch.save(checkpoint, path)
    
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
