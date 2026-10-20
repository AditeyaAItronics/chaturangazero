"""
Dual-head residual network for AlphaZero.

Architecture per §3.3:
    input  C_in × H × W
      ↓  conv 3×3, C_in→channels, BN, ReLU
      ↓  N × residual block [conv3×3, BN, ReLU, conv3×3, BN, +skip, ReLU]
      ├─ policy head
      │     conv 1×1, channels→policy_channels → reshape → action_size logits
      │     → mask → log_softmax
      └─ value head
            conv 1×1, channels→value_channels → flatten → FC → ReLU → FC → tanh

CRITICAL: The policy head uses 1×1 convolution, NOT flatten-to-linear.
A Linear(8192, 4096) layer would have ~34M parameters, dwarfing the 2.5M trunk.
The (B, 64, 8, 8) output is flattened channel-major, so logit index
c*64 + (r*8 + f) = action from_sq*64 + to_sq: the CHANNEL is the from-square
and the spatial position (r, f) is the to-square. (Plan §3.3 describes the
transpose; either assignment works, this is what the code does.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Standard residual block: conv-BN-ReLU-conv-BN + skip + ReLU."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class DualHeadNet(nn.Module):
    """
    Dual-head residual network for AlphaZero-style training.
    
    Configurable for both Chaturanga (14×8×8 input, 4096 actions)
    and Connect-4 (3×6×7 input, 7 actions).
    """
    
    def __init__(
        self,
        input_planes: int = 14,
        board_h: int = 8,
        board_w: int = 8,
        action_size: int = 4096,
        n_blocks: int = 8,
        channels: int = 128,
        policy_channels: int = 64,
        value_channels: int = 8,
        value_hidden: int = 256,
    ):
        super().__init__()
        
        # Saved into checkpoints so loaders can rebuild the exact architecture
        self.config = dict(
            input_planes=input_planes, board_h=board_h, board_w=board_w,
            action_size=action_size, n_blocks=n_blocks, channels=channels,
            policy_channels=policy_channels, value_channels=value_channels,
            value_hidden=value_hidden,
        )
        
        self.board_h = board_h
        self.board_w = board_w
        self.action_size = action_size
        self.channels = channels
        
        # Input convolution
        self.input_conv = nn.Conv2d(input_planes, channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(n_blocks)
        ])
        
        # --- Policy Head ---
        # 1×1 conv producing policy_channels planes
        # Reshape to get action_size logits
        # For Chaturanga: policy_channels * 8 * 8 = 64 * 64 = 4096 ✓
        # For Connect-4: need different approach
        self.policy_conv = nn.Conv2d(channels, policy_channels, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        
        policy_flat_size = policy_channels * board_h * board_w
        if policy_flat_size != action_size:
            # Need a linear layer to map to action_size (e.g., Connect-4)
            self.policy_fc = nn.Linear(policy_flat_size, action_size)
            self._policy_needs_fc = True
        else:
            # Direct reshape works (Chaturanga: 64*8*8 = 4096)
            self.policy_fc = None
            self._policy_needs_fc = False
        
        # --- Value Head ---
        self.value_conv = nn.Conv2d(channels, value_channels, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_channels)
        value_flat_size = value_channels * board_h * board_w
        self.value_fc1 = nn.Linear(value_flat_size, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)
    
    def forward(
        self, x: torch.Tensor, legal_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (B, C_in, H, W) input tensor
            legal_mask: (B, action_size) boolean mask. If provided,
                        illegal actions get -inf before softmax.
        
        Returns:
            policy_logits: (B, action_size) — raw logits (apply mask externally if needed)
            value: (B, 1) — value in [-1, +1]
        """
        # Input
        out = F.relu(self.input_bn(self.input_conv(x)))
        
        # Residual tower
        for block in self.res_blocks:
            out = block(out)
        
        # Policy head
        p = self.policy_bn(self.policy_conv(out))
        p = p.view(p.size(0), -1)  # Flatten
        if self._policy_needs_fc:
            p = self.policy_fc(F.relu(p))
        # Direct-reshape head (Chaturanga): p is already the logits, so no ReLU —
        # it would clamp them to >= 0 and the network could never push a bad
        # move below the rest.
        # p is now (B, action_size)
        
        # Apply legal mask: set illegal to -inf BEFORE softmax
        if legal_mask is not None:
            p = p.masked_fill(~legal_mask.bool(), float('-inf'))
        
        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_chaturanga_net(n_blocks: int = 8, channels: int = 128) -> DualHeadNet:
    """Create a network configured for Chaturanga."""
    return DualHeadNet(
        input_planes=14,
        board_h=8,
        board_w=8,
        action_size=4096,
        n_blocks=n_blocks,
        channels=channels,
        policy_channels=64,   # 64 * 8 * 8 = 4096 actions
        value_channels=8,
        value_hidden=256,
    )


def create_connect4_net(n_blocks: int = 3, channels: int = 64) -> DualHeadNet:
    """Create a network configured for Connect-4."""
    return DualHeadNet(
        input_planes=3,
        board_h=6,
        board_w=7,
        action_size=7,
        n_blocks=n_blocks,
        channels=channels,
        policy_channels=2,    # 2 * 6 * 7 = 84 != 7, so uses FC
        value_channels=4,
        value_hidden=64,
    )


# Board size implied by the input encoding, for checkpoints saved before
# architecture configs were stored (Chaturanga: 14 planes, Connect-4: 3).
_BOARD_FOR_PLANES = {14: (8, 8), 3: (6, 7)}


def net_config_from_state_dict(state_dict: dict) -> dict:
    """Infer DualHeadNet constructor arguments from a state_dict's weight shapes."""
    input_planes = state_dict['input_conv.weight'].shape[1]
    if input_planes not in _BOARD_FOR_PLANES:
        raise ValueError(f"cannot infer board size for {input_planes} input planes")
    board_h, board_w = _BOARD_FOR_PLANES[input_planes]
    policy_channels = state_dict['policy_conv.weight'].shape[0]
    if 'policy_fc.weight' in state_dict:
        action_size = state_dict['policy_fc.weight'].shape[0]
    else:
        action_size = policy_channels * board_h * board_w
    n_blocks = len({k.split('.')[1] for k in state_dict if k.startswith('res_blocks.')})
    return dict(
        input_planes=input_planes, board_h=board_h, board_w=board_w,
        action_size=action_size, n_blocks=n_blocks,
        channels=state_dict['input_conv.weight'].shape[0],
        policy_channels=policy_channels,
        value_channels=state_dict['value_conv.weight'].shape[0],
        value_hidden=state_dict['value_fc1.weight'].shape[0],
    )


def load_net_from_checkpoint(path: str, device=None) -> tuple[DualHeadNet, dict]:
    """
    Rebuild the network a checkpoint was trained with and load its weights.

    Uses the architecture stored in the checkpoint ('net_config'); older
    checkpoints without it have the architecture inferred from weight shapes,
    so a cloud (8x128) checkpoint loads just as well as a local (5x64) one.

    Returns (net in eval mode on `device`, checkpoint dict).
    """
    from utils.device import get_device
    device = get_device(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    config = checkpoint.get('net_config') if isinstance(checkpoint, dict) else None
    if config is None:
        config = net_config_from_state_dict(state_dict)
    net = DualHeadNet(**config)
    net.load_state_dict(state_dict)
    net.to(device).eval()
    return net, checkpoint
