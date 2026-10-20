"""
Replay buffer and mirror augmentation.

- Position-level sampling, NOT game-level (§3.3)
- Fixed-size numpy ring buffer: O(1) random access (a deque is O(n) mid-way)
- Compact storage, roughly 40x smaller than dense float32 at 4096 actions:
    * state: every input plane is either binary (piece planes) or constant-filled
      (ply plane, all-ones plane). Binary planes are bit-packed; constant planes
      are stored as one scalar. Lossless.
    * policy: π is non-zero only on visited legal moves (~20-40 of 4096), so it
      is stored sparsely as (indices, probabilities).
- Mirror augmentation (§3.5) is applied at SAMPLE time with probability 0.5,
  so every slot holds a unique position instead of half the buffer being
  mirrored copies.
- Each position's legal moves are stored too (same frame as π), so training
  can mask illegal logits to -inf before the softmax (§3.1).
"""

import numpy as np
from typing import Optional


class ReplayBuffer:
    """
    Replay buffer storing (state_encoding, policy_target, value_target) triples.

    Samples at the POSITION level, not the game level — each position is
    equally likely to be drawn regardless of which game it came from.
    """

    def __init__(self, max_size: int = 1_000_000, mirror_perm: Optional[np.ndarray] = None):
        """
        Args:
            max_size: capacity in positions.
            mirror_perm: action permutation for left-right mirroring. If given,
                each sampled position is mirrored with probability 0.5 (the
                state is flipped along its width axis and π is permuted).
                None disables augmentation.
        """
        self.max_size = max_size
        self.mirror_perm = None if mirror_perm is None else np.asarray(mirror_perm, dtype=np.int64)

        self._size = 0
        self._next = 0           # ring-buffer write position
        self._total_added = 0    # monotone counter, for buffer-age diagnostics

        # Allocated lazily on the first add, once the state shape is known
        self._shape = None       # (C, H, W)
        self._action_size = None
        self._bits = None        # (N, ceil(C*H*W/8)) uint8 — packed binary planes
        self._consts = None      # (N, C) float32 — value of each constant plane
        self._is_const = None    # (N, C) bool — which planes are constant
        self._values = None      # (N,) float32
        self._added_at = None    # (N,) int64
        self._pi_idx = None      # (N,) object — int16 arrays of action indices
        self._pi_val = None      # (N,) object — float32 arrays of probabilities
        self._legal = None       # (N,) object — int16 arrays of legal action indices

    def __len__(self) -> int:
        return self._size

    def _allocate(self, state: np.ndarray, policy: np.ndarray):
        self._shape = tuple(state.shape)
        self._action_size = int(policy.shape[0])
        n, c = self.max_size, self._shape[0]
        n_bits = int(np.prod(self._shape))
        self._bits = np.zeros((n, (n_bits + 7) // 8), dtype=np.uint8)
        self._consts = np.zeros((n, c), dtype=np.float32)
        self._is_const = np.zeros((n, c), dtype=bool)
        self._values = np.zeros(n, dtype=np.float32)
        self._added_at = np.zeros(n, dtype=np.int64)
        self._pi_idx = np.empty(n, dtype=object)
        self._pi_val = np.empty(n, dtype=object)
        self._legal = np.empty(n, dtype=object)

    def add(self, state: np.ndarray, policy: np.ndarray, value: float,
            legal: Optional[np.ndarray] = None):
        """
        Add a single (state, policy, value) triple. π must be in the same frame as the state.

        Args:
            legal: legal actions in the same frame as π — either an index array
                or a (action_size,) bool mask. If None, the non-zero entries of π
                are used, which can miss legal moves MCTS never visited.
        """
        state = np.asarray(state, dtype=np.float32)
        policy = np.asarray(policy, dtype=np.float32)
        if self._shape is None:
            self._allocate(state, policy)

        c = self._shape[0]
        flat = state.reshape(c, -1)
        first = flat[:, :1]
        is_const = np.all(flat == first, axis=1)
        binary = np.where(is_const[:, None], 0.0, flat)
        if not np.all((binary == 0.0) | (binary == 1.0)):
            raise ValueError("ReplayBuffer expects every input plane to be binary "
                             "or constant-filled; got a plane that is neither.")

        i = self._next
        self._bits[i] = np.packbits(binary.astype(np.uint8).ravel())
        self._consts[i] = first[:, 0]
        self._is_const[i] = is_const
        self._values[i] = value
        self._added_at[i] = self._total_added
        nz = np.flatnonzero(policy)
        self._pi_idx[i] = nz.astype(np.int16)
        self._pi_val[i] = policy[nz]
        if legal is None:
            self._legal[i] = self._pi_idx[i]
        else:
            legal = np.asarray(legal)
            if legal.dtype == bool:
                legal = np.flatnonzero(legal)
            self._legal[i] = legal.astype(np.int16)

        self._advance()

    def _advance(self):
        self._next = (self._next + 1) % self.max_size
        self._size = min(self._size + 1, self.max_size)
        self._total_added += 1

    def _ordered_slots(self) -> np.ndarray:
        """Slot indices from oldest to newest."""
        n = len(self)
        return (np.arange(n) + (self._next if n == self.max_size else 0)) % self.max_size

    def merge(self, other: 'ReplayBuffer'):
        """
        Append every position of `other` (oldest first) without decoding it.
        Used to collect the buffers that self-play worker processes send back.
        """
        if len(other) == 0:
            return
        if self._shape is None:
            self._allocate(np.zeros(other._shape, np.float32),
                           np.zeros(other._action_size, np.float32))
        if other._shape != self._shape or other._action_size != self._action_size:
            raise ValueError(f"cannot merge buffers with shapes {other._shape} and {self._shape}")
        for j in other._ordered_slots():
            i = self._next
            self._bits[i] = other._bits[j]
            self._consts[i] = other._consts[j]
            self._is_const[i] = other._is_const[j]
            self._values[i] = other._values[j]
            self._added_at[i] = self._total_added
            self._pi_idx[i] = other._pi_idx[j]
            self._pi_val[i] = other._pi_val[j]
            self._legal[i] = other._legal[j]
            self._advance()

    def add_game(
        self,
        states: list[np.ndarray],
        policies: list[np.ndarray],
        result: float,
        legal: Optional[list] = None,
    ):
        """
        Add all positions from a completed game.

        Args:
            states: list of encoded board states (C, H, W)
            policies: list of policy targets π (action_size,), same frame as states
            result: game outcome z (+1 win, -1 loss, 0 draw) from player 1's perspective
            legal: optional list of legal actions per position (same frame as π)
        """
        # Assign values: z from the perspective of the side that moved at each position.
        # Positions alternate sides, so signs alternate.
        for i, (s, pi) in enumerate(zip(states, policies)):
            # Position i was played by the side who moved at ply i.
            # If i is even, it was player 1's move (result = z)
            # If i is odd, it was player 2's move (result = -z)
            z = result if i % 2 == 0 else -result
            self.add(s, pi, z, legal=None if legal is None else legal[i])

    def _decode_states(self, idx: np.ndarray) -> np.ndarray:
        c = self._shape[0]
        n_bits = int(np.prod(self._shape))
        bits = np.unpackbits(self._bits[idx], axis=1)[:, :n_bits]
        states = bits.reshape(len(idx), c, -1).astype(np.float32)
        states = np.where(self._is_const[idx][:, :, None],
                          self._consts[idx][:, :, None], states)
        return states.reshape((len(idx),) + self._shape)

    def sample(self, batch_size: int, return_masks: bool = False):
        """
        Sample a random batch of positions (uniform, with replacement).

        Returns:
            states: (B, C, H, W) float32
            policies: (B, action_size) float32
            values: (B, 1) float32
            legal_masks: (B, action_size) bool — only if return_masks=True
        """
        n = len(self)
        if n < batch_size:
            batch_size = n

        idx = np.random.randint(0, n, size=batch_size)
        states = self._decode_states(idx)
        policies = np.zeros((batch_size, self._action_size), dtype=np.float32)
        masks = np.zeros((batch_size, self._action_size), dtype=bool) if return_masks else None

        if self.mirror_perm is not None:
            flip = np.random.random(batch_size) < 0.5
            states[flip] = states[flip][..., ::-1]
        else:
            flip = np.zeros(batch_size, dtype=bool)

        for b, j in enumerate(idx):
            a = self._pi_idx[j].astype(np.int64)
            if flip[b]:
                a = self.mirror_perm[a]
            policies[b, a] = self._pi_val[j]
            if return_masks:
                legal = self._legal[j].astype(np.int64)
                if flip[b]:
                    legal = self.mirror_perm[legal]
                masks[b, legal] = True

        values = self._values[idx][:, None].copy()
        if return_masks:
            return states, policies, values, masks
        return states, policies, values

    def age_histogram(self, n_bins: int = 10) -> list[int]:
        """
        Histogram of buffer entry ages, in positions added since each entry
        was written (0 = newest). Used for diagnostics.
        """
        n = len(self)
        if n == 0:
            return [0] * n_bins
        ages = self._total_added - 1 - self._added_at[:n]
        counts, _ = np.histogram(ages, bins=n_bins, range=(0, max(int(ages.max()), 1) + 1))
        return counts.tolist()

    def save(self, path: str):
        """Save buffer to disk (compact format, in insertion order)."""
        n = len(self)
        if n == 0:
            np.savez_compressed(path, size=np.array(0))
            return
        # Oldest -> newest, so a reload restores the same eviction order
        order = (np.arange(n) + (self._next if n == self.max_size else 0)) % self.max_size
        lengths = np.array([len(self._pi_idx[i]) for i in order], dtype=np.int32)
        np.savez_compressed(
            path,
            size=np.array(n),
            shape=np.array(self._shape),
            action_size=np.array(self._action_size),
            bits=self._bits[order],
            consts=self._consts[order],
            is_const=self._is_const[order],
            values=self._values[order],
            pi_len=lengths,
            pi_idx=np.concatenate([self._pi_idx[i] for i in order]),
            pi_val=np.concatenate([self._pi_val[i] for i in order]),
            legal_len=np.array([len(self._legal[i]) for i in order], dtype=np.int32),
            legal_idx=np.concatenate([self._legal[i] for i in order]),
        )

    def load(self, path: str):
        """Load a buffer saved by save(). Also accepts the old dense format."""
        data = np.load(path)
        self._size = self._next = self._total_added = 0
        self._shape = None

        if 'policies' in data:  # legacy dense format
            for s, p, v in zip(data['states'], data['policies'], data['values']):
                self.add(s, p, float(v))
            return

        n = int(data['size'])
        if n == 0:
            return
        self._shape = tuple(int(x) for x in data['shape'])
        self._action_size = int(data['action_size'])
        self._allocate(np.zeros(self._shape, np.float32), np.zeros(self._action_size, np.float32))

        keep = slice(max(0, n - self.max_size), n)   # newest entries if capacity shrank
        bits, consts, is_const = data['bits'][keep], data['consts'][keep], data['is_const'][keep]
        values = data['values'][keep]
        offsets = np.concatenate([[0], np.cumsum(data['pi_len'])])
        pi_idx, pi_val = data['pi_idx'], data['pi_val']
        has_legal = 'legal_idx' in data
        if has_legal:
            legal_offsets = np.concatenate([[0], np.cumsum(data['legal_len'])])
            legal_idx = data['legal_idx']

        m = len(values)
        self._bits[:m], self._consts[:m], self._is_const[:m] = bits, consts, is_const
        self._values[:m] = values
        start = keep.start
        for k in range(m):
            lo, hi = offsets[start + k], offsets[start + k + 1]
            self._pi_idx[k] = pi_idx[lo:hi]
            self._pi_val[k] = pi_val[lo:hi]
            if has_legal:
                self._legal[k] = legal_idx[legal_offsets[start + k]:legal_offsets[start + k + 1]]
            else:   # saved before legal moves were stored
                self._legal[k] = self._pi_idx[k]
        self._added_at[:m] = np.arange(m)
        self._size = m
        self._total_added = m
        self._next = m % self.max_size
