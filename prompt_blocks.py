# prompt_blocks.py

import torch.nn as nn

class VisionToTextualPrompt(nn.Module):
    def __init__(self, num_patches, dv, dt, m):
        """
        num_patches: number of image patches (e.g. 49 for 224×224 @32)
        dv:         vision‐encoder hidden size (e.g. 768)
        dt:         text‐encoder hidden size  (e.g. 512)
        m:          number of prompt tokens to generate
        """
        super().__init__()
        # first Linear: in = num_patches * dv, out = dt
        self.mapper = nn.Sequential(
            nn.Linear(num_patches * dv, dt),
            nn.ReLU(),
            # then produce m tokens of length dt
            nn.Linear(dt, m * dt)
        )
        self.m, self.dt = m, dt

    def forward(self, E0):
        # E0: (batch, num_patches, dv)
        b, n, dv = E0.shape
        # flatten to (batch, num_patches*dv)
        flat = E0.reshape(b, -1)
        t    = self.mapper(flat)              # (batch, m*dt)
        return t.view(b, self.m, self.dt)     # → (batch, m, dt)


class TextualToVisualPrompt(nn.Module):
    def __init__(self, j_minus1, dt, dv):
        super().__init__()
        self.mapper = nn.Linear(j_minus1 * dt, j_minus1 * dv)
        self.jm1, self.dt, self.dv = j_minus1, dt, dv

    def forward(self, Wprime):
        b, _, dt = Wprime.shape
        v = self.mapper(Wprime.reshape(b, -1))
        return v.view(b, self.jm1, self.dv)


class VisionTextConjunction(nn.Module):
    def __init__(self, M, dt, n, dv):
        super().__init__()
        self.mapper = nn.Sequential(
            nn.Linear(M * dt, dv),
            nn.ReLU(),
            nn.Linear(dv, n * dv)
        )
        self.n, self.dv = n, dv

    def forward(self, Wl):
        b, M, dt = Wl.shape
        v = self.mapper(Wl.reshape(b, -1))
        return v.view(b, self.n, self.dv)