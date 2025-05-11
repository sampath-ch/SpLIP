# model_wrapper.py (Fixed Version)
from jigsaw import JigsawSolver, compute_jigsaw_loss
import torch
import torch.nn as nn
import clip
from clip import tokenize
from prompt_blocks import (
    VisionToTextualPrompt,
    TextualToVisualPrompt,
    VisionTextConjunction,
)


class SpLIP_CLIPRepo(nn.Module):
    def __init__(
            self,
            base_model: str = "ViT-B/32",
            m: int = 4,
            n: int = 2,
            device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        # 1) Load CLIP and force full precision
        self.clip_model, _ = clip.load(base_model, device=device, jit=False)
        self.clip_model = self.clip_model.float()

        # 2) Rebuild the text-transformer attn_mask to the correct context length
        self.M = self.clip_model.context_length  # e.g. 77
        new_attn_mask = (
            torch.empty(self.M, self.M, device=device)
            .fill_(float("-inf"))
            .triu(1)
        )
        for block in self.clip_model.transformer.resblocks:
            block.attn_mask = new_attn_mask

        # 3) Extract dims & instantiate prompt blocks
        dt = self.clip_model.transformer.width  # text hidden dim
        dv = self.clip_model.visual.conv1.weight.shape[0]  # vision hidden dim
        patch_size = self.clip_model.visual.conv1.kernel_size[0]
        num_patches = (self.clip_model.visual.input_resolution // patch_size) ** 2
        self.J = 5  # prompt length suffix

        self.Bt = VisionToTextualPrompt(
            num_patches=num_patches,
            dv=dv,
            dt=dt,
            m=m
        ).to(device)

        self.Bv = TextualToVisualPrompt(
            j_minus1=self.J - 1,
            dt=dt,
            dv=dv
        ).to(device)

        self.Bvt = VisionTextConjunction(
            M=self.M,
            dt=dt,
            n=n,
            dv=dv
        ).to(device)

        # 4) Freeze CLIP backbone except prompt blocks + all LayerNorms
        for p in self.clip_model.parameters():
            p.requires_grad = False
        for module in (self.Bt, self.Bv, self.Bvt):
            for p in module.parameters():
                p.requires_grad = True
        for module in self.clip_model.modules():
            if isinstance(module, nn.LayerNorm):
                for p in module.parameters():
                    p.requires_grad = True

        # placeholders for later
        self.class_text_feats = None
        self.class_names = None

    def encode_branch(
            self,
            images: torch.Tensor,
            prompt_type: str = "photo",
            return_patches: bool = False
    ):
        b = images.size(0)

        # — Vision stem → raw patches
        x = self.clip_model.visual.conv1(images)  # (b, dv, √P, √P)
        H, W = x.shape[-2], x.shape[-1]
        P = H * W
        E0 = x.flatten(2).transpose(1, 2)  # (b, P, dv)
        cls_v = self.clip_model.visual.class_embedding \
            .unsqueeze(0).expand(b, -1, -1)  # (b, 1, dv)

        # — Build vision→text prompts T
        T = self.Bt(E0)  # (b, m, dt)
        T_perm = T.permute(1, 0, 2)  # (m, b, dt)

        # — Prepare text tokens + positional embeddings
        prompts = (["photo of a [CLS]"] * b
                   if prompt_type == "photo"
                   else ["sketch of a [CLS]"] * b)
        toks = tokenize(prompts, context_length=self.M).to(self.device)
        W0 = (self.clip_model.token_embedding(toks)
              + self.clip_model.positional_embedding)  # (b, M, dt)

        # — Permute for CLIP's text transformer
        W = W0.permute(1, 0, 2)  # (M, b, dt)

        # — Run text transformer with injected T_perm
        all_W = []
        for block in self.clip_model.transformer.resblocks:
            W = torch.cat([T_perm, W[T_perm.size(0):]], dim=0)  # (M, b, dt)
            W = block(W)  # (M, b, dt)
            all_W.append(W.permute(1, 0, 2))  # store (b, M, dt)

        # — Vision transformer with injected Bv & Bvt
        Vl = None
        last_Vms = None
        for i, vblock in enumerate(self.clip_model.visual.transformer.resblocks):
            Wl = all_W[i]  # (b, M, dt)
            Wl_prime = Wl[:, 1:1 + (self.J - 1), :]  # (b, J-1, dt)
            Vtg = self.Bv(Wl_prime)  # (b, J-1, dv)
            Vms = self.Bvt(Wl)  # (b, n, dv)
            last_Vms = Vms  # save prompt tokens

            rest = E0 if i == 0 else Vl[:, -P:, :]  # frozen stem tokens
            Vl = torch.cat([cls_v, Vtg, Vms, rest], dim=1)  # (b, 1+J-1+n+P, dv)
            Vl = vblock(Vl)  # (b, ..., dv)

        # — Pooled [CLS] output
        feats = Vl[:, 0, :]  # (b, dv)
        pooled = self.clip_model.visual.ln_post(feats) @ \
                 self.clip_model.visual.proj  # (b, dv)

        if return_patches:
            # return the vision‐side prompt tokens (Vms), not the frozen patches
            return pooled, last_Vms

        return pooled

    def encode_class_prompts(self, class_names: list):
        """
        Compute CLIP's text features for each seen‐class prompt
        via clip_model.encode_text().

        FIXED: Removed torch.no_grad() to allow for gradient flow during training.
        The class features will now be part of the computation graph.
        """
        self.class_names = class_names
        prompts = [f"photo of a {c}" for c in class_names]
        toks = tokenize(prompts, context_length=self.M).to(self.device)

        # FIXED: Removed torch.no_grad() to enable gradient flow
        text_feats = self.clip_model.encode_text(toks)

        # Make sure to normalize the features as CLIP expects normalized features
        text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)

        self.class_text_feats = text_feats  # (num_classes, dt)
        return text_feats

    def forward(self, images: torch.Tensor, sketches: torch.Tensor):
        img_feats = self.encode_branch(images, prompt_type="photo")
        skt_feats = self.encode_branch(sketches, prompt_type="sketch")
        return img_feats, skt_feats