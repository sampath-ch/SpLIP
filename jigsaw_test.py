# debug_gradient_flow.py
# Run this script to debug gradient flow through your model

import torch
import torch.nn.functional as F
from model_wrapper import SpLIP_CLIPRepo
from jigsaw import JigsawSolver, compute_jigsaw_loss


def test_gradient_flow():
    print("Testing gradient flow in SpLIP...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model and solver
    model = SpLIP_CLIPRepo(base_model="ViT-B/32", m=4, n=2, device=device).to(device)
    model.train()

    dv = model.clip_model.visual.conv1.weight.shape[0]
    n = model.Bvt.n
    solver = JigsawSolver(dv=dv, num_patches=n).to(device)
    solver.train()

    # Create optimizer
    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, model.parameters())) +
        list(solver.parameters()),
        lr=1e-3
    )

    # Register hooks to track gradient flow
    gradient_magnitudes = {}

    def save_grad(name):
        def hook(grad):
            gradient_magnitudes[name] = grad.norm().item()

        return hook

    # Register hooks on key parameters
    for name, param in model.named_parameters():
        if param.requires_grad and any(n in name for n in ("Bt.mapper", "Bv.mapper", "Bvt.mapper")):
            param.register_hook(save_grad(name))

    # Create dummy batch
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224, device=device, requires_grad=True)
    sketches = torch.randn(batch_size, 3, 224, 224, device=device, requires_grad=True)

    # Clear gradients
    optimizer.zero_grad()

    # Forward pass
    print("Running forward pass...")
    with torch.set_grad_enabled(True):
        # Get features and patches
        img_feats, photo_patches = model.encode_branch(images, prompt_type="photo", return_patches=True)
        sk_feats, sketch_patches = model.encode_branch(sketches, prompt_type="sketch", return_patches=True)

        # Compute jigsaw loss
        print(f"Sketch patches shape: {sketch_patches.shape}, requires_grad: {sketch_patches.requires_grad}")
        print(f"Photo patches shape: {photo_patches.shape}, requires_grad: {photo_patches.requires_grad}")

        # Try with explicit float conversion
        sketch_patches_float = sketch_patches.float()
        photo_patches_float = photo_patches.float()

        print(f"After float conversion - requires_grad: {sketch_patches_float.requires_grad}")

        # Compute loss
        l_jig = compute_jigsaw_loss(sketch_patches_float, photo_patches_float, solver)

        # Print the loss value
        print(f"Jigsaw Loss: {l_jig.item()}")

    # Backward pass
    print("Running backward pass...")
    l_jig.backward()

    # Print gradient norms
    print("\nPrompt-mapper gradients:")
    for name, param in model.named_parameters():
        if param.requires_grad and any(n in name for n in ("Bt.mapper", "Bv.mapper", "Bvt.mapper")):
            if param.grad is not None:
                print(f"  {name:40s} grad norm = {param.grad.norm().item():.6f}")
            else:
                print(f"  {name:40s} grad = None")

    # Print the saved gradients from hooks
    print("\nGradients from hooks:")
    for name, norm in gradient_magnitudes.items():
        print(f"  {name:40s} hook grad norm = {norm:.6f}")

    # Check jigsaw solver gradients
    print("\nJigsaw solver gradients:")
    for name, param in solver.named_parameters():
        if param.grad is not None:
            print(f"  {name:40s} grad norm = {param.grad.norm().item():.6f}")
        else:
            print(f"  {name:40s} grad = None")


if __name__ == "__main__":
    test_gradient_flow()