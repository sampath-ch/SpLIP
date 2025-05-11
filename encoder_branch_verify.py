# check_encode_branch.py
# This script checks if the encode_branch function preserves gradients properly

import torch
from model_wrapper import SpLIP_CLIPRepo


def check_encode_branch():
    """Verify if encode_branch is properly preserving gradients"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create the model
    model = SpLIP_CLIPRepo(base_model="ViT-B/32", m=4, n=2, device=device).to(device)
    model.train()  # Ensure model is in training mode

    # Create dummy input with gradients
    dummy_image = torch.randn(2, 3, 224, 224, device=device, requires_grad=True)

    # Register hooks to track gradient flow through the model
    activations = {}
    gradients = {}

    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output

        return hook

    def save_gradient(name):
        def hook(grad):
            gradients[name] = grad

        return hook

    # Register hooks on key modules/tensors
    # You'll need to adapt these to match the actual module names in your model
    if hasattr(model.clip_model.visual, 'transformer'):
        model.clip_model.visual.transformer.register_forward_hook(save_activation('transformer'))
    elif hasattr(model.clip_model.visual, 'layers'):
        model.clip_model.visual.layers[-1].register_forward_hook(save_activation('last_layer'))

    # Forward pass
    print("Running forward pass with encode_branch...")
    features, patches = model.encode_branch(dummy_image, prompt_type="photo", return_patches=True)

    # Check if patches require gradients
    print(f"Patches shape: {patches.shape}")
    print(f"Patches require gradient: {patches.requires_grad}")

    # Register gradient hook on patches
    patches.register_hook(save_gradient('patches'))

    # Create a simple loss and backpropagate
    loss = patches.sum()
    loss.backward()

    # Check original input gradients
    print(f"Input has gradient: {dummy_image.grad is not None}")
    if dummy_image.grad is not None:
        print(f"Input gradient norm: {dummy_image.grad.norm().item()}")

    # Check gradients of model parameters
    print("\nModel parameter gradients:")
    for name, param in model.named_parameters():
        if param.requires_grad and any(n in name for n in ("Bt.mapper", "Bv.mapper", "Bvt.mapper")):
            if param.grad is not None:
                print(f"  {name:40s} grad norm = {param.grad.norm().item():.6f}")
            else:
                print(f"  {name:40s} grad = None")

    # Print the stored gradients
    print("\nStored gradients:")
    for name, grad in gradients.items():
        print(f"  {name}: {grad.norm().item():.6f}")

    # Now try with a different prompt type
    print("\nRunning with sketch prompt type...")
    features2, patches2 = model.encode_branch(dummy_image, prompt_type="sketch", return_patches=True)

    print(f"Sketch patches shape: {patches2.shape}")
    print(f"Sketch patches require gradient: {patches2.requires_grad}")

    # Make sure the model implementation isn't detaching patches
    # Check the encode_branch method if this test fails
    if not patches.requires_grad or not patches2.requires_grad:
        print("\n⚠️ WARNING: Patches do not require gradients after encode_branch!")
        print("Check your model_wrapper.py implementation to ensure gradients are being preserved.")
        print("Look for any detach() calls or operations that might break the computation graph.")


if __name__ == "__main__":
    check_encode_branch()