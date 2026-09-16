import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for visual explanation.
    Hooks into target convolutional layer to compute class attribution heatmaps.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        
        if target_layer is None:
            # Default to the last convolutional layer in features
            self.target_layer = self.model.features[-1]
        else:
            self.target_layer = target_layer
            
        self.gradients = None
        self.activations = None

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generates a normalized heatmap (0..1 array of shape HxW) for the given input tensor.
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        score.backward()

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Interpolate to input resolution
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize 0 to 1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class

def overlay_heatmap_on_image(pil_image, heatmap, alpha=0.5, colormap=plt.cm.jet):
    """
    Overlays normalized 2D heatmap on a PIL Image.
    Returns a PIL Image with heatmap blended.
    """
    img_np = np.array(pil_image.resize((224, 224)).convert("RGB")) / 255.0
    
    # Map scalar heatmap values to RGB via colormap
    color_mapped = colormap(heatmap)[:, :, :3]
    
    blended = alpha * color_mapped + (1.0 - alpha) * img_np
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blended)
