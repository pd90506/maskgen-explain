import torch
from captum.attr import GradientShap
import matplotlib.pyplot as plt

# Assuming pred_model is your trained model
# Assuming test_ds is your test dataset

def model_forward(input):
    return pred_model(input)

gradient_shap = GradientShap(pred_model)

baseline_dist = torch.randn(50, *test_ds[0][0].shape)  # Example for 50 samples

input_tensor = test_ds[0][0].unsqueeze(0)  # Example for the first sample

attributions = gradient_shap.attribute(input_tensor, baselines=baseline_dist)

# Example visualization for a single input
plt.imshow(attributions.squeeze().cpu().detach().numpy(), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()