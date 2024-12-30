import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


def generate_mask(mask_size, mask_probability, batch_size, seed=None):
    # Generating the mask
    if seed is not None:
        mask = torch.rand((batch_size, mask_size), generator=torch.manual_seed(seed)) < mask_probability
    else:
        mask = torch.rand((batch_size, mask_size)) < mask_probability
    mask = mask.float()  # Convert boolean mask to float (0.0, 1.0)
    return mask


def get_pyx_prime(model, outputs):
    """
    Obtain p(y|x) and p(y|x'), where x' is the input with the ith entry missing.
    Args:
        model: a CLIP model
        outputs: the outputs of the CLIP model given input x
    Returns:
        pyx: p(y|x)
        pyx_prime: p(y|x')
    """
    text_embeds = outputs.text_embeds # [N_t, d]
    image_embeds = outputs.vision_model_output.last_hidden_state # [N_v, L+1, d']
    image_embeds = model.vision_model.post_layernorm(image_embeds)
    image_embeds = model.visual_projection(image_embeds) #[N_v, L+1, d]

    text_embeds = text_embeds.unsqueeze(0).unsqueeze(0) # [1, 1, N_t, d]
    image_embeds = image_embeds.unsqueeze(2) # [N_v, L+1, 1, d]

    # normalized features
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    logit_scale = model.logit_scale.exp()
    logits_per_image = (image_embeds * text_embeds).sum(-1) * logit_scale # [N_v, L+1, N_t]

    # probs_per_image = torch.softmax(logits_per_image, dim=-1)
    probs_per_image = logits_per_image

    pyx = probs_per_image[:, 0:1, :] # [N_v, 1, N_t]
    pyx_prime = probs_per_image[:, 1:, :] # [N_v, L, N_t]
    return pyx, pyx_prime

def get_heatmap(pyx, pyx_prime):
    """
    Given p(y|x) and p(y|x'), where x' is the input with the ith entry missing.
    Args:
        pyx: [N_v, 1, N_t]
        pyx_prime: [N_v, L, N_t]

    Returns: 
        heatmap: [N_v, 14, 14, N_t]
    """
    res = (pyx - pyx_prime) # [N_v, L, N_t]
    N_v, L, N_t = res.shape
    res = (res>0).float() * res
    
    heatmap = res.reshape(N_v,14,14, N_t).detach().cpu().numpy()
    return heatmap

def unnormalize(img, mean, std):
    mean = np.array(mean).reshape(1,1,3)
    std = np.array(std).reshape(1,1,3)
    return img * std + mean

def convert_to_255_scale(img):
    return (img * 255).astype(np.uint8)

def unnormalize_and_255_scale(img, mean, std):
    return convert_to_255_scale(unnormalize(img,mean,std))

def show_superimposed(img, heatmap):
    cv2_image = cv2.cvtColor(img.transpose(1,2,0), cv2.COLOR_RGB2BGR)
    blur = cv2.GaussianBlur(heatmap,(13,13), 11)

def normalize_and_rescale(heatmap):
    max_value = np.max(heatmap)
    min_value = np.min(heatmap)
    heatmap_ft = (heatmap - min_value) / (max_value - min_value) # float point
    return convert_to_255_scale(heatmap_ft) # int8

def get_overlap(image, heatmap):
    return cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)

def plot_overlap(image, heatmap):
    overlap = get_overlap(image, heatmap)
    plt.imshow(overlap)
    plt.axis('off')
    plt.show()

def plot_overlap_both(image, heatmap):
    overlap = get_overlap(image, heatmap)
    # Plot the overlap image
    plt.subplot(1, 2, 1)
    plt.imshow(overlap)
    plt.axis('off')

    # Plot the original image
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def plot_overlap_np(image, heatmap, img_mean, img_std):
    shape = image.shape[:2]
    heatmap = normalize_and_rescale(heatmap)
    resized_heatmap = cv2.resize(heatmap, shape)
    blur = cv2.blur(resized_heatmap ,(13,13), 11)
    heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
    heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)

    image = unnormalize_and_255_scale(image, img_mean, img_std)
    
    plot_overlap_both(image, heatmap_img)
    return image, heatmap_img


