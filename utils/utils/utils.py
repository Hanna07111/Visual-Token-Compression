from xml.parsers.expat import model
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

def visualize_selected_tokens(image, indices, grid_size=24, color='megenta', alpha=0.5):
    if hasattr(indices, 'cpu'):
        indices = indices.cpu().numpy()
    if hasattr(indices, 'numpy'):
        indices = indices.numpy()
    
    colors = {
        'red': np.array([1.0, 0.0, 0.0]),
        'green': np.array([0.0, 1.0, 0.0]),
        'blue': np.array([0.0, 0.0, 1.0]),
        'yellow': np.array([1.0, 1.0, 0.0]),
        'cyan': np.array([0.0, 1.0, 1.0]),
        'magenta': np.array([1.0, 0.0, 1.0]),
        'white': np.array([1.0, 1.0, 1.0]),
        'black': np.array([0.0, 0.0, 0.0])
    }

    if isinstance(color, str):
        highlight_color = colors.get(color.lower(), colors['red'])
    else:
        highlight_color = np.array(color)

    w, h = image.size
    
    mask = np.zeros(grid_size * grid_size, dtype=np.float32)
    mask[indices] = 1.0
    mask = mask.reshape(grid_size, grid_size)
    
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_resized = mask_img.resize((w, h), resample=Image.NEAREST)
    mask_array = np.array(mask_resized) / 255.0
    
    img_array = np.array(image) / 255.0
    
    overlay = img_array.copy()
    overlay[mask_array == 0] = overlay[mask_array == 0] * 0.3
    
    overlay[mask_array == 1] = overlay[mask_array == 1] * (1 - alpha) + highlight_color * alpha

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title(f"Selected Tokens ({len(indices)}/{grid_size*grid_size})")
    plt.axis('off')
    
    plt.suptitle("Visualization of Selected Image Tokens", fontsize=15, weight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def restore_image_from_tensor(tensor, processor):
    image_tensor = tensor.detach().clone().cpu()

    if image_tensor.ndim == 4:
        image_tensor = image_tensor.squeeze(0)

    mean = torch.tensor(processor.image_mean).view(3, 1, 1)
    std = torch.tensor(processor.image_std).view(3, 1, 1)

    image_tensor = (image_tensor * std) + mean
    image_tensor = image_tensor * 255.0

    image_numpy = image_tensor.clamp(0, 255).byte().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))

    restored_image = Image.fromarray(image_numpy)

    return restored_image


# Helper: compress input_ids so number of image placeholder tokens == num_vis_tokens
def compress_inputs_for_num_vis_tokens(batch, num_vis_tokens, model_config, pad_token_id=None):
    """Modify batch['input_ids'] and batch['attention_mask'] so that for each sample,
    the number of image placeholder tokens equals `num_vis_tokens` by removing excess placeholders.
    This keeps other tokens in-place and pads sequences to the same length after compression.
    """
    image_token_id = model_config.image_token_id
    input_ids = batch['input_ids']
    attention_mask = batch.get('attention_mask', None)
    device = input_ids.device
    B, L = input_ids.shape
    new_input_ids = []
    new_attention = [] if attention_mask is not None else None
    for i in range(B):
        ids = input_ids[i].tolist()
        img_positions = [j for j, t in enumerate(ids) if t == image_token_id]
        if len(img_positions) == 0 or len(img_positions) <= num_vis_tokens:
            # nothing to remove (or too few placeholders) -> keep as is
            new_input_ids.append(input_ids[i])
            if attention_mask is not None:
                new_attention.append(attention_mask[i])
            continue
        # keep first 'num_vis_tokens' placeholder positions, remove the rest
        keep_img = set(img_positions[:num_vis_tokens])
        keep_mask = [ (j in keep_img) or (j not in img_positions) for j in range(L) ]
        keep_indices = torch.tensor([idx for idx, v in enumerate(keep_mask) if v], device=device)
        new_input_ids.append(input_ids[i].index_select(0, keep_indices))
        if attention_mask is not None:
            new_attention.append(attention_mask[i].index_select(0, keep_indices))
    # pad sequences to same length
    max_len = max(x.size(0) for x in new_input_ids) if len(new_input_ids)>0 else L
    pad_id = pad_token_id if pad_token_id is not None else getattr(model_config, 'pad_token_id', 0)
    padded_ids = []
    padded_att = [] if attention_mask is not None else None
    for i in range(len(new_input_ids)):
        cur = new_input_ids[i]
        pad_len = max_len - cur.size(0)
        if pad_len > 0:
            pad_tensor = torch.full((pad_len,), pad_id, dtype=cur.dtype, device=device)
            cur = torch.cat([cur, pad_tensor], dim=0)
        padded_ids.append(cur.unsqueeze(0))
        if attention_mask is not None:
            cur_att = new_attention[i]
            if pad_len > 0:
                pad_att = torch.zeros((pad_len,), dtype=cur_att.dtype, device=device)
                cur_att = torch.cat([cur_att, pad_att], dim=0)
            padded_att.append(cur_att.unsqueeze(0))
    batch['input_ids'] = torch.cat(padded_ids, dim=0)
    if attention_mask is not None:
        batch['attention_mask'] = torch.cat(padded_att, dim=0)
    return batch