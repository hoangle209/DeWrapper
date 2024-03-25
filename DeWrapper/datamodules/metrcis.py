from skimage.metrics import structural_similarity
import torch


def SSIM(im, ref):
    if isinstance(im, torch.Tensor):
        im = im.cpu().numpy()
    if isinstance(ref, torch.Tensor):
        ref = ref.cpu().numpy()
        
    score, _ = structural_similarity(im, ref, channel_axis=1, full=True, data_range=255.0)
    return score