import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.fft as fft
import torchvision.transforms as transforms
import kornia.color as color 

class FFT: # This is used when input is np.ndarray
    def __init__(self, cfg) -> None:
        w = cfg.target_width
        h = cfg.target_height
        blank = np.full(shape=(h, w), fill_value=255, dtype=np.float32)
        self.blank_fft = cv2.dft(blank, flags=cv2.DFT_COMPLEX_OUTPUT)

        self.mask = np.ones(shape=(h, w, 1))
        if cfg.train:
            self.beta = cfg.fourier_converter.beta_train
        else:
            self.beta = cfg.fourier_converter.beta_test
        
        masked_w = int(self.beta * w)
        masked_h = int(self.beta * h)
        cy, cx = int(h//2), int(w//2)
        self.mask[-masked_h + cy : masked_h + cy,
                  -masked_w + cx : masked_w + cx] = 0

    @staticmethod
    def fft(x):
        """
        Parameters:
        -----------
            x, np.ndarray, in gray mode, (h, w)
        """
        x_ = cv2.dft(np.float32(x), flags=cv2.DFT_COMPLEX_OUTPUT) # TODO: check number channels of x
                                                                  # x has to be in gray mode
        return x_

    @staticmethod
    def ifft(x):
        x_ = cv2.idft(x)
        x_ = cv2.magnitude(x_[:,:,0], x_[:,:,1])
        return x_
    
    def converter(self, x):
        x_fft = FFT.fft(x)
        x_ = x_fft * self.mask + self.blank_fft * (1 - self.mask)
        x_ishift = np.fft.ifftshift(x_) # 
        x_ = FFT.ifft(x_ishift)

        return x_


class FourierConverter(nn.Module):
    """Fourier Converter to extract high-frequency information from document images captured by cameras.
    Given a Document Image, the Fourier Converter first transforms it into Fourier space via Fast Fourier Transform (FFT).
    Next, the document's low-frequency information is replaced with the low-frequency information of a Blank Paper. 
    The modified spectral signals are finally transformed back to the spatial space 
    (through inverse Fast Fourier Transform (iFFT)), 
    which produces the OCR-friendly document images with most appearance noises successfully removed.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.invNormalize = transforms.Compose([
                transforms.Normalize(mean = [0., 0., 0.], std = [1/0.229, 1/0.224, 1/0.225]),
                transforms.Normalize(mean = [-0.485, -0.456, -0.406], std = [1., 1., 1.]),
                color.rgb_to_grayscale
        ])

        w = cfg.target_width
        h = cfg.target_height
        blank = torch.full((1, 1, h, w), fill_value=255.0, dtype=torch.float32)
        self.blank_fft = fft.fft2(blank) # (1, 1, h, w)
        
        self.mask = torch.ones(1, 1, h, w)
        if cfg.train:
            self.beta = cfg.fourier_converter.beta_train
        else:
            self.beta = cfg.fourier_converter.beta_test
        
        masked_w = int(self.beta * w)
        masked_h = int(self.beta * h)
        cy, cx = int(h//2), int(w//2)
        self.mask[:, :, -masked_h + cy : masked_h + cy,
                        -masked_w + cx : masked_w + cx] = 0.0
    
    def forward(self, x):
        """
        Parameters:
        -----------
            x, Tensor, (b, 3, h, w), normalized
        """
        x_denormalize = self.invNormalize(x) * 255. # (b, 1, h, w)
        x_fft = fft.fft2(x_denormalize) # (b, 1, h, w)
        x_fft_high_freq = x_fft * self.mask + self.blank_fft * (1. - self.mask)
        
        # shift the origin to the beginning of the vector (top-left in 2D case)
        x_ishift = fft.ifftshift(x_fft_high_freq)
        x_ifft = fft.ifft2(x_ishift)

        return x_ifft.abs() # return image back





        