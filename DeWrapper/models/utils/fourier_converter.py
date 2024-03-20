import torch.nn as nn
import torch.fft as fft


class FourierConverter(nn.Module):
    """Fourier Converter to extract high-frequency information from document images captured by cameras.
    Given a Document Image, the Fourier Converter first transforms it into Fourier space via Fast Fourier Transform (FFT).
    Next, the document's low-frequency information is replaced with the low-frequency information of a Blank Paper. 
    The modified spectral signals are finally transformed back to the spatial space 
    (through inverse Fast Fourier Transform (iFFT)), 
    which produces the OCR-friendly document images with most appearance noises successfully removed.
    """
    def __init__(self, beta=0.008):
        super().__init__()
        self.beta = beta
    
    def forward(self, x, is_normalized=True):
        """
        Parameters:
        -----------
            x, Tensor, (b, 3, h, w)
            is_normalized, bool, default=True
                whether input Tensor is normalize
        """
        x_denormalize = x * 255. if is_normalized else x 
        x_fft = fft.fft2(x_denormalize, dim=(-2, -1)) 

        _, _, h, w = x_fft.size()
        masked_w = int(self.beta * w)
        masked_h = int(self.beta * h)
        cy, cx = int(h//2), int(w//2)

        x_fft[:, :, 
             -masked_h + cy : masked_h + cy,
             -masked_w + cx : masked_w + cx] = 0.0
        
        # shift the origin to the beginning of the vector (top-left in 2D case)
        # x_ishift = fft.ifftshift(x_fft_high_freq)
        x_ifft = fft.ifft2(x_fft, dim=(-2, -1))

        return x_ifft.real # return image back