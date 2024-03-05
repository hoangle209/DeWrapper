import numpy as np
import cv2

class FFT:
    def __init__(self, cfg) -> None:
        w = cfg.target_width
        h = cfg.target_height
        blank = np.full(shape=(h, w), fill_value=255, dtype=np.float32)
        self.blank_fft = cv2.dft(blank, flags=cv2.DFT_COMPLEX_OUTPUT)

        if cfg.train:
            self.beta = cfg.fourier_converter.beta_train
        else:
            self.beta = cfg.fourier_converter.beta_test
        
        self.mask = np.ones(shape=(h, w, 1))

        masked_w = int(self.beta * w)
        masked_h = int(self.beta * h)
        cy, cx = int(h//2), int(w//2)
        self.mask[-masked_h + cy : masked_h + cy,
                  -masked_w + cx : masked_w + cx] = 0

    @staticmethod
    def fft(x):
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
