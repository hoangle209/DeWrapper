"""Common blur operations. 
Blur may be caused by unstable camera sensor, dirty lens, relative motion 
between the camera and the subject, insufficient illumination, out of focus 
settings, imaging while zooming, subject behind a frosted glass window, 
or shallow depth of field. The Blur group includes: 
1) GaussianBlur, 
2) DefocusBlur, 
3) MotionBlur, 
4) GlassBlur 
and 
5) ZoomBlur.

Reference: https://github.com/hendrycks/robustness
Hacked together for STR by: Rowel Atienza
"""

from io import BytesIO

import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from skimage.filters import gaussian
from wand.image import Image as WandImage


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        coords = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        coords = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    x, y = np.meshgrid(coords, coords)
    aliased_disk = np.asarray((x ** 2 + y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


class GaussianBlur:
    def __init__(self, rng=None, mag=-1, prob=1.):
        self.rng = np.random.default_rng() if rng is None else rng
        self.mag = mag
        self.prob = prob

    def __call__(self, img, ):
        if self.rng.uniform(0, 1) > self.prob:
            return img

        w, h = img.size
        # kernel = [(31,31)] prev 1 level only
        ksize = int(min(w, h) / 2) // 4
        ksize = (ksize * 2) + 1
        kernel = (ksize, ksize)
        sigmas = [.5, 1, 2]
        if self.mag < 0 or self.mag >= len(sigmas):
            index = self.rng.integers(0, len(sigmas))
        else:
            index = self.mag

        sigma = sigmas[index]
        return transforms.GaussianBlur(kernel_size=kernel, sigma=sigma)(img)


class DefocusBlur:
    def __init__(self, rng=None, mag=-1, prob=1.):
        self.rng = np.random.default_rng() if rng is None else rng
        self.mag = mag
        self.prob = prob

    def __call__(self, img):
        if self.rng.uniform(0, 1) > self.prob:
            return img

        n_channels = len(img.getbands())
        isgray = n_channels == 1
        # c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)]
        c = [(2, 0.1), (3, 0.1), (4, 0.1)]  # , (6, 0.5)] #prev 2 levels only
        if self.mag < 0 or self.mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = self.mag
        c = c[index]

        img = np.asarray(img) / 255.
        if isgray:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)
            n_channels = 3
        kernel = disk(radius=c[0], alias_blur=c[1])

        channels = []
        for d in range(n_channels):
            channels.append(cv2.filter2D(img[:, :, d], -1, kernel))
        channels = np.asarray(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

        # if isgray:
        #    img = img[:,:,0]
        #    img = np.squeeze(img)

        img = np.clip(channels, 0, 1) * 255
        img = Image.fromarray(img.astype(np.uint8))
        if isgray:
            img = ImageOps.grayscale(img)

        return img


class MotionBlur:
    def __init__(self, rng=None, mag=-1, prob=1.):
        self.rng = np.random.default_rng() if rng is None else rng
        self.mag = mag
        self.prob = prob

    def __call__(self, img):
        if self.rng.uniform(0, 1) > self.prob:
            return img

        n_channels = len(img.getbands())
        isgray = n_channels == 1
        # c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)]
        c = [(10, 3), (12, 4), (14, 5)]
        if self.mag < 0 or self.mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = self.mag
        c = c[index]

        output = BytesIO()
        img.save(output, format='PNG')
        img = WandImage(blob=output.getvalue())

        img.motion_blur(radius=c[0], sigma=c[1], angle=self.rng.uniform(-45, 45))
        img = cv2.imdecode(np.frombuffer(img.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img.astype(np.uint8))

        if isgray:
            img = ImageOps.grayscale(img)

        return img


class GlassBlur:
    def __init__(self, rng=None, mag=-1, prob=1.):
        self.rng = np.random.default_rng() if rng is None else rng
        self.mag = mag
        self.prob = prob

    def __call__(self, img):
        if self.rng.uniform(0, 1) > self.prob:
            return img

        w, h = img.size
        # c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
        c = [(0.45, 1, 1), (0.6, 1, 2), (0.75, 1, 2)]  # , (1, 2, 3)] #prev 2 levels only
        if self.mag < 0 or self.mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = self.mag

        c = c[index]

        img = np.uint8(gaussian(np.asarray(img) / 255., sigma=c[0], multichannel=True) * 255)

        # locally shuffle pixels
        for i in range(c[2]):
            for y in range(h - c[1], c[1], -1):
                for x in range(w - c[1], c[1], -1):
                    dx, dy = self.rng.integers(-c[1], c[1], size=(2,))
                    y_prime, x_prime = y + dy, x + dx
                    # swap
                    img[y, x], img[y_prime, x_prime] = img[y_prime, x_prime], img[y, x]

        img = np.clip(gaussian(img / 255., sigma=c[0], multichannel=True), 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))


class ZoomBlur:
    def __init__(self, rng=None, mag=-1, prob=1.):
        self.rng = np.random.default_rng() if rng is None else rng
        self.mag = mag
        self.prob = prob

    def __call__(self, img):
        if self.rng.uniform(0, 1) > self.prob:
            return img

        w, h = img.size
        c = [np.arange(1, 1.11, .01),
             np.arange(1, 1.16, .01),
             np.arange(1, 1.21, .02)]
        if self.mag < 0 or self.mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = self.mag

        c = c[index]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        uint8_img = img
        img = (np.asarray(img) / 255.).astype(np.float32)

        out = np.zeros_like(img)
        for zoom_factor in c:
            zw = int(w * zoom_factor)
            zh = int(h * zoom_factor)
            zoom_img = uint8_img.resize((zw, zh), Image.BICUBIC)
            x1 = (zw - w) // 2
            y1 = (zh - h) // 2
            x2 = x1 + w
            y2 = y1 + h
            zoom_img = zoom_img.crop((x1, y1, x2, y2))
            out += (np.asarray(zoom_img) / 255.).astype(np.float32)

        img = (img + out) / (len(c) + 1)

        img = np.clip(img, 0, 1) * 255
        img = Image.fromarray(img.astype(np.uint8))

        return img