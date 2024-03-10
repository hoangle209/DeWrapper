import math
import numpy as np
import cv2

class ClassifyLetterBox:
    """
    YOLOv8 LetterBox class for image preprocessing, designed to be part of a transformation pipeline, e.g.,
    T.Compose([LetterBox(size), ToTensor()]).

    Attributes:
        h (int): Target height of the image.
        w (int): Target width of the image.
        auto (bool): If True, automatically solves for short side using stride.
        stride (int): The stride value, used when 'auto' is True.
    """

    def __init__(self, size=(640, 640), auto=False, stride=32):
        """
        Initializes the ClassifyLetterBox class with a target size, auto-flag, and stride.

        Args:
            size (Union[int, Tuple[int, int]]): The target dimensions (height, width) for the letterbox.
            auto (bool): If True, automatically calculates the short side based on stride.
            stride (int): The stride value, used when 'auto' is True.
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto  # pass max size integer, automatically solve for short side using stride
        self.stride = stride  # used with auto

    def __call__(self, im):
        """
        Resizes the image and pads it with a letterbox method.

        Args:
            im (numpy.ndarray | PIL.Image): The input image as a numpy array of shape HWC.

        Returns:
            (numpy.ndarray): The letterboxed and resized image as a numpy array.
        """
        im = np.array(im)
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)  # ratio of new/old dimensions
        h, w = round(imh * r), round(imw * r)  # resized image dimensions

        # Calculate padding dimensions
        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else (self.h, self.w)
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)

        # Create padded image
        im_out = np.full((hs, ws, 3), 114, dtype=im.dtype)
        im_out[top : top + h, left : left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im_out