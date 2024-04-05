import albumentations as A
import cv2

def get_appearance_transform(transform_types):
    """
    Returns an albumentation compose augmentation.

    transform_type is a list containing types of pixel-wise data augmentation to use.
    Possible augmentations are 'shadow', 'blur', 'visual', 'noise', 'color'.
    """

    transforms = []
    if "shadow" in transform_types:
        transforms.append(A.RandomShadow(p=0.1))
    if "blur" in transform_types:
        transforms.append(
            A.OneOf(
                transforms=[
                    A.Defocus(p=5),
                    A.Downscale(p=15, interpolation=cv2.INTER_LINEAR),
                    A.GaussianBlur(p=65),
                    A.MedianBlur(p=15),
                ],
                p=0.75,
            )
        )
    if "visual" in transform_types:
        transforms.append(
            A.OneOf(
                transforms=[
                    A.ToSepia(p=15),
                    A.ToGray(p=20),
                    A.Equalize(p=15),
                    A.Sharpen(p=20),
                ],
                p=0.5,
            )
        )
    if "noise" in transform_types:
        transforms.append(
            A.OneOf(
                transforms=[
                    A.GaussNoise(var_limit=(10.0, 20.0), p=70),
                    A.ISONoise(intensity=(0.1, 0.25), p=30),
                ],
                p=0.6,
            )
        )
    if "color" in transform_types:
        transforms.append(
            A.OneOf(
                transforms=[
                    A.ColorJitter(p=5),
                    A.HueSaturationValue(p=10),
                    A.RandomBrightnessContrast(brightness_limit=[-0.05, 0.25], p=85),
                ],
                p=0.95,
            )
        )

    return A.Compose(transforms=transforms)