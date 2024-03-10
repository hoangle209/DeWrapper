import torchvision.transforms.v2 as T
from PIL import Image

if __name__ == "__main__":
    img = Image.open("images\\0000.jpg")
    print(img.size)
    random_resize = T.RandomResize(1000, 1500)
    img = random_resize(img)
    print(img.size)