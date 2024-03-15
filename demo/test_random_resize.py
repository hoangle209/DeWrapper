import torchvision.transforms.v2 as T
from PIL import Image

if __name__ == "__main__":
    import torch

    x = torch.tensor([3]).to("cuda")
    y = torch.tensor([10]).to("cuda")
    a = torch.tensor([1.], requires_grad=True).to("cuda")
    b = torch.tensor([2.], requires_grad=True).to("cuda")

    y_hat = a*x + b
    z = y_hat - y
    L = z**2
    a.retain_grad()
    del x
    torch.cuda.empty_cache()

    

    L.backward()
    print(a.grad)