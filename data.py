from torchvision import datasets
from torchvision.transforms import transforms


def get_transform():
    t=transforms.Compose([
        #transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
        ])
    return t

def get_data():
    t = get_transform()
    train= datasets.MNIST(root='./data', train=True, download=True, transform=t)
    test = datasets.MNIST(root='./data', train=False, download=True, transform=t)
    return train, test

