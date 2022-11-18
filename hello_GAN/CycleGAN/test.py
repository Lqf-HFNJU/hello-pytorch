from CycleGAN import G
import torch
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

A_res_path = './data/monet2photo/testA_res'
B_res_path = './data/monet2photo/testB_res'
A_test_path = './data/monet2photo/testA'
B_test_path = './data/monet2photo/testB'


def testB(img_path):
    if img_path.endswith('.png'):
        img = cv2.imread(os.path.join(B_test_path, img_path))
        img = img[:, :, ::-1]
    else:
        img = Image.open(os.path.join(B_test_path, img_path))

    transforms = transform.Compose([
        transform.ToTensor(),
        transform.Resize((256, 256)),
        transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = transforms(img.copy())
    img = img[None].to(device)  # [1,3,256,256]

    # 实例化网络 todo
    Gb = G(img[0].shape, 9).to(device)
    # 加载预训练权重
    ckpt = torch.load('weights/cycle_monent2photo.pth', map_location=device)
    Gb.load_state_dict(ckpt['Gb_model'], strict=False)

    Gb.eval()
    out = Gb(img)[0]
    out = out.permute(1, 2, 0)
    out = (0.5 * (out + 1)).cpu().detach().numpy()
    plt.figure()
    plt.imshow(out)
    plt.savefig(os.path.join(B_res_path, img_path))
    # plt.show()


def testA(img_path):
    if img_path.endswith('.png'):
        img = cv2.imread(os.path.join(A_test_path, img_path))
        img = img[:, :, ::-1]
    else:
        img = Image.open(os.path.join(A_test_path, img_path))

    transforms = transform.Compose([
        transform.ToTensor(),
        transform.Resize((256, 256)),
        transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = transforms(img.copy())
    img = img[None].to(device)  # [1,3,256,256]

    # 实例化网络
    Ga = G(img[0].shape, 9).to(device)
    # 加载预训练权重
    ckpt = torch.load('weights/cycle_monent2photo.pth', map_location=device)
    Ga.load_state_dict(ckpt['Ga_model'], strict=False)

    Ga.eval()
    out = Ga(img)[0]
    out = out.permute(1, 2, 0)
    out = (0.5 * (out + 1)).cpu().detach().numpy()
    plt.figure()
    plt.imshow(out)
    plt.savefig(os.path.join(A_res_path, img_path))
    # plt.show()


if __name__ == '__main__':
    if not os.path.exists(A_res_path):
        os.mkdir(A_res_path)
    if not os.path.exists(B_res_path):
        os.mkdir(B_res_path)

    for file in os.listdir(A_test_path):
        testA(file)

    for file in os.listdir(B_test_path):
        testB(file)
