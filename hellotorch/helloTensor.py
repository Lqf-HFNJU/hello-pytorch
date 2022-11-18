import torch

if __name__ == '__main__':
    a = torch.randn(2, 5)
    print(a)

    print(torch.argmax(a, dim=0))
    print(torch.max(a, dim=0))
    print(a.shape[0])

    test_y = torch.randn(5, 2)
    test_x = torch.randn(5, 2)

    aa = torch.where(torch.argmax(test_y, dim=1) == torch.argmax(test_x, dim=1), True, False)

    print(test_y)
    print(test_x)
    print(aa)
    print(aa.sum())
