import torchvision
from tqdm import tqdm  # 进度条提示模块
import torch
import os

"""
A：莫奈风格
B：原图
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_one_epoch(Ga, Da, Gb, Db, train_loader, optim_G, optim_D, writer, loss, device, plot_every, epoch, l1_loss,
                    A_fake_pool, B_fake_pool):
    pd = tqdm(train_loader)
    loss_D, loss_G = 0, 0
    step = 0
    Ga.train()
    Da.train()
    Gb.train()
    Db.train()
    for idx, data in enumerate(train_loader):
        A_real = data[0].to(device)
        B_real = data[1].to(device)

        # 前向传递
        B_fake = Ga(A_real)  # Ga生成的假B
        A_rec = Gb(B_fake)  # Gb重构回的A
        A_fake = Gb(B_real)  # Gb生成的假A
        B_rec = Ga(A_fake)  # Ga重构回的B

        # 训练G   => G包含六部分损失
        set_required_grad([Da, Db], requires_grad=False)  # 不更新D
        optim_G.zero_grad()
        ls_G = train_G(Da=Da, Db=Db, B_fake=B_fake, loss=loss, A_fake=A_fake, l1_loss=l1_loss,
                       A_rec=A_rec,
                       A_real=A_real, B_rec=B_rec, B_real=B_real, Ga=Ga, Gb=Gb)
        ls_G.backward()
        optim_G.step()

        # 训练D
        set_required_grad([Da, Db], requires_grad=True)
        optim_D.zero_grad()
        A_fake_p = A_fake_pool.query(A_fake)
        B_fake_p = B_fake_pool.query(B_fake)
        ls_D = train_D(Da=Da, Db=Db, B_fake=B_fake_p, B_real=B_real, loss=loss, A_fake=A_fake_p, A_real=A_real)
        ls_D.backward()
        optim_D.step()

        loss_D += ls_D
        loss_G += ls_G

        pd.desc = 'train_{} G_loss: {} D_loss: {}'.format(epoch, ls_G.item(), ls_D.item())

        # 绘制训练结果
        if idx % plot_every == 0:
            writer.add_images(tag='epoch{}_Ga'.format(epoch), img_tensor=0.5 * (torch.cat([A_real, B_fake], 0) + 1),
                              global_step=step)
            writer.add_images(tag='epoch{}_Gb'.format(epoch), img_tensor=0.5 * (torch.cat([B_real, A_fake], 0) + 1),
                              global_step=step)
            step += 1
    mean_lsG = loss_G / len(train_loader)
    mean_lsD = loss_D / len(train_loader)
    return mean_lsG, mean_lsD


@torch.no_grad()
def val(Ga, Da, Gb, Db, val_loader, loss, device, l1_loss, epoch):
    pd = tqdm(val_loader)
    loss_D, loss_G = 0, 0
    Ga.eval()
    Da.eval()
    Gb.eval()
    Db.eval()
    all_loss = 10000
    for idx, item in enumerate(pd):
        A_real_img = item[0].to(device)
        B_real_img = item[1].to(device)

        B_fake_img = Ga(A_real_img)
        A_fake_img = Gb(B_real_img)

        A_rec = Gb(B_fake_img)
        B_rec = Ga(A_fake_img)

        # D的loss
        ls_D = train_D(Da=Da, Db=Db, B_fake=B_fake_img, B_real=B_real_img, loss=loss, A_fake=A_fake_img,
                       A_real=A_real_img)
        # G的loss
        ls_G = train_G(Da=Da, Db=Db, B_fake=B_fake_img, loss=loss, A_fake=A_fake_img, l1_loss=l1_loss,
                       A_rec=A_rec,
                       A_real=A_real_img, B_rec=B_rec, B_real=B_real_img, Ga=Ga, Gb=Gb)

        loss_G += ls_G
        loss_D += ls_D
        pd.desc = 'val_{}: G_loss:{} D_Loss:{}'.format(epoch, ls_G.item(), ls_D.item())

        # 保存最好的结果
        all_ls = ls_G + ls_D
        if all_ls < all_loss:
            all_loss = all_ls
            best_image = torch.cat([A_real_img, B_fake_img, B_real_img, A_fake_img], 0)
    result_img = (best_image + 1) * 0.5
    if not os.path.exists('./results'):
        os.mkdir('./results')

    torchvision.utils.save_image(result_img, './results/val_epoch{}_cycle.jpg'.format(epoch))


def set_required_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for params in net.parameters():
                params.requires_grad = requires_grad


"""训练G"""


# (判别器a, 判别器b, B的假图, 损失函数, A的假图, l1, A的重构, A的真图, B的重构, B的真图, 生成器a, 生成器b)
def train_G(Da, Db, B_fake, loss, A_fake, l1_loss, A_rec, A_real, B_rec, B_real, Ga, Gb):
    # GAN loss
    Da_out_fake = Da(B_fake)
    Db_out_fake = Db(A_fake)
    Ga_gan_loss = loss(Da_out_fake, torch.ones(Da_out_fake.size()).to(device))
    Gb_gan_loss = loss(Db_out_fake, torch.ones(Db_out_fake.size()).to(device))

    # Cycle loss
    Cycle_A_loss = l1_loss(A_rec, A_real) * 10
    Cycle_B_loss = l1_loss(B_rec, B_real) * 10

    # identity loss
    # A风格的图像 放在 B -> A 生成器中，生成的图像也要是 A风格
    # B风格的图像 放在 A -> B 生成器中，生成的图像也要是 B风格
    Ga_id_out = Ga(B_real)
    Gb_id_out = Gb(A_real)
    Ga_id_loss = l1_loss(Ga_id_out, B_real) * 10 * 0.5
    Gb_id_loss = l1_loss(Gb_id_out, A_real) * 10 * 0.5

    # G的总损失
    ls_G = Ga_gan_loss + Gb_gan_loss + Cycle_A_loss + Cycle_B_loss + Ga_id_loss + Gb_id_loss

    return ls_G


"""训练D"""


# (判别器a, 判别器b, B的假图, B的真图, 损失函数, A的假图, A的真图)
def train_D(Da, Db, B_fake, B_real, loss, A_fake, A_real):
    # Da的loss
    Da_fake_out = Da(B_fake.detach()).squeeze()
    Da_real_out = Da(B_real).squeeze()
    ls_Da1 = loss(Da_fake_out, torch.zeros(Da_fake_out.size()).to(device))
    ls_Da2 = loss(Da_real_out, torch.ones(Da_real_out.size()).to(device))

    ls_Da = (ls_Da1 + ls_Da2) * 0.5
    # Db的loss
    Db_fake_out = Db(A_fake.detach()).squeeze()
    Db_real_out = Db(A_real.detach()).squeeze()
    ls_Db1 = loss(Db_fake_out, torch.zeros(Db_fake_out.size()).to(device))
    ls_Db2 = loss(Db_real_out, torch.ones(Db_real_out.size()).to(device))

    ls_Db = (ls_Db1 + ls_Db2) * 0.5

    # D的总损失
    ls_D = ls_Da + ls_Db
    return ls_D
