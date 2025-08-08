import InceptionV2
import loss
import dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import os
import torch

# ---------------------------------
# 保存模型（训练时调用）
# ---------------------------------
def save_checkpoint(
        epoch,               # 当前 epoch
        model,               # 网络
        optimizer,           # 优化器
        ckpt_dir='checkpoints',
        ckpt_name='last.pth'
    ):
    """
    保存模型权重、优化器状态、epoch 等信息。
    默认目录：./checkpoints/ckpt_name
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, ckpt_name)
    torch.save({
        'epoch'            : epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f'Checkpoint saved at {path}')

# ---------------------------------
# 加载模型（训练前或推理前调用）
# ---------------------------------
def load_checkpoint(
        model,
        optimizer=None,      # 如果仅推理可设为 None
        ckpt_path='checkpoints/last.pth',
        device='cpu'
    ):
    """
    加载权重到 model 与 optimizer。
    返回：epoch, loss（便于继续训练）
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss  = checkpoint.get('loss', None)
    print(f'Checkpoint loaded from {ckpt_path} (epoch {epoch})')
    return epoch, loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
file_root = 'F:/study/wxg/6detection/VOCdevkit/VOC2012'
batch_size = 8
learning_rate = 0.0001
num_epochs = 10
type = "train"

train_dataset = dataset.plnDataset(root=file_root,  train=True,
                            transform=[transforms.ToTensor()])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,pin_memory=True)
net = InceptionV2.InceptionV2().to(device)
optimizer = torch.optim.SGD(
    net.parameters(),
    # 学习率
    lr=learning_rate,
    # 动量
    momentum=0.9,
    # 正则化
    weight_decay=5e-4
)
criterion = loss.plnLoss(14,2,w_coord=2,w_link=.5,w_class=.5).to(device)

for epoch in range(num_epochs):
    # 计算损失
    total_loss = 0.
    # 开始迭代训练
    for i, (images, target) in enumerate(train_loader):
        # print("training",images.shape,target.shape)
        images = images.to(device)
        target = target.to(device)
        pred = net(images)
        # target torch.Size([batch, 4, 14, 14, 204])
        # pred torch.Size([4, batch, 204, 14, 14])
        target = target.permute(1, 0 ,2, 3, 4)
        batch_size = pred[0].shape[0]
        loss0 = criterion(pred[0], target[0])
        loss1 = criterion(pred[1], target[1])
        loss2 = criterion(pred[2], target[2])
        loss3 = criterion(pred[3], target[3])
        loss3 = (loss0 + loss1 + loss2 + loss3)/batch_size
        total_loss += loss3.item()

        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss3.backward()
        # 参数优化
        optimizer.step()
        if (i + 1) % 5 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' % (epoch +1, num_epochs, i + 1, len(train_loader), loss3.item(), total_loss / (i + 1)))
    save_checkpoint(epoch, net, optimizer, 
                    ckpt_dir='runs/exp1',
                    ckpt_name=f'epoch_{epoch:03d}.pth')