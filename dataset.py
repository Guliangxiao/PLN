import os
import os.path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset


pd.set_option('display.max_rows', None)  
pd.set_option('display.max_columns', None)  

# 根据txt文件制作ground truth
CLASS_NUM = 20  # 使用其他训练集需要更改


class plnDataset(Dataset):
    image_size = 448

    def __init__(self, root, train, transform):
        # 下面这些都保持原样
        self.root      = root
        self.train     = train
        self.transform = transform
        self.fnames    = []
        self.boxes     = []
        self.labels    = []
        self.S = 14
        self.B = 2
        self.C = CLASS_NUM
        self.mean = (123, 117, 104)

        # ====== 唯一改动：不再用 list_file，直接扫描 ======
        from pathlib import Path
        root = Path(root)
        label_root = root / 'PLNLabels'          # 假设 labels 在 img_root/labels 里
        ext = '*.jpg'
        img_paths = []

        img_paths.extend((root/'JPEGImages').glob(ext))
        img_paths = sorted(img_paths)

        # 逐张图处理
        for img_p in img_paths:
            txt_p = label_root / (img_p.stem + '.txt')
            if not txt_p.exists():
                continue

            # 把图片路径存进去
            self.fnames.append(str(img_p))

            # 读 label txt
            boxes, labels = [], []
            with open(txt_p) as f:
                for line in f:
                    cls, xc, yc, w, h = map(float, line.strip().split())
                    
                    boxes.append([xc, yc, w, h])
                    labels.append(int(cls))

            self.boxes.append(torch.tensor(boxes))
            self.labels.append(torch.tensor(labels, dtype=torch.long))

        self.num_samples = len(self.boxes)
    def __getitem__(self, idx):
        try:
            # 获取一张图像
            from pathlib import Path
            root = Path(self.root)
            fname = Path(self.fnames[idx])
            # 读取这张图像
            img = cv2.imread(root/'JPEGImages'/fname)
            # 拷贝一份，避免在对数据进行处理时对原始数据进行修改
            boxes = self.boxes[idx].clone()
            labels = self.labels[idx].clone()
            #print("boxes:", boxes)
            #print("labels:", labels)
    
            """
            数据增强里面的各种变换用pytorch自带的transform是做不到的，因为对图片进行旋转、随即裁剪等会造成bbox的坐标也会发生变化，
            所以需要自己来定义数据增强,这里推荐使用功albumentations更简单便捷
            """

            # cv2读取的图像是BGR，转成RGB
            img = self.BGR2RGB(img)
            # 减去均值，帮助网络更快地收敛并提高其性能
            img = self.subMean(img, self.mean)
            # 调整图像到统一大小
            img = cv2.resize(img, (self.image_size, self.image_size))
            loader = Label_loader( "", 0, 448, 14)
            target = []
            for i in range(4):
                t = loader.load_label(i,boxes,labels)
                target.append(t)
            # 得到一个[4,14,14,204]维的tensor
            target = torch.stack(target)
        except Exception as e:
            print(f"❌ 坏样本 idx={idx}, fname={self.fnames[idx]}")
            print("boxes:", self.boxes[idx])
            print("labels:", self.labels[idx])
            raise e

        # 进行数据增强操作
        for t in self.transform:
            img = t(img)
        
        return img, target
    
    def __len__(self):
        return self.num_samples
        # 编码图像标签为7x7*30的向量，输入的boxes为一张图的归一化形式位置坐标(X1,Y1,X2,Y2)

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr


class Label_loader():
    def __init__(self,  loader_type, seed, pic_width, S=14) -> None:

        self.boxes = []
        self.labels = []
        self.difficulties = []
        self.loader_type = loader_type

        self.eval_label = []
        self.eval_box = []

        self.s = S
        self.classes = 20
        self.B = 2  # the number of objects
        self.infinite = 100000000  # c
        self.pic_width = pic_width

        # random.seed(seed)  # set random seed
        torch.manual_seed(seed)

    def pij_pro(self, branch,boxes):#这个是p，也就是概率维度为1
        posi = []
        posi_ct = []

        # 角点+中心点
        p_tensor = torch.zeros((self.s, self.s, 2))
        p_ct_tensor = torch.zeros((self.s, self.s, 2))

        #p_tensor1 = torch.zeros((self.s, self.s, 1))
        #p_ct_tensor1 = torch.zeros((self.s, self.s, 1))
        i = 0
        for box in boxes:
            if i >=2:
                continue
            xc, yc, w, h = box
            xmin = xc - w*0.5
            ymin = yc - h*0.5
            xmax = xc + w*0.5
            ymax = yc + h*0.5
            if xmax >= 0.99:
                xmax -= 0.001
            if ymax >= 0.99:
                ymax -= 0.001
            if (branch == 0):
                p_tensor[int(ymax * self.s), int(xmin * self.s),i] = 1  # left-bot
                #p_tensor1[int(ymax * self.s), int(xmin * self.s)] = 1  # left-bot
            elif (branch == 1):
                p_tensor[int(ymin * self.s), int(xmin * self.s),i] = 1  # left-top
                #p_tensor1[int(ymin * self.s), int(xmin * self.s)] = 1  # left-top
            elif (branch == 2):
                p_tensor[int(ymax * self.s), int(xmax * self.s),i] = 1  # right-bot
                #p_tensor1[int(ymax * self.s), int(xmax * self.s)] = 1  # right-bot
            elif (branch == 3):
                p_tensor[int(ymin * self.s), int(xmax * self.s),i] = 1  # right-top
                #p_tensor1[int(ymin * self.s), int(xmax * self.s)] = 1  # right-top
            p_ct_tensor[int(yc * self.s), int(xc * self.s),i] = 1  # center
            #p_ct_tensor1[int((ymin + ymax) / 2 * self.s), int((xmin + xmax) / 2 * self.s)] = 1  # center
            i +=1
        posi.append(p_tensor)
        #posi.append(p_tensor1)
        posi_ct.append(p_ct_tensor)
        #posi_ct.append(p_ct_tensor1)

        return posi, posi_ct

 
    def boxes_tensor_pro(self, boxes) -> torch.tensor:
        ''' process the boxes to result
        Args:
            idx: idx-th image
            boxes: processed box
        Returns:
            boxes: tensor. [self.B,points(4),2,2] precise coordinate of corner
        '''
        boxes_list = []
        points = []
        i = 0
        for box in boxes:
            if i >=2:
                continue
            xc, yc, w, h = box
            xmin = xc - w*0.5
            ymin = yc - h*0.5
            xmax = xc + w*0.5
            ymax = yc + h*0.5
            if xmax >= 0.99:
                xmax -= 0.001
            if ymax >= 0.99:
                ymax -= 0.001
            center_x = xc
            center_y = yc
            points.append(
                torch.tensor([(center_x * self.s, center_y * self.s), (xmin * self.s, ymax * self.s)]))  # left-top
            points.append(
                torch.tensor([(center_x * self.s, center_y * self.s), (xmin * self.s, ymin * self.s)]))  # left-bot
            points.append(
                torch.tensor([(center_x * self.s, center_y * self.s), (xmax * self.s, ymax * self.s)]))  # right-top
            points.append(
                torch.tensor([(center_x * self.s, center_y * self.s), (xmax * self.s, ymin * self.s)]))  # right-bot
            #points_tensor = torch.stack(points)
            i+=1
        if len(boxes) != 0:
            stacked_tensor=torch.stack(points)
            #stacked_tensor = torch.stack(boxes_list).squeeze(0)
            boxes_1 = stacked_tensor.view(int(stacked_tensor.shape[0] / 4), 4, 2, 2)
            # [N,4,2,2]
            return boxes_1
        else:
            return torch.zeros((2, 4, 2, 2))

    def LxLy_tensor_pro(self, ct_pt, corner_pt) -> torch.tensor:
        ''' process the L to result
        Args:
            ct_pt: tensor center point
            corner_pt: tensor corner point
        Returns:
            Lx: tensor [s]
            Ly: tensor [s]
        '''
        Lx_ct = torch.zeros(self.s)
        Ly_ct = torch.zeros(self.s)

        Lx_ct[int(corner_pt[0])] = self.infinite
        Ly_ct[int(corner_pt[1])] = self.infinite

        L_ct = torch.cat((Lx_ct, Ly_ct))

        Lx_cr = torch.zeros(self.s)
        Ly_cr = torch.zeros(self.s)

        Lx_cr[int(ct_pt[0])] = self.infinite
        Ly_cr[int(ct_pt[1])] = self.infinite

        L_cr = torch.cat((Lx_cr, Ly_cr))

        return L_ct, L_cr

    def L_tensor_pro(self, branch, boxes) -> list:
        ''' process the L to result
        Args:
            idx: idx-th image
            branch: branch-th result range(0,3)
            boxes: processed boxes
        Returns:
            Link_ct_list: list_ct includeing ct tensor [self.s,self.s,2*self.s]
            Link_cr_list: list_cr includeing cr tensor [self.s,self.s,2*self.s]
        '''
        Link_ct_list = []
        Link_cr_list = []

        Link_tmp_cr = torch.zeros((self.s, self.s, 4 * self.s))
        Link_tmp_ct = torch.zeros((self.s, self.s, 4 * self.s))

        #Link_tmp_cr1 = torch.zeros((self.s, self.s, 2 * self.s))
        #Link_tmp_ct1 = torch.zeros((self.s, self.s, 2 * self.s))
        box_tensor = self.boxes_tensor_pro(boxes)
        box_branch = box_tensor[:, branch, ...]  # [self.B,2,2]
        for obj_idx, obj_data in enumerate(box_branch):#obj_data 是[2,2],为中心点xy与角点xy
            L_ct, L_cr = self.LxLy_tensor_pro(obj_data[0], obj_data[1])
            Link_tmp_ct[int(obj_data[1][1]), int(obj_data[1][0]),14*obj_idx:14*(obj_idx+2)] = L_cr
            #Link_tmp_cr1[int(obj_data[1][1]),int(obj_data[1][0]) ] = L_cr
            Link_tmp_cr[int(obj_data[0][1]), int(obj_data[0][0]),14*obj_idx:14*(obj_idx+2)] = L_ct
            #Link_tmp_ct1[int(obj_data[0][1]), int(obj_data[0][0])] = L_ct

        Link_tmp_ct[..., :self.s] = F.softmax(Link_tmp_ct[..., :self.s], dim=-1)
        Link_tmp_ct[..., self.s:2*self.s] = F.softmax(Link_tmp_ct[..., self.s:2*self.s], dim=-1)
        Link_tmp_ct[..., 2*self.s:3*self.s] = F.softmax(Link_tmp_ct[..., 2*self.s:3*self.s], dim=-1)
        Link_tmp_ct[..., 3*self.s:4*self.s] = F.softmax(Link_tmp_ct[..., 3*self.s:4*self.s], dim=-1)

        #Link_tmp_ct1[..., :self.s] = F.softmax(Link_tmp_ct1[..., :self.s], dim=-1)
        #Link_tmp_ct1[..., self.s:] = F.softmax(Link_tmp_ct1[..., self.s:], dim=-1)
        Link_tmp_cr[..., :self.s] = F.softmax(Link_tmp_cr[..., :self.s], dim=-1)
        Link_tmp_cr[..., self.s:2*self.s] = F.softmax(Link_tmp_cr[..., self.s:2*self.s], dim=-1)
        Link_tmp_cr[..., 2*self.s:3*self.s] = F.softmax(Link_tmp_cr[..., 2*self.s:3*self.s], dim=-1)
        Link_tmp_cr[..., 3*self.s:4*self.s] = F.softmax(Link_tmp_cr[..., 3*self.s:4*self.s], dim=-1)
        #Link_tmp_cr1[..., :self.s] = F.softmax(Link_tmp_cr1[..., :self.s], dim=-1)
        #Link_tmp_cr1[..., self.s:] = F.softmax(Link_tmp_cr1[..., self.s:], dim=-1)
        Link_ct_list.append(Link_tmp_ct)
        #Link_ct_list.append(Link_tmp_ct1)

        Link_cr_list.append(Link_tmp_cr)
        #Link_cr_list.append(Link_tmp_cr1)
        # 返回lxij+lyij[14,14,2*14]
        return Link_ct_list, Link_cr_list

    def Qij_tensor_pro(self, branch, boxes, labels) -> torch.tensor:
        ''' conbine the classes with position
        Args:
            idx: idx-th image
            branch: branch-th result
            boxes: processed box
            labels: mixed label
        Returns:
            Q_list: list. including [S,S,classes*B] corner point
            Q_ct_list: list. including [S,S,classes*B] ct point
        '''

        Q_list = []
        Q_ct_list = []
        Q_tensor = torch.zeros((self.s, self.s, 20*2))
        Q_ct_tensor = torch.zeros((self.s, self.s, 20*2))

        #Q_tensor1 = torch.zeros((self.s, self.s, 20))
        #Q_ct_tensor1 = torch.zeros((self.s, self.s, 20))
        for idx_ele, item in enumerate(labels):
            if idx_ele>=2:
                continue
            # self.boxes[idx][idx_ele] [xmin, ymin, xmax, ymax]
            box = boxes[idx_ele]
            xc, yc, w, h = box
            xmin = xc - w*0.5
            ymin = yc - h*0.5
            xmax = xc + w*0.5
            ymax = yc + h*0.5
            if xmax >= 0.99:
                xmax -= 0.001
            if ymax >= 0.99:
                ymax -= 0.001


            if (branch == 0):
                Q_tensor[int(ymax * self.s), int(xmin * self.s), item+20*idx_ele] = self.infinite
                #Q_tensor1[int(ymax * self.s), int(xmin * self.s), item ] = self.infinite
            elif (branch == 1):
                Q_tensor[int(ymin * self.s), int(xmin * self.s), item+20*idx_ele] = self.infinite
                #Q_tensor1[int(ymin * self.s), int(xmin * self.s), item] = self.infinite
            elif (branch == 2):
                Q_tensor[int(ymax * self.s), int(xmax * self.s), item+20*idx_ele ] = self.infinite
                #Q_tensor1[int(ymax * self.s), int(xmax * self.s), item] = self.infinite
            elif (branch == 3):
                Q_tensor[int(ymin * self.s), int(xmax * self.s), item+20*idx_ele] = self.infinite
                #Q_tensor1[int(ymin * self.s), int(xmax * self.s), item ] = self.infinite
            Q_ct_tensor[int(yc * self.s), int(xc * self.s), item+20*idx_ele] = self.infinite
            #Q_ct_tensor1[int((ymin + ymax) / 2 * self.s), int((xmin + xmax) / 2 * self.s), item] = self.infinite
        Q_ct_tensor[:,:,:20] = F.softmax(Q_ct_tensor[:,:,:20], dim=-1).clone()
        Q_ct_tensor[:,:,20:] = F.softmax(Q_ct_tensor[:,:,20:], dim=-1).clone()

        Q_tensor[:,:,:20] = F.softmax(Q_tensor[:,:,:20], dim=-1).clone()
        Q_tensor[:,:,20:] = F.softmax(Q_tensor[:,:,20:], dim=-1).clone()

        #Q_ct_tensor1 = F.softmax(Q_ct_tensor1, dim=-1).clone()
        #Q_tensor1 = F.softmax(Q_tensor1, dim=-1).clone()
        Q_list.append(Q_tensor)
        #Q_list.append(Q_tensor1)
        Q_ct_list.append(Q_ct_tensor)
        #Q_ct_list.append(Q_ct_tensor1)
        return Q_list, Q_ct_list

    def relative_pos_tensor_pro(self, branch, boxes) -> torch.tensor:
        ''' get relative coordinate to the same grid
        Args:
            idx: idx-th image
            branch: branch-th result range(0,3)
            boxes: processed boxes
        Returns:
            pos_list: list. including [s,s,2*B] corner precise
            pos_ct_list: list. including [s,s,2*B] center precise
        '''
        pos_list = []
        pos_ct_list = []

        pos_tensor = torch.zeros((self.s, self.s, 4))
        pos_ct_tensor = torch.zeros((self.s, self.s, 4))
        #pos_tensor1 = torch.zeros((self.s, self.s, 2))
        #pos_ct_tensor1 = torch.zeros((self.s, self.s, 2))
        i = 0
        for box in boxes:
            if i >=2:
                continue
            xc, yc, w, h = box
            xmin = xc - w*0.5
            ymin = yc - h*0.5
            xmax = xc + w*0.5
            ymax = yc + h*0.5
            if xmax >= 0.99:
                xmax -= 0.001
            if ymax >= 0.99:
                ymax -= 0.001
            if (branch == 0):
                pos_tensor[int(ymax * self.s), int(xmin * self.s),2*i:2*i+2] = torch.tensor(
                    [ymax * self.s - int(ymax * self.s), xmin * self.s - int(xmin * self.s)])  # left-top
                # pos_tensor1[int(xmin * self.s), int(ymax * self.s)] = torch.tensor(
                #     [xmin * self.s - int(xmin * self.s), ymax * self.s - int(ymax * self.s)])  # left-top
                #pos_tensor1[int(ymax * self.s), int(xmin * self.s)] = torch.tensor(
                    #[ymax * self.s - int(ymax * self.s), xmin * self.s - int(xmin * self.s)])
            elif (branch == 1):
                pos_tensor[int(ymin * self.s), int(xmin * self.s),2*i:2*i+2] = torch.tensor(
                    [ymin * self.s - int(ymin * self.s), xmin * self.s - int(xmin * self.s)])  # left-bot
                #pos_tensor1[int(ymin * self.s), int(xmin * self.s)] = torch.tensor(
                    #[ymin * self.s - int(ymin * self.s), xmin * self.s - int(xmin * self.s)])  # left-bot
            elif (branch == 2):
                pos_tensor[int(ymax * self.s), int(xmax * self.s),2*i:2*i+2] = torch.tensor(
                    [ymax * self.s - int(ymax * self.s), xmax * self.s - int(xmax * self.s)])  # right-top
                #pos_tensor1[int(ymax * self.s), int(xmax * self.s)] = torch.tensor(
                    #[ymax * self.s - int(ymax * self.s), xmax * self.s - int(xmax * self.s)])  # right-top
            elif (branch == 3):
                pos_tensor[int(ymin * self.s),int(xmax * self.s),2*i:2*i+2 ] = torch.tensor(
                    [ymin * self.s - int(ymin * self.s),xmax * self.s - int(xmax * self.s) ])  # right-bot
                #pos_tensor1[int(ymin * self.s),int(xmax * self.s) ] = torch.tensor(
                    #[ymin * self.s - int(ymin * self.s),xmax * self.s - int(xmax * self.s) ])  # right-bot

            ctx = int(xc*self.s)
            cty = int(yc*self.s)
            pos_ct_tensor[cty,ctx,2*i:2*i+2] = torch.tensor([xc*self.s - ctx, yc*self.s - cty])  # center
            #pos_ct_tensor1[cty,ctx] = torch.tensor([ctx_p - ctx, cty_p - cty])  # center
            i+=1
        pos_list.append(pos_tensor)
        #pos_list.append(pos_tensor1)
        pos_ct_list.append(pos_ct_tensor)
        #pos_ct_list.append(pos_ct_tensor1)
        return pos_list, pos_ct_list

    def combine_list(self, Q_list, Q_ct_list, posi_list, posi_ct_list, Link_ct_list, Link_cr_list, pos_list,
                     pos_ct_list) -> torch.tensor:
        ''' combine the ct with cr points -> [14,14,204]
        Args:
            Q_list: list
            Q_ct_list: list
            posi_list: list
            posi_ct_list: list
            Link_ct_list: list
            Link_cr_list: list
            pos_list: list
            pos_ct_list: list
        Returns:
            final label tensor: [14,14,204]
        '''
        list_feature = []
        list_ct_feature = []
        for i in range(self.B):
            # zip_tensor_ct = torch.cat((Q_ct_list[i], posi_ct_list[i], Link_ct_list[i], pos_ct_list[i]), dim=-1)
            zip_tensor_ct = torch.cat((posi_ct_list[0][:,:,i:i+1], pos_ct_list[0][:,:,2*i:2*i+2],Link_ct_list[0][:,:,i*28:i*28+28],Q_ct_list[0][:,:,i*20:i*20+20]), dim=-1)
            # zip_tensor = torch.cat((posi_list[i], pos_list[i],Link_cr_list[i],Q_list[i]), dim=-1)
            zip_tensor = torch.cat((posi_list[0][:,:,i:i+1], pos_list[0][:,:,2*i:2*i+2], Link_cr_list[0][:,:,i*28:i*28+28], Q_list[0][:,:,i*20:i*20+20]), dim=-1)
            list_feature.append(zip_tensor)
            list_ct_feature.append(zip_tensor_ct)
        feature_tensor = torch.cat((list_feature), dim=-1)
        feature_ct_tensor = torch.cat((list_ct_feature), dim=-1)
        return torch.cat((feature_ct_tensor, feature_tensor), dim=-1)

    def load_label(self,branch,boxes,labels):
        if len(boxes) != 0:
            # cal Q
            Q_tensor, Q_ct_tensor = self.Qij_tensor_pro(branch,boxes,labels)  # [S,S,classes*B],[S,S,classes*B]
            # cal P,mixed p and note it with lamda
            P_tensor, P_ct_tensor = self.pij_pro(branch, boxes)  # [s,s,1*B],[s,s,1*B]
            # cal L
            Link_ct_list, Link_cr_list = self.L_tensor_pro( branch, boxes)  # [s,s,2*s*B],[s,s,2*s*B]
            # cal x
            x_tensor, x_ct_tensor = self.relative_pos_tensor_pro(branch, boxes)  # [s,s,2*B], [s,s,2*B]

            final_tensor = self.combine_list(Q_tensor, Q_ct_tensor, P_tensor, P_ct_tensor, Link_ct_list, Link_cr_list,
                                             x_tensor, x_ct_tensor)
            # return
            # cr_pt, ct_pt = self.pos_tensor_pro(branch, boxes)
            # final_pt_temp = torch.cat((ct_pt, cr_pt), dim=-1)
            # final_pt = final_pt_temp.reshape(final_pt_temp.shape[0] // 2, 2)
            # return final_tensor, final_pt
            return final_tensor
        else:
            return torch.zeros(14,14,204)


if __name__ == '__main__':
    device = 'cuda'
    file_root = 'VOCdevkit/VOC2012/PLNLabels/'
    batch_size = 1

    learning_rate = 0.001
    num_epochs = 100
    # 自定义训练数据集
    train_dataset = plnDataset(root=file_root,  train=True,
                                transform=[transforms.ToTensor()])
    img, target = train_dataset.__getitem__(1)
    print("target encode success")
        #
        # img = img.to("cuda")
        # target = target.to("cuda")
        #
        # model = inceptionresnetv2(num_classes=20, pretrained='imagenet').cuda()
        # # # 加载权重，就是在train.py中训练生成的权重文件yolo.pth
        # model.load_state_dict(torch.load("pln.pth"))
        # # 测试模式
        # model.eval()
        #
        # img = img.unsqueeze(0)
        # print(img.shape)
        # torch.set_printoptions(profile="full")
        # p = model(img)
        # # p = p.permute(0, 2, 3, 1)
        #
        # print(p)
        # print(target.shape)
        # print(target.unsqueeze(0).expand_as(p).shape)
        # target = target.unsqueeze(0).expand_as(p)
        # loss = plnLoss.plnLoss(14, 2, 1, 1, w_link=1)
        # a = loss.forward(p, target)
        # print(a)
