import datetime
import os
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import joint_transforms
from config import train_data
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir
from DANet import RGBD_sal
from torch.backends import cudnn
import torch.nn.functional as functional

cudnn.benchmark = True

torch.manual_seed(2018)
torch.cuda.set_device(0)

##########################hyperparameters###############################
ckpt_path = './model'
exp_name = 'model_vgg16_DANet'
args = {
    'iter_num':20500,
    'train_batch_size': 4,
    'last_iter': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'snapshot': ''
}
##########################data augmentation###############################
joint_transform = joint_transforms.Compose([
    joint_transforms.RandomCrop(384,384),
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10)
])
img_transform = transforms.Compose([
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
target_transform = transforms.ToTensor()
##########################################################################
train_set = ImageFolder(train_data, joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=12, shuffle=True)
criterion = nn.BCEWithLogitsLoss().cuda()
criterion_BCE = nn.BCELoss().cuda()
criterion_MAE = nn.L1Loss().cuda()
criterion_MSE = nn.MSELoss().cuda()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')


def main():
    model = RGBD_sal()
    net = model.cuda().train()
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])
    if len(args['snapshot']) > 0:
        print ('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']
    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)

def train(net, optimizer):
    curr_iter = args['last_iter']
    while True:
        total_loss_record, loss1_record, loss2_record,loss3_record,loss4_record,loss5_record,loss6_record,loss7_record,loss8_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(),AvgMeter(),AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']
            inputs, depth, labels= data
            labels[labels>0.5] = 1
            labels[labels!=1] = 0
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            depth = Variable(depth).cuda()
            labels = Variable(labels).cuda()
            outputs,outputs_fg,outputs_bg,attention1,attention2,attention3,attention4,attention5 =  net(inputs,depth) #hed
            ##########loss#############
            optimizer.zero_grad()
            labels1 = functional.interpolate(labels, size=24, mode='bilinear')
            labels2 = functional.interpolate(labels, size=48, mode='bilinear')
            labels3 = functional.interpolate(labels, size=96, mode='bilinear')
            labels4 = functional.interpolate(labels, size=192, mode='bilinear')
            loss1 = criterion_BCE(attention1, labels1)
            loss2 = criterion_BCE(attention2, labels2)
            loss3 = criterion_BCE(attention3, labels3)
            loss4 = criterion_BCE(attention4, labels4)
            loss5 = criterion_BCE(attention5, labels)
            loss6 = criterion(outputs_fg, labels)
            loss7 = criterion(outputs_bg, (1-labels))
            loss8 = criterion(outputs, labels)
            total_loss = loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8
            total_loss.backward()
            optimizer.step()
            total_loss_record.update(total_loss.item(), batch_size)
            loss1_record.update(loss1.item(), batch_size)
            loss2_record.update(loss2.item(), batch_size)
            loss3_record.update(loss3.item(), batch_size)
            loss4_record.update(loss4.item(), batch_size)
            loss5_record.update(loss5.item(), batch_size)
            loss6_record.update(loss6.item(), batch_size)
            loss7_record.update(loss7.item(), batch_size)
            loss8_record.update(loss8.item(), batch_size)
            curr_iter += 1
            #############log###############
            if curr_iter %2050==0:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_path, exp_name, '%d_optim.pth' % curr_iter))
            log = '[iter %d], [total loss %.5f],[loss1 %.5f],,[loss2 %.5f],[loss3 %.5f],[loss4 %.5f],[loss5 %.5f],[loss6 %.5f],[loss7 %.5f],[loss8 %.5f],[lr %.13f] '  % \
                     (curr_iter, total_loss_record.avg, loss1_record.avg,loss2_record.avg,loss3_record.avg,loss4_record.avg,loss5_record.avg,loss6_record.avg,loss7_record.avg,loss8_record.avg,optimizer.param_groups[1]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')
            if curr_iter == args['iter_num']:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_path, exp_name, '%d_optim.pth' % curr_iter))
                return
            #############end###############

if __name__ == '__main__':
    main()
