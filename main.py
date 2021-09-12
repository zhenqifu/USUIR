import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from torch.utils.data import DataLoader
from net.net import net
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lrs
from data import get_training_set, get_eval_set
from utils import *
import random
import time
from net.losses import ColorLoss

# Training settings
parser = argparse.ArgumentParser(description='PyTorch UIE')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--decay', type=int, default='10000', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--data_train', type=str, default='../Dataset/UIE/UIEBD/train/image')
parser.add_argument('--label_train', type=str, default='../Dataset/UIE/UIEBD/train/image')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--data_test', type=str, default='../Dataset/UIE/UIEBD/test/image')
parser.add_argument('--label_test', type=str, default='../Dataset/UIE/UIEBD/test/image')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--patch_size', type=int, default=128, help='Size of cropped HR image')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--output_folder', default='results/', help='Location to save checkpoint models')


opt = parser.parse_args()


def seed_torch(seed=opt.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_torch()
cudnn.benchmark = True

mse_loss = torch.nn.MSELoss().cuda()
color_loss = ColorLoss()


def train():
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, label = batch[0], batch[1]

        input = input.cuda()
        label = label.cuda()

        t0 = time.time()
        j_out, t_out = model(input)

        a_out = get_A(input).cuda()
        I_rec = j_out * t_out + (1 - t_out) * a_out
        loss_1 = mse_loss(I_rec, input)

        lam = np.random.beta(1, 1)
        input_mix = lam * input + (1 - lam) * j_out

        j_out_mix, t_out_mix = model(input_mix)
        loss_2 = mse_loss(j_out_mix, j_out.detach())

        loss_3 = color_loss(j_out)

        total_loss = 1 * loss_1 + 1 * loss_2 + 0.01 * loss_3

        optimizer.zero_grad()
        total_loss.backward()
        epoch_loss += total_loss.item()
        optimizer.step()
        t1 = time.time()
        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Learning rate: lr={} || Timer: {:.4f} sec.".format(epoch,
                iteration, len(training_data_loader), total_loss.item(), optimizer.param_groups[0]['lr'], (t1 - t0)))


def test(testing_data_loader):

    torch.set_grad_enabled(False)
    model.eval()

    print('\nEvaluation:')

    for batch in testing_data_loader:
        with torch.no_grad():
            input, label, name = batch[0], batch[1], batch[2]

        input = input.cuda()

        with torch.no_grad():
            j_out, t_out = model(input)
            a_out = get_A(input).cuda()

            if not os.path.exists(opt.output_folder):
                os.mkdir(opt.output_folder)
                os.mkdir(opt.output_folder + 'J/')
                os.mkdir(opt.output_folder + 'A/')
                os.mkdir(opt.output_folder + 'T/')
            j_out_np = np.clip(torch_to_np(j_out), 0, 1)
            t_out_np = np.clip(torch_to_np(t_out), 0, 1)
            a_out_np = np.clip(torch_to_np(a_out), 0, 1)
            my_save_image(name[0], j_out_np, opt.output_folder + 'J/')
            my_save_image(name[0], t_out_np, opt.output_folder + 'T/')
            my_save_image(name[0], a_out_np, opt.output_folder + 'A/')
    torch.set_grad_enabled(True)


def checkpoint(epoch):
    model_out_path = opt.save_folder+"epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")


print('===> Loading datasets')

test_set = get_eval_set(opt.data_test, opt.label_test)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

train_set = get_training_set(opt.data_train, opt.label_train, opt.patch_size, opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print('===> Building model ')

model = net().cuda()

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

milestones = []
for i in range(1, opt.nEpochs+1):
    if i % opt.decay == 0:
        milestones.append(i)

        
scheduler = lrs.MultiStepLR(optimizer, milestones, opt.gamma)
                            
for epoch in range(opt.start_iter, opt.nEpochs + 1):

    train()
    scheduler.step()

    if (epoch+1) % opt.snapshots == 0:
        checkpoint(epoch)
        # test(testing_data_loader)


