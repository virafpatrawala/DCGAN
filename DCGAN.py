################### Importing the libraries ####################
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt
import os
import imageio
import pickle


################### General Functions #####################
class AverageMeter(object):
    """ Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def plot_loss(d_loss, g_loss, num_epoch, epoches, save_dir):
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, epoches + 1)
    ax.set_ylim(0, max(np.max(g_loss), np.max(d_loss)) * 1.1)
    plt.xlabel('Epoch {}'.format(num_epoch))
    plt.ylabel('Loss')
    
    plt.plot([i for i in range(1, num_epoch + 1)], d_loss, label='Discriminator', color='red', linewidth=3)
    plt.plot([i for i in range(1, num_epoch + 1)], g_loss, label='Generator', color='mediumblue', linewidth=3)
    
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'DCGAN_loss_epoch_{}.png'.format(num_epoch)))
    plt.close()

############## Modifying the CIFAR10 Dataset Loading ################
class MyCIFAR10(dset.CIFAR10):

	def __init__(self, root, label, train=True, transform=None, target_transform=None, download=False):

		self.root = os.path.expanduser(root)
		self.transform = transform
		self.target_transform = target_transform
		self.train = train  # training set or test set

		if download:
			self.download()

		if not self._check_integrity():
			raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

		# now load the picked numpy arrays
		if self.train:
			self.train_data = []
			self.train_labels = []
			for fentry in self.train_list:
				f = fentry[0]
				file = os.path.join(self.root, self.base_folder, f)
				fo = open(file, 'rb')

				entry = pickle.load(fo, encoding='latin1')

				data = entry['data']
				if 'labels' in entry:
					targets = entry['labels']
				else:
					targets = entry['fine_labels']

				targets = np.array(targets)
				data = data[targets == label]
				targets = targets[targets == label]
				targets = list(targets)

				self.train_data.append(data)
				self.train_labels += targets

				fo.close()

			self.train_data = np.concatenate(self.train_data)
			self.train_data = self.train_data.reshape((5000, 3, 32, 32))
			self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

		else:
			f = self.test_list[0][0]
			file = os.path.join(self.root, self.base_folder, f)
			fo = open(file, 'rb')
			entry = pickle.load(fo, encoding='latin1')

			data = entry['data']

			if 'labels' in entry:
				targets = entry['labels']
			else:
				targets = entry['fine_labels']

			targets = np.array(targets)
			data = data[targets == label]
			targets = targets[targets == label]
			targets = list(targets)

			self.test_data = data
			self.test_labels = targets

			fo.close()
			self.test_data = self.test_data.reshape((1000, 3, 32, 32))
			self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC



batchSize = 64
imageSize = 64 

# Transformations for the input images
transform = transforms.Compose([transforms.Resize(imageSize), transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

# We download the training set in the ./data folder and we apply the previous transformations on each image.
train_dataset = MyCIFAR10(root = './data', download = True, train=True, transform = transform, label=5)
val_dataset = MyCIFAR10(root = './data', download = True, train=False, transform = transform, label=5)

# We use dataLoader to get the images of the training set batch by batch.
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batchSize, shuffle = True, num_workers = 2)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = batchSize, shuffle = True, num_workers = 2)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the weights of the two models
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

############################ Architectures #####################
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output

netG = Generator().to(device)
netG.apply(weights_init)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(64, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(256, 512, 4, 2, 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(512, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)

netD = Discriminator().to(device)
netD.apply(weights_init)


######################### Define the Parameters #######################################
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))


# Initialize variables and lists
D_losses = AverageMeter()
G_losses = AverageMeter()
training_obj = AverageMeter()

D_loss_list = []
G_loss_list = []
Training_obj_list = []


######################### Training
num_epochs = 100
for epoch in range(num_epochs):
    # Training the DC GAN
    for i, data in enumerate(dataloader, 0):
                
        netD.zero_grad()
        
        #Calculate Real Disc Loss
        real, _ = data
        input = Variable(real.to(device))
        target = Variable(torch.ones(input.size()[0]).to(device))
        output = netD(input)
        errD_real = criterion(output, target).cuda()
        
        #Calculate Fake Disc Loss
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1).to(device))
        fake = netG(noise)
        target = Variable(torch.zeros(input.size()[0]).to(device))
        output = netD(fake.detach())
        errD_fake = criterion(output, target).cuda()
        
        #Backpropogate total Disc loss
        errD = errD_real + errD_fake
        D_losses.update(errD.data[0])
        errD.backward()
        optimizerD.step()

        #Calculate Gen Loss and Backpropogate
        netG.zero_grad()
        target = Variable(torch.ones(input.size()[0]).to(device))
        output = netD(fake)
        errG = criterion(output, target).cuda()
        G_losses.update(errG.data[0])
        errG.backward()
        optimizerG.step()
        

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch+1, num_epochs, i+1, len(dataloader),
                                                            errD.data[0], errG.data[0]))
        if i % 10 == 0:
            real_grid = vutils.make_grid(real[:50,:,:,:], nrow=5)
            vutils.save_image(real_grid, '%s/real_samples.png' % "./results", normalize = True)
            fake = netG(noise)
            fake_grid = vutils.make_grid(fake.data[:50,:,:,:], nrow=5)
            vutils.save_image(fake_grid, '%s/fake_samples_epoch_%03d_iter_%03d.png' % ("./results", epoch, i),
                              normalize = True)
    D_loss_list.append(D_losses.avg)
    G_loss_list.append(G_losses.avg)
    D_losses.reset()
    G_losses.reset()

    #Plot loss after every epoch
    plot_loss(D_loss_list, G_loss_list, epoch + 1, num_epochs, "./results")

