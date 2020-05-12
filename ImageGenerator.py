import argparse
import os.path
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import imageio
import pickle as pkl

########################
# ImageGenerator Class #
########################
class ImageGenerator:
    def __init__(self):
        # Device to use
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Device to bring back to cpu
        self.device_cpu = torch.device("cpu")

        ##################
        # Input settings #
        ##################
        # data_directory expects: /folder/*.jpg
        self.data_directory = None
        # dataset folder name
        self.dataset_name = 'Dataset'

        ## Transforms for the input images
        self.transform = None
        ## ImageFolder data training
        self.dataset = None
        ## DataLoader with batched data for training
        self.loader = None
        ## The size of each batch = the number of images in a batch
        self.batch_size = 32 # DCGAN Paper = 128
        ## Square size of the image data (x, y)
        self.img_size = 32

        ##############
        # GAN Models #
        ##############
        ## Discriminator network
        self.D = None
        ## Generator network
        self.G = None
        # Criterion: binary cross entropy with logits loss
        self.criterion = nn.BCEWithLogitsLoss()

        ######################
        # Models hyperparams #
        ######################
        self.d_conv_dim = 128   # DCGAN Paper = 128
        self.g_conv_dim = 128   # DCGAN Paper = 128
        self.z_size = 100       # DCGAN Paper = 100

        #####################
        # Optimizers (Adam) #
        #####################
        self.opti_lr = 0.0002   # DCGAN Paper = 0.0002
        self.opti_beta1 = 0.5   # Adam default value = 0.9 (DCGAN Paper = 0.5)
        self.opti_beta2 = 0.999 # Adam default value = 0.999
        self.d_optimizer = None
        self.g_optimizer = None

        ############
        # Training #
        ############
        ## number of epochs to train for    
        self.n_epochs = 100
        ## when to print and record the models' losses 
        self.print_every = 50
        ## Record the losses for future display
        self.losses = []

        ####################
        # Outputs settings #
        ####################
        self.dir_output = 'output'
        self.path_output = None     # path_output will be: dir_output/datset_name

        self.dir_output_train = 'train'
        self.path_output_train = None
        self.train_img_saved = 0

        self.save_generated_dir = 'generated'
        self.path_output_generated = None
        self.generated_img_saved = 0                                      


    def use_gpu(self, gpu):
        """If gpu is true request the use of GPU if available"""
        if gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")


    def load_data(self, data_directory, num_workers=0):      
        """Prepare directories and the loader"""
        # Expand user path
        self.data_directory = os.path.expanduser(data_directory)

        # dataset_name for future use is the parent folder name
        self.dataset_name = os.path.basename(os.path.abspath(self.data_directory))

        # create the required folder for the output of the training
        self.create_train_path()

        # resize and normalize the images
        self.transform = transforms.Compose([transforms.Resize((self.img_size, self.img_size)),
                                             transforms.ToTensor()])
        # dataset using ImageFolder
        self.dataset = datasets.ImageFolder(self.data_directory, self.transform)

        # create DataLoader
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)


    def weights_init_normal(self, m, mean=0.0, std=0.02):
        """Applies initial weights to certain layers in a model.
        The weights are taken from a normal distribution 
        DCGAN paper recommends mean = 0, std dev = 0.02.
        :param m: A module or layer in a network
        :param mean: normal distribution mean
        :param std: normal distribution standard deviation"""
        
        # classname are something like:`Conv`, `BatchNorm2d`, `Linear`, etc.
        classname = m.__class__.__name__
        #print(classname)

        # Apply initial weights to convolutional and linear layers
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            m.weight.data.normal_(mean, std)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)


    def build_network(self):
        """Create the discriminator and generator with default values and initialize model weights"""
        self.D = Discriminator(self.d_conv_dim, self.img_size)
        self.G = Generator(z_size=self.z_size, conv_dim=self.g_conv_dim, img_size=self.img_size)

        # initialize model weights
        self.D.apply(self.weights_init_normal)
        self.G.apply(self.weights_init_normal)
        
        print(self.D, '\n', self.G, '\n')


    def create_optimizer(self):
        """Create the optimizers with default values"""
        self.d_optimizer = optim.Adam(self.D.parameters(), self.opti_lr, [self.opti_beta1, self.opti_beta2])
        self.g_optimizer = optim.Adam(self.G.parameters(), self.opti_lr, [self.opti_beta1, self.opti_beta2])


    def real_loss(self, D_out):
        '''Calculates how close discriminator outputs are to being real.
        param, D_out: discriminator logits
        return: real loss'''
        
        batch_size = D_out.size(0)
        # real labels = 1
        labels = torch.ones(batch_size).to(self.device)

        # calculate loss
        loss = self.criterion(D_out.squeeze(), labels)
        return loss


    def fake_loss(self, D_out):
        '''Calculates how close discriminator outputs are to being fake.
        param, D_out: discriminator logits
        return: fake loss'''
        
        batch_size = D_out.size(0)     
        # fake labels = 0 
        labels = torch.zeros(batch_size).to(self.device) 

        # calculate loss
        loss = self.criterion(D_out.squeeze(), labels)  
        return loss


    def evaluate(self, z):
        """Forward pass to the generator network without gradiets
        param, z: batch of input vectors
        return: generated fake images"""
        self.G.eval()
        with torch.no_grad():
            out = self.G(z)
            self.save_samples(out)
        self.G.train()
        return out


    def train(self):
        """Trains adversarial networks for n_epochs
        Prints statistics every print_every batches
        Saves the generated images during evaluation in the output folder"""

        # keep track of loss
        self.losses = []
        
        # Generte fixed data for sampling. 
        # Fixed throughout training to inspect the model's performance
        sample_size=16
        fixed_z = np.random.uniform(-1, 1, size=(sample_size, self.z_size))
        fixed_z = torch.from_numpy(fixed_z).float().to(self.device)

        start_train = time.time()
        last_time = time.time()
        try:
            # move models to GPU if available
            self.D, self.G = self.D.to(self.device), self.G.to(self.device)

            for epoch in range(self.n_epochs):
                for batch_i, (real_images, _) in enumerate(self.loader):
                    
                    batch_size = real_images.size(0)
                    real_images = self.scale(real_images)

                    # ========================================================
                    #    1. Train the discriminator on real and fake images
                    # ========================================================
                    self.d_optimizer.zero_grad()
                    
                    # 1.1. Train with real images
                    # Compute the discriminator losses on real images 
                    real_images = real_images.to(self.device)
                    D_real = self.D(real_images)
                    d_real_loss = self.real_loss(D_real)

                    # 1.2. Train with fake images
                    # 1.2.1. Generate fake images
                    z = np.random.uniform(-1, 1, size=(batch_size, self.z_size))
                    z = torch.from_numpy(z).float().to(self.device)

                    fake_images = self.G(z) # TODO: with torch.no_grad() <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ??

                    # 1.2.2. Compute the discriminator losses on fake images            
                    D_fake = self.D(fake_images)
                    d_fake_loss = self.fake_loss(D_fake)

                    # 1.2.3. add up loss and perform backprop
                    d_loss = d_real_loss + d_fake_loss
                    
                    # perform backprop
                    d_loss.backward()
                    self.d_optimizer.step()

                    # ========================================================
                    #    2. Train the generator with an adversarial loss
                    # ========================================================
                    self.g_optimizer.zero_grad()
                    
                    # 2.1. Train with fake images and flipped labels
                    # 2.1.1. Generate fake images
                    z = np.random.uniform(-1, 1, size=(batch_size, self.z_size))
                    z = torch.from_numpy(z).float().to(self.device)
                    fake_images = self.G(z)

                    # 2.1.2. Compute the discriminator losses on fake images using flipped labels!
                    D_fake = self.D(fake_images)
                    g_loss = self.real_loss(D_fake) # use real loss to flip labels

                    # perform backprop
                    g_loss.backward()
                    self.g_optimizer.step()


                    # ========================================================
                    #    3. Print statistics
                    # ========================================================

                    if batch_i % self.print_every == 0:
                        # append discriminator loss and generator loss
                        self.losses.append((d_loss.item(), g_loss.item()))
                        
                        t = time.time()-last_time
                        last_time = time.time()
                        b_left = len(self.loader)-batch_i+len(self.loader)*(self.n_epochs-epoch)
                        t_left = int(float(t)/self.print_every*b_left)
                        # print discriminator and generator loss                  
                        print('Epoch [{:5d}/{:5d}] batch[{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f} | t:{}m{}s/{}m{}s'.format(
                                epoch+1, self.n_epochs, 
                                batch_i, len(self.loader),
                                d_loss.item(), g_loss.item(),
                                int(t//60), int(t%60), t_left//60, t_left%60))
                                                
                # ========================================================
                #    4. After each epoch generate and save fake images
                # ========================================================
                
                #self.view_samples(self.evaluate(fixed_z))
                o = self.evaluate(fixed_z)

        except KeyboardInterrupt:
            print("Exiting training: KeyboardInterrupt")
            
            self.view_samples(self.evaluate(fixed_z))

        except Exception as e:
            # In case we get CUDA out of memory
            print("Exiting training: {}".format(str(e)))

        t = int(time.time() - start_train)
        print('Training time: {}m{}s'.format(t//60, t%60))
               
        fig, ax = plt.subplots()
        losses_np = np.array(self.losses)
        plt.plot(losses_np.T[0], label='Discriminator', alpha=0.5)
        plt.plot(losses_np.T[1], label='Generator', alpha=0.5)
        plt.title("Training Losses")
        plt.legend()
        plt.show()
            

    def train_pipeline(self, data_dir):
        """Runs through all the steps of the training
        param data_dir: input dataset folder. Expects /folder/*.jpg"""
        self.load_data(data_dir)
        self.build_network()
        self.create_optimizer()
        self.train()
        self.save_checkpoint()


    def generate_images(self, generator_path, nimages):
        """Generates images from a pre-trained generator
        param generator_path: path to the pre-trained generator model.
        param nimages: number of images to generate in the default output folder"""

        self.create_generated_path()
        self.load_generator(generator_path)

        nbatches = nimages//self.batch_size
        remaining = nimages%self.batch_size
        size_per_batch=[self.batch_size]*nbatches + [remaining]

        self.G.eval()
        for bs in size_per_batch:
            z = torch.from_numpy(np.random.uniform(-1, 1, size=(bs, self.G.z_size))).float()
            z = z.to(self.device)
            with torch.no_grad():   
                out = self.G(z)
            self.save_generated_images(out)
        

    def save_checkpoint(self):
        """Saves a checkpoint.
        Not fully implemented. 
        This will save a checkpoint to resume training or generate outputs"""

        fp = os.path.join(self.path_output, self.dataset_name+'_D.pkl')
        with open(fp, 'wb') as f:
            pkl.dump(self.D, f)   

        fp = os.path.join(self.path_output, self.dataset_name+'_G.pkl')
        with open(fp, 'wb') as f:
            pkl.dump(self.G, f)


    def load_generator(self, generator_path):
        """Loads a pre-trained generator network"""

        with open(generator_path, 'rb') as f:
            self.G = pkl.load(f)
            self.G = self.G.to(self.device)


    #############################################################
    #################         Helpers            ################
    #############################################################

    def create_output_path(self):
        if not os.path.isdir(self.dir_output):
            os.mkdir(self.dir_output)
        self.path_output = os.path.join(self.dir_output, self.dataset_name)
        if not os.path.isdir(self.path_output):
            os.mkdir(self.path_output)

    def create_train_path(self):
        self.create_output_path()
        self.path_output_train = os.path.join(self.path_output, self.dir_output_train)
        if not os.path.isdir(self.path_output_train):
            os.mkdir(self.path_output_train)
        
    def next_output_train_path(self):       
        self.train_img_saved += 1
        return os.path.join(self.path_output_train, 'sample-{:06d}.png'.format(self.train_img_saved))

    def create_generated_path(self):
        self.create_output_path()
        self.path_output = os.path.join(self.dir_output, self.dataset_name)
        self.path_output_generated = os.path.join(self.path_output, self.save_generated_dir)
        if not os.path.isdir(self.path_output_generated):
            os.mkdir(self.path_output_generated)
        
    def next_output_generated_path(self):       
        self.generated_img_saved += 1
        return os.path.join(self.path_output_generated, '{}-{:06d}.png'.format(self.dataset_name, self.generated_img_saved))


    def scale(self, x, feature_range=(-1, 1)):
        ''' 
        Takes in an image x and returns it scaled to feature_range
        This function assumes that the input x is already scaled from 0-1.
        '''       
        vmin, vmax = feature_range
        x = x * (vmax - vmin) + vmin
        return x

    '''
    def imshow(self, img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    '''
    def to_data(self, x):
        x = x.to(self.device_cpu)
        x = x.data.numpy()   
        x = ((x +1)*255 / (2)).astype(np.uint8) # rescale to 0-255
        return x
   
    def view_samples(self, samples):
        fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)
        for ax, img in zip(axes.flatten(), samples):
            img = self.to_data(img)
            img = np.transpose(img, (1, 2, 0))
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            im = ax.imshow(img.reshape((self.img_size,self.img_size,3)))
        plt.show()

    def merge_images(self, sources):
        batch_size, _, h, w = sources.shape
        row = int(np.sqrt(batch_size))
        merged = np.zeros([3, row*h, row*w])
        for idx, s in enumerate(sources):
            i = idx // row
            j = idx % row
            merged[:, i*h:(i+1)*h, j*h:(j+1)*h] = s
        return merged

    def save_samples(self, samples, iteration=None):
        samples = self.to_data(samples)
        merged = self.merge_images(samples)
        merged = np.transpose(merged, (1, 2, 0))
        merged = merged.astype(np.uint8)
        imageio.imwrite(self.next_output_train_path(), merged)

    def save_generated_images(self, images):
        images = self.to_data(images)
        for img in images:    
            img = np.transpose(img, (1, 2, 0))
            img = img.astype(np.uint8)
            p = self.next_output_generated_path()
            print("Saving:",p)
            imageio.imwrite(p, img)

###############################
# END OF ImageGenerator Class #
###############################

        
###############################
# Discriminator Network Class #
###############################
class Discriminator(nn.Module):
    def __init__(self, conv_dim, img_size):
        """
        Initialize the Discriminator Module
        :param conv_dim: The depth of the first convolutional layer
        :param img_size: Size of the input images
        """
        super(Discriminator, self).__init__()
        
        negative_slope = 0.2 # DCGAN Paper suggestion
        layers = []
        
        # first layer has *no* batch_norm
        layers += conv(3, conv_dim, 4, batch_norm=False)
        layers.append(nn.LeakyReLU(negative_slope=negative_slope))
        
        # Required # layers to bring the image down in size
        self.nlayers = int(np.log2(img_size))-2
        for i in range(0, self.nlayers):
            layers += conv(conv_dim*(2**i), conv_dim*(2**(i+1)), 4)
            layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            last_i = i
        
        # Classification layer has *no* batch_norm
        layers += conv(conv_dim*(2**(last_i+1)), 1, 4, stride=1, batch_norm=False)
        #print(layers)
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: Discriminator logits; the output of the neural network
        """
        # define feedforward behavior
        batch_size = x.shape[0]       
        out = self.layers(x)       
        out = out.view(batch_size, -1)
        return out

###########################
# Gemerator Network Class #
###########################
class Generator(nn.Module): 
    def __init__(self, z_size, conv_dim, img_size):
        """
        Initialize the Generator Module
        :param z_size: The length of the input latent vector, z
        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
        :param img_size: Size of the input images
        """
        super(Generator, self).__init__()

        # complete init function
        self.conv_dim = conv_dim
        self.z_size = z_size
        self.nlayers = int(np.log2(img_size))-2 ## CAREFUL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        # first, fully-connected layer
        self.fc = nn.Linear(z_size, conv_dim*(2**(self.nlayers-1))*4*4) #out to reshape to: conv_dim*4, 4, 4

        layers = []
        # Required # trans_conv layers to bring the image up in size
        for i in range(self.nlayers-1, 0, -1):
            layers += t_conv(conv_dim*(2**i), conv_dim*(2**(i-1)), 4)
            layers.append(nn.ReLU())
        
        layers += t_conv(conv_dim, 3, 4, batch_norm=False)
        layers.append(nn.Tanh())
        
        self.layers = nn.Sequential(*layers)
                                 
    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: A (img_size, img_size, 3) Tensor image as output
        """
        # define feedforward behavior
        
        out = self.fc(x)
        out = out.view(-1, self.conv_dim*(2**(self.nlayers-1)), 4, 4) # (batch_size, depth, 4, 4)
        out = self.layers(out)
        
        return out

#############################
# Helpers to build networks #
#############################

def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization. """

    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return layers

def t_conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transpose convolutional layer, with optional batch normalization."""

    layers = []
    # append transpose conv layer
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    # optional batch norm layer
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return layers


####################################################################################################################

def parse_input():
    parser = argparse.ArgumentParser(
        description='Train and generate images using GANs'
    )
    
    parser.add_argument('--data_dir', action='store',
                        dest='data_dir',
                        help='Path to dataset to train')

    parser.add_argument('--epochs', action='store',
                        dest='epochs', type=int,
                        help='Number of epochs for training')

    parser.add_argument('--print_every', action='store',
                        dest='print_every', type=int,
                        help='Print every x steps in the training')

    parser.add_argument('--generator_path', action='store',
                        dest='generator_path',
                        help='Path to a pre-trained generator')

    parser.add_argument('--nimages', action='store',
                        dest='nimages', type=int, default=10,
                        help='Num images to generate')

    parser.add_argument('--gpu', action='store_true',
                        dest='gpu', default=False,
                        help='Train using CUDA:0')

    results = parser.parse_args()
    return results

if __name__ == "__main__":
    print("Hola!")
    # Get cmd args
    args = parse_input()
    
    # Instanciate ImageGenerator Class
    ig = ImageGenerator()

    # Request GPU if available
    ig.use_gpu(args.gpu)

    if args.data_dir is not None:
        ig.train_pipeline(args.data_dir)

    if args.generator_path is not None:
        ig.generate_images(args.generator_path, args.nimages)

    if args.data_dir is None and args.generator_path is None:
        print("[ERR] Provide data_dir to train or generator_path to generate images")