import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import Lambda
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import sys

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
#import medmnist

from itertools import product
import multiprocessing as mp

from dataset import load_mnist_dataset, load_MVTEC
from metrics import pixel_accuracy, IOU, dice_coefficient, AUPRO
from ignite.contrib import metrics


class Classical_autoencoder(nn.Module):
    
    def __init__(self, image_size, params):
        super().__init__()

        self.device = "cpu"
        self.patch_size = params['patch_size']
        self.patch_stride = params['patch_stride']
        self.patch_padding = params['patch_padding']
        self.kernel_size = params['kerlen_size']
        self.kernel_stride = params['kernel_stride']
        self.kernel_padding = params['kernel_padding']
        self.out_channels = params['out_channels']
        
        self.conv_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.kernel_stride, padding=self.kernel_padding),
            torch.nn.ReLU(True)
        )
        self.maxpooling = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(2, 2),
            torch.nn.ReLU(True),
            #torch.nn.Linear(1, 2),
            #torch.nn.ReLU(True),
            #torch.nn.Linear(2, 1),
            #torch.nn.ReLU(True),
            nn.Unflatten(dim=1, unflattened_size=(2, 1, 1))
        )
        self.unpooling = nn.MaxUnpool2d(kernel_size=self.kernel_size, stride=2, padding=0)
        self.conv_decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=2, stride=2, padding=0),
            torch.nn.Sigmoid(),
            torch.nn.Flatten()
        )
        self.linear_bottleneck_linear = torch.nn.Sequential(
            torch.nn.Flatten(),
            #torch.nn.Linear(self.kernel_size*self.kernel_size, self.kernel_size*self.kernel_size),
            #torch.nn.ReLU(True),
            torch.nn.Linear(self.kernel_size*self.kernel_size, 4),
            torch.nn.ReLU(True),
            torch.nn.Linear(4, 4),
            torch.nn.ReLU(True),
            torch.nn.Linear(4, self.kernel_size*self.kernel_size),
            torch.nn.ReLU(True),
            #torch.nn.Linear(self.kernel_size*self.kernel_size, self.kernel_size*self.kernel_size),
            #torch.nn.Sigmoid(),
            torch.nn.Flatten()
        )
        
        
    def forward(self, img):
        bs, ch, h, w = img.size()

        patch_no = ((h - self.patch_size) // self.patch_stride + 1) * ((h - self.patch_size) // self.patch_stride + 1)    
        patch_padding = self.patch_padding
        if patch_padding > 0:
            img = nn.ZeroPad2d(patch_padding)(img)
            h = h + patch_padding*2
            w = w + patch_padding*2
        out = torch.zeros((bs, 1, h, w), device=self.device) 
        count_image = torch.zeros((bs, 1, w, h), device=self.device)
            #print(b,bs)
        for b in range(bs):
            for j in range(0, h - self.patch_size + 1, self.patch_stride):
                for k in range(0, w - self.patch_size + 1, self.patch_stride):
                    patch_id = (j // self.patch_stride) * (h // self.patch_stride) + (k // self.patch_stride)
                    c = torch.tensor([patch_id/patch_no], device=self.device)
                    inputs = [img[b, 0, j + i, k + l] for i in range(self.patch_size) for l in range(self.patch_size)]
                    inputs = torch.tensor(inputs, device=self.device)
                    inputs = torch.reshape(inputs, (1,1,self.patch_size,self.patch_size))
                    #inputs = self.linear_bottleneck_linear(inputs)
                    #inputs = self.linear_encoder(inputs)
                    #inputs = self.bottleneck2(torch.reshape(torch.cat((torch.reshape(inputs, (3,)), c)),(1,4)))
                    #inputs = self.bottleneck(inputs)
                    #inputs = self.linear_decoder(inputs)
                    
                    
                    inputs = self.conv_encoder(inputs)
                    inputs, idx = self.maxpooling(inputs)
                    #print(inputs.shape)
                    inputs = self.bottleneck(inputs)
                    #inputs = self.bottleneck(torch.reshape(torch.cat((torch.reshape(inputs, (1,)), c)),(1,2)))
                    inputs = self.unpooling(inputs, idx)
                    inputs = self.conv_decoder(inputs)
                    
                    for i in range(self.patch_size):
                        for l in range(self.patch_size):
                            out[b, 0, j + i, k + l] += inputs[0][i * self.patch_size + l]
                    count_image[b, 0, j:j+self.patch_size, k:k+self.patch_size] += 1
        out /= count_image

        if patch_padding > 0:
            out = out[:, :, patch_padding:-patch_padding, patch_padding:-patch_padding]           
        return out
####################################

def run(chunk, seeds, lr, num_epochs, image_size, training_size, noise, thresholds):
    for i, conf in enumerate(chunk):
        metrics = np.zeros((len(thresholds), len(seeds), 5)) # (thresholds, seeds, metrics)
        train_losses = [[]]*len(seeds)
        val_losses = [[]]*len(seeds)
        for s, seed in enumerate(seeds):
            conf['seed'] = seed
            dataset_name = conf['dataset'] 

            if dataset_name == 'carpet':
                n_normal = 28
            elif dataset_name == 'wood':
                n_normal = 19
            else:
                n_normal = 32

            model_name = "Classical_"+dataset_name+"_KS"+str(conf['patch_size'])+"_ST"+str(conf['patch_stride'])+"_BD"+str(conf['bottleneck_dim'])+"_s"+str(seed)
            print("train: " + model_name)
            
            train_loader, valid_loader, test_loader, mask_loader = load_MVTEC(dataset_name, training_size, image_size)
            loss_fn = torch.nn.MSELoss()

            device = "cpu"

            autoencoder = Classical_autoencoder(image_size, conf)
            
            params_to_optimize = [
                {'params': autoencoder.parameters()},
            ]
            optim = torch.optim.Adam(params_to_optimize, lr=lr)
        
            autoencoder.to(device)

            #diz_loss = {'train_loss':[],'val_loss':[]}
            train_loss_seed = []
            val_loss_seed = []
            for epoch in range(num_epochs):
                train_loss = train_epoch(autoencoder, device, train_loader, loss_fn, optim, noise=noise)
                val_loss = test_epoch(autoencoder, device, valid_loader, loss_fn, noise=noise)
                print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs, train_loss, val_loss))
                train_loss_seed.append(train_loss)
                val_loss_seed.append(val_loss)
                plot_ae_outputs(device, epoch, autoencoder, test_loader, n=10, model_name=model_name)
                if epoch == int(num_epochs/2)-1:
                    torch.save(autoencoder.state_dict(), './models/10epochs_' + model_name + '.pt')
            print("test: " + model_name)

            train_losses[s] = train_loss_seed
            val_losses[s] = val_loss_seed

            pd.DataFrame({'train_loss': train_losses[s], 'val_loss': val_losses[s]}).to_csv("./results/" + model_name + "_losses.csv")
            torch.save(autoencoder.state_dict(), './models/20epochs_' + model_name + '.pt')
            
            plot_loss_curve(train_losses[s], val_losses[s], model_name)
            
            autoencoder = Classical_autoencoder(image_size, conf)
            autoencoder.load_state_dict(torch.load("./models/20epochs_" + model_name + ".pt"))
            autoencoder.to(device)
            autoencoder.eval()

            #test_encoder_with_reconstruction(autoencoder, model_name, test_loader, conf, image_size, device, n_normal)
            
            for t, threshold in enumerate(thresholds):
                acc, dice, iou, aupro, auroc = test_with_mask(autoencoder, model_name, test_loader, mask_loader, conf, image_size, device, threshold, n_normal)
                metrics[t][s][0] = acc
                metrics[t][s][1] = dice
                metrics[t][s][2] = iou
                metrics[t][s][3] = aupro
                metrics[t][s][4] = auroc
                
        mean_metrics = np.mean(metrics, axis=1)
        std_metrics = np.std(metrics, axis=1)
        filename = "./results/Classical_" + dataset_name+"_KS"+str(conf['patch_size'])+"_ST"+str(conf['patch_stride'])+"_BD"+str(conf['bottleneck_dim'])+"_metrics.csv"
        df_means = pd.DataFrame(mean_metrics)
        df_means.columns = ["mean_acc","mean_dice","mean_iou","mean_aupro","mean_auroc"]
        df_std = pd.DataFrame(std_metrics)
        df_std.columns = ["std_acc","std_dice","std_iou","std_aupro","std_auroc"]
        df = pd.concat([df_means, df_std], axis=1)
        df['threshold'] = thresholds
        df = df[["threshold", "mean_acc", "std_acc", "mean_dice", "std_dice", "mean_iou", "std_iou", "mean_aupro", "std_aupro", "mean_auroc", "std_auroc"]]
        df.to_csv(filename)
        plot_loss_curve_avg(train_losses, val_losses, dataset_name+"_KS"+str(conf['patch_size'])+"_ST"+str(conf['patch_stride'])+"_BD"+str(conf['bottleneck_dim']))

def par_runs(params, seeds, lr, num_epochs, image_size, training_size, noise, thresholds, n_processes=2):
    
    keys, values = zip(*params.items())
    params_list = [dict(zip(keys, v)) for v in product(*values)]

    list_chunks = np.array_split(params_list, n_processes)
    
    if len(params_list) == 1:
        processes = [None]
        run(list_chunks[0], seeds, lr, num_epochs, image_size, training_size, noise, thresholds)    
    else:
        processes = [mp.Process(target=run, args=(chunk, seeds, lr, num_epochs, image_size, training_size, noise, thresholds))  for i, chunk in enumerate(list_chunks)]

        for p in processes:
            p.start()
        for p in processes:
            p.join()
            print("process ", p, " terminated")
    

def train_epoch(autoencoder, device, dataloader, loss_fn, optimizer, noise):
    # set train moder for both the encoder and the decoder
    autoencoder.train()
    train_loss = []
    # iterate the dataloader
    for image_batch, _ in dataloader: 
        if noise:
            corrupted_image_batch = image_batch.clone()
            noise = torch.randn(corrupted_image_batch.shape) * 0.15
            corrupted_image_batch = corrupted_image_batch + noise
            image_batch = corrupted_image_batch.to(device)
    
        image_batch = image_batch.to(device)
        # decode data
        decoded_data = autoencoder(image_batch)
        # normalize data to the range (0,1)
        #decoded_data = (decoded_data - torch.min(decoded_data)) / (torch.max(decoded_data) - torch.min(decoded_data))
        # evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print batch loss
        #print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())
    
    return np.mean(train_loss)

def test_epoch(autoencoder, device, dataloader, loss_fn, noise):
    # set evaluation mode for encoder and decoder
    autoencoder.eval()
    with torch.no_grad(): # no need to track the gradients
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            if noise:
                corrupted_image_batch = image_batch.clone()
                noise = torch.randn(corrupted_image_batch.shape) * 0.15
                corrupted_image_batch = corrupted_image_batch + noise
                corrupted_image_batch = corrupted_image_batch.to(device)
            
            # move tensor to proper device
            image_batch = image_batch.to(device)
            # decode data
            if noise:
                decoded_data = autoencoder(corrupted_image_batch)
            else:
                decoded_data = autoencoder(image_batch)
            # normalize data to the range (0,1)
            #decoded_data = (decoded_data - torch.min(decoded_data)) / (torch.max(decoded_data) - torch.min(decoded_data))
            # append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # create a single tensor with all the values in the list
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data

def plot_ae_outputs(device, epoch, autoencoder, test_loader, n=10, model_name="AE"):
    plt.figure(figsize=(16,4.5))
    targets = []
    for i, (_, target) in enumerate(test_loader.dataset):
        targets.append(target)
    targets = np.array(targets)

    t_idx = {}
    for i in range(10):
        if i in targets:
            t_idx[i] = np.where(targets==i)[0][0]


    for i in range(n):
        if i in targets:
            ax = plt.subplot(2, n, i+1)
            img = test_loader.dataset[t_idx[i]][0].unsqueeze(0).to(device)
            autoencoder
            with torch.no_grad():
                rec_img = autoencoder(img).to(device)
                #rec_img  = decoder(encoder(img))
            plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)  
            if i == n//2:
                ax.set_title('Original images')
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)  
            if i == n//2:
                ax.set_title('Reconstructed images')
    #plt.show()
    plt.savefig("./output_images/" + model_name + "_epoch_" + str(epoch) + ".png")  
    plt.close() 

def plot_loss_curve(train_loss, val_loss, model_name):
    # Plot losses
    plt.figure(figsize=(10,8))
    plt.plot(train_loss, label='Train')
    plt.plot(val_loss, label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    #plt.grid()
    plt.legend()
    #plt.title('loss')
    #plt.show()
    plt.savefig("./plots/" + model_name + "_loss_curves.png") 
    plt.close()

def plot_loss_curve_avg(train_losses, val_losses, model_name):
    # mean and std
    mean_train = np.mean(np.array(train_losses), axis=0)
    std_train = np.std(np.array(train_losses), axis=0)
    mean_val = np.mean(np.array(val_losses), axis=0)
    std_val = np.std(np.array(val_losses), axis=0)

    # Plot losses
    plt.figure(figsize=(10,8))
    x = np.arange(20)
    line_1, = plt.plot(x, mean_train, 'b-', label='Train')
    fill_1 = plt.fill_between(x, mean_train - std_train, mean_train + std_train, color='b', alpha=0.2)
    line_2, = plt.plot(x, mean_val, 'r-', label='Valid')
    fill_2 = plt.fill_between(x, mean_val - std_val, mean_val + std_val, color='r', alpha=0.2)

    plt.legend([(line_1, fill_1), (line_2, fill_2)], ['Training', 'Validation'])
    
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    #plt.grid()
    #plt.title('loss')
    #plt.show()
    plt.savefig("./plots/" + model_name + "_loss_curves.png") 
    plt.close()

def test_with_mask(autoencoder, model_name, test_loader, mask_loader, params, image_size, device='cpu', threshold=0.025, n_normal=28):

    autoencoder = Classical_autoencoder(image_size, params)
    autoencoder.load_state_dict(torch.load(model_name + ".pt"))
    autoencoder.to(device)

    # compute accuracy
    accuracies = []
    dice_scores = []
    iou_scores = []
    aupro = AUPRO()
    auroc = metrics.ROC_AUC()
    for i in range(len(test_loader.dataset)):
        img = test_loader.dataset[i][0].unsqueeze(0).to(device)
        mask = np.array(mask_loader.dataset[i][0].unsqueeze(0).to(device)[0,0,:,:]) #####
        mask = np.where(mask > 0.5, 1, 0)
        with torch.no_grad():
            rec_img = autoencoder(img).to(device)
        diff = (rec_img.cpu().squeeze().numpy() - img.cpu().squeeze().numpy())
        #print(np.min(map))
        #print(np.max(map))
        diff = np.where(map > threshold, 0, 1) #anomalies are white in the masks
        acc = pixel_accuracy(mask, diff)
        dice = dice_coefficient(mask, diff)
        iou = IOU(mask, diff)
        map = 1 - map
        aupro.update(torch.Tensor(map).unsqueeze(0), torch.Tensor(mask).unsqueeze(0))
        auroc.update((torch.Tensor(map).ravel(), torch.Tensor(mask).ravel()))
        #print("(" + str(i) + "/" + str(len(test_loader.dataset)) + ") Label: " + str(test_loader.dataset[i][1]) + " - Accuracy: " + str(acc) + " - Dice: " + str(dice) + " - IoU: " + str(iou))
        accuracies.append(acc)
        dice_scores.append(dice)
        iou_scores.append(iou)

    #print("Accuracy: " + str(np.mean(accuracies)))
    #print("Dice Score: " + str(np.mean(dice_scores)))
    #print("IoU: " + str(np.mean(iou_scores)))
    aupro = aupro.compute()
    auroc = auroc.compute()
    #print("AUPRO: " + str(aupro))
    '''
    plt.figure(figsize=(16, 4.5))
    for i in range(10):
        ax = plt.subplot(3, 10, i+1)
        img = test_loader.dataset[i+n_normal][0].unsqueeze(0).to(device)
        mask = np.array(mask_loader.dataset[i+n_normal][0].unsqueeze(0).to(device)[0,0,:,:])
        mask = np.where(mask > 0.5, 1, 0)
        with torch.no_grad():
            rec_img = autoencoder(img).to(device)
        diff = (rec_img.cpu().squeeze().numpy() - img.cpu().squeeze().numpy())
        diff = np.where(map > threshold, 0, 1) #anomalies are white in the masks
        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        #ax.set_title("label: " + str(test_loader.dataset[i][1]), y=0, pad=-15)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        ax = plt.subplot(3, 10, i + 1 + 10)
        plt.imshow(diff, cmap='gist_gray')  
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        #ax.set_title(round(loss_fn(img, ).item(), 4), y=0, pad=-15)
        ax = plt.subplot(3, 10, i + 1 + 20)
        plt.imshow(mask.squeeze().numpy(), cmap='gist_gray')  
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False) 
        
    plt.savefig("./output_images/" + model_name + "_t" + str(threshold) + "_masks.png")
    '''
    return np.mean(accuracies), np.mean(dice_scores), np.mean(iou_scores), aupro, auroc
 

if __name__ == "__main__":
    

    if len(sys.argv) != 2:
        print("ERROR: type '" + str(sys.argv[0]) + " n_processes' to execute the test")
        exit()
   
    try:
        processes = int(sys.argv[1])
    except ValueError:
        print("ERROR: specify a positive integer for the number of processes")
        exit()
    
    if processes < 0:
        print("ERROR: specify a positive integer for the number of processes")
        exit()

    # fixed parameters
    lr = 0.005
    num_epochs = 20
    #image_size = 64
    #training_size = 280
    image_size = 64
    training_size = 125
    noise = False

    # variable parameters
    params = {
        'dataset': ["wood"],
        'patch_size': [4],
        'patch_stride': [1],
        'patch_padding': [0],
        'kernel_size': [2],
        'kernel_stride': [1],
        'kernel_padding': [0],
        'out_channels': [1]
    }

    seeds = [123,456,789]
    thresholds = [0.999, 0.995, 0.990]
    par_runs(dict(params), seeds, lr, num_epochs, image_size, training_size, noise, thresholds, n_processes=processes)

   
