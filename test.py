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

import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers, StronglyEntanglingLayers
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
#import medmnist

from itertools import product
import multiprocessing as mp

from dataset import load_mnist_dataset, load_MVTEC
from metrics import pixel_accuracy, IOU, dice_coefficient, AUPRO

def MPS_BLOCK(weights, wires):
    qml.RY(weights[0], wires=wires[0])
    qml.RY(weights[1], wires=wires[1])
    qml.CNOT(wires=[wires[0],wires[1]])

class Encoder(nn.Module):
    
    def __init__(self, image_size, kernel_size, stride, padding, wires, bottleneck_dim, discarded_wires, mps_layers, n_block_wires, n_params_block, seed):
        super(Encoder, self).__init__()
 
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.wires = wires 
        self.mps_layers = mps_layers
        self.bottleneck_dim = bottleneck_dim
        self.discarded_wires = discarded_wires
        self.stride = stride
        self.padding = padding
        self.n_params_block = n_params_block
        self.n_block_wires = n_block_wires

        self.seed = seed
    
        n_blocks = qml.MPS.get_n_blocks(range(self.wires), n_block_wires)
      
        dev = qml.device("default.qubit", wires=range(self.wires + self.discarded_wires + 1))
        
        @qml.transforms.merge_amplitude_embedding
        def AE(inputs):
            if torch.count_nonzero(inputs[:]):
                qml.AmplitudeEmbedding(inputs[:], wires=range(self.discarded_wires, wires + self.discarded_wires), normalize=True, pad_with=0)
            #if torch.count_nonzero(inputs[-2:]):
                #qml.AmplitudeEmbedding(inputs[-2:], wires=range(wires + self.discarded_wires, wires + self.discarded_wires + 1), normalize=True, pad_with=0)

        
        @qml.qnode(dev, diff_method="backprop")
        def circuit(inputs, weights_mps):
                if torch.count_nonzero(inputs[:]):                   
                    qml.AmplitudeEmbedding(inputs[:], wires=range(self.discarded_wires, self.wires + self.discarded_wires), normalize=True, pad_with=0)
                
                #qml.RY(np.pi/4 * inputs[-2], wires=self.wires + self.discarded_wires)
                #qml.RY(np.pi/4 * inputs[-1], wires=self.wires + self.discarded_wires+1)

                qml.Hadamard(wires=self.wires + self.discarded_wires)
                '''
                qml.CNOT(wires=[self.wires + self.discarded_wires, self.wires + self.discarded_wires - 1])
                qml.CNOT(wires=[self.wires + self.discarded_wires + 1, self.wires + self.discarded_wires - 2])
                '''
                #Apply MPS
                for i in range(self.mps_layers):
                    qml.MPS(range(self.discarded_wires, self.wires + self.discarded_wires), n_block_wires, MPS_BLOCK, n_params_block, weights_mps[i])
                
                
                
                
                #qml.CNOT(wires=[self.discarded_wires - 1, self.wires + self.discarded_wires])
                #qml.CNOT(wires=[self.discarded_wires - 2, self.wires + self.discarded_wires + 1])
                

                for i in range(self.discarded_wires):    
                    qml.CSWAP(wires=[self.wires + self.discarded_wires, i, i + self.discarded_wires])
                #qml.CSWAP(wires=[self.wires*2 + 1 + self.discarded_wires, 2, 7])
            
                qml.Hadamard(wires=self.wires + self.discarded_wires)
 
                p = qml.probs(op=qml.PauliZ(wires=self.wires + self.discarded_wires))
                return p
                #return qml.expval(qml.PauliZ(wires=self.wires + self.discarded_wires + 2))

 
        
        weight_shapes = {'weights_mps': [self.mps_layers, n_blocks, n_params_block]}
        
        torch.manual_seed(self.seed)
        
        init_method = {
            'weights_mps': torch.nn.init.normal_
        }
 
        self.circuit = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes, init_method=init_method)
             
        '''
        fig, ax = qml.draw_mpl(circuit, expansion_strategy="device")(torch.rand(size=(self.wires,1)).flatten(), torch.rand(size=(self.mps_layers, n_blocks, n_params_block)))
        plt.savefig("circuit.png")
        plt.close()
        '''
        
    
    def forward(self, img):

        bs, ch, h, w = img.size()
 
        kernel_size = self.kernel_size
 
        padding = self.padding
        if padding > 0:
            img = nn.ZeroPad2d(padding)(img)
            h = h + padding*2
            w = w + padding*2

        patch_no = ((h - kernel_size) // self.stride + 1) * ((h - kernel_size) // self.stride + 1)    
        out = torch.zeros((bs, 1, patch_no))
        
        for b in range(bs):
            #print(b,bs)
            idx = 0
            for j in range(0, h - kernel_size + 1, self.stride):
                for k in range(0, w - kernel_size + 1, self.stride):
                    #get patch id
                    patch_id = (j // self.stride) * ((w - kernel_size) // self.stride + 1) + (k // self.stride)

                    a = torch.tensor([img[b, 0, j + i, k + l] for i in range(kernel_size) for l in range(kernel_size)])
                    #out[b, 0, idx] = self.circuit(torch.cat((a, x_patch, y_patch)))[1]
                    out[b, 0, idx] = self.circuit(a)[0]
                    idx = idx + 1
        return out
 
class Autoencoder(nn.Module):
    
    def __init__(self, image_size, params):
        super().__init__()


        self.image_size = image_size
        self.kernel_size = params['kernel_size']
        self.stride = params['stride']
        self.padding = self.kernel_size - self.stride
        self.padding = 0

        self.wires = int(np.ceil(np.log2(self.kernel_size*self.kernel_size)))
        self.bottleneck_dim = params['bottleneck_dim']
        self.discarded_wires = self.wires - self.bottleneck_dim

        self.mps_layers = params['mps_layers']
        self.n_block_wires = params['n_block_wires']
        self.n_params_block = params['n_params_block']

        self.seed = params['seed']
        
        self.autoencoder = torch.nn.Sequential(
            Encoder(
                image_size=self.image_size, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                padding=self.padding, 
                wires=self.wires, 
                bottleneck_dim=self.bottleneck_dim, 
                discarded_wires=self.discarded_wires,
                mps_layers=self.mps_layers, 
                n_block_wires=self.n_block_wires,
                n_params_block=self.n_params_block,
                seed=self.seed, 
                )
        )
        
    def forward(self, x):
        x = self.autoencoder(x)
        return x     

################### FOR RECONSTRUCTION #########################

class Encoder_Decoder():

    def __init__(self, weights, image_size, params):
        self.image_size = image_size
        self.weights = weights
        self.kernel_size = params['kernel_size']
        self.stride = params['stride']
        self.padding = self.kernel_size - self.stride
        self.padding = 0

        self.wires = int(np.ceil(np.log2(self.kernel_size*self.kernel_size)))
        self.bottleneck_dim = params['bottleneck_dim']
        self.discarded_wires = self.wires - self.bottleneck_dim

        self.mps_layers = params['mps_layers']
        self.n_block_wires = params['n_block_wires']
        self.n_params_block = params['n_params_block']

        self.seed = params['seed']

    def reconstruction(self, img):
        
        @qml.transforms.merge_amplitude_embedding
        def AE(inputs, discarded_wires, wires):
            qml.AmplitudeEmbedding(inputs[:], wires=range(discarded_wires, wires + discarded_wires), normalize=True, pad_with=0)
            qml.AmplitudeEmbedding(inputs[:], wires=range(wires + discarded_wires, wires*2 + discarded_wires), normalize=True, pad_with=0)

        dev = qml.device("default.qubit", wires=range(self.wires*2 + self.discarded_wires + 1)) 
        @qml.qnode(dev)
        def circuit_rec(inputs):

                if torch.count_nonzero(inputs[:]):                   
                    AE(inputs[:], self.discarded_wires, self.wires)
                
                qml.Hadamard(wires=self.wires*2 + self.discarded_wires)

                #Apply MPS
                for i in range(int(self.mps_layers)):
                    #qml.TTN(range(discarded_wires, wires + discarded_wires+2),n_block_wires, block2, n_params_block, weights_mps[i])
                    qml.MPS(range(self.discarded_wires, self.wires + self.discarded_wires), self.n_block_wires, MPS_BLOCK, self.n_params_block, self.weights[i])
                    
                #Swap wires in the range (0, self.discarded_wires) with wires in the range (self.discarded_wires, self.wires + self.discarded_wires) (only the wire (self.wires + self.discarded_wires -1) is not swapped)\
                for i in range(self.discarded_wires):
                    qml.SWAP(wires=[i, i + self.discarded_wires])
                

                #Apply adjoint MPS
                for i in range(int(self.mps_layers)):
                    #qml.adjoint(qml.TTN(range(discarded_wires, wires + discarded_wires+2),n_block_wires, block2, n_params_block, weights_mps[i]))
                    qml.adjoint(qml.MPS(range(self.discarded_wires, self.wires + self.discarded_wires), self.n_block_wires, MPS_BLOCK, self.n_params_block, self.weights[i]))

                for i in range(self.wires):
                    qml.CSWAP(wires=[self.wires*2 + self.discarded_wires, i + self.discarded_wires, i + self.discarded_wires + self.wires])
                
                qml.Hadamard(wires=self.wires*2 + self.discarded_wires)

                return qml.probs(op=qml.PauliZ(wires=self.wires*2 + self.discarded_wires))
            
        
        bs, ch, h, w = img.size()
    
        if self.padding > 0:
            img = nn.ZeroPad2d(self.padding)(img)
            h = h + self.padding*2
            w = w + self.padding*2

        patch_no = ((h - self.kernel_size) // self.stride + 1) * ((h - self.kernel_size) // self.stride + 1)    
        out = torch.zeros((bs, 1, patch_no))
        
        for b in range(bs):
            idx = 0
            for j in range(0, h - self.kernel_size + 1, self.stride):
                for k in range(0, w - self.kernel_size + 1, self.stride):
                    patch_id = (j // self.stride) * ((w - self.kernel_size) // self.stride + 1) + (k // self.stride)

                    a = torch.tensor([img[b, 0, j + i, k + l] for i in range(self.kernel_size) for l in range(self.kernel_size)])

                    #out[b, 0, idx] = circuit_rec(torch.cat((a, x_patch, y_patch)), weights, wires, discarded_wires, mps_layers, n_block_wires, n_params_block)
                    out[b, 0, idx] = circuit_rec(a)[0]
                    idx = idx + 1
                    '''
                    if idx == 40:
                        fig, ax = qml.draw_mpl(circuit_rec, expansion_strategy="device")(a)
                        plt.savefig("circuit_rec.png")
                        plt.close()
                    '''
                    
                
        return out


def test_encoder_with_reconstruction(autoencoder, model_name, test_loader, params, image_size, device='cpu', n_normal=28):  
    
    loss_fn = loss_function

    gen = list(autoencoder.autoencoder[0].circuit.parameters())[0]
    weights = []
    for w in gen:
        weights.append(w)

    full_autoencoder = Encoder_Decoder(weights, image_size, params)

    loss_anomalies = []
    loss_normal = []
    
 
    for i in range(len(test_loader.dataset)):
        img = test_loader.dataset[i][0].unsqueeze(0).to(device)
        with torch.no_grad():
            patches_scores = full_autoencoder.reconstruction(img)
        if test_loader.dataset[i][1] == 0:
            loss_normal.append(round(loss_fn(patches_scores).item(), 4))
        else:
            loss_anomalies.append(round(loss_fn(patches_scores).item(), 4))

    plt.plot(loss_normal, label="loss_normal")
    plt.plot(loss_anomalies, label="loss_anomalies")
    plt.legend()
    plt.savefig("./output_images/" + model_name + "_losses_rec.png")
    plt.close()
    
    plt.figure(figsize=(16,4.5))
    for i in range(10):
        ax = plt.subplot(2, 10, i+1)
        img = test_loader.dataset[i+n_normal][0].unsqueeze(0).to(device)
        with torch.no_grad():
            #patches_scores = autoencoder(img).to(device)
            patches_scores = full_autoencoder.reconstruction(img)
            #rec_img  = decoder(encoder(img))
        map = build_map(patches_scores, image_size, autoencoder.kernel_size, autoencoder.stride, autoencoder.padding)
        #print(patches_scores)
        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.set_title("label: " + str(test_loader.dataset[i+n_normal][1]), y=0, pad=-15)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        ax = plt.subplot(2, 10, i + 1 + 10)
        plt.imshow(map, cmap='hot')
        #print(map)
        #print(test_loader.dataset[i][1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        ax.set_title(round(loss_fn(patches_scores).item(), 4), y=0, pad=-15)
    plt.savefig("./output_images/" + model_name + "_test_rec.png")
    plt.close()

####################################

def run(chunk, seeds, lr, num_epochs, image_size, training_size, noise):
    
    for i, conf in enumerate(chunk):
        metrics = np.zeros((2,3,4)) # (seeds, thresholds, metrics)
        train_losses = [[]]*len(seeds)
        val_losses = [[]]*len(seeds)
        for s, seed in enumerate(seeds):
            conf['seed'] = seed
            dataset_name = conf['dataset'] 

            if dataset_name == 'carpet':
                n_normal = 28
            else:
                n_normal = 19

            model_name = dataset_name+"_KS"+str(conf['kernel_size'])+"_ST"+str(conf['stride'])+"_BD"+str(conf['bottleneck_dim'])+"_s"+str(seed)
            print("train: " + model_name)
            
            train_loader, valid_loader, test_loader, mask_loader = load_MVTEC(dataset_name, training_size, image_size)
            loss_fn = loss_function

            device = "cpu"

            autoencoder = Autoencoder(image_size, conf)
            
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
                #print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs, train_loss, val_loss))
                train_loss_seed.append(train_loss)
                val_loss_seed.append(val_loss)
                
                #if (epoch == 0) or (epoch == num_epochs-1) or (epoch % 5 == 0):
                #plot_ae_outputs_with_reconstruction(autoencoder, model_name, test_loader, conf, image_size, epoch, device, n=10) 
                if epoch == 9:
                    torch.save(autoencoder.state_dict(), './models/10epochs_' + model_name + '.pt')
            print("test: " + model_name)

            train_losses[s] = train_loss_seed
            val_losses[s] = val_loss_seed

            pd.DataFrame({'train_loss': train_losses[s], 'val_loss': val_losses[s]}).to_csv("./results/" + model_name + "_losses.csv")
            torch.save(autoencoder.state_dict(), './models/20epochs_' + model_name + '.pt')
            
            plot_loss_curve(train_losses[s], val_losses[s], model_name)
            
            autoencoder = Autoencoder(image_size, conf)
            autoencoder.load_state_dict(torch.load("./models/20epochs_" + model_name + ".pt"))
            autoencoder.to(device)
            autoencoder.eval()

            #test_encoder_with_reconstruction(autoencoder, model_name, test_loader, conf, image_size, device, n_normal)
            
            thresholds = [0.99]
            for t, threshold in enumerate(thresholds):
                #print("Threshold: " + str(threshold))
                acc, dice, iou, auroc = test_with_mask(autoencoder, model_name, test_loader, mask_loader, conf, image_size, device, threshold, n_normal)
                metrics[t][s][0] = acc
                metrics[t][s][1] = dice
                metrics[t][s][2] = iou
                metrics[t][s][3] = auroc
            #print(accuracies)
            #print(dice_scores)
            #print(iou_scores)
            #print(aurocs)
                
        mean_metrics = np.mean(metrics, axis=1)
        std_metrics = np.std(metrics, axis=1)
        filename = "./results/" + dataset_name+"_KS"+str(conf['kernel_size'])+"_ST"+str(conf['stride'])+"_BD"+str(conf['bottleneck_dim'])+"_metrics.csv"
        df = pd.concat([pd.DataFrame(mean_metrics), pd.DataFrame(std_metrics)])
        df.columns = ["acc","dice","iou","aupro"]
        df.to_csv(filename)
        plot_loss_curve_avg(train_losses, val_losses, dataset_name+"_KS"+str(conf['kernel_size'])+"_ST"+str(conf['stride'])+"_BD"+str(conf['bottleneck_dim']))

def par_runs(params, seeds, lr, num_epochs, image_size, training_size, noise, n_processes=2):
    
    keys, values = zip(*params.items())
    params_list = [dict(zip(keys, v)) for v in product(*values)]

    list_chunks = np.array_split(params_list, n_processes)
    
    if len(params_list) == 1:
        processes = [None]
        run(list_chunks[0], seeds, lr, num_epochs, image_size, training_size, noise)    
    else:
        processes = [mp.Process(target=run, args=(chunk, seeds, lr, num_epochs, image_size, training_size, noise))  for i, chunk in enumerate(list_chunks)]

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
        loss = loss_fn(decoded_data)
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
        val_loss = loss_fn(conc_out)
    return val_loss.data

def build_map(patches_scores, image_size, patch_size, stride, padding):
    map = np.zeros((image_size + 2*padding, image_size + 2*padding))
    count_image = np.zeros((image_size + 2*padding, image_size + 2*padding))
    ppr = (image_size + 2*padding - patch_size) // stride + 1
    for p in range(patches_scores.shape[2]):
        row_index = p // ppr
        col_index = p % ppr
        start_row = row_index * stride
        start_col = col_index * stride
       
        for k in range(patch_size):
            for l in range(patch_size):
                map[start_row+k, start_col+l] += float(patches_scores[0, 0, p])
                count_image[start_row+k, start_col+l] += 1
 
    map /= count_image
 
    if padding > 0:
        map = map[padding:-padding, padding:-padding]
 
    return map

def plot_ae_outputs_with_reconstruction(autoencoder, model_name, test_loader, params, image_size, epoch, device, n=10):
    
    plt.figure(figsize=(16,4.5))
    targets = []
    for i, (_, target) in enumerate(test_loader.dataset):
        targets.append(target)
    targets = np.array(targets)

    t_idx = {}
    for i in range(10):
        if i in targets:
            t_idx[i] = np.where(targets==i)[0][0]

    autoencoder.eval()
    gen = list(autoencoder.autoencoder[0].circuit.parameters())[0]
    weights = []
    for w in gen:
        weights.append(w)
    full_autoencoder = Encoder_Decoder(weights, image_size, params)

    for i in range(n):
        if i in targets:
            ax = plt.subplot(2, n, i+1)
            img = test_loader.dataset[t_idx[i]][0].unsqueeze(0).to(device)
            with torch.no_grad():
                patches_scores = full_autoencoder.reconstruction(img)
                #rec_img  = decoder(encoder(img))
            plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)  
            if i == n//2:
                ax.set_title('Original images')
            ax = plt.subplot(2, n, i + 1 + n)
            map = build_map(patches_scores, autoencoder.image_size, autoencoder.kernel_size, autoencoder.stride, autoencoder.padding)
            plt.imshow(map, cmap='hot')  
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)                
            if i == n//2:
                ax.set_title('Reconstructed images')
    #plt.show()
    plt.savefig("./output_images/" + model_name + "_epoch_" + str(epoch) + ".png")  
    plt.close()   

def loss_function(output):
    #print(torch.mean(output, 2))    
    return 1-torch.mean(output)

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

    gen = list(autoencoder.autoencoder[0].circuit.parameters())[0]
    weights = []
    for w in gen:
        weights.append(w)

    full_autoencoder = Encoder_Decoder(weights, image_size, params)

    # compute accuracy
    accuracies = []
    dice_scores = []
    iou_scores = []
    aupro = AUPRO()
    for i in range(len(test_loader.dataset)):
        img = test_loader.dataset[i][0].unsqueeze(0).to(device)
        mask = np.array(mask_loader.dataset[i][0].unsqueeze(0).to(device)[0,0,:,:]) #####
        mask = np.where(mask > 0, 1, 0)
        with torch.no_grad():
            #rec_img = autoencoder(img).to(device)
            patches_scores = full_autoencoder.reconstruction(img)
        map = build_map(patches_scores, image_size, autoencoder.kernel_size, autoencoder.stride, autoencoder.padding)
        #print(np.min(map))
        #print(np.max(map))
        diff = np.where(map > threshold, 0, 1) #anomalies are white in the masks
        acc = pixel_accuracy(mask, diff)
        dice = dice_coefficient(mask, diff)
        iou = IOU(mask, diff)
        aupro.update(torch.Tensor(diff).flatten(), torch.Tensor(mask).flatten())
        #print("(" + str(i) + "/" + str(len(test_loader.dataset)) + ") Label: " + str(test_loader.dataset[i][1]) + " - Accuracy: " + str(acc) + " - Dice: " + str(dice) + " - IoU: " + str(iou))
        accuracies.append(acc)
        dice_scores.append(dice)
        iou_scores.append(iou)

    #print("Accuracy: " + str(np.mean(accuracies)))
    #print("Dice Score: " + str(np.mean(dice_scores)))
    #print("IoU: " + str(np.mean(iou_scores)))
    aupro = aupro.compute()
    #print("AUPRO: " + str(aupro))
    
    plt.figure(figsize=(16, 4.5))
    for i in range(10):
        ax = plt.subplot(3, 10, i+1)
        img = test_loader.dataset[i+n_normal][0].unsqueeze(0).to(device)
        mask = np.array(mask_loader.dataset[i+n_normal][0].unsqueeze(0).to(device)[0,0,:,:])
        mask = np.where(mask > 0, 1, 0)
        with torch.no_grad():
            #rec_img = autoencoder(img).to(device)
            patches_scores = full_autoencoder.reconstruction(img)
        map = build_map(patches_scores, image_size, autoencoder.kernel_size, autoencoder.stride, autoencoder.padding)
        diff = np.where(map > threshold, 0, 1) #anomalies are white in the masks
        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        #ax.set_title("label: " + str(test_loader.dataset[i][1]), y=0, pad=-15)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        ax = plt.subplot(3, 10, i + 1 + 10)
        plt.imshow(diff, cmap='gist_gray')  
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        #ax.set_title(round(loss_fn(img).item(), 4), y=0, pad=-15)
        ax = plt.subplot(3, 10, i + 1 + 20)
        plt.imshow(mask.squeeze().numpy(), cmap='gist_gray')  
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False) 
        
    plt.savefig("./output_images/" + model_name + "_masks.png")

    return np.mean(accuracies), np.mean(dice_scores), np.mean(iou_scores), aupro
 

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
    image_size = 32
    training_size = 50
    noise = False

    # variable parameters
    params = {
        'dataset': ["carpet"],
        'kernel_size': [4],
        'stride': [1,4],
        'bottleneck_dim': [1],
        'mps_layers': [1],
        'n_block_wires': [2], 
        'n_params_block': [2],
    }

    seeds = [123,456,789]
    par_runs(dict(params), seeds, lr, num_epochs, image_size, training_size, noise, n_processes=processes)

   
