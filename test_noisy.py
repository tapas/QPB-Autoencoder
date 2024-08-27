import torch
from torch import nn
import sys

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

from itertools import product
import torch.multiprocessing as mp

from dataset import load_dataset
from metrics import pixel_accuracy, IOU, dice_coefficient, AUPRO
from ignite.contrib import metrics


from functools import partial

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
      
        dev = qml.device('default.mixed', wires=range(self.wires + self.discarded_wires + 1))
        
        @qml.qnode(dev)
        @partial(qml.transforms.insert, op=qml.DepolarizingChannel, op_args=0.2, position="all")
        def circuit(inputs, weights_mps):
                if torch.count_nonzero(inputs[:]):                   
                    qml.AmplitudeEmbedding(inputs[:], wires=range(self.discarded_wires, self.wires + self.discarded_wires), normalize=True, pad_with=0)

                qml.Hadamard(wires=self.wires + self.discarded_wires)

                #Apply MPS
                for i in range(self.mps_layers):
                    qml.MPS(range(self.discarded_wires, self.wires + self.discarded_wires), n_block_wires, MPS_BLOCK, n_params_block, weights_mps[i])

                for i in range(self.discarded_wires):    
                    qml.CSWAP(wires=[self.wires + self.discarded_wires, i, i + self.discarded_wires])

                qml.Hadamard(wires=self.wires + self.discarded_wires)

                return qml.expval(qml.PauliZ(wires=self.wires + self.discarded_wires))

 
        
        weight_shapes = {'weights_mps': [self.mps_layers, n_blocks, n_params_block]}
        
        torch.manual_seed(self.seed)
        
        init_method = {
            'weights_mps': torch.nn.init.normal_
        }
 
        self.circuit = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes, init_method=init_method)
             

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
            idx = 0
            for j in range(0, h - kernel_size + 1, self.stride):
                for k in range(0, w - kernel_size + 1, self.stride):
                    a = torch.tensor([img[b, 0, j + i, k + l] for i in range(kernel_size) for l in range(kernel_size)])
                    out[b, 0, idx] = self.circuit(a)
                    idx = idx + 1

        map = torch.zeros((bs, self.image_size + 2*self.padding, self.image_size + 2*self.padding))
        count_image = torch.zeros((bs, self.image_size + 2*self.padding, self.image_size + 2*self.padding))
        ppr = (self.image_size + 2*self.padding - self.kernel_size) // self.stride + 1
        for b in range(bs):
            for p in range(patch_no):
                row_index = p // ppr
                col_index = p % ppr
                start_row = row_index * self.stride
                start_col = col_index * self.stride
            
                for k in range(self.kernel_size):
                    for l in range(self.kernel_size):
                        map[b, start_row+k, start_col+l] += out[b, 0, p]
                        count_image[b, start_row+k, start_col+l] += 1
    
        map /= count_image
        if padding > 0:
            map = map[:, padding:-padding, padding:-padding]
    
        return map

 
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

        dev = qml.device("default.mixed", wires=range(self.wires*2 + self.discarded_wires + 1)) 
        
        @qml.qnode(dev)
        @partial(qml.transforms.insert, op=qml.DepolarizingChannel, op_args=0.2, position="all")
        def circuit_rec(inputs):

                if torch.count_nonzero(inputs[:]):                   
                    AE(inputs[:], self.discarded_wires, self.wires)
                
                qml.Hadamard(wires=self.wires*2 + self.discarded_wires)

                #Apply MPS
                for i in range(int(self.mps_layers)):
                    qml.MPS(range(self.discarded_wires, self.wires + self.discarded_wires), self.n_block_wires, MPS_BLOCK, self.n_params_block, self.weights[i])
                    
                for i in range(self.discarded_wires):
                    qml.SWAP(wires=[i, i + self.discarded_wires])
                
                #Apply adjoint MPS
                for i in range(int(self.mps_layers)):
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
                    a = torch.tensor([img[b, 0, j + i, k + l] for i in range(self.kernel_size) for l in range(self.kernel_size)])
                    out[b, 0, idx] = circuit_rec(a)[0]
                    idx = idx + 1
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
            patches_scores = full_autoencoder.reconstruction(img)
        map = build_map(patches_scores, image_size, autoencoder.kernel_size, autoencoder.stride, autoencoder.padding)
        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.set_title("label: " + str(test_loader.dataset[i+n_normal][1]), y=0, pad=-15)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        ax = plt.subplot(2, 10, i + 1 + 10)
        plt.imshow(map, cmap='hot')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        ax.set_title(round(loss_fn(patches_scores).item(), 4), y=0, pad=-15)
    plt.savefig("./output_images/" + model_name + "_test_rec.png")
    plt.close()

####################################

def run(chunk, seeds, lr, num_epochs, image_size, training_size, noise, thresholds, train):
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

            model_name = dataset_name+"_KS"+str(conf['kernel_size'])+"_ST"+str(conf['stride'])+"_BD"+str(conf['bottleneck_dim'])+"_s"+str(seed)
            train_loader, valid_loader, test_loader, mask_loader = load_dataset(dataset_name, training_size, image_size)
            loss_fn = loss_function
            device = "cpu"
        
            if train:
                print("train: " + model_name)
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
                    print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs, train_loss, val_loss))
                    train_loss_seed.append(train_loss)
                    val_loss_seed.append(val_loss)
                    if epoch == int(num_epochs/2)-1:
                        torch.save(autoencoder.state_dict(), './models/10epochs_' + model_name + '.pt')

                train_losses[s] = train_loss_seed
                val_losses[s] = val_loss_seed

                pd.DataFrame({'train_loss': train_losses[s], 'val_loss': val_losses[s]}).to_csv("./results/" + model_name + "_losses.csv")
                torch.save(autoencoder.state_dict(), './models/20epochs_' + model_name + '.pt')
                
                plot_loss_curve(train_losses[s], val_losses[s], model_name)
            
            print("test: " + model_name)
            autoencoder = Autoencoder(image_size, conf)
            autoencoder.load_state_dict(torch.load("./models/20epochs_" + model_name + ".pt"))
            autoencoder.to(device)
            autoencoder.eval()
            
            for t, threshold in enumerate(thresholds):
                print("Dataset: " + str(dataset_name) + " Tresholds: " + str(threshold) + " Seed: " + str(seed))
                acc, dice, iou, aupro, auroc = test_with_mask(autoencoder, model_name, test_loader, mask_loader, conf, image_size, device, threshold, n_normal)
                metrics[t][s][0] = acc
                metrics[t][s][1] = dice
                metrics[t][s][2] = iou
                metrics[t][s][3] = aupro
                metrics[t][s][4] = auroc
                
        mean_metrics = np.mean(metrics, axis=1)
        std_metrics = np.std(metrics, axis=1)
        filename = "./results/" + dataset_name+"_KS"+str(conf['kernel_size'])+"_ST"+str(conf['stride'])+"_BD"+str(conf['bottleneck_dim'])+"_metrics.csv"
        df_means = pd.DataFrame(mean_metrics)
        df_means.columns = ["mean_acc","mean_dice","mean_iou","mean_aupro","mean_auroc"]
        df_std = pd.DataFrame(std_metrics)
        df_std.columns = ["std_acc","std_dice","std_iou","std_aupro","std_auroc"]
        df = pd.concat([df_means, df_std], axis=1)
        df['threshold'] = thresholds
        df = df[["threshold", "mean_acc", "std_acc", "mean_dice", "std_dice", "mean_iou", "std_iou", "mean_aupro", "std_aupro", "mean_auroc", "std_auroc"]]
        df.to_csv(filename)
        
        if train:
            plot_loss_curve_avg(train_losses, val_losses, dataset_name+"_KS"+str(conf['kernel_size'])+"_ST"+str(conf['stride'])+"_BD"+str(conf['bottleneck_dim']))

def par_runs(params, seeds, lr, num_epochs, image_size, training_size, noise, thresholds, n_processes=2, train=True):
    
    keys, values = zip(*params.items())
    params_list = [dict(zip(keys, v)) for v in product(*values)]

    list_chunks = np.array_split(params_list, n_processes)
    
    if len(params_list) == 1:
        processes = [None]
        run(list_chunks[0], seeds, lr, num_epochs, image_size, training_size, noise, thresholds, train)    
    else:
        processes = [mp.Process(target=run, args=(chunk, seeds, lr, num_epochs, image_size, training_size, noise, thresholds, train))  for i, chunk in enumerate(list_chunks)]

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
 
    return (map + 1)/2

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
    return (1-torch.mean(output))

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
    auroc = metrics.ROC_AUC()
    for i in range(len(test_loader.dataset)):
        img = test_loader.dataset[i][0].unsqueeze(0).to(device)
        mask = np.array(mask_loader.dataset[i][0].unsqueeze(0).to(device)[0,0,:,:]) #####
        mask = np.where(mask > 0.5, 1, 0)
        with torch.no_grad():
            patches_scores = full_autoencoder.reconstruction(img)
        map = build_map(patches_scores, image_size, autoencoder.kernel_size, autoencoder.stride, autoencoder.padding)
        diff = np.where(map > threshold, 0, 1) #anomalies are white (1) in the masks
        acc = pixel_accuracy(mask, diff)
        dice = dice_coefficient(mask, diff)
        iou = IOU(mask, diff)
        map = 1 - map
        aupro.update(torch.Tensor(map).unsqueeze(0), torch.Tensor(mask).unsqueeze(0))
        auroc.update((torch.Tensor(map).ravel(), torch.Tensor(mask).ravel()))
        accuracies.append(acc)
        dice_scores.append(dice)
        iou_scores.append(iou)

    aupro = aupro.compute()
    auroc = auroc.compute()

    return np.mean(accuracies), np.mean(dice_scores), np.mean(iou_scores), aupro, auroc
 
def plot_localizations(conf, image_size, threshold=0.025, device='cpu'):
    
    dataset_name = conf['dataset'] 
    seed = conf['seed']

    if dataset_name == 'carpet':
        n_normal = 28
    elif dataset_name == 'wood':
        n_normal = 19
    else:
        n_normal = 32

    n_normal = 5

    model_name = dataset_name+"_KS"+str(conf['kernel_size'])+"_ST"+str(conf['stride'])+"_BD"+str(conf['bottleneck_dim'])+"_s"+str(seed)
    
    
    train_loader, valid_loader, test_loader, mask_loader = load_dataset(dataset_name, training_size, image_size)
    

    print("test: " + model_name)
    autoencoder = Autoencoder(image_size, conf)
    autoencoder.load_state_dict(torch.load("./models/20epochs_" + model_name + ".pt"))
    autoencoder.to(device)
    autoencoder.eval()
    
    gen = list(autoencoder.autoencoder[0].circuit.parameters())[0]
    weights = []
    for w in gen:
        weights.append(w)

    full_autoencoder = Encoder_Decoder(weights, image_size, conf)

    plt.figure(figsize=(8, 4.5))
    n_images = 5
    for i in range(n_images):
        if i == 0:
            index = 13
        else:
            index = i + n_normal
        ax = plt.subplot(3, n_images, i+1)
        img = test_loader.dataset[index][0].unsqueeze(0).to(device)
        mask = np.array(mask_loader.dataset[index][0].unsqueeze(0).to(device)[0,0,:,:])
        mask = np.where(mask > 0.5, 1, 0)
        with torch.no_grad():
            patches_scores = full_autoencoder.reconstruction(img)
        map = build_map(patches_scores, image_size, autoencoder.kernel_size, autoencoder.stride, autoencoder.padding)
        diff = 1-map
        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        #ax.set_title("label: " + str(test_loader.dataset[i][1]), y=0, pad=-15)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        ax = plt.subplot(3, n_images, i + 1 + n_images)
        plt.imshow(diff, cmap='jet')  
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        #ax.set_title(round(loss_fn(img).item(), 4), y=0, pad=-15)
        ax = plt.subplot(3, n_images, i + 1 + n_images*2)
        plt.imshow(mask.squeeze().numpy(), cmap='gist_gray')  
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False) 
          
    plt.savefig("./output_images/" + model_name + "_t" + str(threshold) + "_masks.png")


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
    image_size = 32 
    training_size = 125
    noise = False

    print("Image size: " + str(image_size))
    print("Training size: " + str(training_size))
    print("Learning rate: " + str(lr))
    print("Epochs: " + str(num_epochs))

    '''
    conf = {
        'seed': 789,
        'dataset': "busi",
        'kernel_size': 4,
        'stride': 1,
        'bottleneck_dim': 1,
        'mps_layers': 1,
        'n_block_wires': 2, 
        'n_params_block': 2,
    }
    plot_localizations(conf, image_size, threshold=0.97, device='cpu')
    exit()
    '''

    # variable parameters
    params = {
        'dataset': ["wood"],
        'kernel_size': [4],
        'stride': [1],
        'bottleneck_dim': [2],
        'mps_layers': [1],
        'n_block_wires': [2], 
        'n_params_block': [2],
    }

    seeds = [123,456,789]
    thresholds = [x/1000 for x in range(1,26,1)]
    par_runs(dict(params), seeds, lr, num_epochs, image_size, training_size, noise, thresholds, n_processes=processes, train=True)
