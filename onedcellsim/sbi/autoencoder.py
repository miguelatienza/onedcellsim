import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class AEconv(nn.Module):
    """Autoencoder class. Takes in a dataset with trajectories and encodes them into a latent space. Then also decodes them back to the original space.

    Args:
        nn (_type_): _description_
    """
    def __init__(self,  encoded_space_dim=16, fc2_input_dim=8, device='cpu', parameters='/project/ag-moonraedler/MAtienza/AE/autoencoder_16_features'):
        self.device=device
        super().__init__()
        # self.criterion = nn.MSELoss()
        # self.optimizer = torch.optim.Adam(lr=1e-3, weight_decay=0.01)

        ### Convolutional section
        self.shape_kernel_size = 3
        n_channels=3
        self.n_features = 8
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(n_channels, 8*n_channels, 3, stride=2, padding=0, groups=n_channels),
            nn.ReLU(True),
            nn.Conv1d(8*n_channels, 16*n_channels, 3, stride=2, padding=0, groups=n_channels),
            nn.BatchNorm1d(16*n_channels),
            nn.ReLU(True),
            nn.Conv1d(16*n_channels, 32*n_channels, 3, stride=2, padding=0, groups=n_channels),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(576*n_channels, 64),#fc2_input_dim),
            nn.ReLU(True),
            #nn.Linear(fc2_input_dim, 64),
            #nn.ReLU(True),
            nn.Linear(64, encoded_space_dim)
        )
        
        self.decoder_lin =  nn.Sequential(
            nn.Linear(encoded_space_dim, 64),
            nn.ReLU(True),
            #nn.Linear(64, fc2_input_dim),
            #nn.ReLU(True),
            nn.Linear(64, 576*n_channels),
            nn.ReLU(True)
        )
        
        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=( 32*n_channels, 18))
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(32*n_channels, 16*n_channels, 3, 
            stride=2, output_padding=0, groups=n_channels, padding=0),
            nn.BatchNorm1d(16*n_channels),
            nn.ReLU(True),
            nn.ConvTranspose1d(16*n_channels, 8*n_channels, 3, stride=2,
            padding=0, output_padding=0, groups=n_channels),
            nn.BatchNorm1d(8*n_channels),
            nn.ReLU(True),
            nn.ConvTranspose1d(8*n_channels, n_channels, 3, stride=2, padding=0, output_padding=0, groups=n_channels)
            
                              )
        
        self.load_state_dict(torch.load(parameters))
        self.to(device=device)
        
    def forward(self, x, noisify=False):
        ##Encode
        #print(x.size())
        if noisify:
            x = self.noisify(x)

        x = self.encoder_cnn(x)
        #print(x.size())
        x = self.flatten(x)
        #print(x.size())
        x = self.encoder_lin(x)
        #print(x.size())
        #print('now decode')
        ##Decode
        #print(x.size())
        #print('linear:')
        x = self.decoder_lin(x)
       # print(x.size())
        #print('flatten:')
        x = self.unflatten(x)
        #print(x.size())
        #print('doconvolve:')
        x = self.decoder_conv(x)
        #print(x.size())
        #x = torch.sigmoid(x)
        #print(x.size())
        return x
    
    def encode(self, x, noisify=False):

        if noisify:
            x = self.noisify(x)

        x = self.encoder_cnn(x)
        #print(x.size())
        x = self.flatten(x)
        #print(x.size())
        x = self.encoder_lin(x)
        return x
    
    def decode(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x
    
    def noisify(self, X, epsilon=3, p=0.05):
        X = X+ epsilon*torch.randn(X.size()).to(X.device)
        X=F.dropout(X, p=p)
        return X

    def train_loader_func(x, batch_size):
    
        N = x.shape[0]
        
        n_batches = np.ceil(N/batch_size).astype(int)
        return (x[batch_size*i: np.clip(int(batch_size*(i+1)),0, N)] for i in range(n_batches))
        
    
    def train_(self, X, input_size, val_frac=0.1, batch_size=1000, n_epochs=1000):
    
        losses=[]
        val_losses=[]
        N = X.shape[0]
        X_val = X[-int(N*val_frac):]
        X = X[:-int(N*val_frac)]
        print(X.size(), X_val.size())
        batch_size_val = int(batch_size*val_frac)
        #min_loss = *0.008
        for epoch in range(n_epochs):
            train_loader = self.train_loader_func(X, batch_size)
            val_loader = self.train_loader_func(X_val, batch_size_val)
            loss = 0
            val_loss=0
            loader_len=np.ceil(N*(1-val_frac)/batch_size)
            
            for batch_features, batch_val_features in zip(train_loader, val_loader):
                
                ##add noise to the features:
                x = self.noisify(batch_features)
                # reshape mini-batch data to [N, 7batch_size] matrix
                # load it to the active device
                #print(batch_features.size())
                # mu = batch_features.mean()
                # var= batch_features.var()
                # batch_features = (batch_features-mu)/torch.sqrt(var**2+1e-05)
                n_features = batch_features.size()[0]
                #sys.exit()
                # reset the gradients back to zero
                # PyTorch accumulates gradients on subsequent backward passes
                self.optimizer.zero_grad()
                
                # compute reconstructions
                outputs = selfmodel(x)

                # compute training reconstruction loss
                train_loss = criterion(outputs, batch_features)
    #             l2_lambda = 0.01
    #             l2_reg = torch.tensor(0., require_)

    #             for param in model.parameters():
    #                 l2_reg += torch.norm(param)

    #             loss += l2_lambda * l2_reg

            
                # compute accumulated gradients
                train_loss.backward()

                # perform parameter update based on current gradients
                optimizer.step()

                # add the mini-batch training loss to epoch loss
                loss += (train_loss.item()*n_features)
                # loss = np.mean([loss, train_loss.item()])
                # val_loss = np.mean([val_loss, val_loss_tensor.item()])
            # compute the epoch training loss
            outputs_val = model(noisify(X_val))
            val_loss = criterion(outputs_val, X_val).item()
            loss = loss / N*(1-val_frac)
            #val_loss = val_loss/(N*val_frac)
            losses.append(loss)
            val_losses.append(val_loss)
            # display the epoch training loss
            #print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, n_epochs, loss*1e5))
            if (epoch%10)==1:
                
                #print(len(x),len(y))
                line.set_data(np.arange(epoch+1), losses)
                line_val.set_data(np.arange(epoch+1), val_losses)
                high = max(max(losses), max(val_losses))
                low = min(min(losses), min(val_losses))
                ax.set_xlim(0,epoch)
                ax.set_ylim(low,high)
                fig.canvas.draw()
                #plt.show()
                #clear_output(wait=True)
        return losses
        
