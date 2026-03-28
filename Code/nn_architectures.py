import torch.nn as nn


class cnn_branch(nn.Module):
    def __init__(self):
        super(cnn_branch, self).__init__()
        self.num_classes=3
        self.conv1 = nn.Conv2d(in_channels=7, out_channels=16, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3))
        self.dense_1 = nn.Linear(128, 64)
        self.dense_2 = nn.Linear(64, self.num_classes)
        self.pool = nn.MaxPool2d((2,2))

    def forward(self, images):
        B, T, C, H, W = images.shape  # [10, 24, 7, 10, 10]                                 
        output = images.view(B * T, C, H, W)  # [240, 7, 10, 10]                            
        output = self.conv1(output)  # [240, 8, 8, 8]                                       
        output = F.relu(output)
        output = self.pool(output)  # [240, 16, 4, 4]                                       
        output = self.conv2(output)  # [240, 16, 2, 2]                                      
        output = F.relu(output)

        # flatten all channels of each image for each training example x timestep          
        output = output.flatten(start_dim=1)  # [240, 64]                                   
        output = self.dense_1(output)  # [240, 16]                                          
        output = F.relu(output)
        output = self.dense_2(output)  # [240, 16]                                          

        output = F.gumbel_softmax(output, tau=1.0, hard=True, dim=1)  # torch.Size([240, 4])
        output = output.view(B, T, -1)  # [10, 24, 4]                                       

        return output


class model_stack_wCNN(nn.Module):
    def __init__(self, ninp, nhid=8, nlayers=2, nout1=1, dropout=0):
        super(pureML_GRU, self).__init__()
        self.num_classes=3
        self.gru = nn.GRU(ninp+self.num_classes, nhid, nlayers,dropout=dropout, batch_first=True)
        self.densor_flux = nn.Linear(nhid, nout1)
        self.nhid = nhid
        self.nlayers = nlayers
        self.drop=nn.Dropout(dropout)
        self.init_weights()
        self.cnn_branch = cnn_branch()

    def init_weights(self):
        initrange = 0.1 #may change to a small value                           
        self.densor_flux.bias.data.zero_()
        self.densor_flux.weight.data.uniform_(-initrange, initrange)

    def forward(self, images, inputs, hidden):
        output = self.cnn_branch(images)  # torch.Size([10, 24, 4])            
        output = torch.cat([inputs, output], dim=-1)  # torch.Size([10, 24, 6])
        output, hidden = self.gru(output, hidden)
        output = self.drop(output)
        output = self.densor_flux(output)  # torch.Size([10, 24, 1])           
        return output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, bsz, self.nhid)
        
    
class gru(nn.Module):
    def __init__(self, ninp, nhid=8, nlayers=2, nout1=1, dropout=0):
        super(pureML_GRU, self).__init__()
        self.gru = nn.GRU(ninp, nhid,nlayers,dropout=dropout, batch_first=True)
        self.densor_flux = nn.Linear(nhid, nout1)
        self.nhid = nhid
        self.nlayers = nlayers
        self.drop=nn.Dropout(dropout)
        self.ReLU=nn.ReLU()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1 
        self.densor_flux.bias.data.zero_()
        self.densor_flux.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, hidden):
        output, hidden = self.gru(inputs, hidden)
        output1 = self.densor_flux(self.drop(output))
        
        return output1, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, bsz, self.nhid)


    
class grad_reverse(torch.autograd.Function):
    # https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/2
    # https://medium.com/@lukas.hauzenberger/adverserial-training-5bb5ea919ae7
    @staticmethod
    def forward(ctx, input_, lmbda=100):
        ctx.lmbda = lmbda
        return input_.view_as(input_)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.lmbda
        #grad_input = grad_output * ctx.lmbda
        return grad_input, None

class feature_extractor(nn.Module):
    def __init__(self, size, nhid):
        super(feature_extractor, self).__init__()
        self.gru = nn.GRU(input_size=size, hidden_size=nhid, num_layers=2, batch_first=True)

    def forward(self, inputs):
        output, _ = self.gru(inputs)
        return output    

class output_branch(nn.Module):
    def __init__(self):
        super(output_branch, self).__init__()
        self.dense = nn.Linear(8,1)

    def forward(self, inputs):
        output = self.dense(inputs)                
        return output

class domain_classifier(nn.Module):
    def __init__(self, lambda_):
        super(domain_classifier, self).__init__()
        self.fc1 = nn.Linear(8, 4) 
        self.fc2 = nn.Linear(4, 1)  # 1 output per month
        self.fc_out = nn.Linear(24, 1)  # 12-mo window
        self.ReLU=nn.ReLU()
        self.lambda_ = lambda_
        
    def forward(self, x):
        x = grad_reverse.apply(x, self.lambda_)
        x = self.ReLU(self.fc1(x))
        x = self.ReLU(self.fc2(x))  # 1 output per month
        x = x.view(x.shape[0], -1)  # flatten        
        x = self.fc_out(x)  # 1 over output for the training example
        x = torch.sigmoid(x)
        return x

class doman_adapt(nn.Module):
    def __init__(self, input_dim, hidden_dim, lambda_):
        super(combined_nn, self).__init__()
        self.feature_extractor = feature_extractor(input_dim, hidden_dim)
        self.output_branch = output_branch()
        self.domain_classifier = domain_classifier(lambda_)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.output_branch(features)
        domain = self.domain_classifier(features)
        return output, domain  # torch.Size([bsz, 12, 1]) torch.Size([bsz, 12, 1])    
