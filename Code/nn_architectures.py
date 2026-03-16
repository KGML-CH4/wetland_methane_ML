import torch.nn as nn

### baseline ML
class pureML_GRU(nn.Module):
    def __init__(self, ninp, nhid=8, nlayers=2, nout1=1, dropout=0):
        super(pureML_GRU, self).__init__()
        if nlayers > 1:
            self.gru = nn.GRU(ninp, nhid,nlayers,dropout=dropout, batch_first=True)
        else:
            self.gru = nn.GRU(ninp, nhid,nlayers, batch_first=True)
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


    
