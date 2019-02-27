import torch
from torch import nn
from layer_norm import LayerNorm


class liGRU(nn.Module):
    """
    https://github.com/mravanelli/pytorch-kaldi/blob/master/neural_networks.py
    """
    
    def __init__(self, inp_dim, out_dim, bidirectional=True, dropout_p=[0.2], use_batchnorm=[True], use_layernorm=[False], use_inp_layernorm=False, use_inp_batchnorm=True, orth_init=[True], ligru_act=nn.ReLU, to_do='train', use_cuda=True):
        super(liGRU, self).__init__()
        
        # Reading parameters
        self.input_dim=inp_dim
        self.out_dim=out_dim
        self.dropout_p=dropout_p
        self.use_batchnorm=use_batchnorm
        self.use_layernorm=use_layernorm
        self.use_inp_layernorm=use_inp_layernorm
        self.use_inp_batchnorm=use_inp_batchnorm
        self.orth_init=orth_init
        self.ligru_act=ligru_act
        self.bidirectional=bidirectional
        self.use_cuda=use_cuda
        self.to_do=to_do
        
        if self.to_do=='train':
            self.test_flag=False
        else:
            self.test_flag=True
        
        
        # List initialization
        self.wh  = nn.ModuleList([])
        self.uh  = nn.ModuleList([])
        
        self.wz  = nn.ModuleList([]) # Update Gate
        self.uz  = nn.ModuleList([]) # Update Gate
              
        
        self.ln  = nn.ModuleList([]) # Layer Norm
        self.bn_wh  = nn.ModuleList([]) # Batch Norm
        self.bn_wz  = nn.ModuleList([]) # Batch Norm


        
        self.act  = nn.ModuleList([]) # Activations
       
  
        # Input layer normalization
        if self.use_inp_layernorm:
           self.ln0=LayerNorm(self.input_dim)
          
        # Input batch normalization    
        if self.use_inp_batchnorm:
           self.bn0=nn.BatchNorm1d(self.input_dim,momentum=0.05)
           
        self.N_ligru_lay=len(self.out_dim)
             
        current_input=self.input_dim
        
        # Initialization of hidden layers
        
        for i in range(self.N_ligru_lay):
             
             # Activations
             self.act += [self.ligru_act()]
            
             add_bias=True
             
             
             if self.use_layernorm[i] or self.use_batchnorm[i]:
                 add_bias=False
             
                  
             # Feed-forward connections
             self.wh.append(nn.Linear(current_input, self.out_dim[i],bias=add_bias))
             self.wz.append(nn.Linear(current_input, self.out_dim[i],bias=add_bias))

             
            
             # Recurrent connections
             self.uh.append(nn.Linear(self.out_dim[i], self.out_dim[i],bias=False))
             self.uz.append(nn.Linear(self.out_dim[i], self.out_dim[i],bias=False))

             if self.orth_init:
             	nn.init.orthogonal_(self.uh[i].weight)
             	nn.init.orthogonal_(self.uz[i].weight)


             
             # batch norm initialization
             self.bn_wh.append(nn.BatchNorm1d(self.out_dim[i],momentum=0.05))
             self.bn_wz.append(nn.BatchNorm1d(self.out_dim[i],momentum=0.05))


                
             self.ln.append(LayerNorm(self.out_dim[i]))
                
             if self.bidirectional:
                 current_input=2*self.out_dim[i]
             else:
                 current_input=self.out_dim[i]
                 
        self.final_dim=self.out_dim[i]+self.bidirectional*self.out_dim[i]
            
             
        
    def forward(self, x):
        
        # Applying Layer/Batch Norm
        if self.use_inp_layernorm:
            x=self.ln0((x))
        
        if self.use_inp_batchnorm:
            x_bn=self.bn0(x.view(x.shape[0]*x.shape[1],x.shape[2]))
            x=x_bn.view(x.shape[0],x.shape[1],x.shape[2])

          
        for i in range(self.N_ligru_lay):
            
            # Initial state and concatenation
            if self.bidirectional:
                h_init = torch.zeros(2*x.shape[1], self.out_dim[i])
                x=torch.cat([x,flip(x,0)],1)
            else:
                h_init = torch.zeros(x.shape[1],self.out_dim[i])
        
               
            # Drop mask initilization (same mask for all time steps)            
            if self.test_flag==False:
                drop_mask=torch.bernoulli(torch.Tensor(h_init.shape[0],h_init.shape[1]).fill_(1-self.dropout_p[i]))
            else:
                drop_mask=torch.FloatTensor([1-self.dropout_p[i]])
                
            if self.use_cuda:
               h_init=h_init.cuda()
               drop_mask=drop_mask.cuda()
               
                 
            # Feed-forward affine transformations (all steps in parallel)
            wh_out=self.wh[i](x)
            wz_out=self.wz[i](x)


            
            # Apply batch norm if needed (all steps in parallel)
            if self.use_batchnorm[i]:

                wh_out_bn=self.bn_wh[i](wh_out.view(wh_out.shape[0]*wh_out.shape[1],wh_out.shape[2]))
                wh_out=wh_out_bn.view(wh_out.shape[0],wh_out.shape[1],wh_out.shape[2])
         
                wz_out_bn=self.bn_wz[i](wz_out.view(wz_out.shape[0]*wz_out.shape[1],wz_out.shape[2]))
                wz_out=wz_out_bn.view(wz_out.shape[0],wz_out.shape[1],wz_out.shape[2])


            
            # Processing time steps
            hiddens = []
            ht=h_init
            
            for k in range(x.shape[0]):
                
                # ligru equation
                zt=torch.sigmoid(wz_out[k]+self.uz[i](ht))
                at=wh_out[k]+self.uh[i](ht)
                hcand=self.act[i](at)*drop_mask
                ht=(zt*ht+(1-zt)*hcand)
                
                
                if self.use_layernorm[i]:
                    ht=self.ln[i](ht)
                    
                hiddens.append(ht)
                
            # Stacking hidden states
            h=torch.stack(hiddens)
            
            # Bidirectional concatenations
            if self.bidirectional:
                h_f=h[:,0:int(x.shape[1]/2)]
                h_b=flip(h[:,int(x.shape[1]/2):x.shape[1]].contiguous(),0)
                h=torch.cat([h_f,h_b],2)
                
            # Setup x for the next hidden layer
            x=h

        return x