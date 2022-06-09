import torch.nn as nn
import torchvision.models
import numpy as np



def convrelu(in_channels, out_channels, kernel, padding):
      return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding),
        nn.ReLU(inplace=True),
      )

def convattn(in_channels,booksize=16,kernel=1, padding=0):
      return nn.Sequential(
        nn.Conv2d(in_channels, booksize, kernel_size=kernel, padding=padding,bias = False)
      )



#Form query
# then you 'dot product them'
# then you soft max
class SelfAttention(nn.Module):
    def __init__(self, n_channel,key_size,side=32,radius=6.):
        super().__init__()
        self.X1 = nn.Parameter(torch.range(0,side-1).float().unsqueeze(1).unsqueeze(2).unsqueeze(3),requires_grad=False)
        self.X2 = nn.Parameter(torch.range(0,side-1).float().unsqueeze(0).unsqueeze(1).unsqueeze(3),requires_grad=False)

        self.Y1 = nn.Parameter(torch.range(0,side-1).float().unsqueeze(0).unsqueeze(2).unsqueeze(3),requires_grad=False)
        self.Y2 = nn.Parameter(torch.range(0,side-1).float().unsqueeze(0).unsqueeze(1).unsqueeze(2),requires_grad=False)

        self.dists = torch.sqrt(torch.square(self.Y1-self.Y2)+torch.square(self.X1-self.X2))
        
        self.x_diff=(self.X1-self.X2)/radius+torch.zeros((side,side,side,side))
        self.y_diff=(self.Y1-self.Y2)/radius+torch.zeros((side,side,side,side))
        self.x_diff = torch.flatten(self.x_diff,start_dim=0,end_dim=1).unsqueeze(0).unsqueeze(1)
        self.y_diff = torch.flatten(self.y_diff,start_dim=0,end_dim=1).unsqueeze(0).unsqueeze(1)

        self.rel_posn = nn.Parameter(torch.cat([self.x_diff,self.y_diff],1),requires_grad=False)



        self.dists = nn.Parameter(torch.flatten(self.dists,start_dim=0,end_dim=1).unsqueeze(0),requires_grad=False)
        self.radius=radius
        self.dist_mask = nn.Parameter(torch.zeros(self.dists.shape,requires_grad=False),requires_grad=False)
        
        self.dist_mask[(self.dists>self.radius)] = float('-inf')
        self.dist_mask =nn.Parameter(self.dist_mask,requires_grad=False)

        self.w3 = nn.Parameter(torch.tensor(0.,requires_grad=True),requires_grad=True)


        self.query = nn.Conv2d(n_channel, key_size, kernel_size=1, padding=0,bias=False)
        self.key = nn.Conv2d(n_channel, key_size, kernel_size=1, padding=0,bias=False)
        self.value = nn.Conv2d(n_channel, key_size,kernel_size=1, padding=0,bias=False)

        self.query_xy = nn.Conv2d(n_channel, 2, kernel_size=1, padding=0,bias=False)
        self.key_xy = nn.Conv2d(n_channel, 2, kernel_size=1, padding=0,bias=False)


        self.output = nn.Conv2d(key_size,n_channel,kernel_size=1, padding=0,bias=False)

    def forward(self,input):
        k = torch.flatten(self.key(input),start_dim = 2).unsqueeze(3).unsqueeze(4)
        q = self.query(input).unsqueeze(2)

        k_xy = torch.flatten(self.key_xy(input),start_dim = 2).unsqueeze(3).unsqueeze(4)
        q_xy = self.query_xy(input).unsqueeze(2)

        v = torch.flatten(self.value(input),start_dim = 2).unsqueeze(3).unsqueeze(4)
        #radial_field = self.gate0(self.w0*self.dists)+self.gate1(self.w1*self.dists)+self.gate2(self.w2*self.dists)
        #radial_field = self.w3*self.local_gate(radial_field)
        radial_field = -torch.abs(self.w3)*self.dists
        
        key_query = torch.sum(k*q,1) # (W*H) x W x H
        key_query =key_query

        #key_query_xy = torch.pow((1+torch.sum((k_xy*q_xy)*self.rel_posn,1))/2.,3.)


        key_query = torch.softmax(key_query_xy+self.dist_mask+key_query,dim=1)
        key_query = key_query.unsqueeze(1) #For viz find the max of this. 1x(W*H) x W x H
        attn_map = self.output(torch.sum(key_query*v,2))
        return attn_map

    def attention_field(self,input):
        k = torch.flatten(self.key(input),start_dim = 2).unsqueeze(3).unsqueeze(4)
        q = self.query(input).unsqueeze(2)

        k_xy = torch.flatten(self.key_xy(input),start_dim = 2).unsqueeze(3).unsqueeze(4)
        q_xy = self.query_xy(input).unsqueeze(2)

        v = torch.flatten(self.value(input),start_dim = 2).unsqueeze(3).unsqueeze(4)
        #radial_field = self.gate0(self.w0*self.dists)+self.gate1(self.w1*self.dists)+self.gate2(self.w2*self.dists)
        #radial_field = self.w3*self.local_gate(radial_field)
        radial_field = -torch.abs(self.w3)*self.dists
        
        key_query = torch.sum(k*q,1) # (W*H) x W x H
        key_query =key_query

        #key_query_xy = torch.pow((1+torch.sum((k_xy*q_xy)*self.rel_posn,1))/2.,3.)


        key_query = torch.softmax(key_query_xy+self.dist_mask+key_query,dim=1)
        key_query = key_query.unsqueeze(1) #For viz find the max of this. 1x(W*H) x W x H
        return key_query

class MultiHeadAttention(nn.Module):
    # TODO: Make return attention field 
    def __init__(self,output_dim,n_head,key_dim,radius=6,side=32):
        super().__init__()
        d = output_dim//n_head
        self.n_head = n_head
        self.attn_maps = nn.ModuleList([nn.Conv2d(output_dim,d,kernel_size=1,padding=0,bias=False) for _ in range(self.n_head)])
        if type(radius) is not list:
            self.attn_heads = nn.ModuleList([SelfAttention(d,key_dim,radius=radius,side=side) for _ in range(self.n_head)])
        else:
            self.attn_heads = nn.ModuleList([SelfAttention(d,key_dim,radius=radius[i],side=side) for i in range(self.n_head)])
    def forward(self,input):
        return torch.cat([self.attn_heads[h](self.attn_maps[h](input)) for h in range(self.n_head)],dim=1)

    def attention_field(self,input):
        return [self.attn_heads[h].attention_field(self.attn_maps[h](input)) for h in range(self.n_head)]


class ResNetUNet(nn.Module):
      def __init__(self, n_class=5):
        super().__init__()

        self.downsize = torchvision.transforms.Resize((32,32),interpolation = transforms.functional.InterpolationMode('nearest'))
        self.layer1 = convrelu(3, 64, 1, 0)
        self.layer1b = nn.Conv2d(64,20,kernel_size=1,padding=0)

        self.multihead1 = MultiHeadAttention(64,16,4,radius=32)#radius=[1.2]*4+[2]*4+[3]*4+[4]*3+[8])
        self.layer_norm2 = nn.BatchNorm2d(64,affine=False)

        self.layer2 = convrelu(64, 32, 1, 0)

        self.resize2 = torchvision.transforms.Resize((8,8),interpolation = transforms.functional.InterpolationMode('nearest'))


        self.multihead_small = MultiHeadAttention(32,4,2,radius=32,side=8)#radius=[1,2,2,4],side=8)
        self.layer_norm_small = nn.BatchNorm2d(32,affine=False)
        self.layer_mid = convrelu(32,16,1,0)


        self.resize3 = torchvision.transforms.Resize((32,32),interpolation = transforms.functional.InterpolationMode('nearest'))


        self.multihead2 = MultiHeadAttention(48,8,4,radius=32)#radius=[2]*2+[4]*2+[6]*2+[8]*2)
        self.layer_norm3 = nn.BatchNorm2d(48,affine=False)
        self.layer3 = convrelu(48,20,1,0)


        self.multihead3 = MultiHeadAttention(40,5,4,radius=32)#radius=[2]+[4]*2+[6]+[8])
        self.layer_norm4 = nn.BatchNorm2d(40,affine=False)
        self.layer4 = convrelu(40,20,1,0)

        self.self_attention =SelfAttention(20,5,radius=32)
        self.layer_norm5 = nn.BatchNorm2d(20,affine=False)
        self.layer5 = convrelu(20,5,1,0)


        self.upsample_full = torchvision.transforms.Resize((256,256),interpolation = transforms.functional.InterpolationMode('nearest'))

      def forward(self, input):
        input= input.float()
        x = input
        #x = self.res_layer(x)
        x=self.layer1(x)
        x = self.downsize(x)

        x1 = self.layer1b(x)

        attn = self.multihead1(x)


        x = self.layer_norm2(x+attn)
        x2 = self.layer2(x)
        x = self.resize2(x2)

        attn = self.multihead_small(x)
        x = self.layer_norm_small(x+attn)
        x = self.layer_mid(x)

        x = self.resize3(x)

        x = torch.cat([x,x2],dim=1)


        attn = self.multihead2(x)
        x = self.layer_norm3(x+attn)
        x = self.layer3(x)


        x = torch.cat([x,x1],dim=1)
        attn = self.multihead3(x)
        x = self.layer_norm4(x+attn)
        x = self.layer4(x)

        attn,key_query = self.self_attention.forward_kq(x)
        x = self.layer_norm5(x+attn)
        x = self.layer5(x)

        out = torch.sigmoid(self.upsample_full(x))
        ycoord = int(random.randint(0,31))
        xcoord = int(random.randint(0,31))


        all_attn = key_query[:,:,:,ycoord,xcoord].view(-1,32,32)

        return out,all_attn, (ycoord,xcoord)