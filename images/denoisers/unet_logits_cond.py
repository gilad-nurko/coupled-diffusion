import torch
import torch.nn as nn


class ChannelShuffle(nn.Module):
    def __init__(self,groups):
        super().__init__()
        self.groups=groups
    def forward(self,x):
        n,c,h,w=x.shape
        x=x.view(n,self.groups,c//self.groups,h,w) # group
        x=x.transpose(1,2).contiguous().view(n,-1,h,w) #shuffle
        
        return x

class ConvBnSiLu(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0):
        super().__init__()
        self.module=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
                                  nn.BatchNorm2d(out_channels),
                                  nn.SiLU(inplace=True))
    def forward(self,x):
        return self.module(x)

class ResidualBottleneck(nn.Module):
    '''
    shufflenet_v2 basic unit(https://arxiv.org/pdf/1807.11164.pdf)
    '''
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.branch1=nn.Sequential(nn.Conv2d(in_channels//2,in_channels//2,3,1,1,groups=in_channels//2),
                                    nn.BatchNorm2d(in_channels//2),
                                    ConvBnSiLu(in_channels//2,out_channels//2,1,1,0))
        self.branch2=nn.Sequential(ConvBnSiLu(in_channels//2,in_channels//2,1,1,0),
                                    nn.Conv2d(in_channels//2,in_channels//2,3,1,1,groups=in_channels//2),
                                    nn.BatchNorm2d(in_channels//2),
                                    ConvBnSiLu(in_channels//2,out_channels//2,1,1,0))
        self.channel_shuffle=ChannelShuffle(2)

    def forward(self,x):
        x1,x2=x.chunk(2,dim=1)
        x=torch.cat([self.branch1(x1),self.branch2(x2)],dim=1)
        x=self.channel_shuffle(x) #shuffle two branches

        return x

class ResidualDownsample(nn.Module):
    '''
    shufflenet_v2 unit for spatial down sampling(https://arxiv.org/pdf/1807.11164.pdf)
    '''
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.branch1=nn.Sequential(nn.Conv2d(in_channels,in_channels,3,2,1,groups=in_channels),
                                    nn.BatchNorm2d(in_channels),
                                    ConvBnSiLu(in_channels,out_channels//2,1,1,0))
        self.branch2=nn.Sequential(ConvBnSiLu(in_channels,out_channels//2,1,1,0),
                                    nn.Conv2d(out_channels//2,out_channels//2,3,2,1,groups=out_channels//2),
                                    nn.BatchNorm2d(out_channels//2),
                                    ConvBnSiLu(out_channels//2,out_channels//2,1,1,0))
        self.channel_shuffle=ChannelShuffle(2)

    def forward(self,x):
        x=torch.cat([self.branch1(x),self.branch2(x)],dim=1)
        x=self.channel_shuffle(x) #shuffle two branches

        return x

class TimeMLP(nn.Module):
    '''
    naive introduce timestep information to feature maps with mlp and add shortcut
    '''
    def __init__(self,embedding_dim,hidden_dim,out_dim):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(embedding_dim,hidden_dim),
                                nn.SiLU(),
                               nn.Linear(hidden_dim,out_dim))
        self.act=nn.SiLU()
    def forward(self,x,t):
        t_emb=self.mlp(t).unsqueeze(-1).unsqueeze(-1)
        x=x+t_emb
  
        return self.act(x)
    
class EncoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,time_embedding_dim):
        super().__init__()
        self.conv0=nn.Sequential(*[ResidualBottleneck(in_channels,in_channels) for i in range(3)],
                                    ResidualBottleneck(in_channels,out_channels//2))

        self.time_mlp=TimeMLP(embedding_dim=time_embedding_dim,hidden_dim=out_channels,out_dim=out_channels//2)
        self.conv1=ResidualDownsample(out_channels//2,out_channels)
    
    def forward(self,x,t=None):
        x_shortcut=self.conv0(x)
        if t is not None:
            x=self.time_mlp(x_shortcut,t)
        x=self.conv1(x)

        return [x,x_shortcut]
        
class DecoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,time_embedding_dim):
        super().__init__()
        self.upsample=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        self.conv0=nn.Sequential(*[ResidualBottleneck(in_channels,in_channels) for i in range(3)],
                                    ResidualBottleneck(in_channels,in_channels//2))

        self.time_mlp=TimeMLP(embedding_dim=time_embedding_dim,hidden_dim=in_channels,out_dim=in_channels//2)
        self.conv1=ResidualBottleneck(in_channels//2,out_channels//2)

    def forward(self,x,x_shortcut,t=None):
        x=self.upsample(x)
        x=torch.cat([x,x_shortcut],dim=1)
        x=self.conv0(x)
        if t is not None:
            x=self.time_mlp(x,t)
        x=self.conv1(x)

        return x        

class Unet(nn.Module):
    def __init__(self, timesteps, time_embedding_dim, in_channels=3, out_channels=2, base_dim=32, dim_mults=[2, 4, 8, 16], is_cond=False, is_y_cond=False, num_classes=10):
        super().__init__()
        assert isinstance(dim_mults, (list, tuple))
        assert base_dim % 2 == 0

        channels = self._cal_channels(base_dim, dim_mults)
        # print(channels)
        self.init_conv = ConvBnSiLu(in_channels, base_dim, 3, 1, 1)
        self.cond_conv = ConvBnSiLu(in_channels, base_dim, 3, 1, 1)
        self.y_cond_conv = ConvBnSiLu(num_classes, base_dim, 3, 1, 1)

        self.time_embedding = nn.Embedding(timesteps, time_embedding_dim)

        self.encoder_blocks = nn.ModuleList([EncoderBlock(c[0], c[1], time_embedding_dim) for c in channels])
        self.cond_encoder_blocks = nn.ModuleList([EncoderBlock(c[0], c[1], time_embedding_dim) for c in channels])
        self.y_cond_encoder_blocks = nn.ModuleList([EncoderBlock(c[0], c[1], time_embedding_dim) for c in channels])

        self.mid_block = nn.Sequential(
            *[ResidualBottleneck(channels[-1][1], channels[-1][1]) for _ in range(2)],
            ResidualBottleneck(channels[-1][1], channels[-1][1] // 2)
        )
        self.cond_mid_block = nn.Sequential(
            *[ResidualBottleneck(channels[-1][1], channels[-1][1]) for _ in range(2)],
            ResidualBottleneck(channels[-1][1], channels[-1][1] // 2)
        )
        self.y_cond_mid_block = nn.Sequential(
            *[ResidualBottleneck(channels[-1][1], channels[-1][1]) for _ in range(2)],
            ResidualBottleneck(channels[-1][1], channels[-1][1] // 2)
        )
        decoder_mul_size = 3 if is_cond and is_y_cond else 2 if is_cond or is_y_cond else 1
        self.decoder_blocks = nn.ModuleList([DecoderBlock(c[1] * decoder_mul_size, c[0] * decoder_mul_size, time_embedding_dim) for c in channels[::-1]])
        # print(channels)
        self.final_conv = nn.Conv2d(in_channels=(channels[0][0] * decoder_mul_size)//2, out_channels=out_channels, kernel_size=1)

    def forward(self, x, t=None, cond=None, y_cond=None):
        x = self.init_conv(x)

        if t is not None:
            t = self.time_embedding(t)

        encoder_shortcuts = []
        if cond is not None:
            cond = self.cond_conv(cond)
            cond_encoder_shortcuts = []
        
        if y_cond is not None:
            # Embed class labels and expand to match input dimensions
            bs, _, w, h = x.size()
            y_cond = y_cond.view(bs, y_cond.size(1), 1, 1).expand(bs, y_cond.size(1), w, h)
            y_cond = self.y_cond_conv(y_cond)
            y_cond_encoder_shortcuts = []

        for i, encoder_block in enumerate(self.encoder_blocks):
            x, x_shortcut = encoder_block(x, t)
            encoder_shortcuts.append(x_shortcut)
            if cond is not None:
                cond, cond_shortcut = self.cond_encoder_blocks[i](cond, t=None) 
                cond_encoder_shortcuts.append(cond_shortcut)
            if y_cond is not None:
                y_cond, y_cond_shortcut = self.y_cond_encoder_blocks[i](y_cond, t=None)
                y_cond_encoder_shortcuts.append(y_cond_shortcut)

        x = self.mid_block(x)
        if cond is not None:
            cond = self.cond_mid_block(cond)
            x = torch.cat([x, cond], dim=1)
            cond_encoder_shortcuts.reverse()
        if y_cond is not None:
            y_cond = self.y_cond_mid_block(y_cond)
            x = torch.cat([x, y_cond], dim=1)
            y_cond_encoder_shortcuts.reverse()

        encoder_shortcuts.reverse()

        for i, decoder_block in enumerate(self.decoder_blocks):
            shortcut = encoder_shortcuts[i]
            if cond is not None:
                cond_shortcut = cond_encoder_shortcuts[i]
                shortcut = torch.cat([shortcut, cond_shortcut], dim=1)
            if y_cond is not None:
                y_cond_shortcut = y_cond_encoder_shortcuts[i]
                shortcut = torch.cat([shortcut, y_cond_shortcut], dim=1)
            x = decoder_block(x, shortcut, t)

        x = self.final_conv(x)

        return x

    def _cal_channels(self, base_dim, dim_mults):
        dims = [base_dim * x for x in dim_mults]
        dims.insert(0, base_dim)
        channels = [(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        return channels


if __name__=="__main__":
    x=torch.randn(3,3,224,224)
    t=torch.randint(0,1000,(3,))
    model=Unet(1000,128)
    y=model(x,t)
    print(y.shape)
