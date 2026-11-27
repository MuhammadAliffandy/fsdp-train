"""Matched four-level plain/C8 U-Nets with native five-slice volume reconstruction."""
import torch
from torch import nn
from torch.nn import functional as F
from physical_unet25d import slice_stack


class BrainUNet25D(nn.Module):
    def __init__(self, backbone='c8', base_fields=2, chunk_size=4):
        super().__init__()
        if backbone not in ('plain4','c8') or not 1 <= base_fields <= 32:
            raise ValueError('Expected plain4/c8 and1..32regular fields')
        self.backbone, self.neighbor_slices, self.chunk_size = backbone, 5, chunk_size
        widths = [base_fields*2**i for i in range(5)]
        self.encoder = nn.ModuleList(); self.decoder = nn.ModuleList(); self.heads = nn.ModuleList()
        if backbone == 'c8':
            from escnn import gspaces, nn as enn
            self.enn = enn
            space = gspaces.rot2dOnR2(N=8)
            self.input_type = enn.FieldType(space, [space.trivial_repr]*5)
            types = [enn.FieldType(space,[space.regular_repr]*c) for c in widths]
            output = enn.FieldType(space,[space.trivial_repr])
            def block(a,b):
                return enn.SequentialModule(enn.R2Conv(a,b,3,padding=1,bias=False), enn.InnerBatchNorm(b), enn.ReLU(b),
                    enn.R2Conv(b,b,3,padding=1,bias=False), enn.InnerBatchNorm(b), enn.ReLU(b))
            self.encoder.extend([block(self.input_type,types[0]), *[block(types[i-1],types[i]) for i in range(1,5)]])
            self.pools = nn.ModuleList([enn.PointwiseMaxPool(t,2) for t in types[:-1]])
            self.ups = nn.ModuleList([enn.R2Upsampling(types[i],scale_factor=2,mode='bilinear',align_corners=False) for i in range(4,0,-1)])
            self.decoder.extend([block(types[i]+types[i-1],types[i-1]) for i in range(4,0,-1)])
            self.heads.extend([enn.R2Conv(t,output,1) for t in types[:3]])
        else:
            channels = [8*c for c in widths]
            def block(a,b):
                return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU(),
                    nn.Conv2d(b,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU())
            self.encoder.extend([block(5,channels[0]), *[block(channels[i-1],channels[i]) for i in range(1,5)]])
            self.decoder.extend([block(channels[i]+channels[i-1],channels[i-1]) for i in range(4,0,-1)])
            self.heads.extend([nn.Conv2d(c,1,1) for c in channels[:3]])

    def plane(self, image):
        shape = image.shape[-2:]
        # Padding to multiples of16is identical in training and inference; no XY resize.
        padded = [max(32, ((n+15)//16)*16) for n in shape]
        value = F.pad(image,(0,padded[1]-shape[1],0,padded[0]-shape[0]))
        if self.backbone == 'c8':
            value = self.enn.GeometricTensor(value,self.input_type)
        features=[]
        for index,block in enumerate(self.encoder):
            if index:
                value = self.pools[index-1](value) if self.backbone=='c8' else F.max_pool2d(value,2)
            value=block(value);features.append(value)
        outputs={}
        for step,(index,block) in enumerate(zip(range(3,-1,-1),self.decoder)):
            if self.backbone=='c8':
                value=self.ups[step](value)
                value=block(self.enn.tensor_directsum([value,features[index]]))
            else:
                value=F.interpolate(value,size=features[index].shape[-2:],mode='bilinear',align_corners=False)
                value=block(torch.cat((value,features[index]),1))
            if index<3:
                logits=self.heads[index](value)
                logits=logits.tensor if self.backbone=='c8' else logits
                logits=F.interpolate(logits,size=features[0].shape[-2:],mode='bilinear',align_corners=False)
                outputs[index]=logits[...,:shape[0],:shape[1]]
        return tuple(outputs[i] for i in range(3))

    def forward(self,image):
        outputs=[[],[],[]]
        for start in range(0,image.shape[-1],self.chunk_size):
            batch=slice_stack(image,range(start,min(start+self.chunk_size,image.shape[-1])),5)
            for collection,value in zip(outputs,self.plane(batch)):
                collection.append(value)
        return tuple(torch.cat(values,0).permute(1,2,3,0)[None] for values in outputs)

    def state_dict(self,*args,**kwargs):
        mode=self.training
        if self.backbone=='c8':
            self.train()  # ESCNN eval caches expanded filters; omit transient caches from checkpoints.
        result=super().state_dict(*args,**kwargs)
        if self.backbone=='c8':
            self.train(mode)
        return result

    def load_state_dict(self,state_dict,strict=True,**kwargs):
        mode=self.training
        if self.backbone=='c8':
            self.train()
        result=super().load_state_dict(state_dict,strict=strict,**kwargs)
        if self.backbone=='c8':
            self.train(mode)
        return result
