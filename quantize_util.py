import torch
import torch.nn as nn


def quantize_dequantize(x, scale, zero, maxq, sym, eps=1e-9):
    if sym:
        q=torch.clamp(torch.round(x/scale.clamp_min(eps)+zero), -int(maxq)//2, int(maxq)//2)
    else:
        q=torch.clamp(torch.round(x/scale.clamp_min(eps)+zero), 0, maxq)
    return scale*(q-zero)


def dequantize(x, scale, zero, eps=1e-9):
    return scale*(x-zero)

# return the literal value of int
def quantize(x, scale, zero, maxq, sym, eps=1e-9):
    if sym:
        q=torch.clamp(torch.round(x/scale.clamp_min(eps)+zero), -int(maxq)//2, int(maxq)//2)
    else:
        q=torch.clamp(torch.round(x/scale.clamp_min(eps)+zero), 0, maxq)
    return q

class Quantizer(nn.Module):
    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits,
        perchannel=False,
        sym=True,
        round_zero: bool=False,
        qq_scale_bits=None,
        qq_zero_bits=None,
        qq_group_size=16,
        qq_zero_sym=False,
        reserved_bins: int=0,
        qqq_params=None
    ):

        self.bits=bits
        self.maxq=torch.tensor(2**bits-1-reserved_bins)
        self.perchannel=perchannel
        self.sym=sym
        self.round_zero=round_zero

        self.qq_scale_bits=qq_scale_bits
        self.qq_zero_bits=qq_zero_bits
        self.qq_zero_sym=qq_zero_sym
        self.qq_group_size=qq_group_size
        self.qqq_params=qqq_params or {}

    def find_params(self, 
        x: torch.Tensor, 
        xmax: torch.Tensor=torch.tensor([]), 
        xmin: torch.Tensor=torch.tensor([]), 
        weight: bool=False
    ):
        dev=x.device
        self.maxq=self.maxq.to(dev)
        maybe_round_zero=torch.round if self.round_zero else lambda x: x

        shape=x.shape
        if self.perchannel:
            if weight:
                x=x.flatten(1)
            else:
                if len(shape)==4:
                    x=x.permute([1, 0, 2, 3])
                    x=x.flatten(1)
                if len(shape)==3:
                    x=x.reshape((-1, shape[-1])).t()
                if len(shape)==2:
                    x=x.t()
        else:
            x=x.flatten().unsqueeze(0)

        if xmin.numel()>0:
            xmin=xmin.reshape_as(x[:,0])
        else:
            xmin=x.min(dim=1).values
        
        if xmax.numel()>0:
            xmax=xmax.reshape_as(x[:,0])
        else:
            xmax=x.max(dim=1).values


        # [xmin, xmax]->[0, maxq]
        if self.sym:
            self.scale=(xmax-xmin)/self.maxq
            self.zero=maybe_round_zero(-(xmax+xmin)/(self.scale*2))
        else:
            self.scale=(xmax-xmin)/self.maxq
            self.zero=maybe_round_zero(-xmin/self.scale)

        xmin_eq_xmax_index=xmin==xmax   
        self.zero[xmin_eq_xmax_index]=1
        self.scale[xmin_eq_xmax_index]=xmin[xmin_eq_xmax_index]


        if not self.perchannel:
            repeat_size=shape[0] if weight else shape[1] if len(shape)!=3 else shape[2]
            self.scale=self.scale.repeat(repeat_size)
            self.zero=self.zero.repeat(repeat_size)
        
        if self.qq_scale_bits is not None:
            scale_groups=self.scale.reshape(-1, self.qq_group_size)
            self.qq_scale=Quantizer(shape=scale_groups.shape)
            self.qq_scale.configure(self.qq_scale_bits, perchannel=True, sym=False, round_zero=False, **self.qqq_params)
            self.qq_scale.find_params(scale_groups, weight=True)
            assert self.qq_scale.scale.shape==(scale_groups.shape[0], 1), self.qq_scale.scale.shape
            self.quant_scale=self.qq_scale.quantize(scale_groups)
            self.scale=self.qq_scale.dequantize(self.quant_scale).reshape_as(self.scale)

        if self.qq_zero_bits is not None and ((not self.round_zero) or self.qq_zero_bits<=self.bits):
            zero_groups=self.zero.reshape(-1, self.qq_group_size)
            self.qq_zero=Quantizer(shape=zero_groups.shape)
            self.qq_zero.configure(
                self.qq_zero_bits, perchannel=True, sym=self.qq_zero_sym, round_zero=False, **self.qqq_params
            )
            self.qq_zero.find_params(zero_groups, weight=True)
            assert self.qq_zero.scale.shape==(zero_groups.shape[0], 1), self.qq_zero.scale.shape
            self.quant_zero=self.qq_zero.quantize(zero_groups)
            self.zero=self.qq_zero.dequantize(self.quant_zero).reshape_as(self.zero)

        if weight:
            shape=[-1]+[1]*(len(shape)-1)
            self.scale=self.scale.reshape(shape)
            self.zero=self.zero.reshape(shape)
            return
       
        if len(shape)==4:
            self.scale=self.scale.reshape((1, -1, 1, 1))
            self.zero=self.zero.reshape((1, -1, 1, 1))
        elif len(shape)==3:
            self.scale=self.scale.reshape((1, 1, -1))
            self.zero=self.zero.reshape((1, 1, -1))
        elif len(shape)==2:
            self.scale=self.scale.unsqueeze(0)
            self.zero=self.zero.unsqueeze(0)




    def quantize_dequantize(self, x, offset=0):
        x_shape=x.shape
        if(len(x.shape)<len(self.scale.shape)):
            x=x.reshape(x.shape+(1,)*(len(self.scale.shape)-len(x.shape)))
        if self.ready():
            return quantize_dequantize(x, self.scale, self.zero+offset, self.maxq, self.sym).reshape(x_shape)
        return x


    def quantize(self, x, offset=0):
        x_shape=x.shape
        if(len(x.shape)<len(self.scale.shape)):
            x=x.reshape(x.shape+(1,)*(len(self.scale.shape)-len(x.shape)))
        if self.ready():
            return quantize(x, self.scale, self.zero+offset, self.maxq, self.sym).reshape(x_shape)
        return x

    def dequantize(self, x, offset=0):
        x_shape=x.shape
        if(len(x.shape)<len(self.scale.shape)):
            x=x.reshape(x.shape+(1,)*(len(self.scale.shape)-len(x.shape)))
        if self.ready():
            return dequantize(x, self.scale, self.zero+offset).reshape(x_shape)
        return x

    def enabled(self):
        return self.maxq>0

    def ready(self):
        return torch.all(self.scale!=0)
    