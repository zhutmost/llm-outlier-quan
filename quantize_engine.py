from __future__ import annotations

import math
from typing import NamedTuple, Optional, Union

import torch
from tqdm.auto import tqdm
torch.set_printoptions(profile="full")

from quantize_util import Quantizer, dequantize, quantize



class QuantizeEngine:
    """Learns GPTQ for a single linear layer"""

    def __init__(self, layer):
        self.layer=layer
        self.dev=layer.weight.device
        self.columns=self.layer.weight.data.shape[1]
        self.h=torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples=0


    def add_batch(self, inp):
        assert self.h is not None, "Already ran quantization; cannot add more data batches"
        if len(inp.shape)==2:
            inp=inp.unsqueeze(0)
        tmp=inp.shape[0]

        if len(inp.shape)==3:
            inp=inp.reshape((-1, inp.shape[-1]))
        inp=inp.t()

        self.h*=self.nsamples/(self.nsamples+tmp)
        self.nsamples+=tmp
        inp=math.sqrt(2/self.nsamples)*inp.float()
        self.h+=inp.matmul(inp.t())

    def quantize(
        self,
        *,
        bits: int=2,
        block_size: int=128,
        qq_group_size: int=16,
        keep_last_columns: int=0,
        per_out_dim: bool=True,
        sym: bool=False,
        save_quant: bool=False,
        percdamp: float=1e-2,
        outlier_mode: bool=False,
        offset_mode: bool=False,
        **kwargs
    ) -> QuantizationResult:
        """
        :param bits: number of bits used at the lowest level (the full model size will be different!)
        :param block_size: take blocks of this many input features at a time for GPTQ
        :param qq_group_size: take groups of this many first-quantized scales and zeros at a time for second-quantization
        :param keep_last_columns: if not None, keep the last (this many) input features un_quantized and return them
        :param per_out_dim: if True, base weight quantization will learn statistics for each output dimension separately
        :param sym: if True, base weight quantization is symmetric
        :param save_quant: if True, save the quantized weights and the mask used for the quantization
        :param percdamp: relative regularizer added to hessian diagonal before inversion
        :param outlier_mode: if True, preserve BF16 outliers with low-bit-width normallers
        :param offset_mode: if True, add an offset to each channel to minimize the number of 1
        :return: weight, mask, save_dict 
        """
        weight=self.layer.weight.detach().to(dtype=torch.float, copy=True)
        error=torch.zeros_like(weight)
        mask=torch.zeros_like(weight, dtype=torch.bool)
        h=self.h 
        save_dict={
            "weight": torch.zeros_like(weight, dtype=torch.bfloat16, device=self.dev),
            "q_weight": torch.zeros_like(weight, dtype=torch.bfloat16, device=self.dev),
            "q_mask": torch.zeros_like(weight, dtype=torch.bool, device=self.dev),
            "q_sacle": torch.zeros((weight.shape[0], weight.shape[1]//block_size), dtype=torch.bfloat16, device=self.dev),
            "q_zero": torch.zeros((weight.shape[0], weight.shape[1]//block_size), dtype=torch.bfloat16, device=self.dev),
            "qq_scale_scale": torch.zeros((weight.shape[0]//qq_group_size, weight.shape[1]//block_size), dtype=torch.bfloat16, device=self.dev),
            "qq_scale_zero": torch.zeros((weight.shape[0]//qq_group_size, weight.shape[1]//block_size), dtype=torch.bfloat16, device=self.dev),
            "qq_zero_zero": torch.zeros((weight.shape[0]//qq_group_size, weight.shape[1]//block_size), dtype=torch.bfloat16, device=self.dev),
            "qq_zero_scale": torch.zeros((weight.shape[0]//qq_group_size, weight.shape[1]//block_size), dtype=torch.bfloat16, device=self.dev),
            "offset": torch.zeros(weight.shape[1], dtype=torch.bfloat16, device=self.dev)
        }
        
        # initial weight permutation by quantization difficulty
        if outlier_mode:    
            block_permute=h.diag().reshape(-1, block_size).abs().max(dim=-1).values.sort(dim=0, descending=True).indices
            block_permute=((block_permute*block_size).unsqueeze(1).repeat(1, block_size)+torch.arange(block_size, device=self.dev)).flatten()
            block_permute_inv=block_permute.sort(dim=0).indices
            weight=weight[:, block_permute]
            h=h[block_permute][:, block_permute]

        # initial hessian up-triangular matrix for leave-one-out iteration
        self.dead=torch.diag(h)==0
        if percdamp>0:
            ix=torch.arange(len(h), device=self.dev)
            h[ix, ix]+=percdamp*abs(torch.diag(h)).mean()
            del ix
        h[self.dead, self.dead]=1
        weight[:, self.dead]=0
        hinvcho=torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
        del h

        # initial quantizer
        quantizer=Quantizer()
        quantizer.configure(bits, per_out_dim=per_out_dim, sym=sym, round_zero=True, **kwargs)
       
        
        # initial dimension and iteration-index information
        assert hinvcho.shape[0]==hinvcho.shape[1]==weight.shape[1], "weight must be [out_features, in_features]"
        if block_size is None:
            block_size=weight.shape[1]
        block_start_iter=range(0, weight.shape[1]-keep_last_columns, block_size)
        block_start_iter=tqdm(block_start_iter, leave=False)

        for block_begin in block_start_iter:
            block_end=min(block_begin+block_size, weight.shape[1])
            block_weight=weight[:, block_begin: block_end].clone()
            block_hinvcho=hinvcho[block_begin: block_end, block_begin: block_end].clone()
            
            def quantize_block(
                capacity_offset: torch.Tensor=torch.tensor([]), 
                save_quant: bool=False
            ):
                if capacity_offset.numel()==0:
                    capacity_offset=torch.zeros(block_size, device=self.dev)

                if outlier_mode:   
                    block_weight_left, block_weight_right=get_minmax_truncate(
                        block_weight.repeat(block_size+1, 1), 
                        block_hinvcho.diag(), 
                        bits, 
                        sym
                    )
                    quantizer.find_params(
                        block_weight.repeat(block_size+1, 1), 
                        xmin=block_weight_left,
                        xmax=block_weight_right,
                        weight=True
                    )
                else:
                    quantizer.find_params(
                        block_weight.repeat(block_size+1, 1), 
                        weight=True
                    )

                # block=>capacity
                capacity_size=block_size
                capacity_weight=block_weight.unsqueeze(0).repeat(capacity_size+1, 1, 1)
                capacity_weight_quant=torch.zeros_like(capacity_weight, device=self.dev, dtype=torch.int16)
                capacity_error=torch.zeros_like(capacity_weight, dtype=torch.float)
                capacity_hinvcho=block_hinvcho.unsqueeze(0).repeat(capacity_size+1, 1, 1)
                capacity_mask=(torch.arange(-1, capacity_size, device=self.dev).unsqueeze(1)==torch.arange(capacity_size, device=self.dev).unsqueeze(0)).unsqueeze(1).repeat(1, weight.shape[0], 1)
                capacity_mask_victim=mask_filter_victim(capacity_mask.flatten(0, 1)).reshape(capacity_size+1, -1, capacity_size)
                if outlier_mode:
                    capacity_permute=torch.arange(capacity_size, device=self.dev).unsqueeze(0).repeat(capacity_size+1, 1)
                    for i in range(0, capacity_size, 4):
                        capacity_permute[i+1, :4]=torch.tensor((i, i+1, i+2, i+3), device=self.dev)
                        capacity_permute[i+2, :4]=torch.tensor((i+1, i, i+2, i+3), device=self.dev)
                        capacity_permute[i+3, :4]=torch.tensor((i+2, i, i+1, i+3), device=self.dev)
                        capacity_permute[i+4, :4]=torch.tensor((i+3, i, i+1, i+2), device=self.dev)
                        capacity_permute[(i+1):(i+5), 4:(i+4)]=torch.arange(i, device=self.dev)
                    capacity_permute_inv=capacity_permute.sort(dim=1).indices
                    for i in range(capacity_size+1):
                        capacity_weight[i]=capacity_weight[i][:, capacity_permute[i]]
                        capacity_hinvcho[i]=capacity_hinvcho[i][capacity_permute[i]][:, capacity_permute[i]]
                        capacity_error[i]=capacity_error[i][:, capacity_permute[i]]
                        capacity_mask[i]=capacity_mask[i][:, capacity_permute[i]]
                        capacity_mask_victim[i]=capacity_mask_victim[i][:, capacity_permute[i]]              
                # capacity_size=>column
                for column_index in range(capacity_size):
                    column_weight=capacity_weight[..., column_index]
                    column_weight_quant=capacity_weight_quant[..., column_index]
                    column_error=capacity_error[..., column_index]
                    column_hinvcho=capacity_hinvcho[:, column_index, column_index]
                    column_mask=capacity_mask[..., column_index]
                    column_mask_victim=capacity_mask_victim[..., column_index]
                    column_offset=capacity_offset[column_index]
                    column_weight_quant_dequant=quantizer.quantize(column_weight.flatten(), column_offset)
                    column_weight_quant[:]=column_weight_quant_dequant.reshape(capacity_size+1, -1)
                    column_weight_quant_dequant=quantizer.dequantize(column_weight_quant_dequant, column_offset)
                    column_weight_quant_dequant=torch.where(column_mask.flatten(), column_weight.flatten(), column_weight_quant_dequant)
                    column_weight_quant_dequant=torch.where(column_mask_victim.flatten(), 0, column_weight_quant_dequant)
                    column_error[:]=(column_weight-column_weight_quant_dequant.reshape(capacity_size+1, -1))/column_hinvcho.unsqueeze(1)
                    column_weight[:]=column_weight_quant_dequant.reshape(capacity_size+1, -1)
                    capacity_weight[..., (column_index+1):]-=\
                        column_error.unsqueeze(2)*capacity_hinvcho[:, column_index, (column_index+1):].unsqueeze(1)
                
                if outlier_mode:
                    capacity_index=capacity_error.square().sum(dim=2).min(dim=0).indices 
                    for i in range(capacity_size+1):
                        capacity_weight[i]=capacity_weight[i, :, capacity_permute_inv[i]]
                        capacity_error[i]=capacity_error[i, :, capacity_permute_inv[i]]
                        capacity_mask[i]=capacity_mask[i, :, capacity_permute_inv[i]]
                else:
                    capacity_index=0
                
                block_weight_result=capacity_weight[capacity_index, torch.arange(weight.shape[0], device=self.dev), :]
                block_weight_quant_result=capacity_weight_quant[capacity_index, torch.arange(weight.shape[0], device=self.dev), :]  
                mask[:, block_begin: block_end]=capacity_mask[capacity_index, torch.arange(weight.shape[0], device=self.dev), :]
                error[:, block_begin: block_end]=capacity_error[capacity_index, torch.arange(weight.shape[0], device=self.dev), :]
                if save_quant:
                    save_dict["weight"][:, block_begin: block_end]=block_weight_result
                    save_dict["q_weight"][:, block_begin: block_end]=block_weight_quant_result
                    save_dict["q_mask"][:, block_begin: block_end]=mask[:, block_begin: block_end]
                    if hasattr(quantizer, "qq_scale"):
                        save_dict["q_sacle"][:, block_begin//block_size]=quantizer.qq_scale.quantize(quantizer.scale.reshape(-1, qq_group_size)).flatten()[:weight.shape[0]]
                        save_dict["qq_scale_scale"][:, block_begin//block_size]=quantizer.qq_scale.scale.flatten()[:weight.shape[0]//qq_group_size]
                        save_dict["qq_scale_zero"][:, block_begin//block_size]=quantizer.qq_scale.zero.flatten()[:weight.shape[0]//qq_group_size]
                    else:
                        save_dict["q_scale"][:, block_begin//block_size]=quantizer.scale[:weight.shape[0]]
                        save_dict["qq_scale_scale"]=None
                        save_dict["qq_scale_zero"]=None
                    if hasattr(quantizer, "qq_zero"):
                        save_dict["q_zero"][:, block_begin//block_size]=quantizer.qq_zero.quantize(quantizer.zero.reshape(-1, qq_group_size)).flatten()[:weight.shape[0]]
                        save_dict["qq_zero_scale"][:, block_begin//block_size]=quantizer.qq_zero.scale.flatten()[:weight.shape[0]//qq_group_size]
                        save_dict["qq_zero_zero"][:, block_begin//block_size]=quantizer.qq_zero.zero.flatten()[:weight.shape[0]//qq_group_size]
                    else:
                        save_dict["q_zero"][:, block_begin//block_size]=quantizer.zero[:weight.shape[0]]
                        save_dict["qq_zero_scale"]=None
                        save_dict["qq_zero_zero"]=None
                    if offset_mode:
                        save_dict["offset"][block_begin: block_end]=capacity_offset
                    else:
                        save_dict["offset"]=None
                return block_weight_result, block_weight_quant_result

            if offset_mode: 
                block_weight_result, block_weight_quant_result=quantize_block()
                capacity_offset=get_offset(block_weight_quant_result, 4)
                if not outlier_mode:
                    capacity_offset=capacity_offset.clamp(min=-1, max=1)
                weight[:, block_begin: block_end], _=quantize_block(capacity_offset, save_quant)
            else:
                weight[:, block_begin: block_end], _=quantize_block(torch.tensor([]), save_quant)
        
            weight[:, block_end:].addmm_(
                error[:, block_begin: block_end],
                hinvcho[block_begin: block_end, block_end:], 
                alpha=-1
            )

        if outlier_mode:
            weight=weight[:, block_permute_inv]
            error=error[:, block_permute_inv]
            mask=mask[:, block_permute_inv]

        return weight, mask, save_dict


def get_minmax_truncate(weight: torch.Tensor, hinvcho_diag: torch.Tensor, bits, sym, max_truncate=1):
    """EXPERIMENTAL! BEWARE - for each weight, fit quantizer without this_one_weight and return this one weight's reconstruction"""
    block_weight_sort=weight.sort(dim=-1).values
    block_hinvcho_diag_sort=hinvcho_diag.unsqueeze(dim=0).repeat(weight.shape[0], 1).gather(dim=-1, index=weight.sort(dim=-1).indices)
    quant_minmax_error=((block_weight_sort[:,-1]-block_weight_sort[:,0]).unsqueeze(dim=1)/((2**(bits+1)-2)*block_hinvcho_diag_sort)).square().sum(dim=-1)
    quant_truncate_mask=torch.zeros(weight.shape[-1], dtype=torch.bool, device=weight.device)
    quant_truncate_error=torch.zeros(weight.shape[0], dtype=torch.float, device=weight.device)
    quant_truncate_index_right=torch.zeros(weight.shape[0], dtype=torch.long, device=weight.device)+weight.shape[-1]
    quant_truncate_index_left=torch.zeros(weight.shape[0], dtype=torch.long, device=weight.device)-1
    quant_truncate_loop_mask=quant_truncate_mask.clone()
    quant_truncate_loop_index=weight.shape[-1]-1
    while not quant_truncate_mask.all() and quant_truncate_loop_index>=weight.shape[-1]-max_truncate:
        quant_truncate_loop_mask[quant_truncate_loop_index]=True
        quant_truncate_error=torch.where(quant_truncate_loop_mask,
            (block_weight_sort-block_weight_sort[:, quant_truncate_loop_index-1].unsqueeze(dim=1))*2/block_hinvcho_diag_sort,
            (block_weight_sort[:, quant_truncate_loop_index-1]-block_weight_sort[:, 0]).unsqueeze(dim=1)/((2**(bits+1)-2)*block_hinvcho_diag_sort)
        ).square().sum(dim=-1)
        quant_truncate_mask=quant_truncate_error>=quant_minmax_error
        quant_truncate_index_right=torch.where((quant_truncate_error>=quant_minmax_error)|quant_truncate_mask,
            quant_truncate_index_right,
            quant_truncate_loop_index
        )
        quant_truncate_loop_index-=1
    quant_truncate_mask=torch.zeros(weight.shape[-1], dtype=torch.bool, device=weight.device)
    quant_truncate_loop_mask=quant_truncate_mask.clone()
    quant_truncate_loop_index=0
    while not quant_truncate_mask.all() and quant_truncate_loop_index<max_truncate:
        quant_truncate_loop_mask[quant_truncate_loop_index]=True
        quant_truncate_error=torch.where(quant_truncate_loop_mask,
            (block_weight_sort[:, quant_truncate_loop_index+1].unsqueeze(dim=1)-block_weight_sort)*2/block_hinvcho_diag_sort,
            (block_weight_sort[:, -1]-block_weight_sort[:, quant_truncate_loop_index+1]).unsqueeze(dim=1)/((2**(bits+1)-2)*block_hinvcho_diag_sort)
        ).square().sum(dim=-1)
        quant_truncate_mask=quant_truncate_error>=quant_minmax_error
        quant_truncate_index_left=torch.where((quant_truncate_error>=quant_minmax_error)|quant_truncate_mask,
            quant_truncate_index_left,
            quant_truncate_loop_index
        )
        quant_truncate_loop_index+=1
    return block_weight_sort.gather(dim=-1, index=quant_truncate_index_left.unsqueeze(dim=1)+1).flatten(), block_weight_sort.gather(dim=-1, index=quant_truncate_index_right.unsqueeze(dim=1)-1).flatten() 

def mask_filter_victim(mask: torch.Tensor):
    mask_seg=mask.reshape(mask.shape[0], mask.shape[1]//4, 4).any(dim=2)
    return mask_seg.unsqueeze(2).repeat(1, 1, 4).reshape_as(mask)&(~mask)
    
def get_offset(weight, bits: int=4):
    one_index=torch.arange(1-2**(bits-1), 2**(bits-1), device=weight.device, dtype=torch.long)
    one_num=one_index.unsqueeze(1)+one_index.unsqueeze(0)
    one_num=((((one_num.clamp(min=1-2**(bits-1), max=2**(bits-1)-1)+(2**bits))%(2**bits)).unsqueeze(2)//(2**torch.arange(bits-1, device=weight.device).reshape(1, 1, -1)))%2).sum(dim=-1)+(one_num<0).to(torch.long)
    one_counter=((weight.unsqueeze(2)==one_index.reshape(1, 1, -1)).to(torch.long).sum(dim=0).unsqueeze(2)*one_num.unsqueeze(0)).sum(dim=1)
    one_counter=2**(bits-1)-1-one_counter.topk(3, dim=1, largest=False).indices
    return one_counter[torch.arange(weight.shape[1], device=weight.device), one_counter.abs().min(dim=1).indices]
