# The Outlier-Aware Quantization Algorithm for LLM

## Introduction

This repository contains quantization algorithm and the model evaluation code for Outlier-Aware Quantization method for LLM compression format. 

It is developed to perform accuracy verification for OA-CIM hardware accepted by JSSC. This work will be published soon.

The scripts can compress model through a Weight-Only method down to INT4 with inference performance preserved. For the weight elements which have significant influence on loss (outliers), they are saved as BF16 replacing the original four INT4. Moreover, this algorithm was also developed to minimize the number of 1 in quantized weight by adding a perchannel offset, in order to meet further demands of hardware.

## Environments

### Packages

Make sure there is `torch>=2.0.0` with `CUDA` support in current environment. 

You can simply install packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Models 

The scripts are expected to preform quantization and evaluation on models of `OPT` and `LLaMA` families so far.  

### W&B

For the sake of convenience, you can optionally log the data to `Weights and Biases` service (W&B). You can install W&B by:

```bash
pip install wandb
```

Set `$WANDB_ENTITY`, `$WANDB_PROJECT`, `$WANDB_NAME` environment variables before running the scripts.

## Launching

### Hardware requirements
The scripts was developed and evaluated on a single RTX3090 GPU with 30GB GPU RAM. It can successfully run on GPUs with 30GB+ RAM for quantization and evaluation of up to `OPT-13B` and `LLaMA-13B` models. 

### Model loading
For models and their tokenizers, loading and caching mechanism by Huggingface is involved, which means   that the models and tokenizers are downloaded from remote only once and cached in the default local path specified by package `huggingface_hub`. 

To downloading models successfully, network connectivity to `huggingface.com` should be ensured. For gated model families such as `LLaMA`, access rights need to be applied from the model developer in advance.

### Perplexity benchmarks
The scripts compresses the model and then evaluates its performance in terms of perplexity on different training and validation/testing subsets of WikiText2, C4, and Penn-Treebank datasets. 

The command to launch the scripts is shown below: 

```
export MODEL=<MODEL_PATH>
export DATASET=<DATASET_NAME_OR_PATH>

python main.py ${MODEL} ${DATASET}\
	--wbits 3 \ 
	--per_out_dim \
	--sym \
	--qq_zero_bits=4 \
	--qq_scale_bits=4 \
	--block_size=32 \
	--outlier_mode \
	--offset_mode
```
Note the launch arguments:
-  `${MODEL}` (str): remote URL on `huggingface_hub` or local path of model to load.
- `${DATASET}` (str): name [c4, pajama, refinedweb, none, etc.] or local path of calibration data.
-  `--model_dtype` (str, default="auto"): Model data type loaded. [auto/float16/float32]
- `--load` (str, default=None): Path to load quantized statistics.
- `--save` (str, default=False): Path to save quantized statistics.
-  `--baseline` (flag): Run with BF16 without quantization.
- `--nearest` (flag): Run simple RTN (round-to-nearest) quantization algorithm.
-  `--outlier_mode` (flag): Preserve outliers.
-  `--offset_mode` (flag): Use per-channel offset to minimize the number of 1 in quantized weight. Offset mode should only be set with symmetric quantization.
- `--wbits` (int, default=16): Weight quantization bits (16 for BF16 base model evaluation)
- `--block_size` (int, default=None): Number of weight columns quantized together. In a block of weights, one or no outlier per row is preserved.
- `--sym` (flag): Use symmetric quantization. In symmetric mode, the quantized weight elements are presented in $[-2^{B-1},\ 2^{B-1}]$,  otherwise in $[0,\ 2^{B-1}]$.
- `--per_out_dim` (flag): Use unique quantizer per output dimension.
- `--qq_scale_bits` (int, default=None): Bits for secondly-quantizing scale factors of weights.
- `--qq_zero_bits` (int, default=None): Bits for secondly-quantizing zero points of weights.
- `--qq_zero_sym` (flag): Use symmetric quantization on secondly-quantized zero points.
- `--qq_group_size` (int, default=16): Group size for secondly-quantization.
- `--wandb` (flag): Enable Weights and Biases logging.
-  `--seed` (int, default=0): Seed for randomly sampling calibration data.
-  `--nsamples` (int, default=128): Number of calibration data samples.
-  `--percdamp` (float, default=0.01): Dampening rate of Hessian diagonal average.
-  `--true-sequential` (flag): Run in true sequential mode.
-  `--offload_activations` (flag): Offload activations to RAM to save GPU memory.

## Evaluation Results

## OPT

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-9wq8">Model</th>
    <th class="tg-9wq8">Dataset</th>
    <th class="tg-9wq8">INT4</th>
    <th class="tg-9wq8">INT4+OUTLIER</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="3">OPT-1.3B</td>
    <td class="tg-9wq8">WikiText2</td>
    <td class="tg-9wq8">14.76</td>
    <td class="tg-9wq8">14.55</td>
  </tr>
  <tr>
    <td class="tg-9wq8">PTB</td>
    <td class="tg-9wq8">17.45</td>
    <td class="tg-9wq8">17.20</td>
  </tr>
  <tr>
    <td class="tg-9wq8">C4</td>
    <td class="tg-9wq8">14.98</td>
    <td class="tg-9wq8">14.89</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="3">OPT-2.7B</td>
    <td class="tg-9wq8">WikiText2</td>
    <td class="tg-9wq8">12.40</td>
    <td class="tg-9wq8">12.24</td>
  </tr>
  <tr>
    <td class="tg-9wq8">PTB</td>
    <td class="tg-9wq8">15.48</td>
    <td class="tg-9wq8">15.30</td>
  </tr>
  <tr>
    <td class="tg-9wq8">C4</td>
    <td class="tg-9wq8">13.36</td>
    <td class="tg-9wq8">13.28</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="3">OPT-6.7B</td>
    <td class="tg-9wq8">WikiText2</td>
    <td class="tg-9wq8">10.91</td>
    <td class="tg-9wq8">10.89</td>
  </tr>
  <tr>
    <td class="tg-9wq8">PTB</td>
    <td class="tg-9wq8">13.27</td>
    <td class="tg-9wq8">13.13</td>
  </tr>
  <tr>
    <td class="tg-9wq8">C4</td>
    <td class="tg-9wq8">11.85</td>
    <td class="tg-9wq8">11.80</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="3">OPT-13B</td>
    <td class="tg-9wq8">WikiText2</td>
    <td class="tg-9wq8">10.15</td>
    <td class="tg-9wq8">10.13</td>
  </tr>
  <tr>
    <td class="tg-9wq8">PTB</td>
    <td class="tg-9wq8">12.42</td>
    <td class="tg-9wq8">12.40</td>
  </tr>
  <tr>
    <td class="tg-9wq8">C4</td>
    <td class="tg-9wq8">11.25</td>
    <td class="tg-9wq8">11.24</td>
  </tr>
</tbody>
</table>

## Llama

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-9wq8">Model</th>
    <th class="tg-9wq8">Dataset</th>
    <th class="tg-9wq8">INT4</th>
    <th class="tg-9wq8">INT4+OUTLIER</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="3">LLaMA-7B</td>
    <td class="tg-9wq8">WikiText2</td>
    <td class="tg-9wq8">5.67</td>
    <td class="tg-9wq8">5.56</td>
  </tr>
  <tr>
    <td class="tg-9wq8">PTB</td>
    <td class="tg-9wq8">21.59</td>
    <td class="tg-9wq8">21.43</td>
  </tr>
  <tr>
    <td class="tg-9wq8">C4</td>
    <td class="tg-9wq8">7.15</td>
    <td class="tg-9wq8">7.06</td>
  </tr>
</tbody>
</table>

## References

- Dettmers, T., Svirschevski, R., Egiazarian, V., Kuznedelev, D., Frantar, E., Ashkboos, S., Borzunov, A., Hoefler, T., & Alistarh, D. (2023). SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression. *arXiv preprint arXiv:2306.03078*. https://doi.org/10.48550/arXiv.2306.03078
- Guo, C., Tang, J., Hu, W., Leng, J., Zhang, C., Yang, F., Liu, Y., Guo, M., & Zhu, Y. (2023). OliVe: Accelerating Large Language Models via Hardware-friendly Outlier-Victim Pair Quantization. *arXiv preprint arXiv:2304.07493*. https://doi.org/10.48550/arXiv.2304.07493 
