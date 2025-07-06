<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
            <img src="figure/b9a58932a39eef8f8dffa29b18d59cc.png" alt="Icon" style="width:40px; vertical-align:middle; margin-right:10px;">      Mamba-Back   <img src="figure/b9a58932a39eef8f8dffa29b18d59cc.png" alt="Icon" style="width:40px; vertical-align:middle; margin-right:10px;">

<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
              Efficience-based-Recommandation-System</h1>   

<p align='center' style="text-align:center;font-size:1.25em;">
   <a href="https://github.com/RichardDuan-shandong" target="_blank" style="text-decoration: none;">WeiCheng Duan</a>, <sup>a,1</sup>, 
    <a href="https://github.com/tj-messi" target="_blank" style="text-decoration: none;">Junze Zhu</a>,<sup>a,1</sup>, 
     &nbsp;<br/>
    <sup>a</sup> <strong>The School of Computer Science and Technology Matched organization, Tongji University, China</strong><br/>
     
</p>


## 📣 Latest Updates


## 🧠 Key Takeaways


## 📝 About this code

## How to apply the work
### 1. Environment

- torch：2.3.1+cu118
- torchaudio：2.3.1+cu118
- torchvision：0.18.1+cu118
- triton：2.3.1
- transformers：4.43.3
- causal-conv1d：1.4.0
- mamba-ssm ：2.2.2
- cuda-nvcc：1.8.89

Solve the environment by doing : 

            conda create -n mamba python=3.10
            conda activate mamba
            
            pip install torch-2.3.1+cu118-cp310-cp310-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
            pip install torchvision-0.18.1+cu118-cp310-cp310-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
            pip install torchaudio-2.3.1+cu118-cp310-cp310-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple

            pip install causal_conv1d-1.4.0+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
            pip install mamba_ssm-2.2.2+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple

            pip install fuxictr==2.3.7 -i https://pypi.tuna.tsinghua.edu.cn/simple

            pip install triton==2.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
            pip install transformers==4.43.3 -i https://pypi.tuna.tsinghua.edu.cn/simple

            conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc

  And change the fuxictr-code in fuxictr/pytorch/model/rank_model.py in line 146:

            self._best_metric = np.inf if self._monitor_mode == "min" else -np.inf

### 2. Training

            # baseline run
            python run_expid.py --config config/DIN_microlens_mmctr_tuner_config_01 --expid DIN_MicroLens_1M_x1_002_1fa8d93d --gpu 0
            
            # ours run (batch_size=256)
            python run_expid.py --config config/Transformer_DCN_microlens_mmctr_tuner_config_01 --expid Transformer_DCN_MicroLens_1M_x1_001_820c435c --gpu 4
            
            # ours run (batch_size=128)
            python run_expid.py --config config/Transformer_DCN_microlens_mmctr_tuner_config_01 --expid Transformer_DCN_MicroLens_1M_x1_001_323m436f --gpu 0







  
