# 1st Place Solution of WWW 2025 EReL@MIR Workshop Multimodal CTR Prediction Challenge from     Team momo

We have released our model checkpoint in [Huggingface](https://huggingface.co/pinskyrobin/WWW2025_MMCTR_momo).

Our technical report is available at [arXiv](https://arxiv.org/abs/2505.03543).

The WWW 2025 Multimodal CTR Prediction Challenge: https://www.codabench.org/competitions/5372/

The overall architecture of our solution is shown below.

![momo](img/momo.png)

Our model (without future work part) achieves a **0.9839** AUC on the test set.

## Notes & Future Work

Our model is roughly based on **Transformer** and **DCNv2**. 
Two future works can also be seen in the right part of the figure.
- **Semantic embeddings with quantization** (Done, tuning)
  - Due to the success of quantization in computer vision, we believe that quantization can also be applied to multimodal RecSys.
**Vector Quantization** (VQ) and **Residual Quantization** (RQ) are used to quantize the original multimodal embeddings.
Quantization transforms the freezed multimodal embeddings into discrete learnable semantic codes.
  - Codes for quantization are provided in `src/Transformer_DCN_Quant.py` and the tuning work is still in progress.
- **Semantic similarity scores as part of the input of Transformer** (In progress)
  - Multimodal embeddings contain rich semantic information, and users have specific preferences for different semantic information.
Semantic similarity information can be explicitly utilized in our model for better performance.
  - We plan to use **semantic similarity scores** as part of the input of the Transformer part.


Both of the two future works are dedicated to utilizing the semantic information of the multimodal representations.

We believe that these two directions have huge potential.
And works on these two directions **are still ongoing** even after the challenge.

## Environment

We run the experiments on a customized 4080 Super GPU server with 32GB memory (vGPU-32G) from [AutoDL](https://www.autodl.com/).

Requirements:

- fuxictr==2.3.7
- numpy==1.26.4
- pandas==2.2.3
- scikit_learn==1.4.0
- torch==1.13.1+cu117

Environment setup:

```bash
conda create -n fuxictr_momo python==3.9
pip install -r requirements.txt
source activate fuxictr_momo
```

## How to Run

### One-click run

```bash
sh ./run.sh
```

This script will run the whole pipeline, including model training and prediction.

### Run step by step

1. Train the model on train and validation sets:

    ```bash
    python run_expid.py --config config/Transformer_DCN_microlens_mmctr_tuner_config_01 --expid Transformer_DCN_MicroLens_1M_x1_001_820c435c --gpu 0
    ```
    
    We got the best validation AUC: 0.976603.

2. Make predictions on the test set:

    ```bash
    python prediction.py --config config/Transformer_DCN_microlens_mmctr_tuner_config_01 --expid Transformer_DCN_MicroLens_1M_x1_001_820c435c --gpu 0
    ```

3. Submission result on [the leaderboard](https://www.codabench.org/competitions/5372/#/results-tab).

   In particular, we got a 0.9814 result (Submission ID: 246140) for un-tuned Residual Quantization (RQ) model (part of future work 1).

4. ![result](img/result.png)