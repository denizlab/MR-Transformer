# MR-Transformer: A Vision Transformer-based Deep Learning Model for Total Knee Replacement Prediction Using MRI

## Introduction

This repo contains the implementation of MR-Transformer.

## Data

This study uses data from two publicly available longitudinal cohort studies:

- **Osteoarthritis Initiative (OAI):** [https://nda.nih.gov/oai](https://nda.nih.gov/oai)

- **Multicenter Osteoarthritis Study (MOST):** [https://most.ucsf.edu](https://most.ucsf.edu)

### Case-Control Selection
- **OAI:** 353 case-control pairs were identified.

- **MOST:** 270 case-control pairs were identified.

Each case participant was matched to a control participant of the same sex, ethnicity, and age, with an additional constraint on the baseline body mass index within a 10% tolerance.

### MRI Sequences

| Dataset | MRI Tissue Contrast | TE (ms) | TR (ms) | TI (ms) | FOV (mm) | ST (mm) | ISR (mm×mm) | Bandwidth (Hz/pixel) | Input Matrix Size | Prediction Task |
|:-------:|:-------------------:|:-------:|:-------:|:-------:|:--------:|:-------:|:-----------:|:--------------------:|:-----------------:|:---------------:|
|OAI|COR-IW-TSE|29|3700|N/A|140|3.0|0.36×0.36|352|36×384×384|TKR in 9 years|
|OAI|SAG-IW-TSE-FS|30|3200|N/A|160|3.0|0.36×0.37|248|36×448×448|TKR in 9 years|
|MOST|COR-STIR|35|4800|100|140|3.0|0.55×0.72|Not Specified|36×256×256|TKR in 7 years|
|MOST|SAG-PD-FAT-SAT|35|4800|N/A|140|3.0|0.27×0.27|139|36×512×512|TKR in 7 years|

Note: COR-IW-TSE = coronal intermediate-weighted turbo spin-echo, SAG-IW-TSE-FS = sagittal intermediate-weighted turbo spin-echo with fat suppression, COR-STIR = coronal short-tau inversion recovery, SAG-PD-FAT-SAT = sagittal proton density fat-saturated, TE = echo time, TR = repetition time, TI = inversion time, FOV = field-of-view, ST = slice thickness, ISR = in-plane spatial resolution.


### Nested Cross-Validation Scheme
We implemented a **sevenfold nested cross-validation** scheme for training, validation, and testing. In this scheme, stratified random sampling was used to partition the subcohort into seven disjoint groups. Each of the seven groups served as a test set to evaluate model performance in the outer loop. Deep learning models were trained and validated using the remaining six groups. Within the inner loop, the prediction model was derived using the validation set from one group and the training set from the other five groups. In this way, six separate prediction models were derived for each test set, with each model applied to predict the progression of knee osteoarthritis to TKR outcome in a test set with data independent of that used to derive models. In each test set, TKR probability predictions were averaged from six models developed within the inner loop.

Repo Structure:
`Data > Nested_Cross_Validation > OAI_COR_IW_TSE (subcohort) > NCV_1 (outer loop) > NCV1_CV2_ train & val & test (inner loop)`


## Environment
Create a virtual environment and install PyTorch and other dependencies.

    conda env create -f environment.yml -n mr_transformer

## Model Training
You can use this repo to train models for predicting the progression of knee osteoarthritis to TKR. Default parameters are defined within `main_train.py`. Note that `NCV` refers to the outer loop and `CV` refers to the inner loop of the nested cross-validation scheme.

Our models were trained on an **Nvidia A100 GPU** with **CUDA 12.4**. If you encounter GPU memory limitations, consider enabling gradient checkpointing by setting `--use_checkpoint True`. This leverages [torch.utils.checkpoint](https://docs.pytorch.org/docs/stable/checkpoint.html) to reduce memory usage at the cost of additional computation.

Train MR-Transformer using COR-IW-TSE MRI sequences:

    python main_train.py \
    --train_df_path Data/Nested_Cross_Validation/OAI_COR_IW_TSE/NCV_1/OAI_TKR_7fold_COR_IW_TSE_NCV1_CV2_train.csv \
    --val_df_path Data/Nested_Cross_Validation/OAI_COR_IW_TSE/NCV_1/OAI_TKR_7fold_COR_IW_TSE_NCV1_CV2_val.csv \
    --output_file OAI_COR_NCV1_CV2.txt \
    --save_model_name OAI_COR_NCV1_CV2 \
    --mr_slice_size 384 \
    --use_checkpoint False

Train MR-Transformer using SAG-IW-TSE-FS MRI sequences:

    python main_train.py \
    --train_df_path Data/Nested_Cross_Validation/OAI_SAG_IW_TSE/NCV_1/OAI_TKR_7fold_SAG_IW_TSE_NCV1_CV2_train.csv \
    --val_df_path Data/Nested_Cross_Validation/OAI_SAG_IW_TSE/NCV_1/OAI_TKR_7fold_SAG_IW_TSE_NCV1_CV2_val.csv \
    --output_file OAI_SAG_NCV1_CV2.txt \
    --save_model_name OAI_SAG_NCV1_CV2 \
    --mr_slice_size 448 \
    --use_checkpoint False

Train MR-Transformer using COR-STIR MRI sequences:

    python main_train.py \
    --train_df_path Data/Nested_Cross_Validation/MOST_COR_STIR/NCV_1/MOST_TKR_7fold_COR_STIR_NCV1_CV2_train.csv \
    --val_df_path Data/Nested_Cross_Validation/MOST_COR_STIR/NCV_1/MOST_TKR_7fold_COR_STIR_NCV1_CV2_val.csv \
    --output_file MOST_COR_NCV1_CV2.txt \
    --save_model_name MOST_COR_NCV1_CV2 \
    --mr_slice_size 256 \
    --use_checkpoint False

Train MR-Transformer using SAG-PD-FAT-SAT MRI sequences:

    python main_train.py \
    --train_df_path Data/Nested_Cross_Validation/MOST_SAG_PD_FAT_SAT/NCV_1/MOST_TKR_7fold_SAG_PD_FAT_SAT_NCV1_CV2_train.csv \
    --val_df_path Data/Nested_Cross_Validation/MOST_SAG_PD_FAT_SAT/NCV_1/MOST_TKR_7fold_SAG_PD_FAT_SAT_NCV1_CV2_val.csv \
    --output_file MOST_SAG_NCV1_CV2.txt \
    --save_model_name MOST_SAG_NCV1_CV2 \
    --mr_slice_size 512 \
    --use_checkpoint True


## Model Performance
MR-Transformer was compared against three representative deep learning models ([TSE-Net](https://www.nature.com/articles/s41598-023-33934-1), [3DMeT](https://link.springer.com/chapter/10.1007/978-3-030-87589-3_36), [MRNet](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699)) for knee MRI diagnosis.

<table>
  <tr>
    <th>MRI Tissue Contrast<br>(TKR Follow-up)</th>
    <th>Model</th>
    <th>AUC</th>
    <th>P Value</th>
  </tr>

  <!-- COR-IW-TSE (9 years) -->
  <tr>
    <td rowspan="4" align="center">COR-IW-TSE<br>(TKR within 9 years)</td>
    <td align="center">TSE-Net</td>
    <td align="center">0.87 (0.84, 0.90)</td>
    <td align="center">0.44</td>
  </tr>
  <tr>
    <td align="center">3DMeT</td>
    <td align="center">0.74 (0.70, 0.77)</td>
    <td align="center">&lt;.001</td>
  </tr>
  <tr>
    <td align="center">MRNet</td>
    <td align="center">0.88 (0.85, 0.90)</td>
    <td align="center">0.69</td>
  </tr>
  <tr>
    <td align="center">MR-Transformer</td>
    <td align="center"><b>0.88 (0.85, 0.91)</b></td>
    <td align="center"><i>*</i></td>
  </tr>

  <!-- SAG-IW-TSE-FS (9 years) -->
  <tr>
    <td rowspan="4" align="center">SAG-IW-TSE-FS<br>(TKR within 9 years)</td>
    <td align="center">TSE-Net</td>
    <td align="center">0.86 (0.84, 0.89)</td>
    <td align="center">0.35</td>
  </tr>
  <tr>
    <td align="center">3DMeT</td>
    <td align="center">0.76 (0.72, 0.79)</td>
    <td align="center">&lt;.001</td>
  </tr>
  <tr>
    <td align="center">MRNet</td>
    <td align="center">0.88 (0.85, 0.90)</td>
    <td align="center">0.92</td>
  </tr>
  <tr>
    <td align="center">MR-Transformer</td>
    <td align="center"><b>0.88 (0.85, 0.90)</b></td>
    <td align="center"><i>*</i></td>
  </tr>

  <!-- COR-STIR (7 years) -->
  <tr>
    <td rowspan="4" align="center">COR-STIR<br>(TKR within 7 years)</td>
    <td align="center">TSE-Net</td>
    <td align="center">0.83 (0.79, 0.86)</td>
    <td align="center">0.03</td>
  </tr>
  <tr>
    <td align="center">3DMeT</td>
    <td align="center">0.77 (0.73, 0.81)</td>
    <td align="center">&lt;.001</td>
  </tr>
  <tr>
    <td align="center">MRNet</td>
    <td align="center"><b>0.86 (0.83, 0.89)</b></td>
    <td align="center">0.63</td>
  </tr>
  <tr>
    <td align="center">MR-Transformer</td>
    <td align="center">0.86 (0.82, 0.89)</td>
    <td align="center"><i>*</i></td>
  </tr>

  <!-- SAG-PD-FAT-SAT (7 years) -->
  <tr>
    <td rowspan="4" align="center">SAG-PD-FAT-SAT<br>(TKR within 7 years)</td>
    <td align="center">TSE-Net</td>
    <td align="center">0.83 (0.79, 0.86)</td>
    <td align="center">0.36</td>
  </tr>
  <tr>
    <td align="center">3DMeT</td>
    <td align="center">0.65 (0.61, 0.70)</td>
    <td align="center">&lt;.001</td>
  </tr>
  <tr>
    <td align="center">MRNet</td>
    <td align="center">0.83 (0.80, 0.86)</td>
    <td align="center">0.36</td>
  </tr>
  <tr>
    <td align="center">MR-Transformer</td>
    <td align="center"><b>0.84 (0.81, 0.87)</b></td>
    <td align="center"><i>*</i></td>
  </tr>
</table>

Note: Data in parentheses are 95% confidence intervals (CIs). CIs were computed across 5000 bootstrap samples. The DeLong test was used to assess the significance of the inter-model differences in AUC. P-values were adjusted for multiple comparisons using the Holm–Bonferroni method. The MR-Transformer model was compared against three baseline deep learning models based on the area under the receiver operating characteristic curve (AUC).

    
## License
This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).  
See [LICENSE](./LICENSE) for more details.
