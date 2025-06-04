# MR-Transformer: A Vision Transformer-based Deep Learning Model for Total Knee Replacement Prediction Using MRI

## Introduction

This repo contains the implementation of MR-Transformer.

## Data

This study uses data from two publicly available longitudinal cohort studies:

- **Osteoarthritis Initiative (OAI):** [https://nda.nih.gov/oai](https://nda.nih.gov/oai)

- **Multicenter Osteoarthritis Study (MOST):** [https://most.ucsf.edu/multicenter-osteoarthritis-study-most-public-data-sharing](https://most.ucsf.edu/multicenter-osteoarthritis-study-most-public-data-sharing)

## Environment
Create a virtual environment and install PyTorch and other dependencies.

    conda env create -f environment.yml -n mr_transformer

## Model Training

    python main_train.py \
    --train_df_path Data/Nested_Cross_Validation/OAI_COR_IW_TSE/NCV_7/OAI_TKR_7fold_COR_IW_TSE_NCV7_CV1_train.csv \
    --val_df_path Data/Nested_Cross_Validation/OAI_COR_IW_TSE/NCV_7/OAI_TKR_7fold_COR_IW_TSE_NCV7_CV1_val.csv \
    --output_file NCV7_CV1.txt \
    --save_model_name NCV7_CV1 \
    --mr_slice_size 384 \
    --use_checkpoint False

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).  
See [LICENSE](./LICENSE-CC-BY-NC-4.0.md) for more details.
