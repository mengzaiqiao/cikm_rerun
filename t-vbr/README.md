# T-VBR: Temporal Variational Bayesian Representation Learning for Grocery Recommendation

>This is our Pytorch implementation for the paper:
>Temporal Variational Bayesian Representation Learning for Grocery Recommendation (submitted to SIGIR 2020)

## Introduction

This paper focuses on the task of grocery shopping recommendation, where users purchase multiple products in sequentialbaskets. It is commonly acknowledged that user product purchasepreferences and product popularities vary over time, however veryfew prior methods account for this, instead representing each userand product via static deterministic points in a low-dimensionalcontinuous space. In contrast, we propose a new model: TemporalVariational Bayesian Representation (T-VBR) for grocery recom-mendation, which can encode and leverage temporal patterns toimprove effectiveness. T-VBR is a novel variational Bayesian modelthat learns the temporal Gaussian representations for users anditems by encoding information from: 1) the basket context; 2) itemside information; and 3) the temporal context from past user-iteminteractions. T-VBR is trained using sampled triples of users withtwo items bought together in baskets during different time windows,via a Bayesian Skip-gram model based on the temporal variationalauto-encoder. Experiments conducted on three large real-world gro-cery shopping datasets show that our proposed T-VBR model cansignificantly outperform existing state-of-the-art grocery recom-mendation methods in terms of effectiveness by up-to 7.1% underNDCG, with larger improvements observed over prior works thatencode temporal evidence.

## Environment Requirement

The code has been tested running under Python 3.7.5 and Pytorch 1.3.1. The required packages can be found at py37torch13.yml

## Usage instruction

notebook_example.ipynb

It shows an example of the notebook, which is able to interactively show the performance state. 

You can also use our codes on terminal by:

```shell
    python ./src/main.py --DATASET dunnhumby --PERCENT 1 --N_SAMPLE 1000000 --MODEL VAE_D --EMB_DIM 60 --INIT_LR 0.0025 --ALPHA 0.005 --TIME_STEP 10 --RESULT_FILE result.csv --EPOCH 120 --BATCH_SIZE 128 --ITEM_FEA_TYPE random --REMARKS full_running --OPTI RMSprop --SAVE_LOG True
```

## Some important arguments:

--DATASET: Specify the datasets.

--PERCENT: Percentage of the dataset. Only used in Instacart dataset.

--N_SAMPLE: Number of sampled triples.

--EMB_DIM: Dimension of embeddings 

--INIT_LR: Inital learning rate

--ALPHA: Parameter for the KL terms. We use the annealing technique to decay the impact of KL terms.

--TIME_STEP: Time step for spliting the sequential baskets

--RESULT_FILE: File name for saving the results.

--EPOCH: Training epoches. 

--ITEM_FEA_TYPE: Item feature type. Can be 'random' 'word2vec' 'bert' or their combinations.
