# CPR

TensorFlow implementation of our paper "Cross Pairwise Ranking for Unbiased Item Recommendation" (WWW'22).

## Requirements

- python=3.6
- tensorflow-gpu=1.15.5
- numpy
- scipy
- pandas
- Cython

And make sure GCC has been installed in your environment.

## Datasets

The preprocessed data have already been placed in the `data/` folder. See [data_preprocess.py](data_preprocess.py) if you want to know how they were generated from original data.

## Compiling

The samplers and evaluators in our codes are mainly implemented as extension modules by Cython & Cpp, which is much faster than Python implementation. Run the following command to compile all the extension modules.

```shell
python setup.py build_ext --inplace
```

You can safely ignore the warnings in the output of this command.

## Reproducing results

Here we list some commands that reproduce the results presented in our paper.

### CPR

The following commands can reproduce the results of CPR on 4 backbones (MF, LightGCN, NeuMF, NGCF) and 2 datasets (MovieLens, Netflix).

#### MF

(MF is equivalent to 0-layer LightGCN.)

```shell
python CPR.py --dataset movielens_10m --lr 0.0001 --reg 0.001 --ks 20 --batch_size 2048 --n_layer 0 --embed_type lightgcn --sample_rate 3 --sample_ratio 3 --eval_types valid test --eval_epoch 4 --early_stop 10 
```

```shell
python CPR.py --dataset netflix --lr 0.0001 --reg 0.001 --ks 20 --batch_size 2048 --n_layer 0 --embed_type lightgcn --sample_rate 3 --sample_ratio 2 --eval_types valid test --eval_epoch 4 --early_stop 10 
```

#### LightGCN

```shell
python CPR.py --dataset movielens_10m --lr 0.001 --reg 0.01 --ks 20 --batch_size 2048 --n_layer 3 --embed_type lightgcn --sample_rate 8 --sample_ratio 7 --eval_types valid test --eval_epoch 1 --early_stop 10 
```

```shell
python CPR.py --dataset netflix --lr 0.001 --reg 0.0001 --ks 20 --batch_size 2048 --n_layer 3 --embed_type lightgcn --sample_rate 7 --sample_ratio 4 --eval_types valid test --eval_epoch 1 --early_stop 10 
```

#### NeuMF

```shell
python CPR.py --dataset movielens_10m --lr 0.001 --reg 0.0001 --weight_reg 0.01 --weight_sizes 256 --ks 20 --batch_size 2048 --eval_batch_size 16 --n_layer 0 --embed_type lightgcn --inference_type mlp --sample_rate 1 --sample_ratio 4 --eval_types valid test --eval_epoch 10 --early_stop 5 
```

```shell
python CPR.py --dataset netflix --lr 0.001 --reg 0.0001 --weight_reg 0.01 --weight_sizes 256 --ks 20 --batch_size 2048 --eval_batch_size 16 --n_layer 0 --embed_type lightgcn --inference_type mlp --sample_rate 1 --sample_ratio 6 --eval_types valid test --eval_epoch 10 --early_stop 5 
```

#### NGCF

```shell
python CPR.py --dataset movielens_10m --lr 0.001 --reg 0.01 --weight_reg 0 --ks 20 --batch_size 2048 --n_layer 3 --embed_type ngcf --sample_rate 1.5 --sample_ratio 5 --eval_types valid test --eval_epoch 1 --early_stop 10 
```

```shell
python CPR.py --dataset netflix --lr 0.0001 --reg 0.001 --weight_reg 0.01 --ks 20 --batch_size 2048 --n_layer 3 --embed_type ngcf --sample_rate 3 --sample_ratio 4 --eval_types valid test --eval_epoch 4 --early_stop 10 
```

### Baselines

We also implement some of the baselines. The following commands can reproduce their results on MF.

#### BPR

```shell
python BPR.py --dataset movielens_10m --lr 0.0001 --reg 0 --ks 20 --batch_size 2048 --n_layer 0 --embed_type lightgcn --eval_types valid test --eval_epoch 4 --early_stop 10 
```

```shell
python BPR.py --dataset netflix --lr 0.0001 --reg 0.00001 --ks 20 --batch_size 2048 --n_layer 0 --embed_type lightgcn --eval_types valid test --eval_epoch 4 --early_stop 10 
```

#### UBPR

```shell
python UBPR.py --dataset movielens_10m --lr 0.0001 --reg 0.001 --ks 20 --batch_size 2048 --n_layer 0 --embed_type lightgcn --ps_pow 0.8 --clip 0 --eval_types valid test --eval_epoch 4 --early_stop 10 
```

```shell
python UBPR.py --dataset netflix --lr 0.0001 --reg 0.0001 --ks 20 --batch_size 2048 --n_layer 0 --embed_type lightgcn --ps_pow 0.7 --clip 0 --eval_types valid test --eval_epoch 4 --early_stop 10 
```

#### DICE

```shell
python DICE.py --dataset movielens_10m --lr 0.0001 --reg 0.01 --ks 20 --batch_size 2048 --n_layer 0 --embed_type lightgcn --int_weight 9 --pop_weight 9 --dis_pen 0.0001 --margin 10 --margin_decay 0.9 --loss_decay 0.9 --eval_types valid test --eval_epoch 4 --early_stop 10 
```

```shell
python DICE.py --dataset netflix --lr 0.0001 --reg 0.01 --ks 20 --batch_size 2048 --n_layer 0 --embed_type lightgcn --int_weight 9 --pop_weight 9 --dis_pen 0.0001 --margin 40 --margin_decay 0.9 --loss_decay 0.9 --eval_types valid test --eval_epoch 4 --early_stop 10 
```

## Output

Here is an example of the output log. It is the output of command:

```shell
python CPR.py --dataset movielens_10m --lr 0.0001 --reg 0.001 --ks 20 --batch_size 2048 --n_layer 0 --embed_type lightgcn --sample_rate 3 --sample_ratio 3 --eval_types valid test --eval_epoch 4 --early_stop 10 
```

```
...
Epoch 1 :    3.51854 s | loss = 0.69320 = 0.69319 + 0.00000
Epoch 2 :    2.98906 s | loss = 0.69284 = 0.69283 + 0.00000
Epoch 3 :    2.98795 s | loss = 0.69080 = 0.69079 + 0.00001
Epoch 4 :    3.20985 s | loss = 0.68585 = 0.68583 + 0.00002
============================================================================================================================================
[ valid set ]
---- Item ----
Recall    @20 :   0.16620
Precision @20 :   0.02216
NDCG      @20 :   0.08712
Rec       @20 :  20.00000
ARP       @20 :1530.23474
[ test set ]
---- Item ----
Recall    @20 :   0.17162
Precision @20 :   0.03469
NDCG      @20 :   0.10235
ARP       @20 :1588.61609
Evaluation :    1.39915 s
============================================================================================================================================
...
Early stopping triggered.
Best epoch: 76.
[ test set ]
---- Item ----
Recall    @20 :   0.20007
Precision @20 :   0.04061
NDCG      @20 :   0.12209
ARP       @20 :1071.91736
============================================================================================================================================
```

## Citation

If you use our codes in your research, please cite our paper.