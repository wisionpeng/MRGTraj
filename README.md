# MRGTraj
The official repo for "MRGTraj: A Novel Non-Autoregressive Approach for Human Trajectory Prediction".

# Overview
![idea](https://github.com/wisionpeng/MRGTraj/assets/134345840/7b80d808-f165-4fc8-9655-30917f085bd5)

**Abstract:** Forecasting human trajectory is an essential technology in intelligent surveillance systems, robot navigation systems, autonomous driving systems, etc. Most of the trajectory prediction models based on RNN and Transformers use autoregressive methods to generate future trajectories, which may accumulate displacement errors and are inefficient for training and test. To address these problems, we propose a novel decoder (MRG decoder) which introduces a Mapping-Refinement-Generation structure in order to generate trajectory in a non-autoregressive manner. Based on the proposed MRG decoder, we design the MRGTraj trajectory prediction model. Firstly, we use a Transformer as an encoder to extract encoded features from the past trajectory. Secondly, we introduce an interaction-aware latent code generator to learn a Gaussian distribution from social context among pedestrians for latent code sampling. Finally, we feed the encoded features to the MRG decoder and sample the latent code multiple times from the learned Gaussian distribution as extra inputs to the MRG decoder to generate multiple socially acceptable future trajectories. The experimental results on two public datasets (ETH \& UCY) demonstrate the effectiveness of the MRGTraj model. Besides, the MRGTraj model achieves superior prediction performance and it wins 3.85\% and 13.21\% improvement on ADE and FDE metrics, as well as 71.29\% speed-up compared with the state-of-the-art models.


## Install requirements

```bash
pip install -r requirements.txt
```

## Get results from pre-trained model

For example, run the evaluation code on eth dataset:

```bash
python val.py -dataset_name eth
```

## Train the model on a dataset
 
Train on the eth-hotel datasets, the dataset_name is set to 'eth', 'hotel', 'univ', 'zara1', 'zara2'.

```bash
python train.py -dataset_name eth
```

Please open an issue if you have any questions.

## Citation
If you find this code useful for your research, please cite our paper:

```bibtex
@ARTICLE{Peng2023MRGTraj,
  author={Peng, Yusheng and Zhang, Gaofeng and Shi, Jun and Li, Xiangyu and Zheng, Liping},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={MRGTraj: A Novel Non-Autoregressive Approach for Human Trajectory Prediction}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2023.3307442}}
```