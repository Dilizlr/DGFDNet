# DGFDNet: Efficient Dual-domain Image Dehazing with Haze Prior Perception
Lirong Zheng, Yanshan Li, Rui Yu, Kaihao Zhang

## Abstract
Transformer-based models exhibit strong global modeling capabilities in single-image dehazing, but their high computational cost limits real-time applicability. Existing methods predominantly rely on spatial-domain features to capture long-range dependencies, which are computationally expensive and often inadequate under complex haze conditions. While some approaches introduce frequency-domain cues, the weak coupling between spatial and frequency branches limits the overall performance.
To overcome these limitations, we propose the Dark Channel Guided Frequency-aware Dehazing Network (DGFDNet), a novel dual-domain framework that performs physically guided degradation alignment across spatial and frequency domains. At its core, the DGFDBlock comprises two key modules: 1) the Haze-Aware Frequency Modulator (HAFM), which generates a pixel-level haze confidence map from dark channel priors to adaptively enhance haze-relevant frequency components, thereby achieving global degradation-aware spectral modulation; 2) the Multi-level Gating Aggregation Module (MGAM), which fuses multi-scale features through diverse convolutional kernels and hybrid gating mechanisms to recover fine structural details.
Additionally, a Prior Correction Guidance Branch (PCGB) incorporates a closed-loop feedback mechanism, enabling iterative refinement of the prior by intermediate dehazed features and significantly improving haze localization accuracy, especially in challenging outdoor scenes. Extensive experiments on four benchmark haze datasets demonstrate that DGFDNet achieves state-of-the-art performance with superior robustness and real-time efficiency.

## Installation
Install warmup scheduler:
~~~
cd pytorch-gradual-warmup-lr/
python setup.py install
cd ..
~~~
Use the following command to create the environment:
~~~
conda env create -f environment.yml
~~~

## Training
~~~
cd XXX
python main.py --mode train --data_dir your_data_path
~~~

## Testing
~~~
cd XXX
python main.py --mode test --data_dir your_data_path --test_model path_of_model
~~~
