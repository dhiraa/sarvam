# Papers:
- [Convolutional Neural Networks for Sentence Classification](http://www.aclweb.org/anthology/D14-1181)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)

# References
- [http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
- [https://github.com/dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)

## How to run?


```
cd path/sarvam/src/

#when asked for parameters values go with default values for test run
#also at the end of training, look for log as below
#Use this model directory for further retraining or prediction (--model-dir) : ...
 
CUDA_VISIBLE_DEVICES=0 python nlp/text_classification/commands/run_experiments.py \
--mode=train \
--dataset-name=spooky \
--data-iterator-name=text_ids \
--batch-size=8 \
--num-epochs=5 \
--model-name=cnn_text_v0 |& tee log.txt

tensorboard --logdir=

CUDA_VISIBLE_DEVICES=0 python nlp/text_classification/commands/run_experiments.py \
--mode=retrain \
--dataset-name=spooky \
--data-iterator-name=text_ids \
--batch-size=8 \
--num-epochs=6 \
--model-name=cnn_text_v0 \
--model-dir= |& tee log.txt

CUDA_VISIBLE_DEVICES=0 python nlp/text_classification/commands/run_experiments.py \
--mode=predict \
--dataset-name=spooky \
--data-iterator-name=text_ids \
--model-name=cnn_text_v0 \
--model-dir= |& tee log.txt
```