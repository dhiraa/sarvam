# References:
- Inspired after: [https://github.com/jiegzhan/multi-class-text-classification-cnn-rnn](https://github.com/jiegzhan/multi-class-text-classification-cnn-rnn)

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