# References:
- Inspired after: [https://github.com/jiegzhan/multi-class-text-classification-cnn-rnn](https://github.com/jiegzhan/multi-class-text-classification-cnn-rnn)

```
cd path/sarvam/src/

#when asked for parameters values go with default values for test run

CUDA_VISIBLE_DEVICES=0 python nlp/text_classification/commands/run_experiments.py \
--mode=train \
--dataset-name=spooky \
--data-iterator-name=text \
--batch-size=8 \
--num-epochs=5 \
--model-name=cnn_text_v0 |& tee log.txt

tensorboard --logdir=experiments/models/spooky_dataset/text/FastTextV0/lr_0.001_wemd_32_keep_0.5/

CUDA_VISIBLE_DEVICES=0 python nlp/text_classification/commands/run_experiments.py \
--mode=retrain \
--dataset-name=spooky \
--data-iterator-name=text \
--batch-size=8 \
--num-epochs=6 \
--model-name=cnn_text_v0 \
--model-dir= |& tee log.txt

CUDA_VISIBLE_DEVICES=0 python nlp/text_classification/commands/run_experiments.py \
--mode=predict \
--dataset-name=spooky \
--data-iterator-name=text \
--model-name=cnn_text_v0 \
--model-dir= |& tee log.txt
```