

## bilstm_var_length_text

```
cd path/sarvam/src/

#when asked for parameters values go with default values for test run

CUDA_VISIBLE_DEVICES=0 python nlp/text_classification/commands/run_experiments.py \
--mode=train \
--dataset-name=spooky \
--data-iterator-name=text_ids \
--batch-size=8 \
--num-epochs=5 \
--model-name=bilstm_var_length_text |& tee log.txt

tensorboard --logdir=experiments/models/spooky_dataset/text/FastTextV0/lr_0.001_wemd_32_keep_0.5/

CUDA_VISIBLE_DEVICES=0 python nlp/text_classification/commands/run_experiments.py \
--mode=retrain \
--dataset-name=spooky \
--data-iterator-name=text_ids \
--batch-size=8 \
--num-epochs=6 \
--model-name=bilstm_var_length_text \
--model-dir= |& tee log.txt

CUDA_VISIBLE_DEVICES=0 python nlp/text_classification/commands/run_experiments.py \
--mode=predict \
--dataset-name=spooky \
--data-iterator-name=text_ids \
--model-name=bilstm_var_length_text \
--model-dir=|& tee log.txt
```

## bilstm_multilabel

```
cd path/sarvam/src/

#when asked for parameters values go with default values for test run
#also at the end of training, look for log as below
# Use this model directory for further retraining or prediction (--model-dir) : ...

CUDA_VISIBLE_DEVICES=0 python nlp/text_classification/commands/run_experiments.py \
--mode=train \
--dataset-name=jigsaw \
--data-iterator-name=text_ids \
--batch-size=8 \
--num-epochs=5 \
--model-name=bilstm_multilabel |& tee log.txt

tensorboard --logdir=

CUDA_VISIBLE_DEVICES=0 python nlp/text_classification/commands/run_experiments.py \
--mode=retrain \
--dataset-name=jigsaw \
--data-iterator-name=text_ids \
--batch-size=8 \
--num-epochs=6 \
--model-name=bilstm_multilabel \
--model-dir= |& tee log.txt

CUDA_VISIBLE_DEVICES=0 python nlp/text_classification/commands/run_experiments.py \
--mode=predict \
--dataset-name=jigsaw \
--data-iterator-name=text_ids \
--model-name=bilstm_multilabel \
--model-dir=|& tee log.txt
```