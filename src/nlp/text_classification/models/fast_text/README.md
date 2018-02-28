# Papers:
- [Bag of Tricks for Efficient Text Classification (Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov)](https://arxiv.org/abs/1607.01759)  

Check the [notebook](notebook/Fast-Text-v0.ipynb) for a simple intro!



```
cd path/sarvam/src/

#when asked for parameters values go with default values for test run

CUDA_VISIBLE_DEVICES=0 python nlp/text_classification/commands/run_experiments.py \
--mode=train \
--dataset-name=spooky \
--data-iterator-name=text_ids \
--batch-size=16 \
--num-epochs=5 \
--model-name=fast_text_v0 |& tee log.txt

tensorboard --logdir=experiments/models/spooky_dataset/text/FastTextV0/lr_0.001_wemd_32_keep_0.5/

CUDA_VISIBLE_DEVICES=0 python nlp/text_classification/commands/run_experiments.py \
--mode=retrain \
--dataset-name=spooky \
--data-iterator-name=text_ids \
--batch-size=16 \
--num-epochs=6 \
--model-name=fast_text_v0 \
--model-dir=experiments/models/spooky_dataset/text/FastTextV0/lr_0.001_wemd_32_keep_0.5/ |& tee log.txt

CUDA_VISIBLE_DEVICES=0 python nlp/text_classification/commands/run_experiments.py \
--mode=predict \
--dataset-name=spooky \
--data-iterator-name=text_ids \
--model-name=fast_text_v0 \
--model-dir=experiments/models/spooky_dataset/text/FastTextV0/lr_0.001_wemd_32_keep_0.5/ |& tee log.txt

CUDA_VISIBLE_DEVICES=0 python nlp/text_classification/commands/run_experiments.py \
--mode=train \
--dataset-name=spooky \
--data-iterator-name=text_ids_lazy_generator \
--batch-size=16 \
--num-epochs=5 \
--model-name=fast_text_v0 |& tee log.txt

```