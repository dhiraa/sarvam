# Papers:
- [Bag of Tricks for Efficient Text Classification (Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov)](https://arxiv.org/abs/1607.01759)  

## How to run?
```bash
source activate sarvam
python run.py
tensorboard --logdir=tmp/fast_text_v0/
```

```
python commands/run_experiments.py --mode=train --dataset-name=jigsaw --data-iterator-name=text --batch-size=32 --num-epochs=5 --model-name=bilstm_var_length_text
```

