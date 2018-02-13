# About 
[]Check the presentation here!](Asariri.pptx.pdf)

# Dataset

```bash
dataset_name
    - audio
        - person_x
            - file_id.wav
    - images
        - person_x
            - file_id.jpeg

```



# Pipeline

Audio File ---> Librosa ---> MFCC ---> 3920 freq samples

# References:
- https://github.com/adeshpande3/Generative-Adversarial-Networks/blob/master/Generative%20Adversarial%20Networks%20Tutorial.ipynb
- https://www.tensorflow.org/api_docs/python/tf/contrib/gan/estimator/GANEstimator


```

python asariri/commands/run_experiments.py \
--mode=train \
--dataset-name=crawled_data \
--data-iterator-name=raw_data_iterators \
--model-name=basic_model \
--batch-size=32 \
--num-epochs=5

python asariri/commands/run_experiments.py \
--mode=predict \
--dataset-name=crawled_data \
--data-iterator-name=raw_data_iterators \
--model-name=basic_model \
--batch-size=32 \
--num-epochs=5 \
--model-dir=experiments/asariri/models/BasicModel/
```