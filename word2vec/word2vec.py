import tensorflow as tf
import argparse

'''
Notes:
Two methods:
- CBOW: Continuous Bag of Word i.e given context words find the target word
- Skip Gram i.e given a word find the target context words

"The cat is sitting on the mat"

CBOW : is   | ( The      cat  )   ( sitting      on  )
       -----------------------------------------------
       w(t)    w(t-2)   w(t-1)       w(t+1)    w(t+2)
       
       Find centre word "is" given "The", "cat", "sitting" and "on"

Skip Gram:   ( The      cat  )   ( sitting      on  ) | is
             ---------------------------------------------
              w(t-2)   w(t-1)       w(t+1)    w(t+2)    w(t)
              
      Find context words of "is", here "The", "cat", "sitting" and "on"
      
      Features:
      (is, The)
      (is, cat)
      (is, sitting)
      (is, given)
      
      
Model: Word Embedding Matrix [Vocab Size, Embedding Size]       

'''

class Word2VecConfig():
    def __init__(self):
        tf.app.flags.FLAGS = tf.app.flags._FlagValues()
        tf.app.flags._global_parser = argparse.ArgumentParser()
        flags = tf.app.flags
        self.FLAGS = flags.FLAGS


class Word2Vec(tf.estimator.Estimator):
    '''
    Skip Gram implementation
    '''
    def __init__(self,
                 config:Word2VecConfig):
        super(Word2Vec, self).__init__(
            model_fn=self._model_fn,
            model_dir=config.FLAGS.MODEL_DIR,
            config=None)

    def _model_fn(self):
        return