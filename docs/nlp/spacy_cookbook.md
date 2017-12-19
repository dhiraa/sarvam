---
layout: page
title:  "Spacy CookBook"
description: "spaCy NLP toolkit Cookbook"
excerpt: "spaCy NLP toolkit Cookbook"
date:   2017-12-18
mathjax: true
comments: true
permalink: /nlp/spacy_cookbook/
---

# References

http://www.winwaed.com/blog/2011/11/08/part-of-speech-tags/  
http://www.clips.ua.ac.be/pages/mbsp-tags  
https://www.analyticsvidhya.com/blog/2017/04/natural-language-processing-made-easy-using-spacy-%E2%80%8Bin-python/
https://dataflume.wordpress.com/2017/07/11/word-vectors-for-non-nlp-data-and-research-people/  
https://www.analyticsvidhya.com/blog/2017/10/essential-nlp-guide-data-scientists-top-10-nlp-tasks/

# A short introduction to NLP in Python with 

- [spaCy](https://spacy.io/)
- [NLTK](http://www.nltk.org/)
- [Gensim](https://radimrehurek.com/gensim/)

Natural Language Processing (NLP) is one of the most interesting sub-fields of data science, and data scientists are increasingly expected to be able to whip up solutions that involve the exploitation of unstructured text data. Despite this, many applied data scientists (both from STEM and social science backgrounds) lack NLP experience.

In this post I explore some fundamental NLP concepts and show how they can be implemented using the popular packages in Python. This post is for the absolute NLP beginner, but knowledge of Python is assumed.


# spaCy, you say?

spaCy is a relatively new package for “Industrial strength NLP in Python” developed by Matt Honnibal at [Explosion AI](https://explosion.ai/). It is designed with the applied data scientist in mind, meaning it does not weigh the user down with decisions over what esoteric algorithms to use for common tasks and it’s fast. Incredibly fast (it’s implemented in Cython). If you are familiar with the Python data science stack, spaCy is your numpy for NLP – it’s reasonably low-level, but very intuitive and performant.



# So, what can it do?

spacy provides a one-stop-shop for tasks commonly used in any NLP project, including:

- Tokenisation
- Stemming
- Lemmatisation
- Part-of-speech tagging
- Entity recognition
- Dependency parsing
- Sentence recognition
- Word-to-vector transformations
- Many convenience methods for cleaning and normalising text

I’ll provide a high level overview of some of these features and show how to access them using spaCy.

# Setup
- python -m spacy download en
- python -m spacy download en_core_web_md
- python -m spacy download parser
- python -m spacy download glove




```python
import numpy as np
from collections import defaultdict
```

# Let’s get started!

First, we load spaCy’s pipeline, which by convention is stored in a variable named nlp. declaring this variable will take a couple of seconds as spaCy loads its models and data to it up-front to save time later. In effect, this gets some heavy lifting out of the way early, so that the cost is not incurred upon each application of the nlp parser to your data. Note that here I am using the English language model, but there is also a fully featured German model, with tokenisation (discussed below) implemented across several languages.

We invoke nlp on the sample text to create a Doc object. The Doc object is now a vessel for NLP tasks on the text itself, slices of the text (Span objects) and elements (Token objects) of the text. It is worth noting that Token and Span objects actually hold no data. Instead they contain pointers to data contained in the Doc object and are evaluated lazily (i.e. upon request). Much of spaCy’s core functionality is accessed through the methods on Doc (n=33), Span (n=29) and Token (n=78) objects.


```python
import spacy
nlp = spacy.load('en_core_web_md')
```


```python
doc = nlp("The big grey dog ate all of the chocolate, but fortunately he wasn't sick! <>")
print(doc)
```

    The big grey dog ate all of the chocolate, but fortunately he wasn't sick! <>


### Tokenization

Tokenisation is a foundational step in many NLP tasks. Tokenising text is the process of splitting a piece of text into words, symbols, punctuation, spaces and other elements, thereby creating “tokens”. A naive way to do this is to simply split the string on white space:


```python
doc.text.split()
```




    ['The',
     'big',
     'grey',
     'dog',
     'ate',
     'all',
     'of',
     'the',
     'chocolate,',
     'but',
     'fortunately',
     'he',
     "wasn't",
     'sick!',
     '<>']



On the surface, this looks fine. But, note that a) it disregards the punctuation and, b) it does not split the verb and adverb (“was”, “n’t”). Put differently, it is naive, it fails to recognise elements of the text that help us (and a machine) to understand its structure and meaning. Let’s see how SpaCy handles this:


```python
[token.orth_ for token in doc]
```




    ['The',
     'big',
     'grey',
     'dog',
     'ate',
     'all',
     'of',
     'the',
     'chocolate',
     ',',
     'but',
     'fortunately',
     'he',
     'was',
     "n't",
     'sick',
     '!',
     '<',
     '>']



Here we access the each token’s .orth_ method, which returns a string representation of the token rather than a SpaCy token object, this might not always be desirable, but worth noting. SpaCy recognises punctuation and is able to split these punctuation tokens from word tokens. Many of SpaCy’s token method offer both string and integer representations of processed text – methods with an underscore suffix return strings, methods without an underscore suffix return integers. For example:



```python
#Take aways _ -> string
[(token, token.orth_, token.orth) for token in doc]
```




    [(The, 'The', 551),
     (big, 'big', 776),
     (grey, 'grey', 4656),
     (dog, 'dog', 1209),
     (ate, 'ate', 3502),
     (all, 'all', 550),
     (of, 'of', 505),
     (the, 'the', 500),
     (chocolate, 'chocolate', 3626),
     (,, ',', 450),
     (but, 'but', 528),
     (fortunately, 'fortunately', 15553),
     (he, 'he', 548),
     (was, 'was', 525),
     (n't, "n't", 513),
     (sick, 'sick', 1732),
     (!, '!', 529),
     (<, '<', 1458735),
     (>, '>', 1216826)]



Here, we return the SpaCy token, the string representation of the token and the integer representation of the token in a list of tuples.

If you want to avoid returning tokens that are punctuation or white space, SpaCy provides convienence methods for this (as well as many other common **text cleaning tasks** – for example, to remove stop words you can call the .is_stop method. 



```python
sentence = [token.orth_ for token in doc if not token.is_punct | token.is_space | token.is_bracket]
' '.join(sentence)
```




    "The big grey dog ate all of the chocolate but fortunately he was n't sick"




```python
sentence = [token.text for token in doc if not token.is_punct | token.is_space | token.is_bracket]
' '.join(sentence)
```




    "The big grey dog ate all of the chocolate but fortunately he was n't sick"



Cool, right?


```python
#Lets see how special characters are handled
abstract= nlp('! @ # $ 5  how this sentence with <special>  < > characters are handled')
```


```python
abstract = [token.text for token in abstract if not token.is_punct | token.is_space | token.is_bracket]
abstract
```




    ['$',
     '5',
     'how',
     'this',
     'sentence',
     'with',
     'special',
     'characters',
     'are',
     'handled']




```python
random_sentence = nlp("D. Mageswaran <Mageswaran1989@gmail.com> likes -------- Tensorflow --------- and <NLP> along with Reinforcement Learning from 2017 while he was working in Pramati TEchnologies")
```


```python
[token.text for token in random_sentence if token.has_vector]
```




    ['D.',
     '<',
     '>',
     'likes',
     '--------',
     '---------',
     'and',
     '<',
     'NLP',
     '>',
     'along',
     'with',
     'Reinforcement',
     'Learning',
     'from',
     '2017',
     'while',
     'he',
     'was',
     'working',
     'in']




```python

```

### Stemming

**What is Stemming?:** Stemming is the process of reducing the words(generally modified or derived) to their word stem or root form. The objective of stemming is to reduce related words to the same stem even if the stem is not a dictionary word. For example, in the English language-

**beautiful** and **beautifully** are stemmed to **beauti**   
**good**, **better** and **best** are stemmed to **good**, **better** and **best** respectively



```python
!pip install stemming
from stemming.porter2 import stem
stem("casually")
```

    Collecting stemming
      Downloading stemming-1.0.1.zip
    Building wheels for collected packages: stemming
      Running setup.py bdist_wheel for stemming ... [?25ldone
    [?25h  Stored in directory: /home/mageswarand/.cache/pip/wheels/3b/fd/d7/a5c5225045c4856ac54a08feace1a9b262fa385ac0fdfd9155
    Successfully built stemming
    Installing collected packages: stemming
    Successfully installed stemming-1.0.1





    'casual'



### Lemmatization

 Lemmatisation is the process of reducing a group of words into their lemma or dictionary form. It takes into account things like POS(Parts of Speech), the meaning of the word in the sentence, the meaning of the word in the nearby sentences etc. before reducing the word to its lemma. For example, in the English Language-  
**beautiful** and **beautifully** are stemmed to **beautiful** and **beautifully**   
**good**, **better** and **best** are stemmed to **good**, **good** and **good** respectively

A related task to tokenisation is lemmatisation. Lemmatisation is the process of reducing a word to its base form, its mother word if you like. Different uses of a word often have the same root meaning. For example, practice, practised and practising all essentially refer to the same thing. It is often desirable to standardise words with similar meaning to their base form. With SpaCy we can access each word’s base form with a token’s .lemma_ method:


```python
practice = "practice practiced practicing"
nlp_practice = nlp(practice)
[print(word.lemma_, word.lemma) for word in nlp_practice]

practice = "beautiful beautifully good better best"
nlp_practice = nlp(practice)
[print(word.lemma_, word) for word in nlp_practice]

```

    practice 1671
    practice 1671
    practice 1671
    beautiful beautiful
    beautifully beautifully
    good good
    better better
    good best





    [None, None, None, None, None]




```python
for token in nlp(u"this is spacy lemmatize testing. programming books are more better than others"):
    print(token, '\t', token.lemma, '\t', token.lemma_)
```

    this 	 530 	 this
    is 	 522 	 be
    spacy 	 173815 	 spacy
    lemmatize 	 1484778 	 lemmatize
    testing 	 2933 	 testing
    . 	 453 	 .
    programming 	 3441 	 programming
    books 	 1045 	 book
    are 	 522 	 be
    more 	 563 	 more
    better 	 649 	 better
    than 	 589 	 than
    others 	 598 	 other


Why is this useful? An immediate use case is in machine learning, specifically text classification. Lemmatising the text prior to, for example, creating a “bag-of-words” avoids word duplication and, therefore, allows for the model to build a clearer picture of patterns of word usage across multiple documents.

### Sentence Tokenize Test or Sentence Segmentation Test:



```python
doc2 = nlp(u"this is spacy sentence tokenize test. this is second sent! is this the third sent? final test.")
```


```python
for sent in doc2.sents:
    print(sent)
```

    this is spacy sentence tokenize test.
    this is second sent!
    is this the third sent?
    final test.


### Pos Tagging:

Part-of-speech tagging is the process of assigning grammatical properties (e.g. noun, verb, adverb, adjective etc.) to words. Words that share the same POS tag tend to follow a similar syntactic structure and are useful in rule-based processes.

For example, in a given description of an event we may wish to determine who owns what. By exploiting possessives, we can do this (providing the text is grammatically sound!). SpaCy uses the popular Penn Treebank POS tags, see https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html. With SpaCy you can access coarse and fine-grained POS tags with the .pos_ and .tag_ methods, respectively. Here, I access the fine grained POS tag:


```python
doc4 = nlp(u"This is pos tagger test for spacy pos tagger")
```


```python
for token in doc4:
    print(token, token.pos, token.pos_)
```

    This 88 DET
    is 98 VERB
    pos 82 ADJ
    tagger 90 NOUN
    test 90 NOUN
    for 83 ADP
    spacy 90 NOUN
    pos 90 NOUN
    tagger 90 NOUN



```python
doc5 = nlp("Conor's dog's toy was hidden under the man's sofa in the woman's house")
pos_tags = [(i, i.tag_) for i in doc5]
pos_tags
```




    [(Conor, 'NNP'),
     ('s, 'POS'),
     (dog, 'NN'),
     ('s, 'POS'),
     (toy, 'NN'),
     (was, 'VBD'),
     (hidden, 'VBN'),
     (under, 'IN'),
     (the, 'DT'),
     (man, 'NN'),
     ('s, 'POS'),
     (sofa, 'NN'),
     (in, 'IN'),
     (the, 'DT'),
     (woman, 'NN'),
     ('s, 'POS'),
     (house, 'NN')]



We can see that the “ ’s ” tokens are labelled as POS. We can exploit this tag to extract the owner and the thing that they own:


```python
owners_possessions = []
...: for i in pos_tags:
...:     if i[1] == "POS":
...:         owner = i[0].nbor(-1)
...:         possession = i[0].nbor(1)
...:         owners_possessions.append((owner, possession))
...:
...: owners_possessions
```




    [(Conor, dog), (dog, toy), (man, sofa), (woman, house)]



This returns a list of owner-possession tuples. If you want to be super Pythonic about it, you can do this in a list comprehenion (which, I think is preferable!):


```python
[(i[0].nbor(-1), i[0].nbor(+1)) for i in pos_tags if i[1] == "POS"]
```




    [(Conor, dog), (dog, toy), (man, sofa), (woman, house)]



Here we are using each token’s .nbor method which returns a token’s neighbouring tokens.

### Named Entity Recognizer (NER):

Entity recognition is the process of classifying named entities found in a text into pre-defined categories, such as persons, places, organizations, dates, etc.  spaCy uses a statistical model to classify a broad range of entities, including persons, events, works-of-art and nationalities / religions (see the documentation for the full list https://spacy.io/docs/usage/entity-recognition).

For example, let’s take the first two sentences from Barack Obama’s wikipedia entry. We will parse this text, then access the identified entities using the Doc object’s .ents method. With this method called on the Doc we can access additional Token methods, specifically .label_ and .label:


```python
doc6 = nlp(u"Rami Eid is studying at Stony Brook University in New York")
```


```python
for ent in doc6.ents:
    print(ent, ent.label, ent.label_)
```

    Rami Eid 377 PERSON
    Stony Brook University 380 ORG
    New York 381 GPE



```python
wiki_obama = "Barack Obama is an American politician who served as the 44th President of the United States from 2009 to 2017. He is the first African American to have served as president, as well as the first born outside the contiguous United States."
nlp_obama = nlp(wiki_obama)
[(i, i.label_, i.label) for i in nlp_obama.ents]
```




    [(Barack Obama, 'ORG', 380),
     (American, 'NORP', 378),
     (the United States, 'GPE', 381),
     (2009 to 2017, 'DATE', 387),
     (first, 'ORDINAL', 392),
     (African, 'NORP', 378),
     (American, 'NORP', 378),
     (first, 'ORDINAL', 392),
     (United States, 'GPE', 381)]



You can see the entities that the model has identified and how accurate they are (in this instance). PERSON is self explanatory, NORP is natianalities or religuos groups, GPE identifies locations (cities, countries, etc.), DATE recognises a specific date or date-range and ORDINAL identifies a word or number representing some type of order.

While we are on the topic of Doc methods, it is worth mentioning spaCy’s sentence identifier. It is not uncommon in NLP tasks to want to split a document into sentences. It is simple to do this with SpaCy by accessing a Doc's  .sents method:


```python
for ix, sent in enumerate(nlp_obama.sents, 1):
    print("Sentence number {}: {}\n".format(ix, sent))
```

    Sentence number 1: Barack Obama is an American politician who served as the 44th President of the United States from 2009 to 2017.
    
    Sentence number 2: He is the first African American to have served as president, as well as the first born outside the contiguous United States.
    


### Noun Chunk Test:


```python
doc6 = nlp(u"Natural language processing (NLP) deals with the application of computational models to text or speech data.")
```


```python
for noun in doc6.noun_chunks:
    print(noun)
```

    Natural language processing (NLP) deals
    the application
    computational models
    text
    speech
    data


### Word Vectors:

https://nlp.stanford.edu/projects/glove/

https://ronxin.github.io/wevi/

https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md

https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

Word vectors are simply vectors of numbers that represent the meaning of a word. For now, that’s not very clear but, we’ll come back to it in a bit. It is useful, first of all to consider why word vectors are considered such a leap forward from traditional representations of words.

Traditional approaches to NLP, such as one-hot encoding and bag-of-words models (i.e. using dummy variables to represent the presence or absence of a word in an observation (e.g. a sentence)), whilst useful for some machine learning (ML) tasks, do not capture information about a word’s meaning or context. This means that potential relationships, such as contextual closeness, are not captured across collections of words. For example, a one-hot encoding cannot capture simple relationships, such as determining that the words “dog” and “cat” both refer to animals that are often discussed in the context of household pets. Such encodings often provide sufficient baselines for simple NLP tasks (for example, email spam classifiers), but lack the sophistication for more complex tasks such as translation and speech recognition. In essence, traditional approaches to NLP, such as one-hot encodings, do not capture syntactic (structure) and semantic (meaning) relationships across collections of words and, therefore, represent language in a very naive way.

In contrast, word vectors represent words as multidimensional continuous floating point numbers where semantically similar words are mapped to proximate points in geometric space. In simpler terms, a word vector is a row of real valued numbers (as opposed to dummy numbers) where each point captures a dimension of the word’s meaning and where semantically similar words have similar vectors. This means that words such as wheel and engine should have similar word vectors to the word car (because of the similarity of their meanings), whereas the word banana should be quite distant. Put differently, words that are used in a similar context will be mapped to a proximate vector space (we will get to how these word vectors are created below). The beauty of representing words as vectors is that they lend themselves to mathematical operators. For example, we can add and subtract vectors – the canonical example here is showing that by using word vectors we can determine that:

king – man + woman = queen

In other words, we can subtract one meaning from the word vector for king (i.e. maleness), add another meaning (femaleness), and show that this new word vector (king – man + woman) maps most closely to the word vector for queen.

The numbers in the word vector represent the word’s distributed weight across dimensions. In a simplified sense each dimension represents a meaning and the word’s numerical weight on that dimension captures the closeness of its association with and to that meaning. **Thus, the semantics of the word are embedded across the dimensions of the vector.**


```python
doc7 = nlp(u"Apples and oranges are similar. Boots and hippos aren't.")
apples = doc7[0]
oranges = doc7[2]
boots = doc7[6]
hippos = doc7[8]
print("Apples vs Oranges: ", apples.similarity(oranges))
print("Boots vs hippos :", boots.similarity(hippos))
```

    Apples vs Oranges:  0.77809414836
    Boots vs hippos : 0.038474555379



```python
from sklearn.decomposition import PCA

animals = "dog cat hamster lion tiger elephant cheetah monkey gorilla antelope rabbit mouse rat zoo home pet fluffy wild domesticated"

animal_tokens = nlp(animals)
animal_vectors = np.vstack([word.vector for word in animal_tokens if word.has_vector])

pca = PCA(n_components=2)
animal_vecs_transformed = pca.fit_transform(animal_vectors)
# animal_vecs_transformed = np.c_[animals.split(), animal_vecs_transformed]
text = animals.split()
```


```python
# animal_vecs_transformed[:,:1], animal_vecs_transformed[:,1:]
```


```python
%matplotlib inline
import sys
sys.path.append('../../examples/')
from dhira_plotly import *
```

    2.0.12



<script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window.Plotly) {{require(['plotly'],function(plotly) {window.Plotly=plotly;});}}</script>



```python
iplot([Scatter(x=animal_vecs_transformed[:,0], y=animal_vecs_transformed[:,1], textposition='bottom', mode='markers+text', text=text)])
```


<div id="97f1f211-8b9b-4bb9-90c3-1170bf9cecfc" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("97f1f211-8b9b-4bb9-90c3-1170bf9cecfc", [{"type": "scatter", "x": [-2.5702973005087766, -2.247249392485344, -2.3505587339870844, 2.2030492013658813, 1.8239401749676434, 2.159262583348045, 2.849375178091706, 0.10010378882396298, 1.9465987640567426, 4.209845579343015, -1.5456673669069048, -2.7125688816928104, -2.436949131636931, 1.3810152258859725, -1.1517439618248442, -2.754355842449194, -1.2830734850308902, 1.5021852733141081, 0.8770883273257035], "y": [-2.3515771973830337, -1.2050567634866058, 1.323629366678258, 0.2780498797509414, 0.4151045854362121, 0.4538388115682919, 0.5343855445609191, 1.9161735957932247, 0.6867759492114917, 0.17859435938382717, 1.0320000927056723, 3.7679339383712063, 3.1304496959269277, -1.2877233733799538, -2.3977564272254064, -2.977657342974969, -0.974776907919425, -0.9156374039085561, -1.6067504031090232], "textposition": "bottom", "mode": "markers+text", "text": ["dog", "cat", "hamster", "lion", "tiger", "elephant", "cheetah", "monkey", "gorilla", "antelope", "rabbit", "mouse", "rat", "zoo", "home", "pet", "fluffy", "wild", "domesticated"]}], {}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>



```python
# !pip install gensim
from gensim.models.keyedvectors import KeyedVectors
word_vectors=KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)
word_vectors['human']
```


```python
#Implementation: Here is how you can train your own word vectors using gensim

sentence=[['first','sentence'],['second','sentence']]
model = gensim.models.Word2Vec(sentence, min_count=1,size=300,workers=4)
```

### Multi-threaded generator


```python
texts = [u'One document.', u'...', u'Lots of documents']
# .pipe streams input, and produces streaming output
iter_texts = (texts[i % 3] for i in range(100000000))
for i, doc in enumerate(nlp.pipe(iter_texts, batch_size=50, n_threads=4)):
    assert doc.is_parsed
    if i == 100:
        break
```

### Deeplearning
https://spacy.io/docs/usage/deep-learning


```python
test_sent = 'Let us see what comes for MACHINE, machine and an outtttoffword'
```


```python
test_sent_parsed = nlp(test_sent)
test_sent_tok = [tok for tok in test_sent_parsed]
```


```python
def get_spacy_embedding_matrix(nlp):
    vocab = nlp.vocab
    max_rank = max(lex.rank for lex in vocab if lex.has_vector)
    vectors = np.ndarray((max_rank + 1, vocab.vectors_length), dtype='float32')
    for lex in vocab:
        if lex.has_vector:
            vectors[lex.rank] = lex.vector
    return vectors
```


```python
def get_word_and_vector(index, nlp):
     return (nlp.vocab[index].text, nlp.vocab[index].vector)
```


```python
[get_word_and_vector(i, nlp) for i in range(10)]
```




    [('', array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], dtype=float32)),
     ('IS_ALPHA',
      array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], dtype=float32)),
     ('IS_ASCII',
      array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], dtype=float32)),
     ('IS_DIGIT',
      array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], dtype=float32)),
     ('IS_LOWER',
      array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], dtype=float32)),
     ('IS_PUNCT',
      array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], dtype=float32)),
     ('IS_SPACE',
      array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], dtype=float32)),
     ('IS_TITLE',
      array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], dtype=float32)),
     ('IS_UPPER',
      array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], dtype=float32)),
     ('LIKE_URL',
      array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], dtype=float32))]




```python
def spacy_word2vec(word, nlp):
    lex = nlp(word)
    if lex.has_vector:
        return lex.vector
    else:
        return nlp.vocab[0].vector #return all zeros for Out of vocab
    
```


```python
spacy_word2vec('out', nlp)
```




    array([  4.96499985e-02,   5.29359989e-02,  -1.97679996e-01,
             7.42449984e-02,   1.59140006e-01,   1.20999999e-02,
            -2.99290001e-01,   1.89219993e-02,   5.11710010e-02,
             2.51090002e+00,  -1.52899995e-01,   2.05960006e-01,
             1.25239998e-01,  -3.28860015e-01,  -3.43309999e-01,
            -1.52260005e-01,  -1.91770002e-01,   8.24530005e-01,
            -4.68430012e-01,   7.05149993e-02,   1.49629995e-01,
             4.56800014e-02,   7.76799978e-04,   1.80930004e-01,
            -7.81830028e-02,   8.67969997e-04,   6.80819973e-02,
            -1.10560000e-01,   7.20390007e-02,  -6.18350029e-01,
             1.24059998e-05,   1.79120004e-01,  -1.98819995e-01,
             4.12929989e-02,   2.91409999e-01,  -1.22970000e-01,
             7.59010017e-02,   2.95679986e-01,  -2.19359994e-03,
             1.19340001e-02,  -1.85090005e-01,   3.61110009e-02,
            -1.82980001e-01,  -3.82050008e-01,   2.73400009e-01,
             1.95230007e-01,  -4.38049994e-02,   5.28290011e-02,
            -1.59729999e-02,  -1.22129999e-01,  -1.45999998e-01,
            -5.15309982e-02,  -2.67509997e-01,   1.87790006e-01,
            -1.36639997e-01,  -1.26629993e-01,   3.10130000e-01,
            -8.86219963e-02,   3.31620015e-02,  -2.45529994e-01,
             1.56379998e-01,  -3.41450013e-02,  -1.78800002e-02,
             2.46879995e-01,   2.62369990e-01,  -1.74800009e-02,
            -4.38669994e-02,   7.45029971e-02,   2.04720005e-01,
             4.54849988e-01,   1.91980004e-01,   6.76179975e-02,
             6.94130003e-01,  -1.38699993e-01,   8.08470044e-03,
            -3.49600017e-02,   1.87140003e-01,  -3.00190002e-01,
            -6.37440011e-02,   1.66520000e-01,  -1.20430002e-02,
             3.94439995e-01,  -2.56770015e-01,  -2.91550010e-01,
             9.27560031e-02,   7.32069984e-02,   5.44500016e-02,
            -1.82290003e-01,   3.20620000e-01,   4.50630009e-01,
             2.44159997e-01,   3.39620002e-02,  -6.46829978e-02,
             3.18379998e-02,   2.46450007e-02,   1.47330001e-01,
             5.95830008e-02,  -9.30199996e-02,   2.72599999e-02,
            -1.55000001e-01,  -6.48320019e-02,  -3.67590010e-01,
            -6.29509985e-02,  -1.05159998e-01,   4.41869982e-02,
            -5.30839980e-01,   3.35249990e-01,   1.75339997e-01,
             6.57769991e-03,  -1.17799997e-01,   1.73620000e-01,
            -6.60229981e-01,   1.18359998e-01,   8.69840011e-02,
             1.58629999e-01,   3.10209990e-01,  -2.78639998e-02,
             9.29879993e-02,   1.94499999e-01,   7.13700010e-03,
             2.07990006e-01,  -1.41649996e-03,  -2.22379994e-02,
            -1.80480003e-01,   1.31040001e-02,   1.24499999e-01,
            -1.30240005e-02,   1.45050004e-01,  -1.42430002e-02,
            -1.64810002e-01,  -2.83410013e-01,   1.42480001e-01,
            -2.01130003e-01,   1.70300007e-01,   2.23030001e-01,
             5.45150004e-02,   9.85139981e-02,   2.36880004e-01,
            -1.39300004e-02,   1.19970001e-01,  -1.95869994e+00,
             1.66529998e-01,   1.39690004e-02,  -1.28649995e-01,
             1.83559999e-01,  -1.96549997e-01,  -4.54490006e-01,
             1.95810005e-01,   2.03970000e-01,   1.19429998e-01,
             3.44729982e-02,  -7.03810006e-02,   8.36620033e-02,
            -3.27540010e-01,  -1.40249997e-01,   7.92799965e-02,
             1.19310003e-02,   1.85900003e-01,  -9.81400013e-02,
             4.47819987e-03,  -1.25409998e-02,  -3.53740007e-02,
             7.79189989e-02,  -1.52970001e-01,  -2.18250006e-01,
            -2.69879997e-01,   1.23010002e-01,  -5.88289984e-02,
             2.63839990e-01,   1.44830003e-01,  -1.44909993e-01,
            -2.92070001e-01,  -4.67760004e-02,  -2.95170009e-01,
            -1.91159993e-01,   1.95040002e-01,  -4.76550013e-02,
             1.43580005e-01,   1.95449993e-01,  -1.06020004e-01,
             3.05620015e-01,   6.23140000e-02,   9.27639976e-02,
             8.92620012e-02,   3.21079999e-01,   1.18280001e-01,
            -8.93419981e-02,   3.48410010e-02,   1.08489998e-01,
             3.39910001e-01,  -5.33939991e-03,   4.40259986e-02,
             1.23329997e-01,   8.94939974e-02,  -1.63839996e-01,
             3.09619993e-01,  -3.76760006e-01,   1.58979997e-01,
            -1.21830001e-01,   9.37760025e-02,  -8.91359970e-02,
            -1.08730003e-01,  -1.29539996e-01,   7.35880015e-03,
             3.65350008e-01,   3.51210013e-02,   1.10720001e-01,
             5.20710014e-02,  -1.65730000e-01,  -1.00910001e-01,
            -8.21970031e-02,  -1.79900005e-01,   9.31150019e-02,
            -2.26909995e-01,  -1.91290006e-01,   1.80360004e-01,
             3.43189985e-02,   1.20449997e-01,  -3.22459996e-01,
            -5.70590012e-02,  -1.91479996e-01,  -2.65120007e-02,
            -9.98840034e-02,  -8.46429989e-02,  -1.33489996e-01,
            -3.02260011e-01,  -1.68660000e-01,   1.26910001e-01,
             1.83589999e-02,   9.22290012e-02,   4.43039984e-02,
            -1.51679993e-01,   4.86389995e-01,   1.53080001e-01,
             2.69749999e-01,  -2.44110003e-01,  -1.22779999e-02,
            -2.27669999e-01,   2.61009997e-03,   1.82429999e-01,
             2.46500000e-01,   1.86900005e-01,   9.20609981e-02,
             4.62430000e-01,   1.81759998e-01,  -1.20549999e-01,
            -1.63619995e-01,  -5.53570017e-02,  -2.32250005e-01,
             2.75840014e-01,  -4.63210000e-03,  -2.82860011e-01,
             1.32280007e-01,   6.17050007e-02,   4.00290012e-01,
             4.04089987e-01,   1.57110006e-01,  -2.17950001e-01,
            -2.94340014e-01,   1.68760002e-01,  -9.70640033e-02,
             3.28249991e-01,  -2.76900008e-02,  -1.25279993e-01,
             6.01610005e-01,  -8.33190009e-02,   1.62189994e-02,
            -1.93550006e-01,   2.56960005e-01,  -4.55330014e-02,
             1.57629997e-01,   2.15399995e-01,   1.42839998e-01,
            -3.07839990e-01,   9.25950035e-02,  -1.56389996e-01,
             8.53279978e-02,   1.55129999e-01,  -1.10320002e-01,
             1.13860004e-01,   8.48370045e-03,   2.92349998e-02,
            -1.60540000e-01,   2.33309995e-02,  -1.13169998e-01,
            -1.80590004e-02,   1.54650003e-01,  -1.05810001e-01,
             5.92680015e-02,  -1.92440003e-01,  -3.56000006e-01,
            -4.58509997e-02,   2.96380013e-01,  -6.42099977e-02,
            -7.69039989e-02,  -2.04539999e-01,   1.37260005e-01,
             9.21899974e-02,   1.12839997e-01,  -2.03460008e-02], dtype=float32)



### A small data indexer


```python
out_of_word = 0
padding = 0
word_to_index = {}
index_to_word= {}
word_to_index[nlp.vocab[0].text] = 0
for i, token in enumerate(set(test_sent_tok),1):
    word_to_index[token.text] = i

print(word_to_index)

for word, i in word_to_index.items():
    index_to_word[i] = word

print(index_to_word)

print(word_to_index['Let'])
print(index_to_word[8])
```

    {'': 0, 'see': 1, 'what': 2, 'comes': 3, 'for': 4, 'machine': 5, 'MACHINE': 6, 'and': 7, 'Let': 8, 'us': 9}
    {0: '', 1: 'see', 2: 'what', 3: 'comes', 4: 'for', 5: 'machine', 6: 'MACHINE', 7: 'and', 8: 'Let', 9: 'us'}
    8
    Let



```python
def get_embedding_matrix(index_to_word, nlp):
    vocab_size = len(index_to_word)
    vectors = np.ndarray((vocab_size, nlp.vocab.vectors_length), dtype='float32')
    for i, word in index_to_word.items():
        vectors[i] = spacy_word2vec(word, nlp)
    return vectors
    
```


```python
embeddings = get_embedding_matrix(index_to_word, nlp)

embeddings.shape
```




    (10, 300)



### Lets see the index value and their values


```python
for i in range(1000): print(i, nlp.vocab[i].text, nlp.vocab[i].has_vector)
```


```python
x = nlp('This is a tessssst')
[w.is_oov for w in x]
```


```python

```


```python
test_sent_parsed[6], test_sent_parsed[6].rank, '----->Glove Vector', test_sent_parsed[6].vector
```


```python
test_sent_parsed[8], test_sent_parsed[8].rank, '----->Glove Vector', test_sent_parsed[8].vector
```


```python
def token_to_index(tokens, max_length):
    Xs = []
    for i, token in enumerate(tokens[:max_length]):
        Xs.append(token.rank if token.has_vector else 0)
    return Xs
```


```python
test_sent_tok
```


```python
token_to_index(test_sent_tok, 10)
```


```python

```


```python
# Convert this notebook for Docs
! jupyter nbconvert --to markdown --output-dir ../../docs/nlp spacy_cookbook.ipynb
```

    [NbConvertApp] Converting notebook spacy_cookbook.ipynb to markdown
    [NbConvertApp] Writing 51359 bytes to ../../docs/nlp/spacy_cookbook.md



```python

```
