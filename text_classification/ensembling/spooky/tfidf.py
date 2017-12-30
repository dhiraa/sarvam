from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def tfidf(train_x, val_x, ngram, analyzer, lowercase):
    # Always start with these features. They work (almost) everytime!
    tfv = TfidfVectorizer(min_df=3,
                          max_df=0.8,
                          lowercase=lowercase,
                          strip_accents='unicode',
                          analyzer=analyzer,
                          # token_pattern=r'\w{1,}',
                          ngram_range=ngram,
                          use_idf=1,
                          smooth_idf=1,
                          sublinear_tf=1,
                          stop_words = 'english')

    # Fitting TF-IDF to both training and test sets (semi-supervised learning)
    tfv.fit(list(train_x) + list(val_x))
    train_x_tfidf =  tfv.transform(train_x)
    val_x_tfidf = tfv.transform(val_x)

    return train_x_tfidf, val_x_tfidf

def count_vec(train_x, val_x, analyzer, lowercase, ngram):
    ctv = CountVectorizer(analyzer=analyzer,
                          # token_pattern=r'\w{1,}',
                          lowercase=lowercase,
                          ngram_range=ngram,
                          stop_words='english',
                          max_df=0.8)

    # Fitting Count Vectorizer to both training and test sets (semi-supervised learning)
    ctv.fit(list(train_x) + list(val_x))
    train_x_ctv = ctv.transform(train_x)
    val_x_ctv = ctv.transform(val_x)
    return train_x_ctv, val_x_ctv