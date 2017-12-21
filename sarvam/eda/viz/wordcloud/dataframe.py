from wordcloud import WordCloud, STOPWORDS
from scipy.misc import imread
import matplotlib.pyplot as plt

def plotWordCloud(df, text_col, category_col, category_name, png_path=None):
    '''

    :param df: Pandasa DataFrame
    :param text_col: Text Column Name
    :param category_col: Class/Category column name
    :param category: Class/Category to be visulaized
    :param png_path: PNG file path for custom Image mask (Fun part!)
    :return:
    '''
    text = df[df[category_col]==category_name][text_col].values
    if png_path is not None:
        img = imread(png_path)
    else:
        img = None
    plt.clf()
    plt.figure(figsize=(16,16))
    wc = WordCloud(background_color="black",
                   max_words=10000,
                   mask=img,
                   stopwords=STOPWORDS,
                   max_font_size=40)
    wc.generate(' ' .join(text))

    plt.title(category_name)

    plt.imshow(wc.recolor( colormap= 'PuBu' , random_state=17), alpha=0.9)
    plt.axis('off')
    return  plt


def plotLDATopicWordCloud(tf_vectorizer, lda_model, topic_num=0):
    '''

    :param tf_vectorizer: Refer workbench.scikit.feature_extraction.LemmaCountVectorizer
    :param lda_model: Fitted LDA model
    :param topic_num: Topic to be visulaized
    :return:
    Example Usage:

    import workbench.scikit.feature_extraction.LemmaCountVectorizer
    text = list(df[text_col].values)
    tf_vectorizer = tfLemmaCountVectorizer(max_df=0.95,
                                        min_df=2,
                                        stop_words='english',
                                        decode_error='ignore')
    tf = tf_vectorizer.fit_transform(text)

    lda = LatentDirichletAllocation(n_components=40, max_iter=5,
                                learning_method = 'online',
                                learning_offset = 50.,
                                random_state = 0)
    lda_model = lda.fit(tf)

    '''
    tf_feature_names = tf_vectorizer.get_feature_names()
    topic = lda_model.components_[topic_num]
    topic_words = [tf_feature_names[i] for i in topic.argsort()[:-50 - 1 :-1]]

    # Generating the wordcloud with the values under the category dataframe
    wc = WordCloud(
        stopwords=STOPWORDS,
        background_color='black',
        width=2500,
        height=1800
    ).generate(" ".join(topic_words))
    plt.imshow(wc)
    plt.axis('off')
    plt.show()


