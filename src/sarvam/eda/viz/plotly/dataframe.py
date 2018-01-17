import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import numpy as np

def classWiseBarChart(df, category_col):
    '''
    Prints the number of sample avaialble in each category/classes
    :param df: Pandas DataFrame
    :param category_col: Column name for classes/categories that needs to be predicted
    :return:
    '''

    data = [go.Bar(
        x=df[category_col].unique(),
        y=df[category_col].value_counts().values,
        marker= dict(colorscale='Jet',
                     color = df[category_col].value_counts().values,
                     ),
        text='Text entries attributed to ' + category_col
    )]

    layout = go.Layout(
        title='Target variable distribution'
    )

    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, filename='classWiseBarChart')

#https://www.kaggle.com/mageswaran/spooky-nlp-and-topic-modelling-tutorial/
def wordFreqBarChart(df, text_col, top_words=50):
    '''
    Prints the frequecy of top N words
    :param df: Pandas DataFrame
    :param text_col: Text column of the dataframe
    :param top_words: Integer value
    :return:
    '''
    all_words = df[text_col].str.split(expand=True).unstack().value_counts()
    data = [go.Bar(
        x = all_words.index.values[2:top_words],
        y = all_words.values[2:top_words],
        marker= dict(colorscale='Jet',
                     color = all_words.values[2:top_words*2]
                     ),
        text='Word Counts'
    )]

    layout = go.Layout(
        title='Top ' + str(top_words) + ' (Uncleaned) Word frequencies in the dataset'
    )

    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, filename='basic-bar')

def termFreqBarChart(tf_vectorizer, tf):
    '''

    :param tf_vectorizer: Refer workbench.scikit.feature_extraction.LemmaCountVectorizer
    :param tf: TermFrequency Vector, i.e fitted text matrix
    :return:
    '''
    feature_names = tf_vectorizer.get_feature_names()
    count_vec = np.asarray(tf.sum(axis=0)).ravel()
    zipped = list(zip(feature_names, count_vec))
    x, y = (list(x) for x in zip(*sorted(zipped, key=lambda x: x[1], reverse=True)))

    # Plotting the Plot.ly plot for the Top 50 word frequencies
    data = [go.Bar(
        x = x[0:50],
        y = y[0:50],
        marker= dict(colorscale='Jet',
                     color = y[0:50]
                     ),
        text='Word counts'
    )]

    layout = go.Layout(
        title='Top 50 Word frequencies after Preprocessing'
    )

    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, filename='basic-bar')

    # Plotting the Plot.ly plot for the Bottom 50 word frequencies
    data = [go.Bar(
        x = x[-50:],
        y = y[-50:],
        marker= dict(colorscale='Portland',
                     color = y[-50:]
                     ),
        text='Word counts'
    )]

    layout = go.Layout(
        title='Bottom 50 Word frequencies after Preprocessing'
    )

    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, filename='basic-bar')