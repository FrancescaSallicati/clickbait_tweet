from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def get_most_common_words(variable, max_words= 30):
    # Get most frequents tokens in variable and plot first most common (default = 30)
    word_frequency = Counter(variable).most_common(max_words)
    words = [word for word, _ in word_frequency]
    counts = [counts for _, counts in word_frequency]
    plt.figure(figsize=(10, 10))
    return plt.barh(words[::-1], counts[::-1], color='dodgerblue')

def get_wordcloud(variable):
    ##Get WordCloud for variable
    text = " ".join(str(post) for post in variable)
    wordcloud = WordCloud(max_words = 1000, background_color="white", colormap="Blues", collocations=True).generate(text)
    plt.figure(figsize=(13, 13))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
