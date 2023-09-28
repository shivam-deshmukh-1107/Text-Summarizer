from heapq import nlargest
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

text = """Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy. 
IBM has a rich history with machine learning. One of its own, Arthur Samuel, is credited for coining the term, “machine learning” with his research (link resides outside ibm.com) around the game of checkers. Robert Nealey, the self-proclaimed checkers master, played the game on an IBM 7094 computer in 1962, and he lost to the computer. Compared to what can be done today, this feat seems trivial, but it's considered a major milestone in the field of artificial intelligence. 
Over the last couple of decades, the technological advances in storage and processing power have enabled some innovative products based on machine learning, such as Netflix's recommendation engine and self-driving cars. Machine learning is an important component of the growing field of data science. Through the use of statistical methods, algorithms are trained to make classifications or predictions, and to uncover key insights in data mining projects. These insights subsequently drive decision making within applications and businesses, ideally impacting key growth metrics. As big data continues to expand and grow, the market demand for data scientists will increase. They will be required to help identify the most relevant business questions and the data to answer them. Machine learning algorithms are typically created using frameworks that accelerate solution development, such as TensorFlow and PyTorch. Since deep learning and machine learning tend to be used interchangeably, it's worth noting the nuances between the two. Machine learning, deep learning, and neural networks are all sub-fields of artificial intelligence. However, neural networks is actually a sub-field of machine learning, and deep learning is a sub-field of neural networks. The way in which deep learning and machine learning differ is in how each algorithm learns. "Deep" machine learning can use labeled datasets, also known as supervised learning, to inform its algorithm, but it doesn't necessarily require a labeled dataset. Deep learning can ingest unstructured data in its raw form (e.g., text or images), and it can automatically determine the set of features which distinguish different categories of data from one another. This eliminates some of the human intervention required and enables the use of larger data sets. You can think of deep learning as "scalable machine learning" as Lex Fridman notes in this MIT lecture (link resides outside ibm.com). Classical, or "non-deep", machine learning is more dependent on human intervention to learn. Human experts determine the set of features to understand the differences between data inputs, usually requiring more structured data to learn. Neural networks, or artificial neural networks (ANNs), are comprised of node layers, containing an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold. If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer of the network by that node. The “deep” in deep learning is just referring to the number of layers in a neural network. A neural network that consists of more than three layers—which would be inclusive of the input and the output—can be considered a deep learning algorithm or a deep neural network. A neural network that only has three layers is just a basic neural network. Deep learning and neural networks are credited with accelerating progress in areas such as computer vision, natural language processing, and speech recognition."""


def summarizer(rawdoc):
    stopwords = list(STOP_WORDS)
    # print(stopwords)
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(rawdoc)
    # print(doc)
    token = [token.text for token in doc]
    # print(token)
    word_frequency = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_frequency.keys():
                word_frequency[word.text] = 1
            else:
                word_frequency[word.text] += 1
    # print(word_frequency)
    max_freq = max(word_frequency.values())
    # print(max_freq)
    for word in word_frequency.keys():
        word_frequency[word] = word_frequency[word] / max_freq
    # print(word_frequency)
    sentence_tokens = [sent for sent in doc.sents]
    # print(sentence_tokens)

    sent_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text in word_frequency.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = word_frequency[word.text]
                else:
                    sent_scores[sent] += word_frequency[word.text]

    # print(sent_scores)

    select_length = int(len(sentence_tokens) * 0.4)
    # print(select_length)

    summary = nlargest(select_length, sent_scores, key=sent_scores.get)
    # print(summary)
    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)

    # print(rawdoc)
    # print(summary)
    # print("Length of original text - ", len(rawdoc.split(' ')))
    # print("Length of summary - ", len(summary.split(' ')))
    return summary, doc, len(rawdoc.split(' ')), len(summary.split(' '))
