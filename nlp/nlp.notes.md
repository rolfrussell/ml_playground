## Podcast:  Taming arVix with Natural Language Processing
* primer.ai
* working on summarizing large numbers of documents
* when you don't have time to read 1000 docs but want the key points and people
* use [spaCy](spacy.io) library for a bunch of nlp


## Podcast:  Information Extraction from Natural Document Formats
* Bloomberg extracts info from documents
* multi step pipeline
* convert pdfs to latex (assuming original came from latex)
* use visual rendering of pdf, not the embedded pdf code because that can contain noisy or misleading data
* able to get higher than human performance for some document classes
* a lot of focus on extracting data from tables and graphs (so far scatter plot)


## Kaggle - Denoising Dirty Documents
* Cleaning scanned documents to make text clear and crisp
* https://www.kaggle.com/c/denoising-dirty-documents
* https://colinpriest.com/2015/08/01/denoising-dirty-documents-part-1/
* https://www.kaggle.com/atrisaxena/autoencoders-denoising-dirty-document


## YouTube NLP with Deep Learning
* https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6


## Datasets
* [NLTK built in corpa](http://www.nltk.org/nltk_data/)
* [UCI: Legal Case Reports Data Set](https://archive.ics.uci.edu/ml/datasets/Legal+Case+Reports) - corpus
* [UCI: NSF Research Award Abstracts](https://archive.ics.uci.edu/ml/datasets/NSF+Research+Award+Abstracts+1990-2003) - 
* [UCI: Reuters-21578 Text Categorization Collection](https://archive.ics.uci.edu/ml/datasets/reuters-21578+text+categorization+collection)
* [obama-white-house.csv](https://www.kaggle.com/jayrav13/obama-white-house) - Kaggle with 20000 document links and text
* [Federal Reserve Greenbook Data](https://www.philadelphiafed.org/research-and-data/real-time-center/greenbook-data/pdf-data-set) - pdfs with lots of charts
* check UPenn.  the are known for document corpa


## Libraries
* [spaCy](spacy.io) - variety of nlp
* [fasttext](https://github.com/facebookresearch/fastText) - great word vectors, text classification, learning word vectors
* [SyntaxNet](https://github.com/tensorflow/models/tree/master/research/syntaxnet) and [CoreNLP](https://stanfordnlp.github.io/CoreNLP/) are research tools, whereas spaCy is really designed for developers
* [NLTK](http://www.nltk.org) - more for research than production; access to corpora
* [Cloud Language API](https://cloud.google.com/natural-language/) - reveal the structure and meaning of text both through pretrained machine learning models and custom models [AutoML Natural Language](https://cloud.google.com/automl/)
* [gensim](https://radimrehurek.com/gensim/) - topic modeling for humans; access to corpora


## Random
* [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759v2.pdf) - Our experiments show that our fast text classifier fastText is often on par with deep learning classifiers in terms of ac- curacy, and many orders of magnitude faster for training and evaluation
* https://medium.com/@souvikghosh_14630/build-a-rasa-nlu-chatbot-with-spacy-with-fasttext-240e192082bd
