import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

sentence = "NLTK es una biblioteca de procesamiento de lenguaje natural en Python."
tokens = word_tokenize(sentence)
print(tokens)