import nltk
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from nltk.tokenize import word_tokenize

sentence = "NLTK es una biblioteca de procesamiento de lenguaje natural en Python."
tokens = word_tokenize(sentence)
tags = pos_tag(tokens)
print(tags)

# NNP - Proper Noun, Singular (Nombre propio, singular): NLTK
# CC - Coordinating Conjunction (Conjunción coordinante): es
# JJ - Adjective (Adjetivo): una, natural
# NN - Noun, Singular (Sustantivo, singular): biblioteca
# IN - Preposition or Subordinating Conjunction (Preposición o conjunción subordinante): de
# FW - Foreign Word (Palabra extranjera): procesamiento, lenguaje