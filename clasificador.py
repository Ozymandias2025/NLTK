import nltk
import random

data = [
    ("I love this movie", "positive"),
    ("This movie is terrible", "negative"),
    ("This movie is great", "positive"),
    ("I can't stand watching this movie", "negative"),
    ("The acting in this movie is phenomenal", "positive"),
    ("I regret wasting my time on this films", "negative"),
    ("I thoroughly enjoyed watching this movie", "positive"),
    ("This movie lacks depth an substance", "negative"),
    ("The plot of this movie was captivating", "positive"),
    ("I found the characters in this movie to be very engaging", "positive"),
    ("The specials effects in this movie were impressive", "positive"),
    ("The storyline was precitable and unoriginal", "negative"),
    ("I was dissapointed by the lack of character development", "negative"),
    ("The cinematography in this film was stunning", "positive"),
    ("The dialogue felt forced and unnatural", "negative"),
    ("The pacing of the movie was too slow for my liking", "negative"),
    ("I was plesantly surprised by how much I enjoyed this movie", "positive"),
    ("The ending left me feeling unsatisfied and confused", "negative"),
    ("The movie exceeded my expectations", "positive"),
    ("The performance by the actors were lackluter", "negative")
]

# Preprocesamiento de datos: tokenización y extracción de características
def preprocess(text):
    tokens = nltk.word_tokenize(text)
    return {word: True for word in tokens}

# Aplicamos el preprocesamiento a los datos
featuresets = [(preprocess(text), sentiment) for (text, sentiment) in data]

# Dividimos los datos en conjuntos de entrenamiento y prueba
train_set, test_set = featuresets[:16], featuresets[16:]

# Entrenamos el clasificador utilizando Naive Bayes
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Evaluamos el clasificador en el conjunto de prueba
accuracy = nltk.classify.accuracy(classifier, test_set)
print("Accuracy:", accuracy)

# Clasificamos una nueva frase
new_text = "This movie is amazing"
new_features = preprocess(new_text)
predicted_label = classifier.classify(new_features)
print("Sentimiento predecido:", predicted_label)