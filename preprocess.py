import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.stem.snowball import SnowballStemmer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def preprocess_text(text, vectorizer=None):
    processed_text = lemmatisatingParagraph(text)
    # Si un vectorizer est fourni, utiliser le vectorizer pour transformer le texte
    if vectorizer:
        processed_text = vectorizer.transform([processed_text])
    
    return processed_text

def TokenisParagraph(paragraph):
    tokens = nltk.word_tokenize(paragraph)
    return tokens

def StopwordsParagraph(tokens):
    new_paragraph = [word for word in tokens if word not in stopwords.words('english')]
    return new_paragraph

def StemmingParagraph(stopwords):
    # Création d'un objet de stemming
    stemmerAnglais = PorterStemmer()
    # Exemple de stemming
    racines = [stemmerAnglais.stem(mot) for mot in stopwords]
    return racines

def get_word_pos(word):
    # Tokenize the word
    tokens = word_tokenize(word)
    # Perform POS tagging
    pos_tags = nltk.pos_tag(tokens)
    # Return the POS tag of the word
    return pos_tags[0][1] if pos_tags else None

def get_word_category(word_pos):
    if word_pos.startswith('V'):
        return 'verb'
    elif word_pos.startswith('N'):
        return 'noun'
    elif word_pos.startswith('J'):
        return 'adjective'
    else:
        return 'other'

def lemmatisatingParagraph(paragraph):
    # Tokenisation && Stopwords && Stemming
    tokens = TokenisParagraph(paragraph)
    stopwords = StopwordsParagraph(tokens)
    steming  = StemmingParagraph(stopwords)
    # Creation d'un objet de lemmatisation
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    for word in steming:
        word_pos = get_word_pos(word)
        if word_pos:
            word_category = get_word_category(word_pos)
        if word_category =="adjective":
            lemma = lemmatizer.lemmatize(word, pos="a")
        elif word_category =="noun":
            lemma = lemmatizer.lemmatize(word, pos="n")
        elif word_category =="verb":
            lemma = lemmatizer.lemmatize(word, pos="v")
        else:
            lemma = word
        lemmatized_words.append(lemma)
    # Retourne les mots lemmatisés comme une seule chaîne de caractères
    return ' '.join(lemmatized_words)
