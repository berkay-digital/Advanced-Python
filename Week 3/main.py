import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN

def custom_lemmatize(word, lemmatizer, pos):
    lemma = lemmatizer.lemmatize(word, pos=pos)
    if word.endswith('ing'):
        verb_lemma = lemmatizer.lemmatize(word, pos=nltk.corpus.wordnet.VERB)
        return verb_lemma if len(verb_lemma) < len(lemma) else lemma
    return lemma

def lemmatize_paragraph(paragraph):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(paragraph.lower())
    pos_tags = pos_tag(tokens)
    
    lemmatized_tokens = []
    for word, pos in pos_tags:
        if word not in stop_words and word.isalnum():
            lemma = custom_lemmatize(word, lemmatizer, get_wordnet_pos(pos))
            lemmatized_tokens.append(lemma)
    
    return ' '.join(lemmatized_tokens)

complex_paragraph = """
The resplendent quetzal's iridescent plumage shimmered in the dappled sunlight 
filtering through the dense canopy of the cloud forest. As the avian marvel 
alighted upon a gnarled branch, its elongated tail feathers cascaded gracefully, 
creating a mesmerizing spectacle for the fortunate observers. The creature's 
piercing gaze seemed to penetrate the very essence of its surroundings, as if 
contemplating the intricate web of life that pulsated within this verdant realm.
"""

lemmatized_paragraph = lemmatize_paragraph(complex_paragraph)

print("Original paragraph:")
print(complex_paragraph)
print("\nLemmatized paragraph:")
print(lemmatized_paragraph)
