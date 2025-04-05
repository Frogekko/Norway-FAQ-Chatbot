# This file contains all the utilities needed to preprocess the data in the intents.json file, we use the natural language toolkit module to tokenize, lower + stem and exclude punctuation marks
import nltk
from nltk.stem.snowball import EnglishStemmer   # A stemmer algorithm developted by Marting Porter, the Snowball Stemmer is an improved version of the Porter Stemmer
stemmer = EnglishStemmer()

# Run the code with this line once to download, once downloaded this line can be commented out or deleted
# nltk.download('punkt_tab')

# This function tokenizes the sentences in the dataset
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# This function stems the words in the sentences of our dataset and lowers them. Stemming cuts the end of the words.
def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    pass



# Example of how the tokenizer works:
a = "Norway has a comprehensive welfare state model, funded by taxes and oil revenues, providing universal healthcare, free education (including university), generous parental leave, and strong social safety nets."
print("Sentence:\n", a)
a = tokenize("Norway has a comprehensive welfare state model, funded by taxes and oil revenues, providing universal healthcare, free education (including university), generous parental leave, and strong social safety nets.")
print("Tokenized Sentence:\n", a)

# Example of how stemming works:
words = a
stemmed_words = [stem(w) for w in words]
print(stemmed_words)

