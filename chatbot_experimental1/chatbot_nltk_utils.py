# This file contains all the utilities needed to preprocess the data in the intents.json file, we use the natural language toolkit module to tokenize, lower + stem and exclude punctuation marks
import nltk
import numpy as np
from nltk.stem.snowball import EnglishStemmer   # A stemmer algorithm developted by Martin Porter, the Snowball Stemmer is an improved version of the Porter Stemmer
stemmer = EnglishStemmer()

# Run the code with this line once to download, once downloaded this line can be commented out or deleted
# nltk.download('punkt_tab')

# This function tokenizes the sentences in the dataset
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# This function stems the words in the sentences of our dataset and lowers them. Stemming cuts the end of the words.
def stem(word):
    return stemmer.stem(word.lower())

# This function returns a bag of words array, 1 for each known word that exists in the sentence, 0 otherwise
def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]    # Applies stemming to the tokenized sentences passed
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag


def main():
    # Example of how the tokenizer works:
    a = "Norway has a comprehensive welfare state model, funded by taxes and oil revenues, providing universal healthcare, free education (including university), generous parental leave, and strong social safety nets."
    print("Sentence:\n", a)
    a = tokenize("Norway has a comprehensive welfare state model, funded by taxes and oil revenues, providing universal healthcare, free education (including university), generous parental leave, and strong social safety nets.")
    print("Tokenized Sentence:\n", a)

    # Example of how stemming works:
    words = a
    stemmed_words = [stem(w) for w in words]
    print(stemmed_words)

    # Example of how bag of words works:
    a_w = stemmed_words
    a = ["oil", "has", "strong", "revenue"]
    print(bag_of_words(a, a_w))    # Returns 1's for the words that were in the sentence

if __name__ == "__main__":
    main()
