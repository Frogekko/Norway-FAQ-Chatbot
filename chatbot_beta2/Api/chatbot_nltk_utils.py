# This file contains all the utilities needed to preprocess the data in the intents.json file, we use the natural language toolkit module to tokenize, lower + stem and exclude punctuation marks
import nltk
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer    # We decided to go for a lemmatizer instead of a stemmer since it
from nltk.corpus import wordnet    # Needed for POS tag mapping

# Run the code with these lines once to download, once downloaded these lines can be commented out or deleted
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger_eng')

lemmatizer = WordNetLemmatizer()

# This function returns the correct Part-of-Speech tag of a word in order to lemmatize it correctly, otherwise it defaults to noun returning the wrong version of the word (f.ex. "has" will become "ha" instead of "have")
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    if treebank_tag.startswith('N'):
        return wordnet.NOUN
    if treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN    # Defaults to noun if not specifically found

# This function tokenizes the sentences in the dataset
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# This function stems the words in the sentences of our dataset and lowers them. Stemming cuts the end of the words.
def lem(word, pos_tag):
    return lemmatizer.lemmatize(word.lower(), pos=pos_tag)

# This function returns a bag of words array, 1 for each known word that exists in the sentence, 0 otherwise
def bag_of_words(tokenized_sentence, all_words):
    pos_tags = nltk.pos_tag(tokenized_sentence)    # Get pos tags for the tokenized sentence
    
    # Lemmatizes the sentence using the correct POS tags
    sentence_lemmas = []
    for word, tag in pos_tags:
        wordnet_tag = get_wordnet_pos(tag)
        lemma = lem(word, wordnet_tag)
        sentence_lemmas.append(lemma)
    
    # Create the bag
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, vocabulary_lemma in enumerate(all_words):
        if vocabulary_lemma in sentence_lemmas:
            bag[idx] = 1.0
    return bag

def main():
    # Test sentence
    sentence = "Norway has a comprehensive welfare state model, funded by taxes and oil revenues, providing universal healthcare, free education (including university), generous parental leave, and strong social safety nets."
    print("Sentence:\n", sentence)
    
    # Example of tokenization works:
    tokens = tokenize(sentence)
    print("Tokenized Sentence:\n", tokens)

    # Example of POS tagging and Lemmatization:
    pos_tags = nltk.pos_tag(tokens)
    print("\nPOS Tags:\n", pos_tags)

    lemmatized_output = []
    for word, tag in pos_tags:
        wordnet_tag = get_wordnet_pos(tag)
        lemma = lem(word, wordnet_tag)
        lemmatized_output.append(lemma)
    print("\nProperly Lemmatized Output:\n", lemmatized_output)

    # Example of bag of words:
    bagged = lemmatized_output
    sentence_to_bag = ["oil", "have", "strong", "revenue"]
    print(bag_of_words(sentence_to_bag, bagged))    # Returns 1's for the words that were in the sentence

if __name__ == "__main__":
    main()
