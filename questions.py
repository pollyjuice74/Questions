import nltk
import sys
import os
import string
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#nltk.download('stopwords')

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = {}
    for file in os.listdir(directory):
        if file.endswith('.txt'):
            path = os.path.join(directory, file)
            with open(path, 'r') as f:
                files[file] = f.read()

    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokens = nltk.word_tokenize(document.lower())

    words = [
        token for token in tokens
        if token not in stopwords.words('english') and # removes determinants etc.
        token not in string.punctuation # removes punctuation
    ]

    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = {}
    all_words = set(word for words in documents.values() for word in words) # set of all words in documents without repeating

    for word in all_words:
        freq = sum(1 for words in documents.values() if word in words)
        idf = math.log(len(documents) / freq)
        idfs[word] = idf

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.

    tf = {}
    for words in documents.values():
        for word in set(words):
            tf[word] = words.count(word) / len(words)
    """
    scores = {}
    for file, words in files.items():
        file_score = 0
        for word in query:
            if word in words:
                tf = words.count(word) / len(words) # calculate tf
                tf_idf_score = tf * idfs[word]
                file_score += tf_idf_score
        scores[file] = file_score
    sorted_files = sorted(files.keys(), key=lambda x: -scores[x]) # sorts files in descending order of tf idf score

    return sorted_files[:n] # return up to n files


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_scores = {}
    for sentence, words in sentences.items():
        score = 0
        num_query_words = 0
        for word in query:
            if word in words:
                score += idfs[word]
                num_query_words += 1
        query_term_density = num_query_words / len(sentences[sentence])
        sentence_scores[sentence] = score, query_term_density # makes tuple of score and query term density

    top_sentences = sorted(sentence_scores.items(), key=lambda x: (-x[1][0], -x[1][0]))[:n] # indexes into value tuple and sorts by score (-x[1][0]) in descending order

    return top_sentences[0]


if __name__ == "__main__":
    main()
