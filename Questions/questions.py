import nltk
import sys
import os
import string
from numpy import log as ln
from numpy.core.defchararray import count

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
    files = dict()
    for file in os.listdir(directory) :
        with open(os.path.join(directory,file), "r") as file_obj :
            print(f"testing.. printing os.path.join(directory,file) : {os.path.join(directory,file)}")
            text = file_obj.read()
        files[file] = text
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = [word for word in nltk.word_tokenize(document.lower()) if word not in set(string.punctuation) and word not in set(nltk.corpus.stopwords.words("english"))]
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    all_words = set()
    for document in documents :
        all_words.update(documents[document])
    idf_words = dict()
    for word in all_words :
        appearing = 0
        for document in documents.values() :
            if word in document :
                appearing += 1
        idf_words[word] = ln(len(documents)/appearing)
    return idf_words


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    file_score_dict = dict()
    for file in files :
        total_file_score = 0
        for word in query :
            if word in files[file] :
                word_score = idfs[word]*files[file].count(word)
                total_file_score += word_score
        file_score_dict[file] = total_file_score
    return [k for k, _ in sorted(file_score_dict.items(), key=lambda item: item[1], reverse=True)][:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_scores = dict()
    for sentence in sentences :
        total_sentence_score = 0
        for word in query :
            if word in sentences[sentence] :
                total_sentence_score += idfs[word]
        sentence_scores[sentence] = total_sentence_score
    sorted_scores = {sentence:score for sentence,score in sorted(sentence_scores.items(),key=lambda item : item[1], reverse=True)}
    sorted_final = list()
    for sentence1 in sorted_scores :
        s1_term_density = len([word for word in query if word in sentences[sentence1]])/len(sentences[sentence1])
        to_be_sorted = [(sentence1,s1_term_density),]
        for sentence2 in sorted_scores :
            if sorted_scores[sentence1] == sorted_scores[sentence2] :
                s2_term_density = len([word for word in query if word in sentences[sentence2]])/len(sentences[sentence2])
                to_be_sorted += [(sentence2, s2_term_density),]
        # Sorting sentences with eqaul idf values based on query term density
        sorted_list = [sentence for sentence, _ in sorted(to_be_sorted, key=lambda item : item[1], reverse=True)]
        sorted_final += sorted_list
    return sorted_final[:n]


if __name__ == "__main__":
    main()
