# coding=utf-8
"""util for data processing"""
import re
import codecs
from common import constant as config
import string
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from common.constant import bos_word, eos_word
from utils.pickle_util import load_pickle_object
import numpy as np
from helper.redis_helper import RedisHelper

wn_lemmatizer = WordNetLemmatizer()
redis_helper = RedisHelper('w2v.aka.corp.amazon.com')


def words_gen(filename, bos=None, eos=None):
    """Generator that yield each word in a line.
    Parameters
    ----------
        filename: str
            data file name
        bos: str
            tag, beginning of sentence
        eos: str
            tag, ending of sentence
    """
    with codecs.open(filename, encoding='utf-8', errors='ignore') as f:
        for line in f:
            tokens = tokenize(line)
            tokens = [bos] + tokens if bos is not None else tokens
            tokens = tokens + [eos] if eos is not None else tokens
            for w in tokens:
                w = w.strip().lower()
                if len(w):
                    yield w


def sentence_gen(files):
    """Generator that yield each sentence in a line.
    Parameters
    ----------
        files: list
            data file list
    """
    if not isinstance(files, list):
        files = [files]
    for filename in files:
        with codecs.open(filename, encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip().lower()
                if len(line):
                    yield line


def aksis_sentence_gen(filename):
    """Generator that yield each sentence in aksis corpus.
    Parameters
    ----------
        filename: str
            data file name
    """
    for line in sentence_gen(filename):
        line = extract_query_title_from_aksis_data(line)
        if len(line):
            yield line


def stem_tokens(tokens, lemmatizer):
    """lemmatizer
    Parameters
    ----------
        tokens: list
            token for lemmatizer
        lemmatizer: stemming model
            default model is wordnet lemmatizer
    """
    return [lemmatizer.lemmatize(token) for token in tokens]


def tokenize(text, lemmatizer=wn_lemmatizer):
    """tokenize and lemmatize the text"""
    text = clean_html(text)
    tokens = word_tokenize(text)
    tokens = [i for i in tokens if i not in string.punctuation]
    stems = stem_tokens(tokens, lemmatizer)
    return stems


def extract_query_title_from_aksis_data(sentence):
    """extract the query and title from aksis raw data, this function is for building up vocabulary
    Aksis data format: MarketplaceId\tAsin\tKeyword\t Score\tActionType\tDate
    ActioType: 1-KeywordsByAdds, 2-KeywordsBySearches, 3-KeywordsByPurchases, 4-KeywordsByClicks
    """
    sentence = sentence.strip().lower()
    items = re.split(r'\t+', sentence)
    if len(items) == 7 and len(items[2]) and len(items[6]):
        return items[2] + " " + items[6]
    else:
        return str()


def extract_raw_query_title_score_from_aksis_data(sentence):
    """extract the query, title and score from aksis raw data, this function is to generate training data
    score gives a rough idea about specificness of a query. For example query1: "iphone" and query2: "iphone 6s 64GB".
    In both the query customer is looking for iphone but query2 is more specific.
    Query specificity score is number which ranges from 0.0 to 1.0.
    Aksis data format: MarketplaceId\tAsin\tKeyword\t Score\tActionType\tDate
    ActioType: 1-KeywordsByAdds, 2-KeywordsBySearches, 3-KeywordsByPurchases, 4-KeywordsByClicks
    """
    sentence = sentence.strip().lower()
    items = re.split(r'\t+', sentence)
    if len(items) == 7 and len(items[2]) and len(items[3]) and len(items[6]):
        return items[2], items[6], items[3]
    else:
        return None, None, None


def query_title_score_generator_from_aksis_data(files):
    """Generator that yield query, title, score in aksis corpus"""
    for line in sentence_gen(files):
        query, title, score = extract_raw_query_title_score_from_aksis_data(line)
        if query and title and score:
            yield query, title, score


def load_vocabulary_from_pickle(path, top_words=40000, special_words=dict()):
    """load vocabulary from pickle
    Parameters
    ----------
        path: str
            corpus path
        top_words: int
            the max words num in the vocabulary
        special_words: dict
         special_words like <unk>, <s>, </s>
    Returns
    -------
        vocabulary
    """
    vocab_pickle = load_pickle_object(path)

    words_num = len(vocab_pickle)
    special_words_num = len(special_words)

    if words_num <= special_words_num:
        raise ValueError("the size of total words must be larger than the size of special_words")

    if top_words <= special_words_num:
        raise ValueError("the value of most_commond_words_num must be larger "
                         "than the size of special_words")

    vocab = dict()
    vocab.update(special_words)
    for word, _ in vocab_pickle:
        if len(vocab) >= top_words:
            break
        if word not in special_words:
            vocab[word] = len(vocab)

    return vocab


def sentence2id(sentence, the_vocab):
    """convert the sentence to the index in vocabulary"""
    words = [the_vocab[w.strip().lower()] if w.strip().lower() in the_vocab else the_vocab[config.unk_word] for w in
             sentence if len(w) > 0]
    return words


def word2id(word, the_vocab):
    """convert the word to the index in vocabulary"""
    word = word.strip().lower()
    return the_vocab[word] if word in the_vocab else the_vocab[config.unk_word]


def convert_data_to_id1(encoder_words, decoder_words, encoder_vocab, decoder_vocab):
    """convert the seq2seq data into id which represent the index for word in vocabulary"""
    encoder = encoder_words
    decoder = [bos_word] + decoder_words
    label = decoder_words + [eos_word]
    encoder_sentence_id = sentence2id(encoder, encoder_vocab)
    decoder_sentence_id = sentence2id(decoder, decoder_vocab)
    label_id = sentence2id(label, decoder_vocab)
    return encoder_sentence_id, decoder_sentence_id, label_id


def convert_data_to_id(encoder_words, decoder_words, encoder_vocab, decoder_vocab):
    """convert the seq2seq data into id which represent the index for word in vocabulary"""
    encoder = encoder_words
    decoder = [bos_word] + decoder_words
    label = decoder_words + [eos_word]
    encoder_sentence_id = words2vectors(encoder)
    decoder_sentence_id = sentence2id(decoder, decoder_vocab)
    label_id = sentence2id(label, decoder_vocab)
    return encoder_sentence_id, decoder_sentence_id, label_id


def words2vectors(sentence):
    return np.array([word2vec(word) for word in sentence])


def word2vec(word, word2vec_dim=128):
    """Convert word to vector"""
    vec = redis_helper.get_data(word).get('result')
    vec = np.fromstring(vec, sep=',') if vec else np.zeros(word2vec_dim)
    return vec


def clean_html(html):
    """
    Copied from NLTK package.
    Remove HTML markup from the given string.

    Parameters
    ----------
        html: str
            the HTML string to be cleaned
    """

    # First we remove inline JavaScript/CSS:
    cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
    # Then we remove html comments. This has to be done before removing regular
    # tags since comments can contain '>' characters.
    cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned)
    # Next we can remove the remaining tags:
    cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
    # Finally, we deal with whitespace
    cleaned = re.sub(r"&nbsp;", " ", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)
    return cleaned.strip()
