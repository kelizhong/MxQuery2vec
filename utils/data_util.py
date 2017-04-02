# coding=utf-8
import pickle
import re
import codecs
from common import constant as config
import string
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from common.constant import bos_word, eos_word
wn_lemmatizer = WordNetLemmatizer()

"""
def read_data(encoder_path, decoder_path, max_line_num=sys.maxsize):
    encoder_data = []
    decoder_data = []
    with codecs.open(encoder_path) as encoder, codecs.open(decoder_path) as decoder:
        for encoder_line, decoder_line in itertools.izip(itertools.islice(encoder, max_line_num),
                                                         itertools.islice(decoder, max_line_num)):
            try:
                # dcoder the line from utf-8 to unicode. assume all the data file is utf-8 format
                encoder_line = encoder_line.decode('utf8').strip().lower()
                decoder_line = decoder_line.decode('utf8').strip().lower()
                encoder_line = tokenize(encoder_line)
                decoder_line = tokenize(decoder_line)
            except Exception:
                # ignore the error line
                continue
            if len(encoder_line) and len(decoder_line):
                encoder_data.append(encoder_line)
                decoder_data.append(decoder_line)
    return encoder_data, decoder_data
"""


def words_gen(filename, bos=None, eos=None):
    num = 1
    """Return each word in a line."""
    with codecs.open(filename, encoding='utf-8', errors='ignore') as f:
        for line in f:
            num += 1
            if num % 10000 == 0:
                print(num)
            tokens = tokenize(line)
            tokens = [bos] + tokens if bos is not None else tokens
            tokens = tokens + [eos] if eos is not None else tokens
            for w in tokens:
                w = w.strip().lower()
                if len(w):
                    yield w


def sentence_gen(files):
    """Return each sentence in a line."""
    if not isinstance(files, list):
        files = [files]
    for filename in files:
        with codecs.open(filename, encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip().lower()
                if len(line):
                    yield line


def aksis_sentence_gen(filename):
    for line in sentence_gen(filename):
        line = extract_query_title_from_aksis_data(line)
        if len(line):
            yield line


def stem_tokens(tokens, lemmatizer):
    return [lemmatizer.lemmatize(token) for token in tokens]


def tokenize(text, lemmatizer=wn_lemmatizer):
    text = clean_html(text)
    tokens = word_tokenize(text)
    tokens = [i for i in tokens if i not in string.punctuation]
    stems = stem_tokens(tokens, lemmatizer)
    return stems


def extract_query_title_from_aksis_data(sentence):
    sentence = sentence.strip().lower()
    items = re.split(r'\t+', sentence)
    if len(items) == 7 and len(items[2]) and len(items[6]):
        return items[2] + " " + items[6]
    else:
        return str()


def extract_query_title_score_from_aksis_data(sentence):
    sentence = sentence.strip().lower()
    items = re.split(r'\t+', sentence)
    if len(items) == 7 and len(items[2]) and len(items[3]) and len(items[6]):
        return tokenize(items[2]), tokenize(items[6]), items[3]
    else:
        return None, None, None


def extract_raw_query_title_score_from_aksis_data(sentence):
    sentence = sentence.strip().lower()
    items = re.split(r'\t+', sentence)
    if len(items) == 7 and len(items[2]) and len(items[3]) and len(items[6]):
        return items[2], items[6], items[3]
    else:
        return None, None, None


def query_title_score_generator_from_aksis_data(files):
    for line in sentence_gen(files):
        query, title, score = extract_raw_query_title_score_from_aksis_data(line)
        if query and title and score:
            yield query, title, score


def load_pickle_object(path):
    """
    Load vocab from file, the 0, 1, 2, 3 should be reserved for pad, <unk>, <s>, </s>

    Args:
        path: the vocab path

    Returns:
        vocab dict
    """
    with open(path, 'rb') as f:
        obj = pickle.load(f)

    return obj


def load_vocabulary_from_pickle(path, top_words=40000, special_words=dict()):
    vocab_pickle = load_pickle_object(path)

    words_num = len(vocab_pickle)
    special_words_num = len(special_words)

    assert words_num > special_words_num, "the size of total words must be larger than the size of special_words"

    assert top_words > special_words_num, "the value of most_commond_words_num must be larger than the size of special_words"

    vocab = dict()
    vocab.update(special_words)
    for word, _ in vocab_pickle:
        if len(vocab) >= top_words:
            break
        if word not in special_words:
            vocab[word] = len(vocab)

    return vocab


def sentence2id(sentence, the_vocab):
    words = [the_vocab[w.strip().lower()] if w.strip().lower() in the_vocab else the_vocab[config.unk_word] for w in
             sentence if len(w) > 0]
    return words


def word2id(word, the_vocab):
    word = word.strip().lower()
    return the_vocab[word] if word in the_vocab else the_vocab[config.unk_word]


def convert_data_to_id(encoder_words, decoder_words, encoder_vocab, decoder_vocab):
    """convert the data into id which represent the index for word in vocabulary"""
    encoder = encoder_words
    decoder = [bos_word] + decoder_words
    label = decoder_words + [eos_word]
    encoder_sentence_id = sentence2id(encoder, encoder_vocab)
    decoder_sentence_id = sentence2id(decoder, decoder_vocab)
    label_id = sentence2id(label, decoder_vocab)
    return encoder_sentence_id, decoder_sentence_id, label_id


def clean_html(html):
    """
    Copied from NLTK package.
    Remove HTML markup from the given string.

    Args:
        html: str
            the HTML string to be cleaned

    Returns: str
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