# coding=utf-8
import pickle
import re
import sys
import codecs
from common import constant as config
from nltk.tokenize import word_tokenize
import itertools


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
                encoder_line = word_tokenize(encoder_line)
                decoder_line = word_tokenize(decoder_line)
            except Exception:
                # ignore the error line
                continue
            if len(encoder_line) and len(decoder_line):
                encoder_data.append(encoder_line)
                decoder_data.append(decoder_line)
    return encoder_data, decoder_data


def words_gen(filename, bos=None, eos=None):
    num = 1
    """Return each word in a line."""
    with codecs.open(filename, encoding='utf-8', errors='ignore') as f:
        for line in f:
            num += 1
            if num % 10000 == 0:
                print(num)
            tokens = word_tokenize(line)
            tokens = [bos] + tokens if bos is not None else tokens
            tokens = tokens + [eos] if eos is not None else tokens
            for w in tokens:
                w = w.strip().lower()
                if len(w):
                    yield w


def sentence_gen(filename):
    """Return each sentence in a line."""
    with codecs.open(filename, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip().lower()
            if len(line):
                yield line


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
    vocab = load_pickle_object(path)
    words_num = len(vocab)
    special_words_num = len(special_words)

    assert words_num > len(
       special_words), "the size of total words must be larger than the size of special_words"

    assert top_words > len(
       special_words), "the value of most_commond_words_num must be larger than the size of special_words"

    vocab_count = vocab.most_common(top_words - special_words_num)
    vocab = {}
    idx = special_words_num + 1
    for word, _ in vocab_count:
        if word not in special_words:
            vocab[word] = idx
            idx += 1
    vocab.update(special_words)

    return vocab


def sentence2id(sentence, the_vocab):
    words = [the_vocab[w.strip().lower()] if w.strip().lower() in the_vocab else the_vocab[config.unk_word] for w in
             sentence if len(w) > 0]
    return words


def word2id(word, the_vocab):
    word = word.strip().lower()
    return the_vocab[word] if word in the_vocab else the_vocab[config.unk_word]


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
