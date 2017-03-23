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
    """Return each word in a line."""
    with codecs.open(filename, encoding='utf-8', errors='ignore') as f:
        for line in f:
            tokens = word_tokenize(line)
            tokens = [bos] + tokens if bos is not None else tokens
            tokens = tokens + [eos] if eos is not None else tokens
            for w in tokens:
                w = w.strip().lower()
                if len(w):
                    yield w


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
