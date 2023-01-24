""" from https://github.com/keithito/tacotron """

import re
import numpy as np
from text import cleaners
from text.symbols import symbols
from espnet2.text.cleaner import TextCleaner
from espnet2.text.phoneme_tokenizer import PhonemeTokenizer
from espnet2.text.token_id_converter import TokenIDConverter


_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


def get_arpabet(word, dictionary):
    word_arpabet = dictionary.lookup(word)
    if word_arpabet is not None:
        return "{" + word_arpabet[0] + "}"
    else:
        return word


def text_to_sequence(text, cleaner_names=["english_cleaners"], dictionary=None):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
      dictionary: arpabet class with arpabet dictionary

    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []
    space = _symbols_to_sequence(" ")
    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            clean_text = _clean_text(text, cleaner_names)
            if dictionary is not None:
                clean_text = [get_arpabet(w, dictionary) for w in clean_text.split(" ")]
                for i in range(len(clean_text)):
                    t = clean_text[i]
                    if t.startswith("{"):
                        sequence += _arpabet_to_sequence(t[1:-1])
                    else:
                        sequence += _symbols_to_sequence(t)
                    sequence += space
            else:
                sequence += _symbols_to_sequence(clean_text)
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    # remove trailing space
    if dictionary is not None:
        sequence = sequence[:-1] if sequence[-1] == space[0] else sequence
    return sequence

def ja_text_to_sequence(text, transcript_token_list, unk_symbol: str = "<unk>"):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
      dictionary: arpabet class with arpabet dictionary
    Returns:
      List of integers corresponding to the symbols in the text
    """
    transcript_token_id_converter = TokenIDConverter(
        token_list=transcript_token_list,
        unk_symbol=unk_symbol,
    )

    tokenizer = PhonemeTokenizer(
        g2p_type="pyopenjtalk_prosody",
        non_linguistic_symbols=None,
        space_symbol="<space>",
        remove_non_linguistic_symbols=False,
    )

    text_cleaner = TextCleaner("jaconv")
    text = text_cleaner(text)
    tokens = tokenizer.text2tokens(text)
    text_ints = transcript_token_id_converter.tokens2ids(tokens)
    sequence = np.array(text_ints, dtype=np.int64)

    return sequence


def ja_phonemes_to_sequence(tokens, transcript_token_list, unk_symbol: str = "<unk>"):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
      dictionary: arpabet class with arpabet dictionary
    Returns:
      List of integers corresponding to the symbols in the text
    """
    tokens = tokens.split()
    transcript_token_id_converter = TokenIDConverter(
        token_list=transcript_token_list,
        unk_symbol=unk_symbol,
    )

    # print(tokens)
    text_ints = transcript_token_id_converter.tokens2ids(tokens)
    sequence = np.array(text_ints, dtype=np.int64)

    return sequence


def text_to_sequence_for_editts(text, cleaner_names=["english_cleaners"], dictionary=None):
    sequence = []
    emphases = []
    final_emphases = []
    space = _symbols_to_sequence(" ")
    clean_text = _clean_text(text, cleaner_names)
    
    i = 0
    result = []
    emphasis_interval = []
    for w in clean_text.split(" "):
        if w == "|":
            emphasis_interval.append(i)
            if len(emphasis_interval) == 2:
                emphases.append(emphasis_interval)
                emphasis_interval = []
        else:
            i += 1
            result.append(get_arpabet(w, dictionary))

    clean_text = result
    emphasis_interval = []
    cnt = 0
    for i in range(len(clean_text)):
        t = clean_text[i]
        if cnt < len(emphases) and i == emphases[cnt][0]:
            emphasis_interval.append(len(sequence))
        
        if t.startswith("{"):
            sequence += _arpabet_to_sequence(t[1:-1])
        else:
            sequence += _symbols_to_sequence(t)
        
        if cnt < len(emphases) and i == emphases[cnt][1] -1:
            emphasis_interval.append(len(sequence))
            final_emphases.append(emphasis_interval)
            emphasis_interval = []
            cnt += 1
            
        sequence += space
        
    # remove trailing space
    if sequence[-1] == space[0]:
        sequence = sequence[:-1]

    return sequence, final_emphases


def ja_text_to_sequence_for_editts(
    text, transcript_token_list, unk_symbol: str = "<unk>"
):
    sequence = []
    emphases = []
    final_emphases = []

    transcript_token_id_converter = TokenIDConverter(
        token_list=transcript_token_list,
        unk_symbol=unk_symbol,
    )

    # tokenizer = PhonemeTokenizer(
    #     g2p_type="pyopenjtalk_prosody",
    #     non_linguistic_symbols=None,
    #     space_symbol="<space>",
    #     remove_non_linguistic_symbols=False,
    # )
    tokenizer = PhonemeTokenizer(
        g2p_type="pyopenjtalk",
        non_linguistic_symbols=None,
        space_symbol="<space>",
        remove_non_linguistic_symbols=False,
    )
    text_cleaner = TextCleaner("jaconv")
    clean_text = text_cleaner(text)
    text_for_seq = re.sub("\|", "", clean_text)
    tokens = tokenizer.text2tokens(text_for_seq)
    tokens.insert(0, "^")
    tokens.append("$")
    text_ints = transcript_token_id_converter.tokens2ids(tokens)
    sequence = np.array(text_ints, dtype=np.int64)

    i = 0
    result = []
    emphasis_interval = []
    for c in clean_text:
        if c == "|":
            emphasis_interval.append(i)
            if len(emphasis_interval) == 2:
                emphases.append(emphasis_interval)
                emphasis_interval = []
        else:
            i += 1
            result.append(c)

    final_emphases = []
    for emphasis in emphases:
        left_tokens = len(tokenizer.text2tokens(clean_text[: emphasis[0]]))
        right_tokens = len(tokenizer.text2tokens(clean_text[: emphasis[1]]))
        emphasis_interval = [left_tokens + 1, right_tokens + 1]
        final_emphases.append(emphasis_interval)

    return sequence, final_emphases


def ja_phoneme_to_sequence_for_editts(
    tokens, transcript_token_list, unk_symbol: str = "<unk>"
):
    emphases = []
    text = re.sub("\|", "", tokens)
    sequence = ja_phonemes_to_sequence(
        tokens=re.sub("\|", "", tokens),
        transcript_token_list=transcript_token_list,
        unk_symbol=unk_symbol,
    )

    i = 0
    emphasis_interval = []
    # print(type(tokens))
    # print(tokens.split())
    tokens = tokens.split()
    for token in tokens:
        # print(token)
        if token == "|":
            emphasis_interval.append(i)
            if len(emphasis_interval) == 2:
                emphases.append(emphasis_interval)
                emphasis_interval = []
        else:
            i += 1
    # print(emphases)
    # print("emphases:", text[emphases[0][0]])
    # print("emphases:", text[emphases[0][1]])
    return sequence, emphases

def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == "@":
                s = "{%s}" % s[1:]
            result += s
    return result.replace("}{", " ")


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(["@" + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s != "_" and s != "~"
