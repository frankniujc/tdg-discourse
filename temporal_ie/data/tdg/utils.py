from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer

CONVERT_TOKENS = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
    "-LCB-": "{",
    "-RCB-": "}",
    "``": '"',
    "''": '"',
}

T2T_LABELS = {
    'None':0,
    'Depend-on': 1,
    'included': 2,
}

E2T_LABELS = {
    'None': 0,
    'Depend-on': 1,
    'before': 2,
    'after': 3,
    'overlap': 4,
    'included': 5,
}

E2E_LABELS = {
    'None': 0,
    'before': 1,
    'after': 2,
    'overlap': 3,
}


def detokenize(tokens):
    detok_sent = TreebankWordDetokenizer().detokenize([CONVERT_TOKENS.get(x, x) for x in tokens])
    detok_spans = get_spans(tokens, detok_sent)
    return detok_sent, detok_spans

def get_spans(tokens, detok_sent):
    spans = []
    offset = 0
    for tok_token in tokens:
        token = CONVERT_TOKENS.get(tok_token, tok_token)
        if detok_sent[offset:offset+len(token)] == token:
            spans.append((offset, offset+len(token)))
            offset = offset+len(token)
        else:
            assert detok_sent[offset+1:offset+len(token)+1] == token
            spans.append((offset+1, offset+len(token)+1))
            offset = offset+len(token)+1
    return spans
    
