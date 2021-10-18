from collections import OrderedDict

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Vocabulary:
    def __init__(self, special_tokens: OrderedDict):
        self.special_tokens = special_tokens

        self.token2id = OrderedDict()
        self.id2token = OrderedDict()

        self.filename = None

        for special_token in self.special_tokens.values():
            self.set_id(str(special_token))

    def add_tokens(self, tokens):
        for token in tokens:
            self.set_id(str(token))

    def set_id(self, token):
        if token not in self.token2id:
            id = len(self.token2id)

            self.token2id[token] = id
            self.id2token[id] = token
        else:
            id = self.token2id[token]

        return id

    def get_id(self, token):
        if token in self.token2id:
            id = self.token2id[token]
        else:
            raise Exception(f"token {token} not found")
        return id

    def save_vocab(self, fname):
        self.filename = fname
        with open(fname, "w") as fout:
            for idx in self.id2token:
                token = self.id2token[idx]
                fout.write("%s\n" % token)

    def get_special_tokens(self):
        return AttrDict({k: str(v) for k,v in self.special_tokens.items()})

    def __len__(self):
        return len(self.id2token)

    def __str__(self):
        str_ = 'vocab: [{} tokens]  '.format(len(self))
        return str_
