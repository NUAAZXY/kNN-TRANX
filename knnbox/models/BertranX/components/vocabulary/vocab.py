from collections import Counter
from itertools import chain

class VocabEntry(object):
    def __init__(self):
        self.word2id = dict()
        self.unk_id = 3
        self.word2id['[PAD]'] = 0
        self.word2id['<s>'] = 1
        self.word2id['</s>'] = 2
        self.word2id['<unk>'] = 3

        # self.word2id['[PAD]'] = 0
        # self.word2id['[CLS]'] = 1
        # self.word2id['[MASK]'] = 2
        # self.word2id['[UNK]'] = 3
        # self.word2id['[SEP]'] = 4

        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add_sentence(self, sentences, freq_cutoff):
        print(sentences)
        word_freq = Counter(chain(*sentences))
        non_singletons = [w for w in word_freq if word_freq[w] > 1]
        singletons = [w for w in word_freq if word_freq[w] == 1]
        print('number of word types: %d, number of word types w/ frequency > 1: %d' % (len(word_freq),
                                                                                       len(non_singletons)))
        print('number of singletons: ', len(singletons))

        total_appearance_count = 0
        top_k_words = sorted(word_freq.keys(), reverse=True, key=word_freq.get)
        words_not_included = []
        for word in top_k_words:
            total_appearance_count += word_freq[word]
            if word_freq[word] >= freq_cutoff:
                self.add(word)
            else:
                words_not_included.append(word)

        print('number of words not included: %s' % len(words_not_included))
        appearance_count = 0
        for word in words_not_included:
            appearance_count += word_freq[word]

        print('total token count: ', total_appearance_count)
        print('unk token count: ', appearance_count)

    def add_tokens(self, list_primitives, freq_cutoff):
        word_freq = Counter(chain(*list_primitives))
        non_singletons = [w for w in word_freq if word_freq[w] > 1]
        singletons = [w for w in word_freq if word_freq[w] == 1]
        print('number of word types: %d, number of word types w/ frequency > 1: %d' % (len(word_freq),
                                                                                       len(non_singletons)))
        print('number of singletons: ', len(singletons))

        total_appearance_count = 0
        top_k_words = sorted(word_freq.keys(), reverse=True, key=word_freq.get)
        words_not_included = []
        for word in top_k_words:
            total_appearance_count += word_freq[word]
            if word_freq[word] >= freq_cutoff:
                self.add(word)
            else:
                words_not_included.append(word)

        self.add('Reduce_primitif')  # end of several primitives

        print('number of words not included: %s' % len(words_not_included))
        appearance_count = 0
        for word in words_not_included:
            appearance_count += word_freq[word]

        print('total token count: ', total_appearance_count)
        print('unk token count: ', appearance_count)

    def add_bert(self, vocab):
        self.word2id = vocab
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.unk_id = 100

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def is_unk(self, word):
        return word not in self

    def merge(self, other_vocab_entry):
        for word in other_vocab_entry.word2id:
            self.add(word)


class Vocab(object):
    def __init__(self, **kwargs):
        self.entries = []
        for key, item in kwargs.items():
            # assert isinstance(item, VocabEntry)
            self.__setattr__(key, item)

            self.entries.append(key)

    def __repr__(self):
        return 'Vocab(%s)' % (', '.join('%s %swords' % (entry, getattr(self, entry)) for entry in self.entries))
