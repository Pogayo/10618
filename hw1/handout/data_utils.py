import pandas as pd


def load_data_from_csv(path):
    df = pd.read_csv(path)
    sentences = df['sentences'].tolist()
    tags = df['tags'].tolist()
    return sentences, tags


class Vocabulary:
    def __init__(self, vocab_size=100000):
        self.vocab_size = vocab_size
        self.idx_to_str = {0: '<PAD>', 1: '<UNK>'}
        self.str_to_idx = {j: i for i, j in self.idx_to_str.items()}

    def __len__(self):
        return len(self.idx_to_str)

    def tokenizer(self, text):
        return [token.lower().strip() for token in text.split(' ')]

    def words_vocabulary(self, sentence_list):
        counts = {}
        idx = 2  # idx = 0 and 1 are reserved for <PAD> and <UNK>

        # calculate counts of words
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                if word not in counts.keys():
                    counts[word] = 1
                else:
                    counts[word] += 1

        counts = dict(sorted(counts.items(), key=lambda x: -x[1])[:self.vocab_size-idx])

        # create vocab
        for word in counts.keys():
            self.str_to_idx[word] = idx
            self.idx_to_str[idx] = word
            idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        numericalized_text = []
        for token in tokenized_text:
            if token in self.str_to_idx.keys():
                numericalized_text.append(self.str_to_idx[token])
            else:
                numericalized_text.append(self.str_to_idx['<UNK>'])
        return numericalized_text


class Tags:
    def __init__(self):
        self.idx_to_tag = {}
        self.tag_to_idx = {j: i for i, j in self.idx_to_tag.items()}

    def __len__(self):
        return len(self.idx_to_tag)

    def tags_sentence(self, text):
        return [tag for tag in text.split(' ')]

    def tags_vocabulary(self, sentence_list):
        tags = set()
        idx = 0
        for pos_tags in sentence_list:
            for tag in self.tags_sentence(pos_tags):
                if tag not in tags:
                    tags.add(tag)
                    self.idx_to_tag[idx] = tag
                    self.tag_to_idx[tag] = idx
                    idx += 1

    def numericalize(self, pos_tags):
        tags_list = self.tags_sentence(pos_tags)
        numericalized_tags = []
        for token in tags_list:
            numericalized_tags.append(self.tag_to_idx[token])
        return numericalized_tags

