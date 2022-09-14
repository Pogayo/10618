from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import Vocabulary, Tags, load_data_from_csv
import numpy as np
import random
from tqdm import tqdm
from typing import List, Tuple

import wandb


def pad_sequences(batch, pad_index):
    # get the size of the longest sequence
    max = 0
    for item in batch:
        if len(item) > max:
            max = len(item)
    # pad
    for i in range(len(batch)):
        no_pad_tokens = max - len(batch[i])  # get number of padding tokens
        batch[i] = torch.cat([batch[i], torch.full((no_pad_tokens,), pad_index)]).numpy()
        assert len(batch[i]) == max

    return batch  # Please make sure to return the numericalized sentence and POS tag sequence as numpy arrays.


class parTUTDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, taggings, voc: Vocabulary, tags: Tags):
        self.sentences = np.array(sentences)
        self.taggings = np.array(taggings)

        self.voc = voc
        self.tags = tags

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, ind):
        X = np.array(self.voc.numericalize(self.sentences[ind]))
        y = np.array(self.tags.numericalize(self.taggings[ind]))
        return X, y

    def collate_fn(self, batch: List[Tuple]):
        '''
        A function that "collates" a sequence of input sentences into a single
        batch by padding to the maximum input sentence length

        Inputs:
            batch is a List of Tuples such that:
                len(batch) is the batch size
                batch[i] is a Tuple of size 2 containing the ith datapoint
                batch[i][0] is an np.ndarray of shape (L,) of dtype np.int64, where L is the number of tokens in the ith sentence
                    - consists of encoded input tokens (tokens converted into their token IDs)
                batch[i][1] is an np.ndarray of shape (L,) of dtype np.int64, where L is the number of tokens in the ith sentence
                    - consists of encoded POS tag IDs (POS tags converted into their IDs)

        Outputs:
            batch_x: pytorch tensor of token IDs
                - of shape (len(batch), max_seq_len) and dtype torch.int64
            batch_y: pytorch tensor of POS tag IDs
                - of shape (len(batch), max_seq_len) and dtype torch.int64

        '''

        batch_x = np.array([torch.from_numpy(x) for x, y in batch])
        # TODO: Zero-pad the sequences in batch_x and collate into a single tensor "batch_x"
        #  of shape (batch_size, max_seq_len) where max_seq_len is the longest sequence in the input
        # raise NotImplementedError("Please implement the TODO here!")
        # get the size of the  longest sequence
        batch_x = pad_sequences(batch_x, 0)

        batch_y = np.array([torch.from_numpy(y) for x, y in batch])
        # TODO: Pad the sequences in batch_y with -100, and collate into a single tensor "batch_y"
        #  of shape (batch_size, max_seq_len) where max_seq_len is the longest sequence in the input
        #  Please see https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html to understand
        #  what ignore_index does, and why we pad labels with -100
        # raise NotImplementedError("Please implement the TODO here!")
        batch_y = pad_sequences(batch_y, -100)

        return batch_x, batch_y


class ConvEncoder(nn.Module):
    '''
    A 1d CNN encoder implemented using nn.Linear, with a GELU activation function

    Args:
        emb_size: The dimension of each token's embedding
        window_size: The convolution operation's window size
            (the number of tokens that each filter looks at)
        pad_idx: The padding token's ID
        vocab_size: The size of the input token vocabulary (including padding)
        out_size: The dimension of the outputs produced by the encoder

    '''

    def __init__(
            self, emb_size: int, window_size: int, out_size: int, vocab_size: int, pad_idx: int
    ):
        super(ConvEncoder, self).__init__()
        assert window_size % 2 == 1, \
            "'window_size' should be odd. We only test with equal left/right contexts."

        # TODO: Save any attributes needed for a forward pass
        self.emb_size = emb_size
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.out_size = out_size
        # raise NotImplementedError("Please implement the TODO here!")

        # TODO: Define an embedding layer, ensuring that the padding token gets a 0 embedding
        self.embeddings = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        # raise NotImplementedError("Please implement the TODO here!")

        # TODO: Use nn.Linear to define a flattened convolution kernel
        # '''calculate window size -- from outputs of get_windows
        # outputs.shape: (B, num_windows, window_size, emb_size)'''
        self.linear_layer = nn.Sequential(
            nn.Linear(emb_size * window_size, out_size),
            nn.GELU())

        # raise NotImplementedError("Please implement the TODO here!")

    @staticmethod
    def get_windows(padded_inputs, window_size):
        '''
        This function takes in padded inputs, and a given window_size to return
        inputs for the linear layer that does the convolution operation.

        padded_inputs.shape: (B, input_len, emb_size)
        outputs.shape: (B, num_windows, window_size, emb_size)

        where:
            B is the batch dimension
            window_size: The size of each convolution filter
            and outputs[:, i] contains the inputs from placing the window at the i'th index from the start
        '''

        # TODO: Slide a window of size 'window_size' along the input tensor and stack slices
        #  such that the outputs for the i'th sample and the j'th window go into outputs[i, j]
        # raise NotImplementedError("Please implement the TODO here!")
        batch_size = len(padded_inputs)
        input_len = len(padded_inputs[0])
        print(f'INPUT LEN{ input_len}')

        embed_size = len(padded_inputs[0][0])

        #working assumption == number of windows = input_length
        no_pads = ((window_size - 1) // 2)*2
        outputs = torch.empty(batch_size,input_len-no_pads, window_size, embed_size)

        for i in range(batch_size):
            for j in range(input_len):
                if (j + window_size) <= input_len:  # check for boundaries
                    output = padded_inputs[i][j: j + window_size]
                    outputs[i][j]=output


        return outputs

    def forward(self, x):
        '''
        Inputs:
            x - shape: (B, max_seq_len), dtype torch.int64

        where:
            x is a pytorch tensor containing a batch of padded sentences encoded into token IDs,
                with each sentence in the batch padded to the length 'max_seq_len' (the maximum
                sentence length in that batch)
            B is the batch dimension
        
        Outputs are of shape (B, max_seq_len, out_size)
        '''
        batch_size = x.size(0)
        max_seq_len = x.size(1)


        # TODO: Obtain embeddings for the input indices
        embeddings = self.embeddings(x, )
        # raise NotImplementedError("Please implement the TODO here!")

        # TODO: Zero-pad the inputs to the convolution layer to ensure we get an output sequence of the
        #  same length as the input sequence
        no_pads = (
                          self.window_size - 1) // 2  # source: https://www.quora.com/What-if-I-want-a-same-dimensional-Output-as-Input-with-filter-F-x-F-stride-2-and-using-zero-padding-in-Convolutional-Neural-Network
        # x dimension so far is  (B, max_seq_len, emb_size)

        pad = (0, 0, no_pads, no_pads)  # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        # assuming x is a tenser
        padded_input = F.pad(embeddings, pad, "constant", 0)
        # raise NotImplementedError("Please implement the TODO here!")

        # TODO: Get inputs for the convolution layer using ConvEncoder.get_windows
        x = ConvEncoder.get_windows(padded_input, self.window_size)
        print(f' expected (B {batch_size} num_windows, window_size {self.window_size}, emb_size {self.emb_size})')
        # print(f'out = {x.shape}')
        # import sys
        # sys.exit(1)

        # raise NotImplementedError("Please implement the TODO here!")

        # Unroll windows to pass through the linear layer
        x = x.view(batch_size, max_seq_len, -1)

        # TODO: Pass the outputs through the flattened conv kernel and a GELU activation
        x = self.linear_layer(x)
        # raise NotImplementedError("Please implement the TODO here!")

        return x


class LSTMEncoder(nn.Module):
    '''
    A bidirectional, two layer, stacked LSTM encoder

    Args:
        emb_size: The dimension of each token's embedding
        vocab_size: The size of the input token vocabulary (including padding)
        hidden_size: The dimension of the outputs produced by the encoder,
            and the hidden dimension size in the LSTM layers
    '''

    def __init__(self, emb_size, hidden_size, vocab_size, pad_idx):
        super(LSTMEncoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # TODO: Define an embedding layer, ensuring that the padding token gets a 0 embedding
        self.embeddings = ...
        raise NotImplementedError("Please implement the TODO here!")

        # TODO: Use nn.LSTM to define a bidirectional LSTM with 2 layers
        # Note, to match the autograder, ensure that you use batch_first=True
        self.lstm = ...
        raise NotImplementedError("Please implement the TODO here!")

    def forward(self, x):
        '''
        Inputs:
            x - shape: (B, max_seq_len), dtype torch.int64

        where:
            x is a pytorch tensor containing a batch of padded sentences encoded into token IDs,
                with each sentence in the batch padded to the length 'max_seq_len' (the maximum
                sentence length in that batch)
            B is the batch dimension
        
        Outputs are of shape (B, max_seq_len, out_size)
        '''
        # TODO: Implement a forward pass
        raise NotImplementedError("Please implement the TODO here!")


class MLP(nn.Module):
    '''
    A feed-forward classification model with one hidden layer, with a 
    tanh non-linearity on the hidden layer, and a final output size equal to the number of classes

    Inputs:
        input_size - The dimension for each token's representation at the input
            (when used with the ConvEncoder, equal to the encoder's out_size,
            and when used with the LSTMEncoder, equal to the encoder's hidden_size)
        hidden_size - The feed-forward hidden layer dimension
        n_classes - The final output dimension (the number of classes being predicted)
    '''

    def __init__(self, input_size: int, hidden_size: int, n_classes: int):
        super(MLP, self).__init__()
        # TODO: Implement a feed-forward model with one hidden layer of size hidden_size,
        #  and a final size 'n_classes', with a tanh non-linearity on the hidden layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.linear = nn.Linear(input_size, hidden_size)
        self.non_linearity = nn.Tanh()
        self.final_linear_layer = nn.Linear(hidden_size, n_classes)
        # raise NotImplementedError("Please implement the TODO here!")

    def forward(self, x):
        '''
        Inputs:
            x - shape: (B, max_seq_len, input_size)

        where:
            x is a pytorch tensor of outputs from an encoder, with representations of size input_size
                corresponding to each input token
            B is the batch dimension
            max_seq_len is the maximum sentence length in that batch, to which inputs have been padded
        
        Outputs are logits of shape (B, max_seq_len, n_classes)
            where output[i][j][k] contains the pre-softmax scores
            for the ith datapoint's jth token, for the kth POS tag
        '''
        # TODO: Implement a forward pass
        #  Note that since we intend to use nn.CrossEntropyLoss, we do not apply a softmax here 
        #  at the outputs. nn.CrossEntropyLoss incorporates a softmax for better numerical stability.
        #  Read: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html


        a = self.linear(x)

        z = self.non_linearity(a)
        logits = self.final_linear_layer(z)

        return logits
        # raise NotImplementedError("Please implement the TODO here!")


class POSTagger(nn.Module):
    '''
    This module puts together the encoder and the mlp to obtain a POS tagger
    '''

    def __init__(self, emb_size, mlp_hidden_size, output_size, vocab_size, encoder_type, pad_idx):
        super(POSTagger, self).__init__()
        self.emb_size = emb_size
        self.encoder_out_dim = 100

        self.mlp_hidden_size = mlp_hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        assert encoder_type in {
            "conv", "lstm"}, f"Unknown encoder type: {encoder_type}"

        self.encoder = None
        self.mlp = None
        if encoder_type == "conv":
            self.encoder = ConvEncoder(
                emb_size,
                window_size=5,
                out_size=self.encoder_out_dim,
                vocab_size=vocab_size,
                pad_idx=pad_idx,
            )
        elif encoder_type == "lstm":
            assert self.encoder_out_dim % 2 == 0, "BiLSTM's output dim needs to be odd!"
            self.encoder = LSTMEncoder(
                emb_size, self.encoder_out_dim // 2, vocab_size, pad_idx=pad_idx)

        self.mlp = MLP(self.encoder_out_dim, mlp_hidden_size, output_size)
        self.model = nn.Sequential(self.encoder, self.mlp)

    def forward(self, x):
        return self.model(x)


def compute_accuracy_util(y_pred, y_true, filter_label_id: int = None):
    '''
    Returns the number of correct predictions and total predictions
    If filter_label_id is not None, returns results for given label id
    '''
    if filter_label_id is None:
        mask = (y_true >= 0)
    else:
        mask = (y_true == filter_label_id)

    correct_predictions = ((y_pred == y_true) & mask).sum()
    total_predictions = mask.sum()

    return correct_predictions.item(), total_predictions.item()


# Procedures for training and evaluation

def evaluate(model, dataloader, tags: Tags):
    '''
    A function that evaluates the model over all the datapoints from the dataloader passed in

    Inputs:
        model: a POSTagger model
        dataloader: A pytorch dataloader, typically the validation/test dataloader
        tags: An instance of the Tags class that contains information regarding the
            input Tags and IDs from the training dataset
    '''
    model.eval()

    correct_predictions = 0
    total_predictions = 0

    correct_predictions_per_label = defaultdict(int)
    total_predictions_per_label = defaultdict(int)

    with torch.no_grad():
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True,
                         leave=False, position=0, desc='Evaluating')
        for i, (X, y_true) in enumerate(dataloader):
            # TODO: Run the pos tagger on the new batch of data
            #  and obtain the predicted model class for each position
            X = np.vstack(X).astype(np.float)
            X = torch.from_numpy(X).type(torch.LongTensor)

            y_true = np.vstack(y_true).astype(np.float)
            y_true = torch.from_numpy(y_true).type(torch.FloatTensor)

            logits = model(X)

            y_pred = torch.argmax(logits, 2).type(torch.FloatTensor)
            # raise NotImplementedError("Please implement the TODO here!")
            assert y_pred.shape == y_true.shape and y_pred.dtype == y_true.dtype, \
                "The calculated predictions should have the same shapes and data types as the true labels"

            # Here we calculate the accuracy averaged across all tags first
            correct_predictions_batch, total_predictions_batch = compute_accuracy_util(
                y_pred, y_true)

            # Here we calculate the accuracy averaged across each tag
            for label_id in tags.idx_to_tag.keys():
                correct_predictions_label, total_predictions_label = compute_accuracy_util(
                    y_pred, y_true, label_id)
                correct_predictions_per_label[label_id] += correct_predictions_label
                total_predictions_per_label[label_id] += total_predictions_label

            correct_predictions += correct_predictions_batch
            total_predictions += total_predictions_batch
            batch_bar.update()
        batch_bar.close()

    accuracy = correct_predictions / total_predictions

    accuracy_per_label = {}
    for label_id in tags.idx_to_tag.keys():
        if total_predictions_per_label[label_id] == 0:
            accuracy_per_label[label_id] = float('nan')
        else:
            accuracy_per_label[label_id] = \
                correct_predictions_per_label[label_id] / \
                total_predictions_per_label[label_id]

    return accuracy, accuracy_per_label


def train_one_epoch(model: POSTagger, loss_fn, optimizer, train_loader, epoch):
    '''
    A function that runs one training epoch, iterating across all the datapoints
    in the training dataloader passed in.

    Inputs:
        model: a POSTagger model
        loss_fn: a pytorch criterion (loss function layer),
        optimizer: a pytorch optimizer
        train_loader: A pytorch dataloader
        epoch: The number of the current epoch being run
    '''

    model.train()

    total_loss = 0.0

    num_batches = len(train_loader)
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True,
                     leave=False, position=0, desc='Train epoch: {}'.format(epoch))
    for i, (X, y_true) in enumerate(train_loader):
        # TODO: Implement the training pass for the model
        #  Run the model to get outputs, calculate the loss + fresh gradients, take an optimizer step
        #  Ensure that the 'loss' variable contains the outputs of the loss_fn
        X = np.vstack(X).astype(np.float)
        X = torch.from_numpy(X).type(torch.LongTensor)

        y_true = np.vstack(y_true).astype(np.float)
        y_true = torch.from_numpy(y_true).type(torch.FloatTensor)

        logits = model(X)

        y_preds = torch.argmax(logits, 2).type(torch.FloatTensor)


        loss = loss_fn(y_preds, y_true)
        loss.requires_grad = True
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # raise NotImplementedError("Please implement the TODO here!")
        total_loss += loss.item()
        batch_bar.update()
    batch_bar.close()

    avg_loss = total_loss / num_batches

    return avg_loss


def train(
        model,
        train_loader,
        dev_loader,
        test_loader,
        tags: Tags,
        learning_rate: float = 1e-3,
        n_epochs: int = 10,
        use_wandb: bool = False,
):
    '''
    A function that trains a model and evaluates it on the validation/test data
    at the end of every training epoch

    Inputs:
        model: a POSTagger model
        train_loader: A pytorch dataloader containing the training data
        dev_loader: A pytorch dataloader containing the validation data
        test_loader: A pytorch dataloader containing the test data
        tags: An instance of the Tags class that contains information regarding the
            input Tags and IDs from the training dataset
        learning_rate: The optimizer learning rate
        n_epochs: The number of epochs to train the model for

    '''

    # We use the cross entropy loss
    loss_fn = nn.CrossEntropyLoss()

    # TODO: Define an Adam optimizer for the model using the given learning_rate (and default parameters)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # raise NotImplementedError("Please implement the TODO here!")

    wandb_log_dict = {}

    for epoch in range(n_epochs):
        avg_loss = train_one_epoch(
            model, loss_fn, optimizer, train_loader, epoch)
        dev_acc, dev_acc_per_label = evaluate(model, dev_loader, tags)
        test_acc, test_acc_per_label = evaluate(model, test_loader, tags)

        print("Epoch: {}, Avg train loss = {}, validation acc = {}, test acc = {}"
              .format(epoch, avg_loss, dev_acc, test_acc))

        wandb_log_dict = {"Metrics/Avg train loss": avg_loss, "Metrics/Validation acc": dev_acc,
                          "Metrics/Test acc": test_acc, 'Metrics/epoch': epoch}
        wandb_log_dict.update(
            {"valid_tag/{}_acc".format(tags.idx_to_tag[k]): v for k, v in dev_acc_per_label.items()})
        wandb_log_dict.update(
            {"test_tag/{}_acc".format(tags.idx_to_tag[k]): v for k, v in test_acc_per_label.items()})

        # Note: the autograder uses this flag to skip wandb logging
        if use_wandb:
            wandb.log(wandb_log_dict)

    print(f"Final scores: {wandb_log_dict}")


def main():
    # We set seeds to ensure reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # experiment config
    batch_size = 64
    learning_rate = 1e-3
    epochs = 20

    wandb.login()

    train_sentences, train_taggings = load_data_from_csv("data/train_data.csv")
    dev_sentences, dev_taggings = load_data_from_csv("data/valid_data.csv")
    test_sentences, test_taggings = load_data_from_csv("data/test_data.csv")

    voc = Vocabulary()
    tags = Tags()

    voc.words_vocabulary(train_sentences)
    tags.tags_vocabulary(train_taggings)

    train_data = parTUTDataset(train_sentences, train_taggings, voc, tags)
    dev_data = parTUTDataset(dev_sentences, dev_taggings, voc, tags)
    test_data = parTUTDataset(test_sentences, test_taggings, voc, tags)

    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=2, collate_fn=train_data.collate_fn)
    dev_loader = DataLoader(dev_data, batch_size=batch_size,
                            shuffle=False, num_workers=2, collate_fn=dev_data.collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                             shuffle=False, num_workers=2, collate_fn=test_data.collate_fn)

    # Train Conv Model
    model_type = 'conv'

    wandb_config = {
        "model_type": model_type,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
    }

    run = wandb.init(project="10418-hw1",
                     config=wandb_config, reinit=True)
    model = POSTagger(100, 75, len(tags), len(voc),
                      model_type, voc.str_to_idx["<PAD>"])
    wandb.watch(model, log="all")
    print(f"Training the {model_type} model..")
    train(model, train_loader, dev_loader, test_loader, tags,
          learning_rate=learning_rate, n_epochs=epochs)
    run.finish()

    print("****************\n")

    # Train LSTM Model
    model_type = 'lstm'

    wandb_config = {
        "model_type": model_type,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
    }

    run = wandb.init(project="10418-hw1",
                     config=wandb_config, reinit=True)
    model = POSTagger(100, 75, len(tags), len(voc),
                      model_type, voc.str_to_idx["<PAD>"])
    wandb.watch(model, log="all")
    print(f"Training the {model_type} model..")
    train(model, train_loader, dev_loader, test_loader, tags,
          learning_rate=learning_rate, n_epochs=epochs)
    run.finish()

    print("****************\n")


if __name__ == '__main__':
    main()
