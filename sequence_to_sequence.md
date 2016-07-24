---
layout: guide
title: Sequence-to-Sequence Learning
---

Many kinds of problems need us to predict a output sequence given an input sequence. This is called a _sequence-to-sequence_ problem.

One such sequence-to-sequence problem is machine translation, which is what we'll try here.

The general idea of sequence-to-sequence learning with neural networks is that we have one network that is an encoder (an RNN), transforming the input sequence into some encoded representation.

This representation is then fed into another network, the decoder (also an RNN), which generates an output sequence for us.

![](/guides/assets/sequence_to_sequence.png){:class="figure", width="280px"}

That's the basic idea, anyway. There are enhancements, most notably the inclusion of an _attention_ mechanism, which doesn't look at the encoder's single final representation but all of its intermediary representations as well. The attention mechanism involves the decoder weighting different parts of these intermediary representations so it "focuses" on certain parts at certain time steps.

Another enhancement is to use a _bidirectional_ RNN - that is, to look at the input sequence from start to finish and from finish to start. This helps because when we represent an input sequence as a single representation vector, it tends to be biased towards later parts of the sequence. We can push back against this a bit by reading the sequence both forwards and backwards.

We'll work through a few variations here on the basic sequence-to-sequence architecture:

- with one-hot encoded inputs
- learning embeddings
- with a bidirectional encoder

The attention mechanism is not very straightforward to incorporate with Keras (in my experience at least), but the [`seq2seq` library](https://github.com/farizrahman4u/seq2seq) includes one (I have not tried it myself).

## Data

For sequence-to-sequence tasks we need a parallel corpus. This is just a corpus with input and output sequences that have been matched up (aligned) with one another.

Note that "translation" doesn't have to just be between two languages - we could take any aligned parallel corpus and train a sequence-to-sequence model on it. It doesn't even have to be text, although what I'm showing here will be tailored for that.

I'm going to be boring - here we'll just do a more conventional translation task.

[OPUS](http://opus.lingfil.uu.se/) (Open Parallel Corpus) provides many free parallel corpora. In particular, we'll use their [English-German Tatoeba corpus](http://opus.lingfil.uu.se/) which consists of phrases translated from English to German or vice-versa.

Some preprocessing was involved to extract just the aligned sentences from the various XML files OPUS provides; I've provided the [processed data for you](/guides/assets/en_de_tatoeba.json).

## Preparing the data

First, let's import what we need.

```python
import numpy as np
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, base_filter
from keras.layers import Activation, Dense, RepeatVector, Input, merge
```

Let's load the corpus. We are going to do some additional processing on it, mainly to filter out sentences that are too long.

Sequence-to-sequence learning can get difficult if the sequences are long; the resulting representation is biased towards later elements of the sequence. Attention mechanisms should help with this, but as I said we aren't going to explore them here (sorry). Fortunately bidirectional RNNs help too.

We'll also limit our vocabulary size and the number of examples we look at to limit memory usage.

```python
import json
data = json.load(open('data/en_de_corpus.json', 'r'))

# to deal with memory issues,
# limit the dataset
# we could also generate the training samples on-demand
# with a generator and use keras models' `fit_generator` method
max_len = 6
max_examples = 80000
max_vocab_size = 10000

def get_texts(source_texts, target_texts, max_len, max_examples):
    """extract texts
    training gets difficult with widely varying lengths
    since some sequences are mostly padding
    long sequences get difficult too, so we are going
    to cheat and just consider short-ish sequences.
    this assumes whitespace as a token delimiter
    and that the texts are already aligned.
    """
    sources, targets = [], []
    for i, source in enumerate(source_texts):
        # assume we split on whitespace
        if len(source.split(' ')) <= max_len:
            target = target_texts[i]
            if len(target.split(' ')) <= max_len:
                sources.append(source)
                targets.append(target)
    return sources[:max_examples], targets[:max_examples]

en_texts, de_texts = get_texts(data['en'], data['de'], max_len, max_examples)
n_examples = len(en_texts)
```

It will help if we explicitly tell our network where sentences begin and end so that it can learn when to start/stop generating words (this is explained a more [here](https://github.com/fchollet/keras/issues/395#issuecomment-150891272)). To do so we'll specify special start and end tokens. Make sure they aren't tokens that are already present in your corpus!

```python
# add start and stop tokens
start_token = '^'
end_token = '$'
en_texts = [' '.join([start_token, text, end_token]) for text in en_texts]
de_texts = [' '.join([start_token, text, end_token]) for text in de_texts]
```

Now we can use Keras' tokenizers to tokenize the source sequences and target sequences (note that "input" and "source" are interchangeable, as are "output" and "target").

```python
# characters for the tokenizers to filter out
# preserve start and stop tokens
filter_chars = base_filter().replace(start_token, '').replace(end_token, '')

source_tokenizer = Tokenizer(max_vocab_size, filters=filter_chars)
source_tokenizer.fit_on_texts(en_texts)
target_tokenizer = Tokenizer(max_vocab_size, filters=filter_chars)
target_tokenizer.fit_on_texts(de_texts)

# vocab sizes
# idx 0 is reserved by keras (for padding)
# and not part of the word_index,
# so add 1 to account for it
source_vocab_size = len(source_tokenizer.word_index) + 1
target_vocab_size = len(target_tokenizer.word_index) + 1
```

Our input sentences are variable in length, but we can't directly input variable length vectors into our network. What we do instead is pad it with a special padding character (Keras takes care of this for us, which I'll explain a bit more below).

We need to figure out the longest input and output sequences so that we make our vectors long enough to fit them.

```python
# find max length (in tokens) of input and output sentences
max_input_length = max(len(seq) for seq in source_tokenizer.texts_to_sequences_generator(en_texts))
max_output_length = max(len(seq) for seq in target_tokenizer.texts_to_sequences_generator(de_texts))
```

The tokenizers will take text and output a sequence of integers (which are mapped to words).

Then we'll pad these sequences so that they are all of the same length (the padding value Keras uses is 0, which is why the tokenizer doesn't assign that value to any words).

For example:

```python
sequences = pad_sequences(source_tokenizer.texts_to_sequences(en_texts[:1]), maxlen=max_input_length)
print(en_texts[0])
# >>> ^ I took the bus back. $
print(sequences[0])
# >>> [  0   0   0   2   4 223   3 461 114   1]
```

The `0` values are padding, the `1` is our start token, the `2` is our end token, and the rest are other words.

The first sequence-to-sequence model we'll build will take one-hot vectors as input, so we'll write a function that takes these sequences and converts them.

(Our [RNN guide](/guides/recurrent_neural_networks) explains more about one-hot vectors.)

```python
def build_one_hot_vecs(sequences):
    """generate one-hot vectors from token sequences"""
    # boolean to reduce memory footprint
    X = np.zeros((len(sequences), max_input_length, source_vocab_size), dtype=np.bool)
    for i, sent in enumerate(sequences):
        word_idxs = np.arange(max_input_length)
        X[i][[word_idxs, sent]] = True
    return X
```

Basically what this does is represent each input sequence as a matrix of one-hot vectors.

This image is from our [RNN guide](/guides/recurrent_neural_networks), which deals with individual characters, but the idea is the same (just imagine words instead of characters):

![](/guides/assets/rnn_3tensor.png){:class="figure"}

You can think of this as a "stack" of "tiers".

Each "tier" is a sequence, i.e. a sentence, each row in a tier is a word, and each element in a row is associated with a particular word.

The "stack" is `n_examples` tall (one tier for each sentence), each tier has `max_input_length` rows (some of these first rows will just be padding), and each row is `source_vocab_size` long.

We'll also encode our target sequences in this way:

```python
def build_target_vecs():
    """encode words in the target sequences as one-hots"""
    y = np.zeros((n_examples, max_output_length, target_vocab_size), dtype=np.bool)
    for i, sent in enuerate(pad_sequences(target_tokenizer.texts_to_sequences(de_texts), maxlen=max_output_length)):
        word_idxs = np.arange(max_output_length)
        y[i][[word_idxs, sent]] = True
    return y
```

## Defining the model

Now we can start defining the sequence-to-sequence model. Since there's a lot of overlap between the one-hot and embedding versions and the bidirectional and unidirectional variations, we'll write a function that can generate a model of either combination.

```python
hidden_dim = 128
embedding_dim = 128

def build_model(one_hot=False, bidirectional=False):
    """build a vanilla sequence-to-sequence model.
    specify `one_hot=True` to build it for one-hot encoded inputs,
    otherwise, pass in sequences directly and embeddings will be learned.
    specify `bidirectional=False` to use a bidirectional LSTM"""
    if one_hot:
        input = Input(shape=(max_input_length,source_vocab_size))
        input_ = input
    else:
        input = Input(shape=(max_input_length,), dtype='int32')
        input_ = Embedding(source_vocab_size, embedding_dim, input_length=max_input_length)(input)

    # encoder; don't return sequences, just give us one representation vector
    if bidirectional:
        forwards = LSTM(hidden_dim, return_sequences=False)(input_)
        backwards = LSTM(hidden_dim, return_sequences=False, go_backwards=True)(input_)
        encoder = merge([forwards, backwards], mode='concat', concat_axis=-1)
    else:
        encoder = LSTM(hidden_dim, return_sequences=False)(input_)

    # repeat encoder output for each desired output from the decoder
    encoder = RepeatVector(max_output_length)(encoder)

    # decoder; do return sequences (timesteps)
    decoder = LSTM(hidden_dim, return_sequences=True)(encoder)

    # apply the dense layer to each timestep
    # give output conforming to target vocab size
    decoder = TimeDistributed(Dense(target_vocab_size))(decoder)

    # convert to a proper distribution
    predictions = Activation('softmax')(decoder)
    return Model(input=input, output=predictions)
```

We're using [Keras's functional API](http://keras.io/getting-started/functional-api-guide/) because it provides a great deal more flexibility when defining models. Layers and inputs can be linked up in ways that the [sequential API](http://keras.io/getting-started/sequential-model-guide/) doesn't support and is in general easier to develop with (you can view the output of intermediary layers, for instance).

In any case, this is what we're doing here:

- we define the input layer (which must be explicitly defined for the functional API)
- then we assemble the encoder, which is an RNN (a LSTM, but you could use, for instance, a GRU)
    - we set `return_sequences` to `False` because we only want the _last_ output of the LSTM, which is its representation of an _entire_ input sequence
    - we then add `RepeatVector` to repeat this representation so that it's available for each of the decoder's inputs
    - if we have `bidirectional=True`, we actually create two LSTMs, one of which reads the sequence backwards (the `go_backwards` parameter), then we concatenate them together
- then we assemble the decoder, which again is an RNN (also doesn't have to be an LSTM)
    - here we set `return_sequences` to `True` because we want all the sequences (timesteps) produced by the LSTM to pass along
    - then we add a `TimeDistributed(Dense)` layer; the `TimeDistributed` wrapper applies the `Dense` layer to each timestep

The result of this final time distributed dense layer is a "stack" similar to the one we inputted. It also has `n_examples` tiers but now each tier has `max_output_length` rows (which again may consist of some padding rows), and each row is of `target_vocab_size` length.

Another important difference is that these rows are not one-hot vectors. They are each a probability distribution over the target vocabulary;the softmax layer is responsible for making sure each row sums to 1 like a proper probability distribution should.

Here's a illustration depicting this for one input example. Note that in this illustration the raw source sequence of indices are passed into the encoder, which is how the embedding variation of this model works; for the one-hot variation there would be an intermediary step where we create the one-hot vectors.

![](/guides/assets/sequence_to_sequence_details.png){:width="100%"}

This "stack" (which is technically called a 3-tensor) basically the translated sequence that we want, except we have to do some additional processing to turn it back into text. In the illustration above, the output of the decoder corresponds to one tier in this stack.

Let's prepare that preprocessing now. Basically, we will take these probabilities and translate them into words, as illustrated in the last two steps above.

```python
target_reverse_word_index = {v:k for k,v in target_tokenizer.word_index.items()}

def decode_outputs(predictions):
    outputs = []
    for probs in predictions:
        preds = probs.argmax(axis=-1)
        tokens = []
        for idx in preds:
            tokens.append(target_reverse_word_index.get(idx))
        outputs.append(' '.join([t for t in tokens if t is not None]))
    return outputs
```

To start, we're preparing a reverse word index which will let us put in a number and get back the associated word.

The `decode_outputs` function then just takes that 3-tensor stack of probability distributions (`predictions`). The variable `probs` represents a tier in that stack. With `argmax` get the indices of the highest-probability words, then we look up each of those in our reverse word index to get the actual word. We join them up with spaces and voilá, we have our translation.

## Training

But first we have to train the model.

To reduce memory usage while training, we're going to write a generator to output training data on-the-fly. This way all the data won't sit around in memory.

It will generate one-hot vectors or output the raw sequences (which we need for the embedding approach) according to the `one_hot` parameter and output them in chunks of the batch size we specify.

In the interest of neater code, we're writing this batch generator so that it can also generate raw sequences if we set `one_hot=False` (we'll need this when we try the embedding approach). So first we'll define a convenience function for that:

```python
def build_seq_vecs(sequences):
    return np.array(sequences)
```

And then define the actual batch generator:

```python
import math
def generate_batches(batch_size, one_hot=False):
    # each epoch
    n_batches = math.ceil(n_examples/batch_size)
    while True:
        sequences = pad_sequences(source_tokenizer.texts_to_sequences(en_texts), maxlen=max_input_length)

        if one_hot:
            X = build_one_hot_vecs(sequences)
        else:
            X = build_seq_vecs(sequences)
        y = build_target_vecs()

        # shuffle
        idx = np.random.permutation(len(sequences))
        X = X[idx]
        y = y[idx]

        for i in range(n_batches):
            start = batch_size * i
            end = start+batch_size
            yield X[start:end], y[start:end]
```

Now let's build the model and train it.

We'll train it using the categorical cross-entropy loss function because this is essentially a classification problem, where we have `target_vocab_size` "categories".

Training will likely take a very long time. 100 epochs took me a couple hours on an Nvidia GTX 980Ti. As I note later, 100 epochs is not enough to get the network performing very well; that choice is more in the interest of trying multiple models and not wanting to wait for days.

```python
n_epochs = 100
batch_size = 128

model = build_model(one_hot=True, bidirectional=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(generator=generate_batches(batch_size, one_hot=True), samples_per_epoch=n_examples, nb_epoch=n_epochs, verbose=1)
```

Since we're going to be trying a few different models, let's also write a function to make it easier to generate translations.

```python
def translate(model, sentences, one_hot=False):
    seqs = pad_sequences(source_tokenizer.texts_to_sequences(sentences), maxlen=max_input_length)
    if one_hot:
        input = build_one_hot_vecs(seqs)
    else:
        input = build_seq_vecs(seqs)
    preds = model.predict(input, verbose=0)
    return decode_outputs(preds)
```

Let's give it a shot:

```python
print(en_texts[0])
print(de_texts[0])
print(translate(model, [en_texts[0]], one_hot=True))
# >>> ^ I took the bus back. $
# >>> ^ Ich nahm den Bus zurück. $
# >>> ^ ich ich die die verloren $
```

That's pretty bad to be honest. As I said before, I don't think you'll have particularly good results unless you train for a significantly longer amount of time.

In the meantime, let's try this task with a model that learns embeddings, instead of using one-hot vectors.

We can just use what we've got, but specifying `one_hot=True`.

```python
model = build_model(one_hot=False, bidirectional=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(generator=generate_batches(batch_size, one_hot=False), samples_per_epoch=n_examples, nb_epoch=n_epochs, verbose=1)
```

And we can try the bidirectional variations, e.g.

```python
model = build_model(one_hot=False, bidirectional=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(generator=generate_batches(batch_size, one_hot=False), samples_per_epoch=n_examples, nb_epoch=n_epochs, verbose=1)
```

We should see some improvement with the bidirectional variant, but again, a significant amount of training time is still likely needed.

## Final words

When preparing this guide I found that I had to train the network for many, many epochs before achieving decent output. I went with 300 epochs and got to ~82% accuracy.

The dataset here is also relatively small - larger, richer parallel corpora should result in a better translation model.

Here are the results from my comparison trainings (for the sake of time I ran each only for 100 epochs) - interestingly the one-hot models performed better (I expected embeddings would be best):

![](/guides/assets/sequence_training.png){:width="100%"}

Here are some examples from the best model, after 300 epochs:

```
---
^ I took the bus back. $
^ Ich nahm den Bus zurück. $
^ ich nahm den bus geländer $
---
^ He had barely enough to eat. $
^ Er hatte kaum genug zu essen. $
^ er hatte kaum genug zu hunger $
---
^ Without air we would die. $
^ Ohne Luft würden wir sterben. $
^ ^ luft luft gesellschaft $
---
^ I thought you'd be older. $
^ Ich hätte Sie für älter gehalten. $
^ ich hätte sie als als hunger $
---
^ Hanako questioned his sincerity. $
^ Hanako zweifelte an seiner Ernsthaftigkeit. $
^ hanako zweifelte jeden mit $
---
^ I study Chinese in Beijing. $
^ Ich lerne in Peking Chinesisch. $
^ ich lerne in peking stimmung $
```

There's definitely still quite a bit of weirdness, but it's not incoherently bad. More training (and of course a larger corpus) would probably help.

## Further reading

- [`seq2seq`](https://github.com/farizrahman4u/seq2seq), a library that implements sequence-to-sequence learning for Keras
- Sequence Modeling With Neural Networks [Part 1](https://indico.io/blog/sequence-modeling-neuralnets-part1/), [Part 2](https://indico.io/blog/sequence-modeling-neural-networks-part2-attention-models/), high-level overview on these problems
- Bahdanau, D., Cho, K., & Bengio, Y. (2014). [Neural machine translation by jointly learning to align and translate](http://arxiv.org/pdf/1409.0473v7.pdf). Describes in detail a neural machine translation model with attention (see Appendix A.2).
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). [Sequence to sequence learning with neural networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf). In Advances in neural information processing systems (pp. 3104-3112). Introduces sequence-to-sequence learning with neural networks
- Jörg Tiedemann, 2012, [Parallel Data, Tools and Interfaces in OPUS](http://www.lrec-conf.org/proceedings/lrec2012/pdf/463_Paper.pdf). In Proceedings of the 8th International Conference on Language Resources and Evaluation (LREC 2012). Source of the data we used here.

