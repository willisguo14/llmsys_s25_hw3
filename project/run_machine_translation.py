from functools import partial
import time
import os
import fire
import tqdm
import json
import random
import datasets
import numpy as np
import argparse
from distutils.util import strtobool

from sacrebleu.metrics import BLEU
from transformers import AutoTokenizer
from tokenizers import ByteLevelBPETokenizer

import minitorch
from minitorch import DecoderLM
from minitorch.cuda_kernel_ops import CudaKernelOps


def get_dataset(dataset_name, model_max_length):
    """
    Obtrain IWSLT (de-en) dataset.
    """
    dataset = {
        split: datasets.load_dataset(dataset_name, split=split, use_auth_token=True)['translation']
        for split in ['train', 'validation', 'test']
    }
    src_key, tgt_key = 'de', 'en'

    dataset = {
        split: [
            example for example in dataset[split]
            if len(example[src_key].split()) + len(example[tgt_key].split()) < model_max_length
        ] for split in dataset.keys()
    }

    dataset['test'] = dataset['test'][:100]             # 6750

    print(json.dumps(
        {'data_size': {split: len(dataset[split]) for split in dataset.keys()}},
        indent=4))

    return dataset, src_key, tgt_key


def get_tokenizer(examples, vocab_size, src_key, tgt_key, workdir):
    """
    Trains a tokenizer on the provided dataset examples and saves the tokenizer configuration.

    Parameters:
    - examples: The dataset examples used for training the tokenizer.
    - vocab_size: The desired vocabulary size for the tokenizer.
    - src_key: The key used to access the source text within the dataset examples.
    - tgt_key: The key used to access the target text within the dataset examples.
    - workdir: The directory where the tokenizer should be saved.

    Returns:
    - tokenizer: The trained tokenizer with special tokens,
        e.g., ("<eos_de>", "<eos_en>", "<pad>") if src_key and tgt_key are "de" and "en", respectively.
    """
    tokenizer = ByteLevelBPETokenizer()

    # Customized training
    tokenizer.train_from_iterator(
        [[example[src_key], example[tgt_key]] for example in examples],
        vocab_size=vocab_size,
        special_tokens=[f'<eos_{src_key}>', f'<eos_{tgt_key}>', '<pad>'])

    tokenizer.save(f'{workdir}/tokenizer.json')
    json.dump({'model_type': 'gpt2'}, open(f'{workdir}/config.json', 'w'))

    tokenizer = AutoTokenizer.from_pretrained(
        workdir,
        eos_token=None,
        bos_token=None,
        pad_token=None,
        unk_token=None)

    return tokenizer


def collate_batch(
        examples, src_key, tgt_key, tokenizer, model_max_length, backend):
    """
    Prepares a batch of examples for model training or evaluation by tokenizing and padding them.

    Parameters:
    - examples: A list of examples to be processed.
    - src_key: The key for accessing source texts in the examples.
    - tgt_key: The key for accessing target texts in the examples.
    - tokenizer: The tokenizer to be used for encoding the texts.
    - model_max_length: The maximum sequence length the model can handle.
    - backend: The backend of minitorch tensors.

    Returns:
    - A dictionary containing keys: 'input_ids', 'labels', 'label_token_weights',
        each indicates a minitorch tensor with shape (len(examples), model_max_length).

    Notes:
    ["input_ids"] for every example in the DE-EN translation, the "input_ids" will be:
        <de_token_ids> + <de_eos_id> + <en_token_ids> + <en_eos_id> + <pad_ids>
    where the pad_ids makes the length of input_ids to be model_max_length.

    ["labels"]: the next tokens to be predicted, which will be used in the cross-entropy
    loss function, e.g., for an example tokenized as [a, b, c, d], "input_ids" and "labels" 
    can be [a, b, c] and [b, c, d], respectively.

    ["label_token_weights"] The 'label_token_weights' are used to differentiate
    calculation purposes. (the MLE loss is computed on target tokens only.)
    between the source (weight = 0) and target (weight = 1) tokens for loss
    """
    token_ids, tgt_token_mask = [], []
    max_length = model_max_length
    pad_token_id = tokenizer.vocab['<pad>']
    for example in examples:
        token_ids_src = tokenizer(
            f'{example[src_key]}<eos_{src_key}>')['input_ids']
        token_ids_tgt = tokenizer(
            f'{example[tgt_key]}<eos_{tgt_key}>')['input_ids']

        example_token_ids = token_ids_src + token_ids_tgt
        example_tgt_token_mask = (
                [0] * len(token_ids_src) + [1] * len(token_ids_tgt))
        example_token_ids = example_token_ids[:max_length]
        example_tgt_token_mask = example_tgt_token_mask[:max_length]
        pad_ids = [pad_token_id] * (max_length - len(example_token_ids))

        token_ids.append(example_token_ids + pad_ids)
        tgt_token_mask.append(example_tgt_token_mask + [0] * len(pad_ids))

    # TODO: make examples in a 1d list, provide shape to initialize minitorch.Tensor
    token_ids = np.array(token_ids)
    tgt_token_mask = np.array(tgt_token_mask)

    input_ids = token_ids[:, :-1]
    labels    = token_ids[:, 1:]
    label_token_weights = tgt_token_mask[:, 1:]

    input_ids = minitorch.tensor_from_numpy(input_ids, backend=backend)
    labels    = minitorch.tensor_from_numpy(labels, backend=backend)
    label_token_weights = minitorch.tensor_from_numpy(label_token_weights, backend=backend)
    
    # input_ids = token_ids[:, :-1].tolist()
    # labels    = token_ids[:, 1:].tolist()
    # label_token_weights = tgt_token_mask[:, 1:].tolist()

    # input_ids = minitorch.tensor(input_ids, backend=backend)
    # labels    = minitorch.tensor(labels, backend=backend)
    # label_token_weights = minitorch.tensor(label_token_weights, backend=backend)

    return {
        'input_ids': input_ids,
        'labels': labels,
        'label_token_weights': label_token_weights
    }


def loss_fn(batch, model):
    """
    The MLE loss for a batch.

    Parameters:
    - batch: The result of collate_fn, a dict with "input_ids", "labels", and "label_token_weights".
    - model: The model to be trained.

    Returns:
    - A scalar loss value for this batch, averaged across all target tokens.
    """

    idx = batch['input_ids']
    idx.requires_grad_(True)
    # print("getting into loss_fn")
    logits = model(idx=idx)
    # print("finish prediction")
    bs, l, c = logits.shape
    logits = logits.view(bs * l, c)
    targets = batch['labels'].view(bs * l)
    label_token_weights = batch['label_token_weights'].view(bs * l)

    targets.requires_grad_(True)
    # print("start calculating loss")
    # import pdb
    # pdb.set_trace()
    loss = minitorch.nn.softmax_loss(
        logits=logits,
        target=targets
    )

    return ((loss * label_token_weights).sum() / label_token_weights.sum())


def train(model, optimizer, examples, n_samples, collate_fn, batch_size, desc):
    """
    Trains the model on the provided examples.

    Parameters:
    - model: The model to be trained.
    - optimizer: The optimizer used for updating the model's parameters.
    - examples: The dataset examples used for training.
    - n_samples: The random samples to train from "examples".
    - collate_fn: The function to collate data examples into batches.
    - batch_size: The number of examples in each batch.
    - desc: Description for the training process (used in progress bars).
    """
    model.train()
    random.shuffle(examples)
    examples = examples[:n_samples]

    for i in (prog_bar := tqdm.trange(
            0, len(examples), batch_size, desc=f'Training ({desc})')):
        batch = collate_fn(examples=examples[i:i + batch_size])

        t0 = time.time()
        optimizer.zero_grad()
        loss = loss_fn(batch=batch, model=model)
        t1 = time.time()

        loss.backward()
        t2 = time.time()

        optimizer.step()
        t3 = time.time()

        print(f"Forward: {t1 - t0}")
        print(f"Backward: {t2 - t1}")
        print(f"Opt.step: {t3 - t2}")

        batch_time = time.time() - t0
        prog_bar.set_postfix(
            tokens_per_sec=np.prod(batch['input_ids'].shape) / batch_time,
            loss=loss.item(),
            lr=optimizer.lr)


def evaluate_loss(model, examples, batch_size, collate_fn, desc):
    """
    Evaluates the model on the provided examples and computes the average loss.

    Parameters:
    - model: The model to be evaluated.
    - examples: The dataset examples used for evaluation.
    - batch_size: The number of examples in each batch.
    - collate_fn: The function to collate data examples into batches.
    - desc: Description for the evaluation process (used in progress bars).

    Returns:
    - The average loss computed over all batches.
    """
    model.eval()
    losses = []

    for i in (prog_bar := tqdm.trange(
        0, len(examples), batch_size, desc=f'Evaluating ({desc})')):
        batch = collate_fn(examples=examples[i:i + batch_size])
        loss = loss_fn(batch=batch, model=model)

        losses.append(loss.item())
        prog_bar.set_postfix(loss=loss.item())

    return np.mean(losses)


def generate(model,
             examples,
             src_key,
             tgt_key,
             tokenizer,
             model_max_length,
             backend,
             desc):
    """
    Generates target sequences for the given source sequences using the model, based on argmax decoding.
    Note that it runs generation on examples one-by-one instead of in a batched manner.

    Parameters:
    - model: The model used for generation.
    - examples: The dataset examples containing source sequences.
    - src_key: The key for accessing source texts in the examples.
    - tgt_key: The key for accessing target texts in the examples.
    - tokenizer: The tokenizer used for encoding texts.
    - model_max_length: The maximum sequence length the model can handle.
    - backend: The backend of minitorch tensors.
    - desc: Description for the generation process (used in progress bars).

    Returns:
    - A list of generated target sequences.
    """

    model.eval()
    gen_sents = []
    for example in tqdm.tqdm(examples, desc=f'Generating {desc}'):
        # Run generation for every single example

        token_ids = tokenizer(f'{example[src_key]}<eos_{src_key}>')['input_ids']
        len_src = len(token_ids)

        while len(token_ids) <= model_max_length:
            # BEGIN ASSIGN2_2
            # TODO
            # run the model with current token_ids, and predict the next token (gen_id)
            # hint: obtain the logits of next token, and take the argmax.
            gen_id = 0
            
            token_ids_numpy = np.array(token_ids).reshape((1, -1))
            token_ids_tensor = minitorch.tensor_from_numpy(token_ids_numpy, backend)
            logits = model(token_ids_tensor)

            gen_id = logits.to_numpy()[0, -1].argmax()
            # END ASSIGN2_2

            if gen_id == tokenizer.vocab[f'<eos_{tgt_key}>']:
                break
            else:
                token_ids.append(gen_id)

        gen_sents.append(tokenizer.decode(token_ids[len_src:]))

    return gen_sents


def evaluate_bleu(examples, gen_sents, tgt_key):
    """
    Evaluates the BLEU score for generated sentences against the target sentences in the examples.

    Parameters:
    - examples: The dataset examples used for evaluation.
    - gen_sents: The generated sentences to be evaluated.
    - tgt_key: The key for accessing target texts in the examples.

    Returns:
    - A dictionary containing the BLEU score.
    """
    return {
        'bleu': BLEU().corpus_score(
            hypotheses=gen_sents,
            references=[[example[tgt_key] for example in examples]]).score
    }

def parse_args():
    def str2bool(x):
        return bool(strtobool(x))
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-fused-kernel', type=str2bool, default=False)
    return parser.parse_args()


def main(dataset_name='bbaaaa/iwslt14-de-en-preprocess',
         model_max_length=40,
         n_epochs=1,
         batch_size=128,
         learning_rate=0.02,
         samples_per_epoch=20000,
         n_vocab=10000,
         n_embd=256,
         seed=11111):
    """
    The main function to train and evaluate the model on a specified dataset.

    Parameters:
    - dataset_name: The name of the dataset to be used.
    - model_max_length: The maximum sequence length the model can handle.
    - n_epochs: The number of training epochs.
    - batch_size: The number of examples in each batch.
    - learning_rate: The learning rate for the optimizer.
    - samples_per_epoch: Samples from the training dataset every epoch.
    - n_vocab: The vocabulary size of the BPE tokenizer.
    - n_embd: The embedding dimension.
    - seed: Random seed.
    """
    args = parse_args()
    np.random.seed(seed)
    random.seed(seed)

    workdir = f'./workdir_vocab{n_vocab}_lr{learning_rate}_embd{n_embd}'
    os.makedirs(workdir, exist_ok=True)

    backend = minitorch.TensorBackend(CudaKernelOps)

    config = {
        'n_vocab'     : n_vocab,  # vocab_size
        'n_embd'      : n_embd,   # n_embed
        'n_head'      : 8,    # n_head
        'n_positions' : model_max_length,  # n_ctx == n_positions
        # 'n_layer'     : 4,    # n_layer
        'p_dropout'   : 0.1,  # x_pdrop
        'ln_eps'      : 1e-5, # layer_norm_epsilon
        'backend'     : backend,
        'use_fused_kernel': args.use_fused_kernel
    }

    model = DecoderLM(**config)
    optimizer = minitorch.Adam(model.parameters(), lr=learning_rate)

    dataset, src_key, tgt_key = get_dataset(
        dataset_name=dataset_name, model_max_length=model_max_length)

    tokenizer = get_tokenizer(
        examples=dataset['train'],
        vocab_size=config['n_vocab'],
        src_key=src_key,
        tgt_key=tgt_key,
        workdir=workdir)

    collate_fn = partial(
        collate_batch,
        src_key=src_key,
        tgt_key=tgt_key,
        tokenizer=tokenizer,
        model_max_length=model_max_length,
        backend=backend)

    for epoch_idx in range(n_epochs):
        desc = f'epoch {epoch_idx} / {n_epochs}'

        train(
            model=model,
            optimizer=optimizer,
            examples=dataset['train'],
            n_samples=samples_per_epoch,
            batch_size=batch_size,
            collate_fn=collate_fn,
            desc=desc)

        validation_loss = evaluate_loss(
            model=model,
            examples=dataset['validation'],
            batch_size=batch_size,
            collate_fn=collate_fn,
            desc=desc)

        print(f'Epoch {epoch_idx}: Validation Loss = {validation_loss}')

        gen_sents = generate(
            model=model,
            examples=dataset['test'],
            src_key=src_key,
            tgt_key=tgt_key,
            tokenizer=tokenizer,
            model_max_length=model_max_length,
            backend=backend,
            desc=desc)

        gen_examples = []
        for example, gen_sent in zip(dataset['test'], gen_sents):
            gen_examples.append({'example': example, 'gen': gen_sent})
        json.dump(gen_examples, open(
            f'{workdir}/gen_epoch{epoch_idx}.json', 'w'), indent=4)

        eval_scores = evaluate_bleu(
            examples=dataset['test'], gen_sents=gen_sents, tgt_key=tgt_key)
        print(f'Epoch {epoch_idx}: {eval_scores}')

        json.dump(
            {'validation_loss': float(validation_loss), **eval_scores},
            open(f'{workdir}/eval_results_epoch{epoch_idx}.json', 'w'))


if __name__ == '__main__':
    fire.Fire(main)
