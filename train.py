import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
import time
import tensorflow_datasets as tfds
import tensorflow as tf
import pickle
from model import *



MAX_LENGTH = 40
EPOCHS=100

BATCH_SIZE = 128
BUFFER_SIZE = 20000

# Hyper-parameters
D_MODEL = 256
NUM_LAYERS = 2
NUM_HEADS = 8
DFF = 512
DROPOUT = 0.1

def train(file_path="aihub.CSV"):
    ## column = [Q, A, label]
    train_data = pd.read_csv(file_path, sep=",", encoding="utf-8")
    train_data = train_data[['Q', 'A']].dropna(axis=0)
    print(train_data.info())
    ## Preprocessing for Q
    questions = []
    for sentence in train_data['Q']:
        # print(sentence)
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = sentence.strip()
        questions.append(sentence)

    ## Preprocessing for A
    answers = []
    for sentence in train_data['A']:
        # 구두점에 대해서 띄어쓰기
        # ex) 12시 땡! -> 12시 땡 !
        # print(sentence)
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = sentence.strip()
        answers.append(sentence)

    # 서브워드텍스트인코더를 사용하여 질문, 답변 데이터로부터 단어 집합(Vocabulary) 생성
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        questions + answers, target_vocab_size=2**13)

    with open('tokenizer.p', 'wb') as file:
        pickle.dump(tokenizer, file)

    # 시작 토큰과 종료 토큰에 대한 정수 부여.
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

    # 시작 토큰과 종료 토큰을 고려하여 단어 집합의 크기를 + 2
    VOCAB_SIZE = tokenizer.vocab_size + 2
    def tokenize_and_filter(inputs, outputs):
        tokenized_inputs, tokenized_outputs = [], []
        
        for (sentence1, sentence2) in zip(inputs, outputs):
            sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
            sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)
        tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_inputs, maxlen=MAX_LENGTH, padding="post"
        )
        tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_outputs, maxlen=MAX_LENGTH, padding="post"
        )
        return tokenized_inputs, tokenized_outputs
    
    questions, answers = tokenize_and_filter(questions, answers)

    

    # 디코더의 실제값 시퀀스에서는 시작 토큰을 제거해야 한다.
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': questions,
            'dec_inputs': answers[:, :-1] # 디코더의 입력. 마지막 패딩 토큰이 제거된다.
        },
        {
            'outputs': answers[:, 1:]  # 맨 처음 토큰이 제거된다. 다시 말해 시작 토큰이 제거된다.
        },
    ))

    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)


    model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        dff=DFF,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT)

    learning_rate = CustomSchedule(D_MODEL)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    def accuracy(y_true, y_pred):
    # 레이블의 크기는 (batch_size, MAX_LENGTH - 1)
        y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

    
    model.summary()
    model.fit(dataset, epochs=EPOCHS)
    model.save_weights("model.h5")
    

train(file_path="aihub.CSV")