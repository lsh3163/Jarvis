import tensorflow as tf
from tensorflow.keras.models import model_from_json
from model import *
import pickle
import re
with open('tokenizer.p', 'rb') as file:
    tokenizer = pickle.load(file)


# Hyper-parameters
D_MODEL = 256
NUM_LAYERS = 2
NUM_HEADS = 8
DFF = 512
DROPOUT = 0.1
MAX_LENGTH = 40
# 시작 토큰과 종료 토큰에 대한 정수 부여.
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# 시작 토큰과 종료 토큰을 고려하여 단어 집합의 크기를 + 2
VOCAB_SIZE = tokenizer.vocab_size + 2


model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        dff=DFF,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT)
model.load_weights("model.h5")

# 시작 토큰과 종료 토큰에 대한 정수 부여.
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# 시작 토큰과 종료 토큰을 고려하여 단어 집합의 크기를 + 2
VOCAB_SIZE = tokenizer.vocab_size + 2

def evaluate(sentence):    
    sentence = preprocess_sentence(sentence)


    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    output = tf.expand_dims(START_TOKEN, 0)

    # 디코더의 예측 시작
    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        # 현재(마지막) 시점의 예측 단어를 받아온다.
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # 만약 마지막 시점의 예측 단어가 종료 토큰이라면 예측을 중단
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

    # 마지막 시점의 예측 단어를 출력에 연결한다.
    # 이는 for문을 통해서 디코더의 입력으로 사용될 예정이다.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)
def predict(sentence):
    prediction = evaluate(sentence)
    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])
    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))
    return predicted_sentence
def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence


# predict("전 코딩을 너무 못해요")
# predict("할줄 아는게 아무 것도 없어요")
# predict("킥복싱을 잘하고 싶어요")
# predict("저는 인공지능 과제 하기 싫어요")
# predict("라쿤과 너구리는 달라요?")
# predict("뭘 하면 좋을까요?")
# predict("저희 소프트웨어 응용 학점 얼마나 나올까요?")
# predict("어떤 취미를 가지면 좋을까?")
# predict("저 요즘 자꾸 안좋은 생각이 들고 마음이 불안해요")
# predict("저 요즘 매일 기분좋고 날아갈 것 처럼 행복해요")