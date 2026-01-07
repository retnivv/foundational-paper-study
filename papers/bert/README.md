# BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding

Created: 2025년 8월 8일 오후 5:44

# Introduction

pre-training은 NLP task에서 효과적임.

pre-trained된 language representation을 downstream task에 적용하는 두 가지 방법 :

1. feature-based

→ task-specific한 아키텍쳐를 사용하고 pre-trained representation은 additional feature로 넣어줌. 사전 학습된 모델은 고정.

1. fine-tuning

→ 사전 학습된 모델의 모든 파라미터를 업데이트함. 사전 학습 모델에 간단한 출력 레이어만 추가하는 방식.

두 방식은 같은 objective function 을 사용하고, unidirectional(단방향) language model으을 사용함.

→ 이런 단방향 방식은 pre-train 시에 사용되는 모델의 구조의 선택 폭을 제한시킴.

→ BERT는 MLM(Masked Language Model)이라는 objective를 설정하여 양방향 context를 잘 파악하게 하였음. 그리고 추가적으로 NSP(Next Sentence Prediction) 라는 objective로 문장 관계 파악 능력도 올림.

Contribution : 양방향 사전학습의 중요성, pre-trained representation은 task-specific architecture의 engineering을 쉽게 해줌, BERT는 11개의 NLP task에서 SOTA 달성.,

# Related Work

### Unsupervised Feature-based Approaches

Pre-trained word embedding은 NLP에서 매우 중요함.

당시까지는 word embedding을 사전학습하기 위해서 left-to-right 방식이 사용되어왔음.

sentence representation을 학습하기 위해서 다음 문장 후보들의 순위를 매기거나, 다음 문장을 left-to-right로 생성하거나, 문장에 노이즈를 넣고 이를 복원하는 방법 등이 사용되어왔음.

word embedding 학습에는 왼→오 언어 모델과 오→왼 언어 모델을 둘 다 학습하고 이를 concat해서 맥락 표현으로 사용함(ELMo). 또는 왼쪽 및 오른쪽 맥락을 사용하여 single word를 예측하는 task, Cloze task(빈칸채우기) 등이 활용되기도 함.

→ 이것들은 완전한 deep bidirectional은 아님. 

### Unsupervised Fine-tuning Approaches

초기에는 feature-based 방식과 유사하게 word embedding parameter만 사전학습하는 방식이 사용됨.

→ 최근에는 sentence or document encoder 전체를 학습함.

→ 이로 인해 새로 학습해야 하는 파라미터가 적어짐. 

이 방식의 모델을 사전학습하는 데는 left-to-right language modeling이나 auto-encoder objective 사용.

### Transfer Learning from Supervised Data

대규모 supervised 데이터셋에서 학습한 모델을 다른 태스크로 전이시켰을 때도 성능이 좋았다는 연구들이 있었음. (NLP, CV 둘 다)

# BERT

pre-training, fine-tuning 이라는 두 단계에 거쳐서 학습을 진행.

Pre-training : unlabeled data로 두 가지 task로 학습.

Fine-tuning : pre-trained 된 파라미터를 그대로 가져와서 labeled data로 다양한 downstream task를 학습.

→ 다양한 task에 대해 통일된 구조를 가지고 있다는 점이 특징.

### Model Architecture

![image.png](image.png)

multi-layer bidirectional Transformer encoder.

L : layer 개수 / H : hidden size / A : self-attention head 개수

BERT_BASE : L = 12 / H = 768 / A = 12 / 총 파라미터 수 = 110M

BERT_LARGE : L = 24 / H = 1024 / A = 16 / 총 파라미터 수 = 340M

(feed-forward/filter size는 4H로 설정.)

### Input/Output Representations

BERT가 다양한 downstream task를 다룰 수 있게 하기 위해, input representation이 단일 문장과 문장 pair를 둘 다 명확하게 표현할 수 있도록 만들었음.

- WordPiece embeddings 사용 (30000 token vocabulary)
- token의 제일 처음 자리는 [CLS] 토큰으로 설정. → 이 자리의 최종 출력이 sequence representation을 종합하도록 학습되며, classificatrion task에 쓰임.
- Sentence pair는 단일 sequence로 묶임.
- 문장 구분 방법 2가지 : [SEP] 토큰으로 분리. / 문장 A,B에 서로 다른 segment embedding을 더함.
- E : 입력 임베딩 / C : [CLS] 최종 hidden vetor / T_i : i번째 최종 hidden vector
- input representation = token + segment + position embedding

![image.png](image%201.png)

## Pre-training BERT

기존의 모델들과는 다르게, BERT를 사전학습할 때는 left-to-right 또는 right-to-left 언어 모델을 사용하지 않음. 대신 MLM, NSP라는 두 가지의 비지도학습 task를 사용함

### Task #1: Masked LM (MLM)

deep bidirectional 모델이 단방향모델보다 좋아보이긴 하지만, 기존의 conditional language model은 양방향 학습을 할 수가 없었음. 양방향 context를 동시에 쓰면 각 단어가 정답(자기 자신)을 참고하여, 너무 쉽게 예측을 수행할 수 있기 때문.

**→ input token들 중 일정 비율을 mask로 가리고, 이것들을 예측하게 함. (Masked LM)**

mask token에 해당하는 최종 hidden vector는 softmax를 거쳐 출력을 내놓게 됨. (원래 단어가 뭐였는지)

이 논문에서는 mask 비율을 15%로 설정.

→ 양방향으로 사전학습된 모델을 얻을 수 있다는 장점은 있지만, fine-tuning 시에는 [mask] token이 없기 때문에 pre-training과의 mismatch가 생길 수 있음.

→ 이를 해결하기 위해 15%의 토큰을 예측 대상으로 선정한 후, 80%는 [mask], 10%는 랜덤 토큰, 10%는 원래 토큰 그대로 둠. 이렇게 하면 [mask]가 없을 때도 모델이 잘 동작할 수 있음.

### Task #2: Next Sentence Prediction (NSP)

Question Answering (QA), Natural Language Inference (NLI) 등에서 문장 간의 관계를 파악하는 것을 기반으로 함.

이것을 학습하기 위해서 Next Sentence Prediction (NSP) 적용.

단일 언어 corpus에서 두 문장 A, B를 선택하는데, 50% 확률로 실제 A 다음 문장인 B를 선택하고(IsNext), 50% 확률로 랜덤 문장을 선택(NotNext)하는 방식으로 데이터셋을 쉽게 만들 수 있음.

그리고 이것들이 실제로 연결된 문장인지 아닌지 판별하게 시키는 방식으로 학습함.

→ QA와 NLI에서 매우 효과적.

![image.png](image%202.png)

→ 이런 식으로 MLM과 NSP를 동시에 적용. MLM loss + NSP loss 로 학습.

NSP는 기존 모델들의 representation-learning objective와 매우 관련이 깊지만, sentence embedding만 downstream task로 가져온 이전 모델과는 다르게 BERT는 모든 파라미터를 최종 task model의 파라미터로 그대로 사용한다는 점이 다름. (그리고 이걸 fine-tuning하면서 다시 업데이트)

### Pre-training data

BERT의 사전학습 절차는 기존 언어 모델의 사전학습 연구의 흐름을 따름.

BooksCorpus(800M words), English Wikipedia(2500M words) 사용.

## Fine-tuning BERT

Transformer의 self-attention은 입력 길이나 형태와 상관없이 모든 토큰 쌍의 관계를 계산할 수 있기 때문에 입력과 출력 포맷을 바꿔주면 다양한 태스크에 바로 적용 가능.

이전 모델들은 문장 pair 처리에서, 각 문장을 독립적으로 인코딩한 후 cross-attention으로 정보를 교환했지만, BERT는 두 문장을 concat하고 한 번에 self-attention 수행. → 모든 레이어에서 A, B의 토큰이 양방향으로 서로 참조 가능.

[Input]

pre-training시에 사용했던 문장 A,B 구조는

Paraphrasing : 문장 pair

Entailment(NLI) : hypothesis(가설)-premise(전제)

QA : quiestion(질문)-passage(지문)

Text classification : A만 입력하고 B는 비움

등의 구조로 재활용 가능.

[Output]

Token-level task : 각 토큰의 hidden vector를 output layer로 전달.

Sentence-level task : [CLS] 토큰의 hidden vector를 전달.

Pre-training에 비해 fine-tuning은 매우 적은 시간이 소요됨. 

# Experiments

11가지 NLP 태스크에 fine-tuning한 결과 소개.

## GLUE

C ([CLS] token 최종 hidden vector)가 classification layer를 거쳐 최종 출력 생성.

classification layer weight W가 fine-tuning에서 새롭게 추가된 유일한 파라미터임.

![image.png](image%203.png)

→ 실험 결과. (Batch size = 32, epoch = 3)

GPT 포함 기존 모델들 << BERT_BASE < BERT_LARGE

## SQuAD v1.1

Standford Question Answering Dataset.

[fine-tuning 처리 방법]

A : question / B : passage로 input 설정.

새로운 학습가능한 벡터 $S,E \in \mathbb R^H$ 설정. (start, end)

![image.png](image%204.png)

→ i번째 word가 start, end가 될 확률. 

$S^\top T_i + E^\top T_j$ 가 최대인 i, j 쌍을 answer의 시작 및 끝 위치로 예측. (i ≤ j)

올바른 시작 및 끝 위치에 대한 log-likelihood 합을 최대로 만들도록 학습. (cross-entropy loss)

epoch = 3, lr = 5e-5, batch size = 32로 설정.

TriviaQA dataset에 대해 미리 fine-tuning을 한 후에 한 번 더 학습시키기도 해봤음.

![image.png](image%205.png)

→ 실험 결과.

TriviaQA data를 사용하니까 최고 수준 달성.

## SQuAD v2.0

SQuAD v1.1에서 ‘no answer’ 경우를 추가한 데이터셋.

no answer인 경우는 start와 end를 [CLS] token으로 예측하도록 함.

$s_{\text{null}} = S^\top C + E^\top C$ → no answer일 때 score 계산

$s_{\text{null}} \ge \hat{s}_{\text{best}} - \tau$ 일 때 no answer라고 예측.

$\tau$ : dev set을 바탕으로 설정한 threshold.

epoch = 2, lr = 5e-5, batch size = 48 사용.

TriviaQA 미사용.

![image.png](image%206.png)

→ 기존보다 훨씬 높은 성능.

### SWAG

Situations With Adversarial Generations

어떤 문장을 보고, 그 뒤에 이어질 문장을 4개의 후보 중에서 판별하는 task.

[CLS] token에 가중치 하나만 곱하고 softmax 처리하는 식으로 학습.

![image.png](image%207.png)

epoch = 3, lr = 2e-5, batch size = 16.

# Ablation Studies

### Effect of Pre-training Task

![image.png](image%208.png)

NSP를 빼거나 LTR (left-to-right)로 학습할 경우 성능이 낮아짐. BiLSTM (Bidirectional LSTM)을 끝에 추가해주니까 SQuAD에서는 성능이 올라갔지만 나머지에서는 비슷하거나 오히려 더 내려가기도 함.

### Effect of Model Size

![image.png](image%209.png)

모든 task에서 모델 크기가 클수록 성능이 올라감.

데이터셋 크기가 3600개밖에 안 되는 소규모 task에도 마찬가지.

이전 연구에서는 모델 크기를 키워도 성능이 향상되지 않는 경우가 있었음.

이유 : 기존 방식은 feature-based 방식인데, BERT는 fine-tuning 방식이라서 모델 전체를 태스크에 맞춰서 조정이 가능하기 때문. → 모델 크기가 커질수록 더 많은 표현력 활용 가능.

대규모 pre-training 이후 fine-tuning을 하는 방식의 중요성 제시.

### Feature-based Approach with BERT

![image.png](image%2010.png)

feature-based 방식의 장점 :

- task-specific한 모델 구조를 추가할 수 있음.
- 연산량이 많은 pre-compute를 한 번에 미리 계산해놓고, 그 이후에 가벼운 모델로 여러 번 실험을 할 수 있음.

feature-based 적용 방법:

BERT의 특정 레이어의 hidden state들을 추출해서 2-layer BiLSTM에 통과시킨 후, NER classification layer로 최종 예측.

결과 : feature-based에서도 fine-tuning과 거의 유사한 정도로 좋은 성능을 보임.

→ BERT는 feature-based 방식에서도 좋은 성능을 냄.

# Conclusion

BERT는 전이학습에 deep bidirectional architecture를 적용하여 다양한 NLP task에서 좋은 성능을 보임.