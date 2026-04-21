import sentencepiece as spm

INPUT_FILE = "corpus/datasets/train.zh"
MODEL_PREFIX = "corpus/datasets/sp_zh"

spm.SentencePieceTrainer.Train(
    input=INPUT_FILE,
    model_prefix=MODEL_PREFIX,
    vocab_size=10000,           # 中文推荐 8000~12000
    character_coverage=0.999,   # 覆盖 99.9% 字符，生僻字自动走 <unk>
    model_type='unigram',
    input_sentence_size=200000,
    shuffle_input_sentence=True,
    max_sentencepiece_length=16,
    num_threads=8
)
print(f"✅ 训练完成: {MODEL_PREFIX}.model / {MODEL_PREFIX}.vocab")