import sentencepiece as spm


spm.SentencePieceTrainer.Train(
    input='corpus/datasets/train.skz',      # 密文文件
    model_prefix='corpus/datasets/sp_src',  # 输出前缀
    vocab_size=256,                         # 56字符+常见组合，256足够
    character_coverage=1.0,                 # ✅ 关键：强制覆盖全部56字符
    model_type='unigram',
    input_sentence_size=200000,             # 采样句数
    shuffle_input_sentence=True,
    split_by_whitespace=False,              # 密文无空格，关闭此切分
    split_digits=False
)
print("trained models: sp_src.model / sp_src.vocab")