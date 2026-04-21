from dl.data import TranslationDataset, DataLoader, src_sp, tgt_sp, collate_fn
from dl.model import Seq2SeqTransformer
from dl.train import train_model
from dl.predict import translate

dataset = TranslationDataset("train.src", "train.tgt", src_sp, tgt_sp)
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# 2. 构建模型（源词表~4000，目标词表=56+4特殊符≈60）
model = Seq2SeqTransformer(src_vocab=src_sp.vocab_size(), tgt_vocab=tgt_sp.vocab_size())

# 3. 训练
model = train_model(model, loader, epochs=10)

# 4. 测试
print(translate(model, "人工智能正在改变世界", src_sp, tgt_sp))