import sys
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase



files = ["wiki.train.tokens", "wiki.valid.tokens", "wiki.test.tokens"]
# files = ["wiki.valid.tokens", "wiki.test.tokens"]
files = [os.path.join(sys.argv[1], f) for f in files]

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()
tokenizer.normalizer = Lowercase()
trainer = BpeTrainer(special_tokens=["<unk>"])
print("Training BPE encoding")
tokenizer.train(files=[files[0]], trainer=trainer)
print("Tokenizer trained. Vocabulary size: {}".format(tokenizer.get_vocab_size()))

for file in files:
    print("Tokenizing file {}".format(file))
    os.rename(file, file+"old")
    with open(file+"old", "r") as f:
        with open(file, "w") as new_file:
            for line in f.readlines():
                output = tokenizer.encode(line).tokens
                words = ' '.join(output)
                new_file.write(words)

