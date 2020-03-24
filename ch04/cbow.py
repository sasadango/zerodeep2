import sys
sys.path.appned('..')
import numpy as np
from common.layers import Embedding
from ch04.negative_sampling_layer import NegativeSmaplingLoss


class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size

        # 重み初期化
        W_in = 0.01 * np.random.randn(V,H).astype('f')
        W_out = 0.01 * np.random.randn(V,H).astype('f')

        # レイヤの生成
        self.in_layers = []
        for i in range(2 * window_size):
            layer = Embedding(W_in)   # Embeddingレイヤを使用
            self.in_layers.append(layer)
        self.ns_loss = NegativeSmaplingLoss(W_out, corpus, power=0.75, sample_size=5)

        # すべての重みと勾配を配列にまとめる
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # メンバ変数に単語の分散表現を設定
        self.word_vecs = W_in