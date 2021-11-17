import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class MemN2N(nn.Module):

    def __init__(self, params, vocab):
        super(MemN2N, self).__init__()
        self.input_size = len(vocab)
        self.embed_size = params.embed_size
        self.memory_size = params.memory_size
        self.num_hops = params.num_hops
        self.use_bow = params.use_bow #使用的bag-of-words或pe
        self.use_lw = params.use_lw #使用adjacent or not
        self.use_ls = params.use_ls #使用linear start or not
        self.vocab = vocab

        # create parameters according to different type of weight tying
        pad = self.vocab.stoi['<pad>']
        self.A = nn.ModuleList([nn.Embedding(self.input_size, self.embed_size, padding_idx=pad)])
        self.A[-1].weight.data.normal_(0, 0.1)
        self.C = nn.ModuleList([nn.Embedding(self.input_size, self.embed_size, padding_idx=pad)])
        self.C[-1].weight.data.normal_(0, 0.1)
        if self.use_lw:
            for _ in range(1, self.num_hops):
                self.A.append(self.A[-1])
                self.C.append(self.C[-1])
            self.B = nn.Embedding(self.input_size, self.embed_size, padding_idx=pad)
            self.B.weight.data.normal_(0, 0.1)
            self.out = nn.Parameter(
                I.normal_(torch.empty(self.input_size, self.embed_size), 0, 0.1))
            self.H = nn.Linear(self.embed_size, self.embed_size)
            self.H.weight.data.normal_(0, 0.1)
        else:
            for _ in range(1, self.num_hops):
                self.A.append(self.C[-1])
                self.C.append(nn.Embedding(self.input_size, self.embed_size, padding_idx=pad))
                self.C[-1].weight.data.normal_(0, 0.1)
            self.B = self.A[0]
            self.out = self.C[-1].weight

        # temporal matrix
        self.TA = nn.Parameter(I.normal_(torch.empty(self.memory_size, self.embed_size), 0, 0.1))
        self.TC = nn.Parameter(I.normal_(torch.empty(self.memory_size, self.embed_size), 0, 0.1))

    def forward(self, story, query):
        sen_size = query.shape[-1] #句子长度
        weights = self.compute_weights(sen_size)
        state = (self.B(query) * weights).sum(1) # 位置权重乘以embedding得到query的表示

        sen_size = story.shape[-1]
        weights = self.compute_weights(sen_size)
        for i in range(self.num_hops):
            # (self.A[i](story.view(-1, sen_size)) * weights).sum(1) 求每个sentence的表示，与state的求法一致
            # view之后应该是sen_num乘emb_size
            memory = (self.A[i](story.view(-1, sen_size)) * weights).sum(1).view(
                *story.shape[:-1], -1)
            memory += self.TA # dummy memories/random noise
            
            output = (self.C[i](story.view(-1, sen_size)) * weights).sum(1).view(
                *story.shape[:-1], -1)
            output += self.TC
            # 求memory和state的相似度
            probs = (memory @ state.unsqueeze(-1)).squeeze() # @等价于Torch.matmul()
            
            #如果不是linear start 在每层都要进行softmax
            if not self.use_ls:
                probs = F.softmax(probs, dim=-1)
                
            # 将weight与output相乘
            response = (probs.unsqueeze(1) @ output).squeeze()
            
            if self.use_lw:
                state = self.H(response) + state
            else:
                state = response + state

        return F.log_softmax(F.linear(state, self.out), dim=-1)

    def compute_weights(self, J):
        d = self.embed_size
        if self.use_bow: # 只使用词袋模型 就全部赋值为1——位置的权重是1
            weights = torch.ones(J, d)
        else:
            func = lambda j, k: 1 - (j + 1) / J - (k + 1) / d * (1 - 2 * (j + 1) / J)    # 0-based indexing 考虑位置编码
            weights = torch.from_numpy(np.fromfunction(func, (J, d), dtype=np.float32))
        return weights.cuda() if torch.cuda.is_available() else weights
