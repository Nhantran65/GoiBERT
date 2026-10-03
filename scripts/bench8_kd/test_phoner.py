import unittest
import numpy as np
import torch
import torch.nn.functional as F
from run import token_pair,kd_loss

class TestPhoNER(unittest.TestCase):
    def test_positions_padding_multisubword_truncation(self):
        # Sentence a: tokens [special, word0, continuation(word0), word1,
        # pad, pad]; word2 is beyond max length. Sentence b is shorter.
        logits=torch.arange(2*6*3,dtype=torch.float32).reshape(2,6,3)
        batch={'ids':['a','b'],'positions':[[(1,0),(3,1)],[(1,0),(2,1)]],'word_count':[3,2]}
        soft={'a':(np.zeros((3,3),np.float32),np.array([1,1,0],bool)),'b':(np.zeros((2,3),np.float32),np.array([1,1],bool))}
        s,t=token_pair(logits,batch,soft)
        self.assertEqual(tuple(s.shape),(4,3))
        self.assertTrue(torch.equal(s,torch.stack([logits[0,1],logits[0,3],logits[1,1],logits[1,2]])))
    def test_sentence_word_lookup_after_shuffle(self):
        logits=torch.zeros(2,4,3);batch={'ids':['c','a'],'positions':[[(1,0)],[(1,0),(2,1)]],'word_count':[1,2]}
        soft={'a':(np.array([[1,0,0],[0,1,0]],np.float32),np.ones(2,bool)),'b':(np.array([[9,9,9]],np.float32),np.ones(1,bool)),'c':(np.array([[0,0,1]],np.float32),np.ones(1,bool))}
        _,t=token_pair(logits,batch,soft)
        self.assertEqual(t.tolist(),[[0,0,1],[1,0,0],[0,1,0]])
    def test_kd_loss_and_gradient(self):
        s=torch.tensor([[.2,-.3,1.],[1.,.4,-.2]],requires_grad=True);gold=torch.tensor([2,0]);t=F.one_hot(gold,3).float()*1000
        ce=F.cross_entropy(s/2,gold)*4;kd=kd_loss(s,t)
        self.assertLess(abs(float(kd-ce)),1e-5)
        kd.backward();self.assertTrue(torch.isfinite(s.grad).all())
        self.assertLess(abs(float(kd_loss(s,s.detach()))),1e-6)

if __name__=='__main__':unittest.main()
