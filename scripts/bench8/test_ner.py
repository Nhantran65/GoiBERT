import unittest
from ner import spans,metrics
class TestNER(unittest.TestCase):
 def test_initial_i(self):self.assertEqual(spans(['I-A','I-A','O']),[('A',0,2)])
 def test_adjacent(self):self.assertEqual(spans(['B-A','I-A','B-A','B-B']),[('A',0,2),('A',2,3),('B',3,4)])
 def test_boundary_and_type(self):
  m=metrics([['B-A','I-A','O','B-B']],[['B-A','O','O','B-A']]);self.assertEqual((m['entity_tp'],m['entity_gold'],m['entity_predicted']),(0,2,2))
 def test_i_type_switch(self):self.assertEqual(spans(['B-A','I-B']),[('A',0,1),('B',1,2)])
if __name__=='__main__':unittest.main()
