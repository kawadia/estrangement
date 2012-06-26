import networkx as nx
import sys
import os

sys.path.append(os.getcwd() + "/..")
import lpa
import utils

class opt:
	resolution = 0.2
	gap_proof_estrangement = True
	delta = 0
	tolerance = 0.4
	precedence_tiebreaking = True

class test_utils:
	def setUp(self):
		self.g0 = nx.Graph()
                self.g1 = nx.Graph()
                self.g2 = nx.Graph()
                self.g3 = nx.Graph()
                self.g4 = nx.Graph()
                self.g5 = nx.Graph()
                self.g5 = nx.Graph()
                self.g0.add_edges_from([(1,2,{'weight':1}),(2,3,{'weight':1}),(3,4,{'weight':1}),(4,5,{'weight':1}),(5,1,{'weight':1})])  # circle
                self.g1.add_edges_from([(1,2,{'weight':1}),(2,3,{'weight':1}),(3,4,{'weight':2}),(2,4,{'weight':1}),(4,5,{'weight':4}),(3,5,{'weight':4})])
		self.label_dict1 = {1:'a',2:'a',3:'a',4:'a',5:'a'}
		self.label_dict2 = {1:'b',2:'a',3:'a',4:'a',5:'a'}
		self.label_dict3 = {1:'b',2:'b',3:'a',4:'a',5:'a'}

	def test_lpa(self):
		out_label_dict = lpa.lpa(self.g0,opt,1,self.label_dict1)  # all are in the same community => no change
		assert out_label_dict == self.label_dict1 
		out_label_dict = lpa.lpa(self.g0,opt,1,self.label_dict2)  # 1 in 'a', others in 'b' => no change
		assert out_label_dict == self.label_dict1 

		out_label_dict = lpa.lpa(self.g0,opt,1,self.label_dict3)  # b---b---a    b---a---a     a---a---a
		out_label_dict2 = lpa.lpa(self.g1,opt,1,out_label_dict)   #     | \ | =>     | \ | =>      | \ |
		assert out_label_dict2 == self.label_dict1             	  #     a---a        a---a         a---a
