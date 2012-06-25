import networkx as nx
import sys
import os

sys.path.append(os.getcwd() + "/..")
import estrangement

class test_estrangement:
	def setUp(self):
	        self.g0 = nx.Graph()
        	self.g1 = nx.Graph()
		self.g2 = nx.Graph()
		self.g3 = nx.Graph()
		self.g4 = nx.Graph()
		self.g5 = nx.Graph()
		self.g5 = nx.Graph()
        	self.g0.add_edges_from([(1,2,{'weight':2}),(1,3,{'weight':1}),(2,4,{'weight':1})])
        	self.g1.add_edges_from([(1,4,{'weight':1}),(2,3,{'weight':1}),(3,4,{'weight':1})])
        	self.g2.add_edges_from([(1,2,{'weight':2}),(2,3,{'weight':1}),(3,4,{'weight':1})])
		self.g3.add_edges_from([(1,2,{'weight':2})])
		self.g4.add_edges_from([(1,2,{'weight':2}),(3,4,{'weight':1})])	
        	self.g5.add_edges_from([(1,2,{'weight':2}),(1,3,{'weight':1}),(2,3,{'weight':1}),(2,4,{'weight':1})])
		self.label_dict1 = {1:'a',2:'a',3:'b',4:'b',5:'c',6:'c'}	
		self.label_dict2 = {1:'a',2:'b',3:'b',4:'b',5:'c',6:'c'}	
		self.label_dict3 = {1:'a',2:'b',3:'c',4:'d',5:'e',6:'f'}	
		self.label_dict4 = {1:'a',2:'a',3:'a',4:'a',5:'a',6:'a'}	
		self.label_dict5 = {1:'b',2:'b',3:'b',4:'b',5:'b',6:'b'}

	def test_make_Zgraph(self):
		self.g6  = estrangement.make_Zgraph(self.g0,self.g2,self.label_dict4)  # Just the edge [1,2]
  		GM = nx.isomorphism.GraphMatcher(self.g3,self.g6)
		assert GM.is_isomorphic()
		self.g6 =  estrangement.make_Zgraph(self.g0,self.g0,self.label_dict4)  # same edges and communties
  		GM = nx.isomorphism.GraphMatcher(self.g0,self.g0)
		assert GM.is_isomorphic()
		self.g6 =  estrangement.make_Zgraph(self.g0,self.g0,self.label_dict3)  # no nodes belong to the same community
		assert len(self.g6.nodes()) == 0
		self.g6 =  estrangement.make_Zgraph(self.g0,self.g1,self.label_dict4)  # no overlapping edges
		assert len(self.g6.nodes()) == 0
		self.g6 =  estrangement.make_Zgraph(self.g0,self.g5,self.label_dict1)  # two overlapping edges only one has same label on both ends
		assert len(self.g6.nodes()) == 2
  		GM = nx.isomorphism.GraphMatcher(self.g6,self.g3)
		assert GM.is_isomorphic()
			
	def test_update_Zgraph(self):
		self.g6 = estrangement.update_Zgraph(self.g0,self.g0,self.g0,self.label_dict4)	# input is same a Zgraph
		GM = nx.isomorphism.GraphMatcher(self.g0,self.g6)	
		assert GM.is_isomorphic()  		
                self.g6 = estrangement.update_Zgraph(self.g3,self.g0,self.g0,self.label_dict4)  # Zgraph is a subgraph of input
                GM = nx.isomorphism.GraphMatcher(self.g0,self.g6)
                assert GM.is_isomorphic()	
                self.g6 = estrangement.update_Zgraph(self.g0,self.g1,self.g0,self.label_dict4)  # Zgraph has no edge in common with input
                GM = nx.isomorphism.GraphMatcher(self.g1,self.g6)
                assert GM.is_isomorphic()
                self.g6 = estrangement.update_Zgraph(self.g0,self.g0,self.g0,self.label_dict3)  # input has no two ends in same community
                assert len(self.g6.edges()) == 0
		self.g6 = estrangement.update_Zgraph(self.g5,self.g2,self.g0,self.label_dict1)  # more complicated example
		print(self.g6.edges())
                GM = nx.isomorphism.GraphMatcher(self.g4,self.g6)
                assert GM.is_isomorphic()
