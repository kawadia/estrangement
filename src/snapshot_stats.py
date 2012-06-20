
class SnapshotStatistics():
  def __init__(self):
      self.VI = {}  # key = time t, val = VI between partitions t and t-1
      self.VL = {}  # key = time t, val = Variation of labels between partitions t and t-1
      self.GD = {}  # key = time t, val = Graph distance between graphs t and t-1
      self.Node_GD = {}  # key = time t, val = Node graph distance between graphs t and t-1
      self.NumComm = {}  # key = time t, val = Number of communities at time t
      self.Q = {}  # key = time t, val = Modularity of partition at t
      self.Qstar = {}  # key = time t, val = Modularity of partition at t with tau=0
      self.F = {}  # key = time t, val = F of partition at t
      self.StrengthConsorts = {} # key = time t, val = strength of consorts at time t
      self.NumConsorts = {} # key = time t, val = Num of conorts at time t
      self.Estrangement = {} # key = time t, val = number of estranged edges at time t
      self.lambdaopt = {} # key = time t, lambdaopt found via solving the dual problem
      self.best_feasible_lambda = {} # key = time t, lambdaopt found via solving the dual problem
      self.numfunc = {} # key = time t, Number of function evaluations needed for solving the dual
      self.ierr = {} # key = time t, convergence of the dual
      self.feasible = {} # key = time t, convergence of the dual
      self.NumNodes = {}
      self.NumEdges = {}
      self.Size = {}
      self.NumComponents = {}
      self.LargestComponentsize = {}
      self.Qdetails = {} # {'time': {'lambduh': {'run_number': Q}}}
      self.Edetails = {} # {'time': {'lambduh': {'run_number': E}}}
      self.Fdetails = {} # {'time': {'lambduh': {'run_number': F}}}


