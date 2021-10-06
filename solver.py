from gurobipy import *
import numpy as np

def obtain_sol(sol, n):
	M = [[0 for j in range(n)] for i in range(n)]
	for i in range(n):
		for j in range(n):
			for v in range(n):
				val = sol[i, j, v]
				if val == 1:
					M[i][j] = v+1
					break
	return M

def solve_sudoku(M):
	m = Model('sudoku')
	Q = [0, 3, 6]
	N = [0,1,2,3,4,5,6,7,8]
	x = m.addVars(9, 9, 9, vtype=GRB.BINARY)
	m.update()
	for i in range(len(M)):
		for j in range(len(M)):
			if M[i][j] > 0:
				m.addConstr(x[i, j, M[i][j]-1] == 1)
	m.update()
	m.setObjective(0, GRB.MINIMIZE)
	m.update()
	[m.addConstr(sum(x[i, j, v] for i in N) == 1) for j in N for v in N]
	[m.addConstr(sum(x[i, j, v] for j in N) == 1) for i in N for v in N]
	[m.addConstr(sum(x[i, j, v] for v in N) == 1) for i in N for j in N]
	[m.addConstr(sum(x[i, j, v]
					 for i in range(a, a+3)
					 for j in range(b, b+3)) == 1)
	 				for v in N for a in Q for b in Q]
	m.update()
	m.optimize()
	sol = m.getAttr('x', x)
	M = obtain_sol(sol, len(N))
	return M


M = [[0, 2, 0, 0, 0, 0, 0, 0, 8],
	 [0, 0, 0, 4, 0, 0, 0, 5, 0],
	 [1, 0, 0, 3, 8, 5, 0, 0, 0],
	 [0, 0, 2, 0, 0, 0, 6, 9, 0],
	 [0, 0, 0, 5, 0, 0, 1, 0, 0],
	 [6, 1, 4, 2, 0, 0, 0, 0, 0],
	 [4, 0, 9, 0, 0, 0, 0, 1, 0],
	 [2, 0, 0, 0, 0, 0, 7, 0, 0],
	 [0, 8, 0, 0, 9, 7, 0, 0, 0]]

M = solve_sudoku(M)
print(np.matrix(M))
