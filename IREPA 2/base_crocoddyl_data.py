
import numpy as np
import crocoddyl as c
c.switchToNumpyArray()
def base_data():
    # get the cost
    positions = []
    cost = []
    model = c.ActionModelUnicycle()
    for _ in range(1000):

        x0 = np.array([np.random.uniform(-2.1, 2.1), np.random.uniform(-2.1, 2.1), np.random.uniform(0,1)])
        T = 30
        problem = c.ShootingProblem(x0.T, [ model ] * T, model)
        ddp = c.SolverDDP(problem)
        ddp.solve()
        positions.append(x0)
        cost.append(np.array([ddp.cost, ddp.iter]))
    positions = np.asarray(positions)
    cost = np.asarray(cost)
    del model
    return positions, cost