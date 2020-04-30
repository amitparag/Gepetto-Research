import numpy as np
import torch 
import crocoddyl

class Datagen:
        
    def cost(self, size:int = 10, as_tensor:bool = True):
        """
        Returns xtrain, ytrain
        """
        x = []
        y = []
        for _ in range(size):
            xyz = np.array([np.random.uniform(-2.1, 2.1),
                            np.random.uniform(-2.1, 2.1),
                            np.random.uniform(-1., 1.)])

            squarederror = sum(xyz **2)
            x.append(xyz)
            y.append([squarederror])

        if as_tensor:
            x = torch.tensor(x, dtype = torch.float32)
            y = torch.tensor(y, dtype = torch.float32)
            return x, y
        else:
            return np.array(x), np.array(y)
            
    def solver_ddp(self, xyz):
        """
        Returns the ddp
        """
        model = crocoddyl.ActionModelUnicycle()
        T = 30
        model.costWeights = np.matrix([1,1]).T
        problem = crocoddyl.ShootingProblem(np.array(xyz).T, [ model ] * T, model)
        ddp = crocoddyl.SolverDDP(problem)
        ddp.solve([], [], 1000)
        return ddp

    def grid_data(self, size:int = 30):
        """
        @params:
            size = number of grid points
        @returns:
            grid array        
        """
        xrange = np.linspace(-2.,2.,size)
        xtest = np.array([ [x1,x2, 0.] for x1 in xrange for x2 in xrange ])
        return xtest
    
    def circular_data(self, r=[2], n=[100]):
        """
        @params:
            r = list of radius
            n = list of points required from each radius

        @returns:
            array of points from the circumference of circle of radius r centered on origin

        Usage: circle_points([2, 1, 3], [100, 20, 40])

        """
        circles = []
        for r, n in zip(r, n):
            t = np.linspace(0, 2* np.pi, n)
            x = r * np.cos(t)
            y = r * np.sin(t)
            z = np.zeros(x.size,)
            circles.append(np.c_[x, y, z])
        return np.array(circles).squeeze()
    
    def constant_data(self, constant:float = 0.99):
        print(f"x = {constant}, theta = 0.")
        y = np.linspace(-1., 1., 100)
        test =  np.array([ [constant,x2, 0.] for x2 in y ])
        return test