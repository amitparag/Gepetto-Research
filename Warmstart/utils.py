
import numpy as np


def plot_trajectories(cost, trajectories, name = "Cost", savename=None, title=None ):
    """
    
    @params:
        cost           = list of keys for cmap
        trajectories   = list of corresponding trajectories
        name           = str, to distinguish between cost and iterations
        
    @ returns plot of trajectories colored according to keys.    
    
    """

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams['figure.dpi'] = 80
    fig = plt.figure(figsize=(8, 6))

    norm = mpl.colors.Normalize(vmin=min(cost), vmax=max(cost))
    cmap = mpl.cm.ScalarMappable(norm = norm, cmap=mpl.cm.plasma)
    cmap.set_array([])


    for key, trajectory in zip(cost, trajectories):
        plt.scatter(trajectory[:, 0], trajectory[:, 1], 
                    marker = '',
                    zorder=2, 
                    s=50,
                    linewidths=0.2,
                    alpha=.8, 
                    cmap = cmap )
        plt.plot(trajectory[:, 0], trajectory[:, 1], c=cmap.to_rgba(key))

    plt.xlabel("X Coordinates", fontsize = 20)
    plt.ylabel("Y Coordinates", fontsize = 20)
    if title:
        plt.title(title)
    plt.colorbar(cmap).set_label(name, labelpad=2, size=15)
    if savename is not None:
        plt.savefig("./images/"+savename+".png")
    plt.show()

    
    
def a2m(a):
    return np.matrix(a).T


def m2a(m):
    return np.array(m).squeeze()


def circular(r=[2], n=[100]):
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
