
import crocoddyl
import numpy as np
import torch
import matplotlib.pyplot as plt
from data import crocoddyl_cost

def a2m(a):
    return np.matrix(a).T


def m2a(m):
    return np.array(m).squeeze()


def validate_and_plot(net, plot_data = True, plot_error= True):
    """
    Returns the validation score(MSE) and scatter of error between V(x) and Net(x) 
    
    """
    net.eval()
    
    xrange = np.linspace(-2.,2.,100)
    xtest = torch.tensor([ [x1,x2, 0.] for x1 in xrange for x2 in xrange ], dtype = torch.float32)

    with torch.no_grad():
        y_pred = net(xtest)
    y_true = crocoddyl_cost(xtest)
    mean_squared_error = (y_true - y_pred) **2
    
    print(f"Mean Squared Error during testing is {torch.mean(mean_squared_error)}") 
    print("......................................................................")
    
    y_pred = y_pred.detach().numpy()
    y_true = y_true.numpy()
    
    if plot_data:
        print("\n Plot of ddp.cost from plain crocoddyl and cost predicted by Neural Network")
        plt.set_cmap('plasma')
        plt.figure(figsize=[10,12])

        trange = [ min(y_true),max(y_true) ]
        prange = [ min(y_pred),max(y_pred) ]
        vrange = [ min(trange+prange),max(trange+prange)]
        vd  = vrange[1]-vrange[0]
        vrange = [ vrange[0]-vd*.1, vrange[1]+vd*.1 ]

        plt.subplot(3,1,1)
        plt.scatter(xtest[:,0],xtest[:,1],c=y_true.flat,vmin=vrange[0],vmax=vrange[1])
        plt.colorbar().set_label("cost",labelpad=2, size=15)
        plt.title('Crocoddyl data')
        
                    

        plt.subplot(3,1,2)
        plt.scatter(xtest[:,0],xtest[:,1],c=y_pred.flat,vmin=vrange[0],vmax=vrange[1])
        plt.colorbar().set_label("cost",labelpad=2, size=15)
        plt.title('Neural Network Predictions')
        #plt.savefig("NetvsCrocoddyl.png")
        plt.subplots_adjust(hspace=0.25)
        plt.show()
        plt.close()
        

        
        
    if plot_error:
        print("Plotting scatter of error between ddp.cost and cost predicted by Neural Net")
        

        plt.figure(figsize=[10,4])
        plt.set_cmap('plasma')

        z = y_true - y_pred
        plt.scatter(xtest[:,0],xtest[:,1],c=z.flat)
        plt.title("Crocoddyl(x) - Neural_Net(x)")
        plt.colorbar().set_label('Error', labelpad=2, size = 15)
        #plt.savefig("NetvsCrocoddylscatterError.png")
        plt.show()
        plt.close()