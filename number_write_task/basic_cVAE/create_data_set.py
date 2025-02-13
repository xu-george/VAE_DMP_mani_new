# ----------------- code to generate trajectory data for the number write task
#  trainning data: random end point offset [-0.3, 0.3], test data set: [-0.3, 0.3]. test data set: [-0.5, 0.5]
import os.path
import sys
import numpy as np 
import matplotlib.pyplot as plt
import torch

grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Add the grandparent directory to sys.path
sys.path.insert(0, grandparent_dir)

from utils.data_loader import DataLoader
from models.dmp import CanonicalSystem, SingleDMP

# class to genereate trajectory with different ends
class Traj_data(DataLoader):
    def __init__(self, dmp, run_time=1.0, dt=0.01, dof=2):
        super(Traj_data, self).__init__(dmp=dmp, run_time=run_time, dt=dt, dof=dof)

    def data_augment(self, number, start=[0, 1], ends=[1, 0],   end_offset=0, end_range=0, aug_number=[1,2,3,6,7]):
        """
        data augmentation using classical DMP
        number: number of augmented data for each trajectory
        """
        all_paths = []
        all_labels = []
        init_size = len(self.paths)
        aug_paths = self.paths.copy()
        start = np.array(start)
        ends = np.array(ends)

        for i in range(init_size):             
            traj, label = aug_paths[i]
            number_label = int(label[0])
            if number_label not in aug_number:
                continue

            new_paths = np.zeros((len(self.time_steps), number, self.dof))
            for j in range(self.dof):

                self.dmp.imitate_path(traj[:, j])   

                begin = start[j]   
                for k in range(number):                              
                    goal = ends[j] + end_range * (np.random.rand() - 0.5) + end_offset
                    new_paths[:, k, j], _, _ = self.dmp.rollout(y0=begin, goal=goal)

            # updata labels
            for x in range(number):
                labels = np.array([int(label[0])])
                labels = np.append(labels, new_paths[0, x, :])
                labels = np.append(labels, new_paths[-1, x, :])
                all_paths.append(new_paths[:, x, :])
                all_labels.append(labels)

        self.paths = all_paths
        self.labels = all_labels


if __name__ == "__main__":
    # test the generated data
    cs = CanonicalSystem(dt=0.01, ax=1)
    dmp = SingleDMP(n_bfs=50, cs=cs, run_time=1.0, dt=0.01)

    print("current path: ", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    data_path = "./data/number_write/"

    # generate training data
    train_traj = Traj_data(dmp=dmp, run_time=1.0, dt=0.01, dof=2)
    train_traj.load_data_all(data_path)  

    # show all the loaded data
    plt.figure(1, figsize=(6, 6))
    # load all the data in the loader
    print("number of training data: ", len(train_traj))
    for i in range(len(train_traj)):
        inter, label = train_traj.paths[i] 
        plt.plot(inter[:, 0], inter[:, 1])
    
    plt.axis("equal")
    plt.title("loaded training data")
    plt.show()
    # save plot
    figure_name = f"loaded_train_data"
    plt.savefig(figure_name)

    # augment the data
    train_traj.data_augment(10, start=[0, 1], ends=[1, 0], end_offset=0, end_range=0)

    # plot the augmented data
    plt.figure(2, figsize=(6, 6))
    # load all the data in the loader
    print("number of training data: ", len(train_traj))
     
    number = 2
    for i in range(len(train_traj)):        
        inter, label = train_traj.paths[i], train_traj.labels[i]    
        if int(label[0]) == number:
            plt.plot(inter[:, 0], inter[:, 1])
    plt.axis("equal")
    plt.title("generated training data")
    plt.show() 
    # save plot
    figure_name = f"train_data"
    plt.savefig(figure_name)