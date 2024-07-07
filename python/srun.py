import numpy as np
from skimage.io import imread, imshow, imsave
import argparse
import pickle
import matplotlib.pyplot as plt
import networkx as nx
cmap = plt.get_cmap('viridis')
from tqdm import tqdm

# Import from internal class
from score import Network
# from sconfig import *
import sconfig as config
# from sio import IO



class NetRead():
    def __init__(self, image_dir = '../../data/10stacks.tif', save_dir = './crop.tif', voxel_size = [0.0901435, 0.0901435, 1.0273500], numStackZ = 10, nnx=2048, nny=2048, cx=111, cy=111):
        """
        Executes various operations within a tif file.

        Parameters:
        image_dir: directory of the original tif image
        save_dir: directory to save the cropped image
        voxel_size: size of the voxel
        numStackZ: number of stack of images in z direction
        nnx, nny = number of voxels in x and y direction (original image)
        cx, cy: number of voxels in x and y direction (cropped image)
        """

        self.image_dir = image_dir
        self.save_dir = save_dir
        self.voxel_size = voxel_size
        self.numStackZ = numStackZ
        self.nnx = nnx
        self.nny = nny
        self.cx = cx
        self.cy = cy

    """ Crop image from the original image """
    def crop_image(self):
        """
        Crops original image into square cx by cy number of indices.
        """

        # Load image (original)
        image = imread(image_dir)
        print("Shaped of loaded image: ", image.shape)

        # Random indices of image to crop
        startx = np.random.randint(0, self.nnx-self.cx)
        starty = np.random.randint(0, self.nny-self.cy)

        # Voxel size
        vx, vy, vz = self.voxel_size
        self.cz = round(vz/vx * self.numStackZ)
        self.volume_shape = (self.cz, self.cx, self.cy)
        self.transform_box = np.diag([vx * 1, vx, vx])
        print("Number of pixels (Z, X, Y): %d, %d, %d" %(self.cz, self.cx, self.cy))

        # Crop image
        crop_image = image[:, startx:startx+self.cx, starty:starty+self.cy]
        imsave(self.save_dir, crop_image)

    """ Run Qiber3D to extract network """
    def read_network(self):
        """
        Runs Qiber3D and reads network and saves 
        graph.pkl - nodes on the lattice grid (networkx)
        segments_smooth.pkl - spline interpolated segments (dict)
        """

        config.extract.voxel_size = self.voxel_size
        net_ex = self.save_dir
        net = Network.load(net_ex)

    """ Plot fiber segments """
    def plot_segments(self):
        """
        Open 'segments_smooth.pkl' and plots it.
        """

        with open('segments_smooth.pkl', 'rb') as f:
            segments_smooth = pickle.load(f)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        for sid, segment in segments_smooth.items():
            points = segment.point
            points = points @ self.transform_box
            
            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]

            color = cmap(np.random.rand())

            ax.plot(x, y, z, marker='o', markersize=2, color=color, label=f'Segment {sid}')

        ax.set_xlabel(r'X ($\mu m$)')
        ax.set_ylabel(r'Y ($\mu m$)')
        ax.set_zlabel(r'Z ($\mu m$)')
        ax.set_title('Spline interpolated fibrin segments')

        # Show the plot
        plt.savefig('segments.png')
        plt.clf()

    """ Plot from graph (sanity check) """
    def plot_graph(self):
        """
        Open 'graph.pkl' and plots it.
        """

        with open('graph.pkl', 'rb') as f:
            graph = pickle.load(f)

        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111, projection='3d')
        dists  =  []

        for edge in graph.edges():
            p1 = np.unravel_index(edge[0], self.volume_shape) @ self.transform_box
            p2 = np.unravel_index(edge[1], self.volume_shape) @ self.transform_box
            dists.append(np.linalg.norm(p1-p2))
            zs = [p1[0], p2[0]]
            ys = [p1[1], p2[1]]
            xs = [p1[2], p2[2]]

            ax2.plot(xs, ys, zs, color='black')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        plt.savefig('graph.png')
        plt.clf()

        plt.figure(3)
        plt.hist(dists, 100)
        plt.xlabel('bond length [um]')
        plt.savefig('hist.png')
        plt.clf()

    """ Generate graph (networkx) """
    def create_graph(self, n, spline=False):
        """
        Open pkl file and plots it.
        Choose between spline interpolated data (segments_smooth.pkl) or pure lattice grid data (graph.pkl)
        nth dataset
        """
        fibrinG = nx.Graph()

        if spline:
            with open('segments_smooth.pkl', 'rb') as f:
                segments_smooth = pickle.load(f)
            
            for sid, segment in segments_smooth.items():
                points = segment.point
                print(points)
                points = points @ self.transform_box
                
                x = points[:, 0]
                y = points[:, 1]
                z = points[:, 2]

                # STILL WORKING ON IT!

        else:
            with open('graph.pkl', 'rb') as f:
                graph = pickle.load(f)

            for edge in graph.edges():
                p1 = np.unravel_index(edge[0], self.volume_shape) @ self.transform_box
                p2 = np.unravel_index(edge[1], self.volume_shape) @ self.transform_box
                p1 = np.array(p1)
                p2 = np.array(p2)
                dist = np.linalg.norm(p1 - p2)

                fibrinG.add_node(edge[0], coordinates=p1)
                fibrinG.add_node(edge[1], coordinates=p2)
                fibrinG.add_edge(edge[0], edge[1], distance=dist)

            with open('fibrinG' + str(n) + '.pkl', 'wb') as f:
                pickle.dump(fibrinG, f)
                print("GRAPH FOR GNN SAVED.")



# Receive arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='../../data/10stacks.tif', help="Directory for images.")
    parser.add_argument('--save_dir', type=str, default='./crop.tif', help="Directory to save images.")
    parser.add_argument('--plot', default=True, help="Plot results.")
    parser.add_argument('--create_dataset', default=False, help="Create multiple datasets.")
    parser.add_argument('--num_data', type=int, default=100, help="Number of datasets to create.")

    return parser.parse_args()

args = parse_args()
image_dir = args.image_dir
save_dir = args.save_dir
plot = args.plot
create_dataset = args.create_dataset
num_data = args.num_data



# Create instance
myNetRead = NetRead(image_dir=image_dir, save_dir=save_dir)

if create_dataset:
    for n in tqdm(range(num_data), desc='Create Dataset'):
        myNetRead.crop_image()
        myNetRead.read_network()    
        myNetRead.create_graph(n=n, spline=False)

else:
    myNetRead.crop_image()
    myNetRead.read_network()
    myNetRead.plot_segments()
    myNetRead.plot_graph()
    myNetRead.create_graph(spline=False)
