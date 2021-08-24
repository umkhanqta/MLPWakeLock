import ml
import os
import numpy as np
from random import shuffle
from tqdm import tqdm
import networkx as nx


class Analysis:
    """ A class to run a classification experiment """

    def __init__(self, dirs, labels, max_files=0, max_node_size=0):
        """ 
        The Analysis class allows to load sets of pickled graoh objects
        from different directories where the objects in each directory
        belong to different classes. It also provide the methods to run
        different types of classification experiments by training and 
        testing a linear classifier on the feature vectors generated
        from the different graph objects.

        :dirs: A list with directories including types of files for
            classification e.g. <[LEAK_DIR, CLEAN_DIR]> or just
            directories with samples from different wake lock leak
        :labels: The labels assigned to samples in each directory.
            For example a number or a string.
        :returns: an Analysis object with the dataset as a set of
            properties and several functions to train, test, evaluate
            or run a learning experiment iteratively
        """

        self.X = []
        self.Y = np.array([])
        self.label_dist = np.zeros(2**15)
        self.sample_sizes = []
        self.neighborhood_sizes = []
        self.class_dist = np.zeros(15)
        self.predictions = []
        self.fnames = []

        for d in zip(dirs, labels):
            files = self.read_files(d[0], "pz", max_files)
            print("Loading samples in dir {0} with label {1}".format(d[0],
                                                                     d[1]))
            # load labels and feature vectors
            for f in tqdm(files):
                try:
                    g = nx.read_gpickle(f)
                    size = g.number_of_nodes()
                    if size < max_node_size or max_node_size == 0:
                        if size > 0:
                            x_i = self.compute_label_histogram(g)
                            # save distribution of generated labels
                            self.label_dist = np.sum([self.label_dist,
                                                      x_i], axis=0)
                            # save sizes of the sample for further analysis
                            # of the dataset properties
                            self.sample_sizes.append(size)
                            self.neighborhood_sizes += ml.neighborhood_sizes(g)
                            for n, l in g.node.items():
                                self.class_dist = np.sum([self.class_dist,
                                                          l["label"]], axis=0)
                            # delete nx object to free memory
                            del g
                            self.X.append(x_i)
                            self.Y = np.append(self.Y, [int(d[1])])
                            self.fnames.append(f)
                except Exception as e:
                    print(e)
                    print("err: {0}".format(f))
                    pass

        # convert feature vectors to its binary representation
        # and make the data matrix sparse
        print("[*] Stacking feature vectors...")
        self.X = np.array(self.X, dtype=np.int32)
        print("X=", self.X.shape)

    ################################
    # Data Preprocessing functions #
    ################################

    def read_files(self, d, file_extension, max_files=0):
        """ Return a random list of N files with a certain extension in dir d
        
        Args:
            d: directory to read files from
            file_extension: consider files only with this extension
            max_files: max number of files to return. If 0, return all files.

        Returns:
            A list of max_files random files with extension file_extension
            from directory d.
        """

        files = []
        # d = os.getcwd()+d
        for f in os.listdir(d):
            if f.lower().endswith(file_extension):
                files.append(os.path.join(d, f))
        shuffle(files)

        # if max_files is 0, return all the files in dir
        if max_files == 0:
            max_files = len(files)
        files = files[:max_files]
        
        return files

    def compute_label_histogram(self, g):
        """ Compute the neighborhood hash of a graph g and return
            the histogram of the hashed labels.
        """

        g_hash = ml.neighborhood_hash(g)
        g_x = ml.label_histogram(g_hash)
        return g_x

    def save_data(self, path, x_name, y_name):
        """ Store npz objects for the data matrix, the labels and
            the name of the original samples so that they can be used
            in a new experiment without the need to extract all
            features again
        """
        x_path = os.getcwd()+path+r'\\'+x_name
        y_path = os.getcwd()+path+r'\\'+y_name
        fnames_path = os.getcwd()+path+r'\fnames.npz'
        print("[*] Saving labels, data matrix and file names at...",x_path)
        np.savez_compressed(x_path, self.X)
        np.savez_compressed(y_path, self.Y)
        np.savez_compressed(fnames_path, self.fnames)


