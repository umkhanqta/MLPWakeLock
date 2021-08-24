""" A module to build NX graph objects from APKs. """

import zipfile
import networkx as nx
import numpy as np

from instructionSet import INSTRUCTION_SET_COLOR
from instructionSet import INSTRUCTION_CLASS_COLOR
from instructionSet import INSTRUCTION_CLASSES

from androguard.core.bytecodes.apk import APK
from androguard.core.analysis.analysis import Analysis
from androguard.core.bytecodes.dvm import DalvikVMFormat
from androguard.misc import AnalyzeDex
from utils import get_sha256
from tqdm import tqdm
import sys
import os

class CG():

    def __init__(self, filename):
        self.filename = filename
        try:
            self.a = APK(filename)
            self.d = DalvikVMFormat(self.a.get_dex())
            self.d.create_python_export()
            self.dx = Analysis(self.d)
        except zipfile.BadZipfile:
            # if file is not an APK, may be a dex object
            _, self.d, self.dx = AnalyzeDex(self.filename)

        self.d.set_vmanalysis(self.dx)
        self.dx.create_xref()
        self.cfg = self.build_cfg()

    def get_cg(self):
        return self.cfg

    def get_cfg(self):
        return self.dx.get_call_graph()

    def build_cfg(self):
        """ Using NX and Androguard, build a directed graph NX object so that:
            - node names are analysis.MethodClassAnalysis objects
            - each node has a label that encodes the method behavior
        """
        cfg = self.get_cfg()  ##/////////My changes///////////////
        for n in cfg.nodes:
            instructions = []
            # print(n)
            try:
                ops = n.get_instructions()
                for i in ops:
                    instructions.append(i.get_name())
                # print(ops)
                encoded_label = self.color_instructions(instructions)
                # print("No Exception")
            except AttributeError:
                encoded_label = np.array([0] * 15)
            cfg.node[n]["label"] = encoded_label
        return cfg

    def color_instructions(self, instructions):
        """ Node label based on coloring technique by Kruegel """

        h = [0] * len(INSTRUCTION_CLASS_COLOR)
        for i in instructions:
            h[INSTRUCTION_SET_COLOR[i]] = 1
        return np.array(h)

    def get_classes_from_label(self, label):
        classes = [INSTRUCTION_CLASSES[i] for i in range(len(label)) if label[i] == 1]
        return classes


def process_dir(read_dir, out_dir, mode='CG'):
    """ Convert a series of APK into graph objects. Load all
    APKs in a dir subtree and create graph objects that are pickled
    for later processing and learning.
    """
    # read_dir = os.getcwd() + read_dir
    # out_dir = os.getcwd() + out_dir
    sys.setrecursionlimit(100000)
    files = []

    # check if pdg doesnt exist yet and mark the file to be processed
    for dirName, subdirList, fileList in os.walk(read_dir):
        for f in fileList:
            if f.endswith(".apk"):
                files.append(os.path.join(dirName, f))

    # loop through .apk files and save them in .pdg.pz format
    print("\nProcessing {} APK files in dir {}".format(len(files), read_dir))
    for f in tqdm(files):
        # nx.DiGraph().nodes
        f = os.path.realpath(f)
        print('[] Loading {0}'.format(f))
        try:
            if mode is 'CG':
                graph = CG(f).get_cg()

        # if an exception happens, save the .apk in the corresponding dir
        except Exception as e:
            err = e.__class__.__name__
            err_dir = err + "/"
            d = os.path.join(read_dir, err_dir)
            if not os.path.exists(d):
                os.makedirs(d)
            cmd = "cp {} {}".format(f, d)
            os.system(cmd)
            print("[*] {} error loading {}".format(err, f))
            continue

        h = get_sha256(f)
        if not out_dir:
            out_dir = read_dir
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        fnx = os.path.join(out_dir, "{}".format(h))
        try:
            nx.write_gpickle(graph, fnx+'.pz')
            print("[*] Saved {}\n".format(fnx))
        except:
            print("[*] Error in pickle cannot save {}\n".format(fnx))
    print("Done.")
