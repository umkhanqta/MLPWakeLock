import analysis
import warnings
warnings.filterwarnings("ignore")
from graphs import process_dir
from NNAlgo import *
from MLAlgo import apply_smote_ML

clean_dir = r'\Dataset\CleanAPK'
clean_out_dir = r'\Dataset\CleanAppsOut'

leak_dir = r'\Dataset\LeakAPK'
leak_out_dir = r'\Dataset\LeakAppsOut'
compressed_path = r'\Dataset\CompressedFiles'

 ##/////////////Step-1: Extracting CG///////////////////////////
# process_dir(leak_dir, leak_out_dir, mode='CFG') ## FCG and CG complete
# process_dir(clean_dir, clean_out_dir, mode='CFG') ## FCG and CG complete

##/////////////Step-2: Labelling and Hashing/////////////////////
# a = analysis.Analysis([leak_out_dir], labels=[1])
# a.save_data(compressed_path, 'LeakCGX.npz', 'LeakCGY.npz') ## CG

# a = analysis.Analysis([clean_out_dir], labels=[0])
# a.save_data(compressed_path, 'CleanDroidCFGX.npz', 'CleanDroidCFGY.npz') ## CG and FCG

##//////////////Step-3: Load Data and Apply MLP and Plot/////////////////
X, y = load_data(compressed_path)
X_smt, y_smt = apply_smote(X, y)
apply_MLP(X_smt, y_smt)

##///////////Step-4: Applying Machine Learning Algorithms////////
# X, y = load_data(compressed_path)
# X_smt, y_smt = apply_smote_ML()
