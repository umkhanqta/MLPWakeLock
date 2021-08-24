import numpy
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import warnings
warnings.filterwarnings("ignore")

model_path = r'\Dataset\Model'

def save_json(model, fileName='model'):
    # serialize model to JSON
    model_json = model.to_json() ##///////////Saving Model////////////////////////
    with open(fileName+'.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(fileName+'h5')
    print("Saved model to disk")

def load_json(fileName='model'):
    # # load json and create model
    json_file = open(fileName+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    from keras.models import model_from_json
    loaded_model = model_from_json(loaded_model_json)##///////////Loading Model/////////////////
    # load weights into new model
    loaded_model.load_weights(fileName+".h5")
    print("Loaded model from disk")


def smote_over_sampling(X, y):
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(sampling_strategy='minority')
    X_sm, y_sm = smote.fit_resample(X, y)
    return X_sm, y_sm

def randomize_dataset(X, Y):
    """ Randomly split the dataset in training and testing sets
    """
    n = len(Y)
    ##/////////For 80 and 20 % split/////////
    train_size = int(0.8 * n)
    index = list(range(n))
    # print(n, train_size, index)
    from random import shuffle
    shuffle(index)
    train_index = sorted(index[:train_size])
    test_index = sorted(index[train_size:])

    X_train = X[train_index, :]
    X_test = X[test_index, :]
    Y_train = Y[train_index]
    Y_test = Y[test_index]
    # print("train=", len(Y_train))
    # print("test=", len(Y_test))
    return X_train, X_test, Y_train, Y_test


def apply_smote(X, y):
    X_smt, y_smt = smote_over_sampling(X, y)
    return X_smt, y_smt

def apply_MLP(X, y):
    # # define the keras model
    from keras.layers import Dropout
    from keras.regularizers import l1
    from sklearn.model_selection import train_test_split
    X1, Xt, y1, yt = randomize_dataset(X, y)
    train_X, test_X, train_y, test_y = train_test_split(X1, y1, test_size=0.2, random_state=2, stratify=y1)

    model = Sequential()
    model.add(Dense(512, kernel_initializer='normal', input_dim=32768, activation='relu',
                    activity_regularizer=l1(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])

    history = model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=500, batch_size=8, verbose=2)

    # print(Xt.shape, yt)
    # print(model.summary())
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    score = model.evaluate(Xt, yt, verbose=2)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    # plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Test'], loc='lower right')
    plt.grid(True)
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    # plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.grid(True)
    plt.show()

def load_data(compressed_dir_path):
    from numpy import load
    import os
    import numpy as np
    dict_data = load(os.path.join(compressed_dir_path, 'LeakCGX.npz'))
    X = dict_data['arr_0']
    dict_lbl = load(os.path.join(compressed_dir_path, 'LeakCGY.npz'))
    y = dict_lbl['arr_0']
    ##///////////////////////////////////////////////////////////////////
    dict_data = load(os.path.join(compressed_dir_path, 'CleanCGX.npz'))
    X = np.append(X, dict_data['arr_0'], 0)
    # print(X)
    dict_lbl = load(os.path.join(compressed_dir_path, 'CleanCGY.npz'))
    y = np.append(y, dict_lbl['arr_0'], 0)
    ##///////////////////////////////////////////////////////////////////////
    dict_data = load(os.path.join(compressed_dir_path, 'CleanDroidCGX.npz'))
    X = np.append(X, dict_data['arr_0'], 0)
    # print(X)
    dict_lbl = load(os.path.join(compressed_dir_path, 'CleanDroidCGY.npz'))
    y = np.append(y, dict_lbl['arr_0'], 0)
    # print(len(X))
    return X, y


# X_smt, y_smt = apply_smote(X, y)
# apply_MLP(X_smt, y_smt)

# import MLAlgo
# X_train, X_test, y_train, y_test = randomize_dataset(X_smt, y_smt)
#
# nb_list = []
# sv_list = []
# knr_list = []
# lr_list = []
# rc_list = []
# bdt_list = []
# rf_list = []
# sgb_list = []
# for i in range(10):
#     nb, sv, knr, lr, rc, bdt, rf, sgb = MLAlgo.apply_different_model(X_train, y_train, X_test, y_test)
#     nb_list.append(nb)
#     sv_list.append(sv)
#     knr_list.append(knr)
#     lr_list.append(lr)
#     rc_list.append(rc)
#     bdt_list.append(bdt)
#     rf_list.append(rf)
#     sgb_list.append(sgb)
# import matplotlib.pyplot as plt
# print('NB=',  nb_list)
# print('SV=',  sv_list)
# print('KNN=',  knr_list)
# print('LR=',  lr_list)
# print('RC=',  rc_list)
# print('BDT=',  bdt_list)
# print('RF=',  rf_list)
# print('SGB=',  sgb_list)
# plt.plot(nb_list)
# plt.plot(sv_list)
# plt.plot(knr_list)
# plt.plot(lr_list)
# plt.plot(rc_list)
# plt.plot(bdt_list)
# plt.plot(rf_list)
# plt.plot(sgb_list)
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Iteration')
# plt.legend(['Naive-Bay', 'Linear SVC', 'K-NearNeighbor', 'Logistic Regression', 'Ridge Classifier', 'Bagged Decision Tree', 'Random Forest', 'Stochastic Gradient Boosting'])
# plt.show()

# apply_basic_NN_validate(X_smt, y_smt, X_val, y_val)
# print(len(y_smt))
##////////////////////////////////////////////////////////////////////////////////////////
##///////////get accuracy of 0.800 and loss of 1.3098/////////////////////////////////////
##////////////////////////////////////////////////////////////////////////////////////////