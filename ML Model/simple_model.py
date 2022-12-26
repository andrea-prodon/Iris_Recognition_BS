from dataset import IrisDataset
import numpy as np
from lbp import lbp_code
import cv2
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, auc, roc_curve
import warnings
warnings.filterwarnings('ignore') # setting ignore as a parameter

def shuffle_in_unison(a: np.ndarray , b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

if __name__ == '__main__':
    rootpath = "Dataset/CASIA_Iris_interval_norm/"
    samples = 108
    lbp_npoint = 24
    lbp_radius = 8
    train_data_set = IrisDataset(data_set_path=rootpath, n_samples = samples)
    #train_data = np.array([lbp_code(sample['image'].astype('float32'),lbp_npoint,lbp_radius)[0].reshape(-1) for sample in train_data_set])
    train_label = np.array([sample['label'] for sample in train_data_set])
    train_data = np.array([sample['image'].reshape(-1) for sample in train_data_set])
    test_data_set = IrisDataset(data_set_path=rootpath, train = False, n_samples = samples)
    #test_data = np.array([lbp_code(sample['image'].astype('float32'),lbp_npoint,lbp_radius)[0].reshape(-1) for sample in test_data_set])
    test_data = np.array([sample['image'].reshape(-1) for sample in test_data_set])
    test_label = np.array([sample['label'] for sample in test_data_set])
    
    #dataset shuffling
    train_data, train_label = shuffle_in_unison(train_data, train_label)
    #print(f'Test performed with lbp npoint = {lbp_npoint}, radius ={lbp_radius}')
    print('Test performed with 1D images')
    #SVM
    SVM = svm.SVC(kernel='linear', C=1)
    SVM.fit(train_data, train_label)
    y_pred = SVM.predict(test_data)
    acc = accuracy_score(y_pred, test_label)
    print(f'linear SVM - Accuracy score: {acc}')
    #MLP
    mlp = MLPClassifier(activation='tanh', alpha= 0.05, hidden_layer_sizes= (50, 50, 50), learning_rate='constant', solver='adam')
    mlp.fit(train_data, train_label)
    y_pred = mlp.predict(test_data)
    acc = accuracy_score(y_pred, test_label)
    print(f'MLP - Accuracy score: {acc}')
    #Random Forest
    rf = RandomForestClassifier(criterion='entropy',max_features='auto',n_estimators=300)
    rf.fit(train_data, train_label)
    y_pred = rf.predict(test_data)
    acc = accuracy_score(y_pred, test_label)
    print(f'Random Forest - Accuracy score: {acc}')