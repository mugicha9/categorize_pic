import os
from skimage import io, color
from skimage.transform import resize

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

dir_apple = "./images/apple/"
dir_orange = "./images/orange/"

data = {'data': [], 'label': []}
value_to_label = {
    0: 'apple',
    1: 'orange'
}

def load_image(path):
    img = io.imread(path)
    if img.shape[2] == 4:
        img = color.rgba2rgb(img)
    resized_img = resize(img, (64, 64))

    return resized_img

def read_images(dir, label):
    list_files = os.listdir(dir)
    for file_name in list_files:
        img = io.imread(dir + file_name)
        if img.ndim == 2:
            continue
        if img.shape[2] == 4:
            img = color.rgba2rgb(img)
        resized_img = resize(img, (64, 64))

        data['data'].append(resized_img)
        data['label'].append(label)

read_images(dir_apple, 0)
read_images(dir_orange, 1)

reshaped_data = np.array(data['data']).reshape(len(data['data']), -1)

X_train, X_test, y_train, y_test = train_test_split(reshaped_data, data['label'])
steps = [('pca', PCA()), ('model', SVC())]
pipe = Pipeline(steps)

pipe.fit(X_train, y_train)

y_test_pred = pipe.predict(X_test)
print(accuracy_score(y_test, y_test_pred))

X_new_data = load_image("./images/test/4140.png")
X_new = np.array(X_new_data).reshape(1, -1)

print(value_to_label[pipe.predict(X_new)[0]])