from pyAudioAnalysis import MidTermFeatures as aF
import os
import numpy as np
import plotly.graph_objs as go 
import plotly
from sklearn.svm import SVC

import torch.nn as nn
import torch.nn.functional as F


dirs = [
        "/mnt/devel/datasets-psychology/wav/design/train/anger",
        #"/mnt/devel/datasets-psychology/wav/design/train/boredom",
        #"/mnt/devel/datasets-psychology/wav/design/train/disgust",
        #"/mnt/devel/datasets-psychology/wav/design/train/fear",
        #"/mnt/devel/datasets-psychology/wav/design/train/happiness",
        #"/mnt/devel/datasets-psychology/wav/design/train/neutral",
        "/mnt/devel/datasets-psychology/wav/design/train/sadness"
        ] 
class_names = [os.path.basename(d) for d in dirs] 
m_win, m_step, s_win, s_step = 1, 1, 0.1, 0.05 

# segment-level feature extraction:
features = [] 
for d in dirs: # get feature matrix for each directory (class) 
    f, files, fn = aF.directory_feature_extraction(d, m_win, m_step, 
                                                   s_win, s_step) 
    features.append(f)

# (each element of the features list contains a 
# (samples x segment features) = (10 x 138) feature matrix)
print(features[0].shape)
print('Feature names:')
for i, nam in enumerate(fn):
    print(f'{i}:{nam}')

# select 2 features and create feature matrices for the two classes:
chosenfeatures = ['spectral_centroid_mean',
                  'energy_entropy_mean'
                 ] 

chosenfeatures = ['spectral_entropy_mean',
                  'energy_entropy_mean'
                 ] 

chosenfeatures = ['mfcc_mean',
                  'energy_entropy_mean'
                 ] 
y = []
plots = []
for i in range(0, len(dirs)):
        #choose = np.array([features[i][:, fn.index(chosenfeatures[0])]])
        #for feature in range(1, len(chosenfeatures)):
        #        choose = np.concatenate((choose, features[i][:, fn.index(chosenfeatures[feature])]))

        choose = np.array([features[i][:, fn.index(chosenfeatures[0])],
               features[i][:, fn.index(chosenfeatures[1])]])

        plot = go.Scatter(x=choose[0, :], y=choose[1, :], name=class_names[i], mode='markers')
        plots.append(plot)
        print ("Plot 2 features for class ", class_names[i])
        print(choose.shape)
        print(np.full(choose.shape[1], i).shape) 
        if i == 0:
                input = choose.T
                output = np.full(choose.shape[1], i)
        else:
                input = np.concatenate((input, choose.T))
                output = np.concatenate((output, (np.full(choose.shape[1], i)))) 
        print(output.shape)

oldlayout = go.Layout(xaxis=dict(title=chosenfeatures[0]),
                     yaxis=dict(title=chosenfeatures[1]))
#plotly.offline.iplot(go.Figure(data=plots, layout=oldlayout))


#SVM example
# train the svm classifier
cl = SVC(kernel='rbf', C=20) 
cl.fit(input, output) 



# apply the trained model on the points of a grid
x_ = np.arange(input[:, 0].min(), input[:, 0].max(), 0.002) 
y_ = np.arange(input[:, 1].min(), input[:, 1].max(), 0.002) 
xx, yy = np.meshgrid(x_, y_) 
Z = cl.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape) / 2 
# and visualize the grid on the same plot (decision surfaces)
cs = go.Heatmap(x=x_, y=y_, z=Z, showscale=False, 
               colorscale= [[0, 'rgba(255, 182, 193, .3)'], 
                           [1, 'rgba(100, 100, 220, .3)']]) 
mylayout = go.Layout(xaxis=dict(title=chosenfeatures[0]),
                     yaxis=dict(title=chosenfeatures[1]))
plotly.offline.iplot(go.Figure(data=[plots[0], plots[1], cs], layout=mylayout))



