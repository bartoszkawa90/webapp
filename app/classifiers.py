# imports

# STANDARD
from .resources import *

# for Kmeans
from sklearn.cluster import KMeans

# for KNN
from sklearn.neighbors import KNeighborsClassifier
# from skimage.transform import resize as skresize
import json

# for SVC
# from skimage.io import imread as skimread
# from skimage.transform import resize as skresize
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score

# for CNN
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
# from tensorflow import constant
# from tensorflow.keras import layers


class ClassifyOperations:
    """
    methods made for purpose of classification functions
    """
    @staticmethod
    def distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    @staticmethod
    def get_mean_rgb_from_cell(cell):
        # MY WAY
        # print('extracting rgb from cell', type(cell))
        # print(cell.shape)
        red = [rgb[0] for rgb in cell]
        green = [rgb[1] for rgb in cell]
        blue = [rgb[2] for rgb in cell]

        rmean, gmean, bmean = np.mean(red), np.mean(green), np.mean(blue)

        return [rmean, gmean, bmean]

    @staticmethod
    def kmeans_class(img, num_of_clusters=2):
        ''' group all pixels on image into n number of groups'''
        X = img.reshape(-1, 3)
        kmeans = KMeans(n_clusters=num_of_clusters, n_init=10)
        kmeans.fit(X)

        segmented_img = kmeans.cluster_centers_[kmeans.labels_]
        segmented_img = segmented_img.reshape(img.shape)/255

        if np.mean(kmeans.cluster_centers_[0]) > np.mean(kmeans.cluster_centers_[1]):
            cell_centroid = kmeans.cluster_centers_[1]
        else:
            cell_centroid = kmeans.cluster_centers_[0]

        # psróbowac ze zwracaniem centroid
        return segmented_img, cell_centroid

    @staticmethod
    def save_set_of_KNN_coordinates(values, file_path):
        file = open(file_path, 'w')
        values = [list(v) for v in values]
        json.dump(values, file, indent=6)
        file.close()

    @staticmethod
    def load_set_of_KNN_coordinates(file_path):
        file = open(file_path)
        values = json.load(file)
        # with open(file_path) as file:
        #     values = json.load(file)
        return values


## ---------------------------------------------------------------------------------------------------------------------
# bez nauczyciela
# kMeans
def kMeans(num_of_clusters=2, cells=[]):
    '''
    number od iteration does not really matter
    num of clusters is what matters , cluster with highest mean value is blue and the rest is ponetialy black
    '''
    # start
    black, blue, blackCenter, blueCenter, blueCenter2 = [], [], [], [], []
    # cells_RGB = [ClassifyOperations.get_mean_rgb_from_cell(ClassifyOperations.kmeans_class(cell)[0]) for cell in cells]
    cells_RGB = [ClassifyOperations.kmeans_class(cell)[1] for cell in cells]
    # kMeans
    k_means = KMeans(n_clusters=num_of_clusters, random_state=0)
    model = k_means.fit(cells_RGB)
    centroids = k_means.cluster_centers_

    # classify
    means = [np.mean(center) for center in centroids]
    blueCenter = centroids[means.index(max(means))]

    for cell_id in range(len(cells_RGB)):
        distances = [ClassifyOperations.distance(center, cells_RGB[cell_id]) for center in centroids]
        nearest = centroids[distances.index(min(distances))]
        if (nearest == blueCenter).all():
            blue.append(cells[cell_id])
        else:
            black.append(cells[cell_id])

    return black, blue, centroids


def simple_color_classyfication(cells):
    black, blue = [], []
    cells_RGB = [ClassifyOperations.get_mean_rgb_from_cell(ClassifyOperations.kmeans_class(cell)[0]) for cell in cells]
    for cell_id in range(len(cells)):
        if cells_RGB[cell_id][2] > 165/255:
            blue.append(cells[cell_id])
        else:
            black.append(cells[cell_id])

    return black, blue


# z nauczycielem
def KNN(cells, blackCellsPath='', blueCellsPath='', k=3, save_reference_coordinates_path_black='',
        save_reference_coordinates_path_blue='',
        load_reference_coordinates_path_black='./KNN_black_reference_coordicates.json',
        load_reference_coordinates_path_blue='./KNN_blue_reference_coordicates.json',
        working_state='load data'):

    # preparing cells for classification
    y, X = [], []
    cells_RGB = [ClassifyOperations.kmeans_class(cell)[1] for cell in cells]

    # opening images and creating data based on saved reference cells
    if working_state == 'create data':
        list_of_blue_cells = [blueCellsPath + img for img in listdir(f'{blueCellsPath}') if ".DS" not in img]
        list_of_black_cells = [blackCellsPath + img for img in listdir(f'{blackCellsPath}') if ".DS" not in img]

        black_cells, blue_cells, X, y = [], [], [], []
        for cell_id in range(len(list_of_black_cells)):
            # black_cells.append(imread(list_of_black_cells[cell_id]))
            black_cells.append(imread(list_of_black_cells[cell_id]))
            y.append(0)
        for cell_id in range(len(list_of_blue_cells)):
            blue_cells.append(imread(list_of_blue_cells[cell_id]))
            y.append(1)

        black_RGB = [ClassifyOperations.kmeans_class(cell)[1] for cell in black_cells]
        blue_RGB = [ClassifyOperations.kmeans_class(cell)[1] for cell in blue_cells]

    # if data was not created , load data from saved json files
    if working_state == 'load data':
        black_RGB = ClassifyOperations.load_set_of_KNN_coordinates(load_reference_coordinates_path_black)
        blue_RGB = ClassifyOperations.load_set_of_KNN_coordinates(load_reference_coordinates_path_blue)
        for _ in black_RGB:
            y.append(0)
        for _ in blue_RGB:
            y.append(1)
    print(f'{len(black_RGB)} black data cells were loaded '
          f' {len(blue_RGB)} blue data cells were loaded')

    # save reference coordinates based on reference cells if save path was specified
    if save_reference_coordinates_path_black != '' and save_reference_coordinates_path_blue != '' \
        and working_state != 'load data':
        ClassifyOperations.save_set_of_KNN_coordinates(black_RGB, save_reference_coordinates_path_black)
        ClassifyOperations.save_set_of_KNN_coordinates(blue_RGB, save_reference_coordinates_path_blue)
        print(f'{len(black_RGB)} black data cells were saved into {save_reference_coordinates_path_black} '
          f' {len(blue_RGB)} blue data cells were saved into {save_reference_coordinates_path_blue}')
    X = black_RGB + blue_RGB

    # test
    # test_blue_path = './Reference/blue_test/'
    # test_black_path = './Reference/black_test/'
    # list_of_blue_cells_test = [test_blue_path + img for img in listdir(f'{test_blue_path}') if ".DS" not in img]
    # list_of_black_cells_test = [test_black_path + img for img in listdir(f'{test_black_path}') if ".DS" not in img]
    #
    # black_test, blue_test, X_test, y_test = [], [], [], []
    # for cell_id in range(len(list_of_black_cells_test)):
    #     black_test.append(imread(list_of_black_cells_test[cell_id]))
    #     y_test.append(0)
    # for cell_id in range(len(list_of_blue_cells_test)):
    #     blue_test.append(imread(list_of_blue_cells_test[cell_id]))
    #     y_test.append(1)
    #
    # black_RGB_test = [ClassifyOperations.get_mean_rgb_from_cell(cell) for cell in black_test]
    # blue_RGB_test = [ClassifyOperations.get_mean_rgb_from_cell(cell) for cell in blue_test]
    # X_test = black_RGB_test + blue_RGB_test

    # KNN
    black, blue = [], []
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)

    for cell_id in range(len(cells)):
        if knn.predict([cells_RGB[cell_id]]) == 0:
            black.append(cells[cell_id])
        else:
            blue.append(cells[cell_id])

    return black, blue


def classification_using_svc(cells, blackCellsPath, blueCellsPath, imageResize=15):
    list_of_blue_cells = [blueCellsPath + img for img in listdir(f'{blueCellsPath}') if ".DS" not in img]
    list_of_black_cells = [blackCellsPath + img for img in listdir(f'{blackCellsPath}') if ".DS" not in img]

    # black_cells, blue_cells, X, y, cells_after_preparations = [], [], [], [], []
    X, y, cells_after_preparations = [], [], []
    for cell_id in range(len(list_of_black_cells)):
        cell = skresize(imread(list_of_black_cells[cell_id]), (imageResize, imageResize))
        # black_cells.append(cell.flatten())
        X.append(cell.flatten())
        y.append(0)
    for cell_id in range(len(list_of_blue_cells)):
        cell = skresize(imread(list_of_blue_cells[cell_id]), (imageResize, imageResize))
        # blue_cells.append(cell.flatten())
        X.append(cell.flatten())
        y.append(1)

    # prepare input data
    for cell_id in range(len(cells)):
        cell = skresize(cells[cell_id], (imageResize, imageResize))
        cells_after_preparations.append(cell.flatten())

    # split data for test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    # classification
    classifier = SVC()
    parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
    grid_search = GridSearchCV(classifier, parameters)

    grid_search.fit(X_train, y_train)

    # test performance
    best_extimator = grid_search.best_estimator_
    y_prediction = best_extimator.predict(X_test)
    score = accuracy_score(y_prediction, y_test)
    print(f"{score*100} % of samples were corretly classified")

    # classify cells
    result = best_extimator.predict(cells_after_preparations)
    black, blue = [], []
    for idx in range(len(cells)):
        if result[idx] == 0:
            black.append(cells[idx])
        if result[idx] == 1:
            blue.append(cells[idx])

    return black, blue


def cnn_classifier(cells, blackCellsPath, blueCellsPath, imageResize=15, model_path='./image_classification.model',
                   working_state='load model', save_model=False, save_path=''):

    # prepare input data
    black, blue, cells_after_preparations = [], [], []
    for cell_id in range(len(cells)):
        cell = skresize(cells[cell_id], (imageResize, imageResize))
        cells_after_preparations.append(cell)

    if working_state == 'create model':
        list_of_blue_cells = [blueCellsPath + img for img in listdir(f'{blueCellsPath}') if "cell" in img]
        list_of_black_cells = [blackCellsPath + img for img in listdir(f'{blackCellsPath}') if "cell" in img]

        X, y = [], []
        for cell_id in range(len(list_of_black_cells)):
            cell = skresize(imread(list_of_black_cells[cell_id]), (imageResize, imageResize))
            X.append(cell)
            y.append(0)
        for cell_id in range(len(list_of_blue_cells)):
            cell = skresize(imread(list_of_blue_cells[cell_id]), (imageResize, imageResize))
            X.append(cell)
            y.append(1)
        # split data for test and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        X_train , X_test, y_train, y_test = constant(X_train), constant(X_test), constant(y_train), constant(y_test)

    ###  CNN model
    ## load model
    if working_state == 'load model':
        model = load_model(model_path)

    ## create and train model
    if working_state == 'create model':
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(imageResize, imageResize, 3)))
        model.add(MaxPooling2D((3, 3)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f" CNN model loss : {loss}, and accuracy : {accuracy}")

    # use loaded or created model
    pred = []
    try:
        pred = model.predict(constant(cells_after_preparations))
    except:
        print("model was not loaded or created")

    for cell_id in range(len(cells)):
        if pred[cell_id][0] > pred[cell_id][1]:
            black.append(cells[cell_id])
        else:
            blue.append(cells[cell_id])

    if save_model and save_path != '':
        model.save('/Users/bartoszkawa/Desktop/REPOS/GitLab/inzynierka/image_classification.model')
    
    return black, blue




# ### TEST
# black_path = "./Reference/black/"
# blue_path = "./Reference/blue/"
# plot_photo(imread('./Reference/black/cell59#new9.jpg'))
# black, blue = cnn_classifier([imread('./Reference/black/cell59#new9.jpg')]
#        , black_path, blue_path, imageResize=15, working_state='create model')
# #
#
# black, blue = cnn_classifier([imread('./Reference/blue/cell23#new10.jpg'),imread('./Reference/black/cell787.jpg')]
#      , black_path, blue_path)

# cell, cent = kmeans_class(imread('./Reference/blue/cell56.jpg'))
# print(f'cent {cent}')
# plot_photo(cell)

# black_path = "./Reference/black/"
# blue_path = "./Reference/blue/"
# blackKNN, blueKNN = KNN([], black_path, blue_path,
#                             save_reference_coordinates_path_black='./KNN_black_reference_coordicates.json',
#                             save_reference_coordinates_path_blue='./KNN_blue_reference_coordicates.json',
#                             working_state='create data')
#
# black = load_set_of_KNN_coordinates('./KNN_black_reference_coordicates.json')
#
# print('')



#
# {
#     "version": 2,
#     "builds": [
#         {
#             "src": "WebApp/wsgi.py",
#             "use": "@vercel/python",
#             "config": {"maxLambdaSize":  "15mb", "runtime":  "python3.9"}
#         },
#         {
#             "src": "build_files.sh",
#             "use": "@vercel/static_build",
#             "config": {
#                 "distDir":  "staticfiles_build"
#             }
#         }
#     ],
#     "routes": [
#       {
#           "src": "/(.*)",
#           "dest": "WebApp/wsgi.py"
#       },
#       {
#           "src": "/(.*)",
#           "dest": "WebApp/wsgi.py"
#       }
#     ]
# }




