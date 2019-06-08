from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler


def main():
    # number of neighbours
    N = 5
    # cross-validation folds
    F = 10

    # reding data from arff
    data = arff.loadarff('choose_your_reptile.arff')
    df = pd.DataFrame(data[0])
    classes = df['class']

    df = df.drop(['class'], axis=1)
    X = df.values.tolist()

    # normalization/scaling
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    Y = []

    # representing classes as integers
    for x in classes:
        if x == b'Argentine Tegu':
            Y.append(0)
        elif x == b'Cornsnake':
            Y.append(1)
        elif x == b'Leopard Gecko':
            Y.append(2)
        elif x == b'Yemen Chameleon':
            Y.append(3)
        elif x == b'Ball Python':
            Y.append(4)
        elif x == b'Blue Tongue Skink':
            Y.append(5)
        elif x == b'Musk Turtle':
            Y.append(6)
        elif x == b'Green Iquana':
            Y.append(7)
        elif x == b'Common Boa':
            Y.append(8)
        else:
            Y.append(9)

    accuracy = []

    for f in range(F):
        # num of records in 1 fold
        records = len(X)/F

        # split into train and test sets
        test_set_begining = int(f*records)
        test_set_ending = int((f+1)*records)

        normalized_x_test = X[test_set_begining:test_set_ending]
        y_test = Y[test_set_begining:test_set_ending]

        normalized_x_train = list(X[:test_set_begining])
        normalized_x_train.extend(list(X[test_set_ending:]))
        y_train = list(Y[:test_set_begining])
        y_train.extend(list(Y[test_set_ending:]))

        x_train = np.array(normalized_x_train)
        y_train = np.array(y_train)
        x_test = np.array(normalized_x_test)
        y_test = np.array(y_test)

        # print(x_train)
        # print(y_train)
        # print(x_test)
        # print(y_test)

        y_predicted = []

        # for every x in test set looking for closest neighbours from training set
        for x in x_test:
            distances = []
            # counting euclidean distance to every neighbour
            for neighbour_x in x_train:
                dist = distance.euclidean(x, neighbour_x)
                distances.append(dist)

            closest_neighbours = y_train.tolist()

            # bubble sort
            for i in range(len(distances)):
                for j in range(0, len(distances) - i - 1):
                    if distances[j + 1] < distances[j]:
                        distances[j], distances[j + 1] = distances[j + 1], distances[j]

                        closest_neighbours[j], closest_neighbours[j + 1] = closest_neighbours[j + 1], closest_neighbours[j]

            # finding which class is appearing the most often among N nearest neighbours
            neighbours_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for i in range(N):
                neighbours_count[closest_neighbours[i]] += 1
            y_predicted.append(neighbours_count.index(max(neighbours_count)))

        accuracy.append(accuracy_score(y_test, y_predicted))
        print('\nAccuracy :')
        print(accuracy[f])
        print('Actual :')
        print(y_test.tolist())
        print('Predicted :')
        print(y_predicted)

    avg_accuracy = np.mean(accuracy)
    print('\nAverage accuracy :')
    print(avg_accuracy)


if __name__ == '__main__':
    main()
