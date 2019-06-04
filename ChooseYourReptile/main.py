from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler


def main():
    #number of neighbours
    N = 3

    #reding data from arff
    data = arff.loadarff('choose_your_reptile.arff')
    df = pd.DataFrame(data[0])
    target = df['class']

    df = df.drop(['class'], axis=1)

    X = df.values.tolist()
    # X = preprocessing.normalize(X)

    #normalizacja/skalowanie danych
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    Y = []

    #reprezentacja klas jako integerow zeby bylo latwiej
    for x in target:
        if (x == b'Argentine Tegu'):
            Y.append(0)
        elif (x == b'Cornsnake'):
            Y.append(1)
        elif (x == b'Leopard Gecko'):
            Y.append(2)
        elif (x == b'Yemen Chameleon'):
            Y.append(3)
        elif (x == b'Ball Python'):
            Y.append(4)
        elif (x == b'Blue Tongue Skink'):
            Y.append(5)
        elif (x == b'Musk Turtle'):
            Y.append(6)
        elif (x == b'Green Iquana'):
            Y.append(7)
        elif (x == b'Common Boa'):
            Y.append(8)
        else:
            Y.append(9)

    #podział na zbiory testowe i treningowe
    normalized_X_test = X[-10:]
    Y_test = Y[-10:]
    normalized_X_train = X[0:40]
    Y_train = Y[0:40]

    x_train = np.array(normalized_X_train)
    y_train = np.array(Y_train)
    x_test = np.array(normalized_X_test)
    y_test = np.array(Y_test)

    print(x_train)
    print(y_train)
    print(x_test)
    print(y_test)

    y_pred = []

    #dla kazdego elementu ze zbioru testowego szukamy najblizszych sasiadow ze zbioru treningowego
    for x in x_test:
        dis = []
        #counting euclidean distance to every neighbour
        for neighbour_x in x_train:
            dist = distance.euclidean(x, neighbour_x)
            dis.append(dist)

        closest_neighbours = y_train.tolist()

        #bubble sort
        for i in range(len(dis)):
            for j in range(0, len(dis) - i - 1):
                if (dis[j + 1] < dis[j]):
                    dis[j], dis[j + 1] = dis[j + 1], dis[j]

                    closest_neighbours[j], closest_neighbours[j + 1] = closest_neighbours[j + 1], closest_neighbours[j]

        #counting which class is appearing the most often in N nearest neighbours
        neighbours_num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(N):
            neighbours_num[closest_neighbours[i]] += 1
        y_pred.append(neighbours_num.index(max(neighbours_num)))

    print('Accuracy :')
    print(accuracy_score(y_test, y_pred))
    print('Actual :')
    print(y_test.tolist())
    print('Predicted :')
    print(y_pred)


if __name__ == '__main__':
    main()
