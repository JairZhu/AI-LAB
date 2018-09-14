import csv
import numpy as np
import math
from collections import Counter
import time

start = time.time()


def classification_KNN(test_set, k):
    labels = []
    for line in test_set:
        words = line[0].split(' ')
        test_vector = np.zeros(len(words_vector))
        for word in words:
            if word in words_vector:
                test_vector[words_vector.index(word)] += 1
        test_matrix = np.abs(np.matrix(np.tile(test_vector, (len(train_set), 1))) - xmatrix)
        distance = np.sum(test_matrix, axis=1).T.tolist()[0]
        maps = {}
        for i in range(len(distance)):
            if distance[i] not in maps.keys():
                maps[distance[i]] = [train_set[i][1]]
            else:
                maps[distance[i]].append(train_set[i][1])
        distance.sort()
        count = 0
        mood = []
        for index in range(len(distance)):
            i = distance[index]
            if len(maps[i]) + count >= k:
                mood += maps[i]
                break
            else:
                mood += maps[i]
                count += len(maps[i])
        top = Counter(mood).most_common(1)[0][0]
        labels.append(top)
    return labels


def read_csv(filename):
    output = []
    with open(filename, 'r') as file:
        all_data = csv.reader(file)
        for row in all_data:
            output.append(row)
    output = output[1:]
    return output


train_set = read_csv('lab1_data\\classification_dataset\\train_set.csv')

for i in range(len(train_set)):
    train_set[i][0] = train_set[i][0].split(' ')

words_vector = []
for line in train_set:
    words_vector += line[0]
words_vector = list(set(words_vector))

xmatrix = np.zeros((len(train_set), len(words_vector)))

for i in range(len(train_set)):
    for word in train_set[i][0]:
        xmatrix[i][words_vector.index(word)] += 1

xmatrix = np.matrix(xmatrix)

validation_set = read_csv('lab1_data\\classification_dataset\\validation_set.csv')

out = classification_KNN(validation_set, 2)
count = 0
for i in range(len(validation_set)):
    if out[i] != validation_set[i][1]:
        count += 1

for element in out:
    print(element)
print(count / len(validation_set))
end = time.time()
print('KNN运行用时:', end - start)
