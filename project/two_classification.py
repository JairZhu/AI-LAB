import string
import numpy as np
import math
from collections import Counter
import time


def filter_data(filename):
    all_data = []
    common_words = ['have', 'had', 'has', 'are', 'was', 'were', 'the', 'this', 'that', 'and', 'etc', 'they', 'them',
                    'their', 'theirs', 'our', 'ours', 'you', 'your', 'yours', 'its', 'she', 'her', 'hers', 'him', 'his']
    ctlword = '\x97\x91\x96\x08\x83\x8e\x9e\x84'
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().replace('<br />', ' ').replace("n't", " not")
            table = str.maketrans(string.punctuation + string.digits + ctlword,
                                  ' ' * (len(string.punctuation) + len(string.digits) + len(ctlword)))
            line = line.translate(table).split()
            row = []
            for item in line:
                if len(item) > 2 and item.lower() not in common_words:
                    row.append(item.lower())
            all_data.append(row)
    return all_data


def write_data(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for i in range(len(data)):
            for j in range(len(data[i])):
                file.write(data[i][j] + ' ')
            file.write('\n')


def read_data(filename):
    output = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            output.append(line.strip().split())
    return output


def read_label(filename):
    output = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            output.append(int(line.strip()))
    return output


def write_result(filename, data):
    with open(filename, 'w', encoding='utf-8') as file:
        for i in range(len(data)):
            file.write(str(data[i]) + '\n')


class Naive_Bayes:
    def __init__(self, train_set, train_labels):
        self.train_set = train_set
        self.trian_labels = train_labels
        self.pos_words_vector = {}
        self.neg_word_vector = {}
        self.words_vector = []
        for word in train_set:
            self.words_vector += word
        self.words_vector = list(set(self.words_vector))
        for word in self.words_vector:
            self.pos_words_vector[word] = 0
            self.neg_word_vector[word] = 0
        self.pos_prob = 0
        for label in self.trian_labels:
            if label == 1:
                self.pos_prob += 1
        self.neg_prob = (len(self.trian_labels) - self.pos_prob) / len(self.trian_labels)
        self.pos_prob = self.pos_prob / len(self.trian_labels)

    def classification(self, test_set):
        pos_words_num = 0
        neg_words_num = 0
        words_num = len(self.words_vector)
        for i in range(len(self.train_set)):
            print('计算第', i, '行')
            if self.trian_labels[i] == 1:
                pos_words_num += len(self.train_set[i])
                for word in self.train_set[i]:
                    self.pos_words_vector[word] += 1
            else:
                neg_words_num += len(self.train_set[i])
                for word in self.train_set[i]:
                    self.neg_word_vector[word] += 1
        for key in self.words_vector:
            self.pos_words_vector[key] = (self.pos_words_vector[key]) / (pos_words_num + words_num)
            self.neg_word_vector[key] = (self.neg_word_vector[key]) / (neg_words_num + words_num)
        labels = []
        for line in test_set:
            is_pos = math.log(self.pos_prob)
            is_neg = math.log(self.neg_prob)
            for word in line:
                if word in self.words_vector:
                    is_pos += math.log(self.pos_words_vector[word])
                    is_neg += math.log(self.neg_word_vector[word])
            if is_pos > is_neg:
                labels.append(1)
            else:
                labels.append(0)
            print('已预测', len(labels), '行')
        return labels

    def validation(self, validation_set, validation_labels):
        results = self.classification(validation_set)
        count = 0
        for i in range(len(results)):
            if results[i] == validation_labels[i]:
                count += 1
        print('验证集准确率:', count / len(results))


class KNN:
    def __init__(self, train_set, train_set_label):
        self.train_row_num = len(train_set)
        self.train_label = train_set_label
        self.words_vector = []
        for row in train_set:
            self.words_vector += row
        self.words_vector = list(set(self.words_vector))
        self.xmatrix = np.zeros((self.train_row_num, len(self.words_vector)))
        for i in range(len(train_set)):
            print('第', i, '行')
            for word in train_set[i]:
                self.xmatrix[i][self.words_vector.index(word)] += 1

    def classification_KNN(self, test_set, k):
        labels = []
        for line in test_set:
            test_vector = np.zeros(len(self.words_vector))
            for word in line:
                if word in self.words_vector:
                    test_vector[self.words_vector.index(word)] += 1
            test_matrix = np.abs((np.tile(test_vector, (self.train_row_num, 1)) - self.xmatrix))
            distance = np.sum(test_matrix, axis=1).T.tolist()
            maps = {}
            for i in range(len(distance)):
                if distance[i] not in maps.keys():
                    maps[distance[i]] = [self.train_label[i]]
                else:
                    maps[distance[i]].append(self.train_label[i])
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
            print('已生成', len(labels), '个')
        return labels

    def validation(self, validation_set, validation_labels, k):
        out = self.classification_KNN(validation_set, k)
        count = 0
        for i in range(len(validation_set)):
            if out[i] == validation_labels[i]:
                count += 1
        print('验证集准确率:', count / self.train_row_num)


if __name__ == '__main__':
    start_time = time.time()

    train_set = read_data('out.txt')
    train_labels = read_label('trainLabel.txt')
    test_set = read_data('in.txt')
    NB = Naive_Bayes(train_set[:18000], train_labels[:18000])
    NB.validation(train_set[18000:], train_labels[18000:])
    # labels = NB.classification(test_set)
    # write_result('16337327_1.txt', labels)

    end_time = time.time()
    print('运行总用时:', end_time - start_time)
