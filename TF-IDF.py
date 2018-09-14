import math
import time

start = time.time()

article = []
with open('semeval.txt', 'r') as file:
    for line in file.readlines():
        data = line.split('\t')[2].strip('\n')
        article.append(data)

TF = []
IDF = {}
for line in article:
    words = line.split(' ')
    tmp = {}
    for word in words:
        if word not in tmp.keys():
            tmp[word] = 1 / len(words)
        else:
            tmp[word] += 1 / len(words)
    TF.append(tmp)

    tmp = []
    for element in words:
        if element not in tmp:
            tmp.append(element)

    for word in tmp:
        if word not in IDF.keys():
            IDF[word] = 1
        else:
            IDF[word] += 1

for key in IDF.keys():
    IDF[key] = math.log(len(article) / (IDF[key] + 1))

TF_IDF = []
for i in range(len(TF)):
    tmp = {}
    for key in TF[i].keys():
        tmp[key] = TF[i][key] * IDF[key]
    TF_IDF.append(tmp)

totalwords = []
for line in article:
    line = line.split(' ')
    for word in line:
        if word not in totalwords:
            totalwords.append(word)

with open('16337341_zhuzhiru_TFIDF.txt', 'w') as file:
    for i in range(len(TF_IDF)):
        for element in totalwords:
            if element in TF_IDF[i].keys():
                file.write(str(round(TF_IDF[i][element], 9)) + ' ')
            else:
                file.write('0.0' + ' ')
        file.write('\n')

end = time.time()
print('用时共计：' + str(end - start) + 's')
