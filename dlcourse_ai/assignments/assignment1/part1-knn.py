#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
    currentWorkingDirectory = os.getcwd()
    print("Current directory: \"{}\"".format(currentWorkingDirectory))

    workingDirectory = os.path.join(
        currentWorkingDirectory,
        "assignments\\assignment1"
    )

    print("Trying to change to {}".format(workingDirectory))
    os.chdir(workingDirectory)
    print("Change successful")
except:
    print("Directory change failed")

#%% [markdown]
# # Задание 1.1 - Метод К-ближайших соседей (K-neariest neighbor classifier)
#
# В первом задании вы реализуете один из простейших алгоритмов машинного обучения - классификатор на основе метода K-ближайших соседей.
# Мы применим его к задачам
# - бинарной классификации (то есть, только двум классам)
# - многоклассовой классификации (то есть, нескольким классам)
#
# Так как методу необходим гиперпараметр (hyperparameter) - количество соседей, мы выберем его на основе кросс-валидации (cross-validation).
#
# Наша основная задача - научиться пользоваться numpy и представлять вычисления в векторном виде, а также ознакомиться с основными метриками, важными для задачи классификации.
#
# Перед выполнением задания:
# - запустите файл `download_data.sh`, чтобы скачать данные, которые мы будем использовать для тренировки
# - установите все необходимые библиотеки, запустив `pip install -r requirements.txt` (если раньше не работали с `pip`, [вам сюда](https://pip.pypa.io/en/stable/quickstart/))
#
# Если вы раньше не работали с numpy, вам может помочь tutorial. [Например этот](http://cs231n.github.io/python-numpy-tutorial/).

#%%
import random
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt

#%%
sys.path.insert(0, "..")
from dataset import load_svhn
from metrics import binary_classification_metrics, multiclass_accuracy

#%% [markdown]
# # Загрузим и визуализируем данные
# 
# В задании уже дана функция `load_svhn`, загружающая данные с диска. Она возвращает данные для тренировки и для тестирования как numpy arrays.
# 
# Мы будем использовать цифры из датасета Street View House Numbers ([SVHN](http://ufldl.stanford.edu/housenumbers/)), чтобы решать задачу хоть сколько-нибудь сложнее MNIST.

#%%
train_X, train_y, test_X, test_y = load_svhn(
    "..\\data",
    max_train=1000,
    max_test=100)

#%%
samples_per_class = 5  # Number of samples per class to visualize
plot_index = 1
for example_index in range(samples_per_class):
    for class_index in range(10):
        plt.subplot(5, 10, plot_index)
        image = train_X[train_y == class_index][example_index]
        plt.imshow(image.astype(np.uint8))
        plt.axis('off')
        plot_index += 1

#%% [markdown]
# # Сначала реализуем KNN для бинарной классификации
# 
# В качестве задачи бинарной классификации мы натренируем модель, которая будет отличать цифру 0 от цифры 9.
# ## Сначала подготовим данные
# Выберем только классы 0 и 9
#%%
def data_by_classes_binary(data_x, data_y, classes):
    selector_mask = False
    for cl in classes:
        selector_mask = selector_mask | (data_y == cl)
    return data_x[selector_mask], (data_y[selector_mask] == classes[0])

# First, let's prepare the labels and the source data
# Only select 0s and 9s
classes = [0,9]
binary_train_X, binary_train_y = data_by_classes_binary(
    train_X,
    train_y,
    classes)

binary_test_X, binary_test_y = data_by_classes_binary(
    test_X,
    test_y,
    classes)

# Reshape to 1-dimensional array [num_samples, 32*32*3]
binary_train_X = binary_train_X.reshape(binary_train_X.shape[0], -1)
binary_test_X = binary_test_X.reshape(binary_test_X.shape[0], -1)

#%% [markdown]
# ## Создадим классификатор и обучим модель на тренировочных данных
# Классификатор KNN просто запоминает все данные

#%%
from knn import KNN
knn_classifier = KNN(k=1)
knn_classifier.fit(binary_train_X, binary_train_y)

#%% [markdown]
# ## Пришло время написать код! 
# 
# Последовательно реализуйте функции `compute_distances_two_loops`, `compute_distances_one_loop` и `compute_distances_no_loops`
# в файле `knn.py`.
# 
# Эти функции строят массив расстояний между всеми векторами в тестовом наборе и в тренировочном наборе.  
# В результате они должны построить массив размера `(num_test, num_train)`, где координата `[i][j]` соотвествует расстоянию между i-м вектором в test (`test[i]`) и j-м вектором в train (`train[j]`).
# 
# **Обратите внимание** Для простоты реализации мы будем использовать в качестве расстояния меру L1 (ее еще называют [Manhattan distance](https://ru.wikipedia.org/wiki/%D0%A0%D0%B0%D1%81%D1%81%D1%82%D0%BE%D1%8F%D0%BD%D0%B8%D0%B5_%D0%B3%D0%BE%D1%80%D0%BE%D0%B4%D1%81%D0%BA%D0%B8%D1%85_%D0%BA%D0%B2%D0%B0%D1%80%D1%82%D0%B0%D0%BB%D0%BE%D0%B2)).
# 
# ![image.png](https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Manhattan_distance.svg/800px-Manhattan_distance.svg.png)

#%%
dists = knn_classifier.compute_distances_two_loops(binary_test_X)
assert np.isclose(dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))

#%%
dists = knn_classifier.compute_distances_one_loop(binary_test_X)
assert np.isclose(dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))

#%%
dists = knn_classifier.compute_distances_no_loops(binary_test_X)
assert np.isclose(dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))

#%%
# Lets look at the performance difference
get_ipython().run_line_magic('timeit', 'knn_classifier.compute_distances_two_loops(binary_test_X)')
get_ipython().run_line_magic('timeit', 'knn_classifier.compute_distances_one_loop(binary_test_X)')
get_ipython().run_line_magic('timeit', 'knn_classifier.compute_distances_no_loops(binary_test_X)')

#%%
prediction = knn_classifier.predict(binary_test_X)
prediction

#%%
def print_samples(samples):
    for i in range(0, samples.shape[0]):
        image = np.reshape(samples[i], (32,32,3))
        plt.imshow(image.astype(np.uint8))
        plt.axis("off")
        plt.show()

#%%
precision, recall, f1, accuracy = binary_classification_metrics(prediction, binary_test_y)
print("KNN with k = %s" % knn_classifier.k)
print("Accuracy: %4.2f, Precision: %4.2f, Recall: %4.2f, F1: %4.2f" % (accuracy, precision, recall, f1)) 

#%%
# Let's put everything together and run KNN with k=3 and see how we do
knn_classifier_3 = KNN(k=3)
knn_classifier_3.fit(binary_train_X, binary_train_y)
prediction = knn_classifier_3.predict(binary_test_X)

precision, recall, f1, accuracy = binary_classification_metrics(prediction, binary_test_y)
print("KNN with k = %s" % knn_classifier_3.k)
print("Accuracy: %4.2f, Precision: %4.2f, Recall: %4.2f, F1: %4.2f" % (accuracy, precision, recall, f1)) 

#%% [markdown]
# # Кросс-валидация (cross-validation)
# 
# Попробуем найти лучшее значение k! 
# 
# Для этого мы воспользуемся [k-fold cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation). Мы разделим тренировочные данные на 5 фолдов (folds), и по очереди будем использовать каждый из них в качестве проверочных данных (validation data), а остальные -- в качестве тренировочных (training data).
# 
# В качестве финальной оценки эффективности k мы усредним значения F1 score на всех фолдах.
# После этого мы просто выберем значение k с лучшим значением метрики.
# 
# *Бонус*: есть ли другие варианты агрегировать F1 score по всем фолдам? Напишите плюсы и минусы в клетке ниже.

#%%
# Find the best k using cross-validation based on F1 score
num_folds = 5
train_folds_X = []
train_folds_y = []

#%%
data_to_fold = binary_train_X
answers_to_fold = binary_train_y
fold_size = binary_train_y.shape[0]//num_folds

indices = random.sample(range(0, answers_to_fold.shape[0]), fold_size)
for fold_number in range(num_folds-1):
    indices_range = range(0, answers_to_fold.shape[0])
    indices = random.sample(indices_range, fold_size)
    left_indices = set(indices_range) - set(indices)
    fold_x = data_to_fold[indices]
    fold_y = answers_to_fold[indices]
    data_to_fold = data_to_fold[list(left_indices)]
    answers_to_fold = answers_to_fold[list(left_indices)]
    train_folds_X.append(fold_x)
    train_folds_y.append(fold_y)
train_folds_X.append(data_to_fold)
train_folds_y.append(answers_to_fold)

[len(fold) for fold in train_folds_X], [len(fold) for fold in train_folds_y]

#%%
k_choices = [1, 2, 3, 5, 8, 10, 15, 20, 25, 50]
k_to_f1 = {}  # dict mapping k values to mean F1 scores (int -> float)

#%%
for k in k_choices:
    for fold_idx in range(num_folds):
        test_fold = train_folds_X[fold_idx]
        test_answers = train_folds_y[fold_idx]
        rest_X = reduce(
            lambda acc, v: np.concatenate((acc, v)),
            [fold for idx, fold in enumerate(train_folds_X) if idx != fold_idx])
        rest_y = reduce(
            lambda acc, v: np.concatenate((acc, v)),
            [fold for idx, fold in enumerate(train_folds_y) if idx != fold_idx])
        knn_classifier = KNN(k=k)
        knn_classifier.fit(rest_X, rest_y)
        cross_prediction = knn_classifier.predict(test_fold)
        precision, recall, f1, accuracy = binary_classification_metrics(
            cross_prediction,
            test_answers)
        k_to_f1[k] = f1
        

for k in sorted(k_to_f1):
    print('k = %d, f1 = %f' % (k, k_to_f1[k]))

#%% [markdown]
# ### Проверим, как хорошо работает лучшее значение k на тестовых данных (test data)

#%%
best_k = 15

best_knn_classifier = KNN(k=best_k)
best_knn_classifier.fit(binary_train_X, binary_train_y)
prediction = best_knn_classifier.predict(binary_test_X)

precision, recall, f1, accuracy = binary_classification_metrics(prediction, binary_test_y)
print("Best KNN with k = %s" % best_k)
print("Accuracy: %4.2f, Precision: %4.2f, Recall: %4.2f, F1: %4.2f" % (accuracy, precision, recall, f1)) 

#%% [markdown]
# # Многоклассовая классификация (multi-class classification)
# 
# Переходим к следующему этапу - классификации на каждую цифру.

#%%
# Now let's use all 10 classes
train_X = train_X.reshape(train_X.shape[0], -1)
test_X = test_X.reshape(test_X.shape[0], -1)

knn_classifier = KNN(k=1)
knn_classifier.fit(train_X, train_y)

#%%
dists = knn_classifier.compute_distances_no_loops(test_X)
num_test = dists.shape[0]
k_closest_indices = np.argpartition(dists, knn_classifier.k, axis=1)[:,:knn_classifier.k]
k_closest_indices.shape

#%%
predict = knn_classifier.predict(test_X)

#%%
accuracy = multiclass_accuracy(predict, test_y)
print("Accuracy: %4.2f" % accuracy)

#%% [markdown]
# Снова кросс-валидация. Теперь нашей основной метрикой стала точность (accuracy), и ее мы тоже будем усреднять по всем фолдам.

#%%
# Find the best k using cross-validation based on accuracy
num_folds = 5
train_folds_X = []
train_folds_y = []

#%%
data_to_fold = train_X
answers_to_fold = train_y
fold_size = train_y.shape[0]//num_folds

for fold_number in range(num_folds-1):
    indices_range = range(0, answers_to_fold.shape[0])
    indices = random.sample(indices_range, fold_size)
    left_indices = set(indices_range) - set(indices)
    fold_x = data_to_fold[indices]
    fold_y = answers_to_fold[indices]
    data_to_fold = data_to_fold[list(left_indices)]
    answers_to_fold = answers_to_fold[list(left_indices)]
    train_folds_X.append(fold_x)
    train_folds_y.append(fold_y)
train_folds_X.append(data_to_fold)
train_folds_y.append(answers_to_fold)

[len(fold) for fold in train_folds_X], [len(fold) for fold in train_folds_y]

#%%
k_choices = [1, 2, 3, 5, 8, 10, 15, 20, 25, 50]
k_to_accuracy = {}

for k in k_choices:
    for fold_idx in range(num_folds):
        test_fold = train_folds_X[fold_idx]
        test_answers = train_folds_y[fold_idx]
        rest_X = reduce(
            lambda acc, v: np.concatenate((acc, v)),
            [fold for idx, fold in enumerate(train_folds_X) if idx != fold_idx])
        rest_y = reduce(
            lambda acc, v: np.concatenate((acc, v)),
            [fold for idx, fold in enumerate(train_folds_y) if idx != fold_idx])
        knn_classifier = KNN(k=k)
        knn_classifier.fit(rest_X, rest_y)
        cross_prediction = knn_classifier.predict(test_fold)
        k_to_accuracy[k] = multiclass_accuracy(
            cross_prediction,
            test_answers)

for k in sorted(k_to_accuracy):
    print('k = %d, accuracy = %f' % (k, k_to_accuracy[k]))

#%% [markdown]
# ### Финальный тест - классификация на 10 классов на тестовой выборке (test data)
# 
# Если все реализовано правильно, вы должны увидеть точность не менее **0.2**.

#%%
best_k = 50

best_knn_classifier = KNN(k=best_k)
best_knn_classifier.fit(train_X, train_y)
prediction = best_knn_classifier.predict(test_X)

# Accuracy should be around 20%!
accuracy = multiclass_accuracy(prediction, test_y)
print("Accuracy: %4.2f" % accuracy)
