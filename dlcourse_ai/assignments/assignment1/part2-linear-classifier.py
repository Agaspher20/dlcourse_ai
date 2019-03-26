#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
import sys

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
    print("Directory change failed", sys.exc_info()[0])

#%% [markdown]
# # Задание 1.2 - Линейный классификатор (Linear classifier)
# 
# В этом задании мы реализуем другую модель машинного обучения - линейный классификатор. Линейный классификатор подбирает для каждого класса веса, на которые нужно умножить значение каждого признака и потом сложить вместе.
# Тот класс, у которого эта сумма больше, и является предсказанием модели.
# 
# В этом задании вы:
# - потренируетесь считать градиенты различных многомерных функций
# - реализуете подсчет градиентов через линейную модель и функцию потерь softmax
# - реализуете процесс тренировки линейного классификатора
# - подберете параметры тренировки на практике
# 
# На всякий случай, еще раз ссылка на туториал по numpy:  
# http://cs231n.github.io/python-numpy-tutorial/

#%%
import numpy as np
import matplotlib.pyplot as plt
# pylint: disable=undefined-variable
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
# pylint: enable=undefined-variable

#%%
from dataset import load_svhn, random_split_train_val
from metrics import multiclass_accuracy

#%% [markdown]
# # Как всегда, первым делом загружаем данные
# 
# Мы будем использовать все тот же SVHN.

#%%
def prepare_for_linear_classifier(train_X, test_X):
    train_flat = train_X.reshape(train_X.shape[0], -1).astype(np.float) / 255.0
    test_flat = test_X.reshape(test_X.shape[0], -1).astype(np.float) / 255.0
    
    # Subtract mean
    mean_image = np.mean(train_flat, axis = 0)
    train_flat -= mean_image
    test_flat -= mean_image
    
    # Add another channel with ones as a bias term
    train_flat_with_ones = np.hstack([train_flat, np.ones((train_X.shape[0], 1))])
    test_flat_with_ones = np.hstack([test_flat, np.ones((test_X.shape[0], 1))])    
    return train_flat_with_ones, test_flat_with_ones
    
train_X, train_y, test_X, test_y = load_svhn("data", max_train=10000, max_test=1000)    
train_X, test_X = prepare_for_linear_classifier(train_X, test_X)
# Split train into train and val
train_X, train_y, val_X, val_y = random_split_train_val(train_X, train_y, num_val = 1000)

#%% [markdown]
# # Играемся с градиентами!
# 
# В этом курсе мы будем писать много функций, которые вычисляют градиенты аналитическим методом.
# 
# Необходимым инструментом во время реализации кода, вычисляющего градиенты, является функция его проверки. Эта функция вычисляет градиент численным методом и сверяет результат с градиентом, вычисленным аналитическим методом.
# 
# Мы начнем с того, чтобы реализовать вычисление численного градиента (numeric gradient) в этой функции.
# Вычислите градиент с помощью численной производной для каждой координаты. Для вычисления производной используйте так называемую [two-point formula](https://en.wikipedia.org/wiki/Numerical_differentiation):
# 
# ![image](https://wikimedia.org/api/rest_v1/media/math/render/svg/22fc2c0a66c63560a349604f8b6b39221566236d)
# 

#%%
from gradient_check import check_gradient

def sqr(x):
    return x*x, 2*x

check_gradient(sqr, np.array([3.0]))

def array_sum(x):
    assert x.shape == (2,), x.shape
    return np.sum(x), np.ones_like(x)

check_gradient(array_sum, np.array([3.0, 2.0]))

def array_2d_sum(x):
    assert x.shape == (2,2)
    return np.sum(x), np.ones_like(x)

check_gradient(array_2d_sum, np.array([
    [3.0, 2.0],
    [1.0, 0.0]
]))

#%% [markdown]
# Теперь реализуем функцию softmax, которая получает на вход оценки для каждого класса и преобразует их в вероятности от 0 до 1:
# ![image](https://wikimedia.org/api/rest_v1/media/math/render/svg/e348290cf48ddbb6e9a6ef4e39363568b67c09d3)
# 
# **Важно:** Практический аспект вычисления этой функции заключается в том, что в ней учавствует вычисление экспоненты от потенциально очень больших чисел - это может привести к очень большим значениям в числителе и знаменателе за пределами диапазона float.
# 
# К счастью, у этой проблемы есть простое решение -- перед вычислением softmax вычесть из всех оценок максимальное значение среди всех оценок:
# ```
# predictions -= np.max(predictions)
# ```
# ([подробнее здесь](http://cs231n.github.io/linear-classify/#softmax), секция `Practical issues: Numeric stability`)

#%%
from linear_classifer import softmax

probs = softmax(np.array([[-10, 0, 10]]))
assert np.isclose(np.sum(probs), 1.0)

# Make sure it works for big numbers too!
probs = softmax(np.array([[1000, 0, 0]]))

assert np.isclose(probs[0][0], 1.0)

#%% [markdown]
# Кроме этого, мы реализуем cross-entropy loss, которую мы будем использовать как функцию ошибки (error function).
# В общем виде cross-entropy определена следующим образом:
# ![image](https://wikimedia.org/api/rest_v1/media/math/render/svg/0cb6da032ab424eefdca0884cd4113fe578f4293)
# 
# где x - все классы, p(x) - истинная вероятность принадлежности сэмпла классу x, а q(x) - вероятность принадлежности классу x, предсказанная моделью.  
# В нашем случае сэмпл принадлежит только одному классу, индекс которого передается функции. Для него p(x) равна 1, а для остальных классов - 0. 
# 
# Это позволяет реализовать функцию проще!

#%%
from linear_classifer import cross_entropy_loss

probs = softmax(np.array([[-5, 0, 5]],))
cross_entropy_loss(probs, np.array([1]))

#%% [markdown]
# После того как мы реализовали сами функции, мы можем реализовать градиент.
# 
# Оказывается, что вычисление градиента становится гораздо проще, если объединить эти функции в одну, которая сначала вычисляет вероятности через softmax, а потом использует их для вычисления функции ошибки через cross-entropy loss.
# 
# Эта функция `softmax_with_cross_entropy` будет возвращает и значение ошибки, и градиент по входным параметрам. Мы проверим корректность реализации с помощью `check_gradient`.

#%%
from linear_classifer import softmax_with_cross_entropy

loss, grad = softmax_with_cross_entropy(np.array([[1, 0, 0]]), np.array([1]))

print(loss, grad)
check_gradient(
    lambda x: softmax_with_cross_entropy(x, np.array([1])),
    np.array([[1, 0, 0]], np.float))

#%% [markdown]
# В качестве метода тренировки мы будем использовать стохастический градиентный спуск (stochastic gradient descent или SGD), который работает с батчами сэмплов. 
# 
# Поэтому все наши фукнции будут получать не один пример, а батч, то есть входом будет не вектор из `num_classes` оценок, а матрица размерности `batch_size, num_classes`. Индекс примера в батче всегда будет первым измерением.
# 
# Следующий шаг - переписать наши функции так, чтобы они поддерживали батчи.
# 
# Финальное значение функции ошибки должно остаться числом, и оно равно среднему значению ошибки среди всех примеров в батче.

#%%
from linear_classifer import cross_entropy_loss, softmax_with_cross_entropy

# Test batch_size = 1
batch_size = 1
predictions = np.zeros((batch_size, 3))
target_index = np.ones(batch_size, np.int)

loss, grad = softmax_with_cross_entropy(predictions, target_index)
print("First test. Loss: {}, Grad: {}".format(loss, grad))
check_gradient(lambda x: softmax_with_cross_entropy(x, target_index), predictions)

print("Second test")
# Test batch_size = 3
batch_size = 3
predictions = np.zeros((batch_size, 3))
target_index = np.ones(batch_size, np.int)

loss, grad = softmax_with_cross_entropy(predictions, target_index)
print("Second test. Loss: {}, Grad: {}".format(loss, grad))
check_gradient(lambda x: softmax_with_cross_entropy(x, target_index), predictions)

#%% [markdown]
# ### Наконец, реализуем сам линейный классификатор!
# 
# softmax и cross-entropy получают на вход оценки, которые выдает линейный классификатор.
# 
# Он делает это очень просто: для каждого класса есть набор весов, на которые надо умножить пиксели картинки и сложить. Получившееся число и является оценкой класса, идущей на вход softmax.
# 
# Таким образом, линейный классификатор можно представить как умножение вектора с пикселями на матрицу W размера `num_features, num_classes`. Такой подход легко расширяется на случай батча векторов с пикселями X размера `batch_size, num_features`:
# 
# `predictions = X * W`, где `*` - матричное умножение.
# 
# Реализуйте функцию подсчета линейного классификатора и градиентов по весам `linear_softmax` в файле `linear_classifer.py`

#%%
# TODO Implement linear_softmax function that uses softmax with cross-entropy for linear classifier
batch_size = 2
num_classes = 2
num_features = 3
np.random.seed(42)
W = np.random.randint(-1, 3, size=(num_features, num_classes)).astype(np.float)
X = np.random.randint(-1, 3, size=(batch_size, num_features)).astype(np.float)
target_index = np.ones(batch_size, dtype=np.int)

loss, dW = linear_classifer.linear_softmax(X, W, target_index)
check_gradient(lambda w: linear_classifer.linear_softmax(X, w, target_index), W)

#%% [markdown]
# ### И теперь регуляризация
# 
# Мы будем использовать L2 regularization для весов как часть общей функции ошибки.
# 
# Напомним, L2 regularization определяется как
# 
# l2_reg_loss = regularization_strength * sum<sub>ij</sub> W[i, j]<sup>2</sup>
# 
# Реализуйте функцию для его вычисления и вычисления соотвествующих градиентов.

#%%
# TODO Implement l2_regularization function that implements loss for L2 regularization
linear_classifer.l2_regularization(W, 0.01)
check_gradient(lambda w: linear_classifer.l2_regularization(w, 0.01), W)

#%% [markdown]
# # Тренировка!
#%% [markdown]
# Градиенты в порядке, реализуем процесс тренировки!

#%%
# TODO: Implement LinearSoftmaxClassifier.fit function
classifier = linear_classifer.LinearSoftmaxClassifier()
loss_history = classifier.fit(train_X, train_y, epochs=10, learning_rate=1e-3, batch_size=300, reg=1e1)


#%%
# let's look at the loss history!
plt.plot(loss_history)


#%%
# Let's check how it performs on validation set
pred = classifier.predict(val_X)
accuracy = multiclass_accuracy(pred, val_y)
print("Accuracy: ", accuracy)

# Now, let's train more and see if it performs better
classifier.fit(train_X, train_y, epochs=100, learning_rate=1e-3, batch_size=300, reg=1e1)
pred = classifier.predict(val_X)
accuracy = multiclass_accuracy(pred, val_y)
print("Accuracy after training for 100 epochs: ", accuracy)

#%% [markdown]
# ### Как и раньше, используем кросс-валидацию для подбора гиперпараметтов.
# 
# В этот раз, чтобы тренировка занимала разумное время, мы будем использовать только одно разделение на тренировочные (training) и проверочные (validation) данные.
# 
# Теперь нам нужно подобрать не один, а два гиперпараметра! Не ограничивайте себя изначальными значениями в коде.  
# Добейтесь точности более чем **20%** на проверочных данных (validation data).

#%%
num_epochs = 200
batch_size = 300

learning_rates = [1e-3, 1e-4, 1e-5]
reg_strengths = [1e-4, 1e-5, 1e-6]

best_classifier = None
best_val_accuracy = None

# TODO use validation set to find the best hyperparameters
# hint: for best results, you might need to try more values for learning rate and regularization strength 
# than provided initially

print('best validation accuracy achieved: %f' % best_val_accuracy)

#%% [markdown]
# # Какой же точности мы добились на тестовых данных?

#%%
test_pred = best_classifier.predict(test_X)
test_accuracy = multiclass_accuracy(test_pred, test_y)
print('Linear softmax classifier test set accuracy: %f' % (test_accuracy, ))
