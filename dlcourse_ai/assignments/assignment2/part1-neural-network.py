#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
    currentWorkingDirectory = os.getcwd()
    print("Current directory: \"{}\"".format(currentWorkingDirectory))

    workingDirectory = os.path.join(
        currentWorkingDirectory,
        "assignments\\assignment2"
    )

    print("Trying to change to {}".format(workingDirectory))
    os.chdir(workingDirectory)
    print("Change successful")
except:
    print("Directory change failed")

#%% [markdown]
# # Задание 2.1 - Нейронные сети
# 
# В этом задании вы реализуете и натренируете настоящую нейроную сеть своими руками!
# 
# В некотором смысле это будет расширением прошлого задания - нам нужно просто составить несколько линейных классификаторов вместе!
# 
# <img src="https://i.redd.it/n9fgba8b0qr01.png" alt="Stack_more_layers" width="400px"/>

#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
sys.path.insert(0, "..")
from gradient_check import check_layer_gradient, check_layer_param_gradient, check_model_gradient
from metrics import multiclass_accuracy
from dataset import load_svhn, random_split_train_val

#%% [markdown]
# # Загружаем данные
# 
# И разделяем их на training и validation.

#%%
def prepare_for_neural_network(train_X, test_X):
    train_flat = train_X.reshape(train_X.shape[0], -1).astype(np.float) / 255.0
    test_flat = test_X.reshape(test_X.shape[0], -1).astype(np.float) / 255.0
    
    # Subtract mean
    mean_image = np.mean(train_flat, axis = 0)
    train_flat -= mean_image
    test_flat -= mean_image
    
    return train_flat, test_flat
    
train_X, train_y, test_X, test_y = load_svhn("..\\data", max_train=10000, max_test=1000)    
train_X, test_X = prepare_for_neural_network(train_X, test_X)
# Split train into train and val
train_X, train_y, val_X, val_y = random_split_train_val(train_X, train_y, num_val = 1000)

#%% [markdown]
# # Как всегда, начинаем с кирпичиков
# 
# Мы будем реализовывать необходимые нам слои по очереди. Каждый слой должен реализовать:
# - прямой проход (forward pass), который генерирует выход слоя по входу и запоминает необходимые данные
# - обратный проход (backward pass), который получает градиент по выходу слоя и вычисляет градиент по входу и по параметрам
# 
# Начнем с ReLU, у которого параметров нет.

#%%
from layers import ReLULayer

X = np.array([
    [1, -2, 3],
    [-1, 2, 0.1]
])

assert check_layer_gradient(ReLULayer(), X)

#%% [markdown]
# А теперь реализуем полносвязный слой (fully connected layer), у которого будет два массива параметров: W (weights) и B (bias).
# 
# Все параметры наши слои будут использовать для параметров специальный класс `Param`, в котором будут храниться значения параметров и градиенты этих параметров, вычисляемые во время обратного прохода.
# 
# Это даст возможность аккумулировать (суммировать) градиенты из разных частей функции потерь, например, из cross-entropy loss и regularization loss.

#%%
from layers import FullyConnectedLayer
assert check_layer_gradient(FullyConnectedLayer(3, 4), X)
assert check_layer_param_gradient(FullyConnectedLayer(3, 4), X, 'W')

#%% [markdown]
# ## Создаем нейронную сеть
# 
# Теперь мы реализуем простейшую нейронную сеть с двумя полносвязным слоями и нелинейностью ReLU. Реализуйте функцию `compute_loss_and_gradients`, она должна запустить прямой и обратный проход через оба слоя для вычисления градиентов.
# 
# Не забудьте реализовать очистку градиентов в начале функции.

#%%
from model import TwoLayerNet

model = TwoLayerNet(
    n_input = train_X.shape[1],
    n_output = 10,
    hidden_layer_size = 3,
    reg = 0)
loss = model.compute_loss_and_gradients(train_X[:2], train_y[:2])

check_model_gradient(model, train_X[:2], train_y[:2])

#%% [markdown]
# Теперь добавьте к модели регуляризацию - она должна прибавляться к loss и делать свой вклад в градиенты.

#%%
from model import TwoLayerNet
model_with_reg = TwoLayerNet(
    n_input = train_X.shape[1],
    n_output = 10,
    hidden_layer_size = 3,
    reg = 1e1)
loss_with_reg = model_with_reg.compute_loss_and_gradients(train_X[:2], train_y[:2])
assert loss_with_reg > loss and not np.isclose(loss_with_reg, loss),     "Loss with regularization (%2.4f) should be higher than without it (%2.4f)!" % (loss, loss_with_reg)

check_model_gradient(model_with_reg, train_X[:2], train_y[:2])

#%% [markdown]
# Также реализуем функцию предсказания (вычисления значения) модели на новых данных.
# 
# Какое значение точности мы ожидаем увидеть до начала тренировки?

#%%
# Finally, implement predict function!

# What would be the value we expect?
multiclass_accuracy(model_with_reg.predict(train_X[:30]), train_y[:30]) 

#%% [markdown]
# # Допишем код для процесса тренировки

#%%
from trainer import Trainer, Dataset
from optim import SGD
from modelGridSearch import search_model

model = TwoLayerNet(
    n_input = train_X.shape[1],
    n_output = 10,
    hidden_layer_size = 100,
    reg = 1e-1)
dataset = Dataset(train_X, train_y, val_X, val_y)
trainer = Trainer(model, dataset, SGD(), learning_rate=1e-4)
loss_history, train_history, val_history = trainer.fit()

#%%
dataset = Dataset(train_X, train_y, val_X, val_y)
model.predict(dataset.val_X)

#%%
plt.plot(loss_history)
#%%
plt.plot(train_history)
plt.plot(val_history)

#%% [markdown]
# # Улучшаем процесс тренировки
# 
# Мы реализуем несколько ключевых оптимизаций, необходимых для тренировки современных нейросетей.
#%% [markdown]
# ## Уменьшение скорости обучения (learning rate decay)
# 
# Одна из необходимых оптимизаций во время тренировки нейронных сетей - постепенное уменьшение скорости обучения по мере тренировки.
# 
# Один из стандартных методов - уменьшение скорости обучения (learning rate) каждые N эпох на коэффициент d (часто называемый decay). Значения N и d, как всегда, являются гиперпараметрами и должны подбираться на основе эффективности на проверочных данных (validation data). 
# 
# В нашем случае N будет равным 1.

#%%
model = TwoLayerNet(
    n_input = train_X.shape[1],
    n_output = 10,
    hidden_layer_size = 100,
    reg = 1e-1)
dataset = Dataset(train_X, train_y, val_X, val_y)
trainer = Trainer(
    model,
    dataset,
    SGD(),
    learning_rate=1e-4,
    learning_rate_decay=0.99)

initial_learning_rate = trainer.learning_rate
loss_history, train_history, val_history = trainer.fit()

assert trainer.learning_rate < initial_learning_rate, "Learning rate should've been reduced"
assert trainer.learning_rate > 0.5*initial_learning_rate, "Learning rate shouldn'tve been reduced that much!"

#%% [markdown]
# # Накопление импульса (Momentum SGD)
# 
# Другой большой класс оптимизаций - использование более эффективных методов градиентного спуска. Мы реализуем один из них - накопление импульса (Momentum SGD).
# 
# Этот метод хранит скорость движения, использует градиент для ее изменения на каждом шаге, и изменяет веса пропорционально значению скорости.
# (Физическая аналогия: Вместо скорости градиенты теперь будут задавать ускорение, но будет присутствовать сила трения.)
# 
# ```
# velocity = momentum * velocity - learning_rate * gradient 
# w = w + velocity
# ```
# 
# `momentum` здесь коэффициент затухания, который тоже является гиперпараметром (к счастью, для него часто есть хорошее значение по умолчанию, типичный диапазон -- 0.8-0.99).
# 
# Несколько полезных ссылок, где метод разбирается более подробно:
#
# - http://cs231n.github.io/neural-networks-3/#sgd
# - https://distill.pub/2017/momentum/

#%%
from optim import MomentumSGD

model = TwoLayerNet(
    n_input = train_X.shape[1],
    n_output = 10,
    hidden_layer_size = 100,
    reg = 1e-1)
dataset = Dataset(train_X, train_y, val_X, val_y)
trainer = Trainer(model, dataset, MomentumSGD(), learning_rate=1e-4, learning_rate_decay=0.99)

# You should see even better results than before!
loss_history, train_history, val_history = trainer.fit()

#%%
model.predict(dataset.val_X)

#%% [markdown]
# # Ну что, давайте уже тренировать сеть!
#%% [markdown]
# ## Последний тест - переобучимся (overfit) на маленьком наборе данных
# 
# Хороший способ проверить, все ли реализовано корректно - переобучить сеть на маленьком наборе данных.  
# Наша модель обладает достаточной мощностью, чтобы приблизить маленький набор данных идеально, поэтому мы ожидаем, что на нем мы быстро дойдем до 100% точности на тренировочном наборе. 
# 
# Если этого не происходит, то где-то была допущена ошибка!

#%%
data_size = 15
model = TwoLayerNet(
    n_input = train_X.shape[1],
    n_output = 10,
    hidden_layer_size = 100,
    reg = 1e-1)
dataset = Dataset(train_X[:data_size], train_y[:data_size], val_X[:data_size], val_y[:data_size])
trainer = Trainer(
    model,
    dataset,
    SGD(),
    learning_rate=1e-1,
    num_epochs=150,
    batch_size=5)

# You should expect this to reach 1.0 training accuracy 
loss_history, train_history, val_history = trainer.fit()

#%% [markdown]
# Теперь найдем гипепараметры, для которых этот процесс сходится быстрее.
# Если все реализовано корректно, то существуют параметры, при которых процесс сходится в **20** эпох или еще быстрее.
# Найдите их!

#%%
# Now, tweak some hyper parameters and make it train to 1.0 accuracy in 20 epochs or less

model = TwoLayerNet(n_input = train_X.shape[1], n_output = 10, hidden_layer_size = 100, reg = 1e-1)
dataset = Dataset(train_X[:data_size], train_y[:data_size], val_X[:data_size], val_y[:data_size])
# TODO: Change any hyperparamers or optimizators to reach training accuracy in 20 epochs
trainer = Trainer(model, dataset, SGD(), learning_rate=1e-1, num_epochs=20, batch_size=5)

loss_history, train_history, val_history = trainer.fit()

#%% [markdown]
# # Итак, основное мероприятие!
# 
# Натренируйте лучшую нейросеть! Можно добавлять и изменять параметры, менять количество нейронов в слоях сети и как угодно экспериментировать. 
# 
# Добейтесь точности лучше **40%** на validation set.

#%%
# Let's train the best one-hidden-layer network we can

learning_rates = 1e-4
reg_strength = 1e-3
learning_rate_decay = 0.999
hidden_layer_size = 128
num_epochs = 200
batch_size = 64

best_classifier = None
best_val_accuracy = None

loss_history = []
train_history = []
val_history = []

# TODO find the best hyperparameters to train the network
# Don't hesitate to add new values to the arrays above, perform experiments, use any tricks you want
# You should expect to get to at least 40% of valudation accuracy
# Save loss/train/history of the best classifier to the variables above

print('best validation accuracy achieved: %f' % best_val_accuracy)


#%%
plt.figure(figsize=(15, 7))
plt.subplot(211)
plt.title("Loss")
plt.plot(loss_history)
plt.subplot(212)
plt.title("Train/validation accuracy")
plt.plot(train_history)
plt.plot(val_history)

#%% [markdown]
# # Как обычно, посмотрим, как наша лучшая модель работает на тестовых данных

#%%
test_pred = best_classifier.predict(test_X)
test_accuracy = multiclass_accuracy(test_pred, test_y)
print('Neural net test set accuracy: %f' % (test_accuracy, ))
