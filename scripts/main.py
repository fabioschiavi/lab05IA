
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

print('Carregando Arquivo de teste')
arquivo = np.load('teste4.npy')
x = arquivo[0]
y = np.ravel(arquivo[1])

media = []
desvio = []

#Execução 1 - 10x / Neuronios =	10 / Camadas = 20 / Iterações = 20000
#Execução 1 - 10x / Neuronios =	10 / Camadas = 3 / Iterações = 23000
#Execução 1 - 10x / Neuronios =	10 / Camadas = 35 / Iterações = 1500

neu = 10
cam = 3
iter = 23000

for i in range(10):
    regr = MLPRegressor(hidden_layer_sizes=(neu,cam),
                        max_iter=iter,
                        activation='relu', #{'identity', 'logistic', 'tanh', 'relu'},
                        solver='adam',
                        learning_rate = 'adaptive',
                        n_iter_no_change=50)
    print('Treinando RNA')
    regr = regr.fit(x,y)

    print('Preditor')
    y_est = regr.predict(x)

    media.append(np.average(y_est))
    desvio.append(np.std(y_est))

    plt.figure(figsize=[14,7])

    #plot curso original
    plt.subplot(1,3,1)
    plt.plot(x,y)

    #plot aprendizagem
    plt.subplot(1,3,2)
    plt.plot(regr.loss_curve_)

    #plot regressor
    plt.subplot(1,3,3)
    plt.plot(x,y,linewidth=1,color='yellow')
    plt.plot(x,y_est,linewidth=2)
    plt.show()


iteracoes=23000
neuro=10
camadas=3


print(f'Média das médias de execução: {iteracoes} iterações, {neuro} Neurônios e {camadas} Camadas é igual a:'" %.2f" %np.average(media))
print(f'Média dos desvios padrões de execução: {iteracoes} iterações, {neuro} Neurônios e {camadas} Camadas: é igual a:'" %.2f" %np.std(desvio))
