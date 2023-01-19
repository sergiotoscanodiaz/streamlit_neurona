import numpy as np

# Clase con las características de la neurona

class Neuron:
  
  # CONSTRUCTOR
  def __init__(self, weights, bias, func):
    self.weights = weights
    self.bias = bias
    self.func = func
  
  # MÉTODO PRINCIPAL PARA EL CÁLCULO DE LA NEURONA
  def run(self, input_data):
    result = np.dot(np.array(input_data), self.weights) + self.bias

    if self.func == "sigmoid":
      return self.__sigmoid_function(result)
    elif self.func == "relu":
      return self.__relu_function(result)
    elif self.func == "tanh":
      return self.__tanh_function(result)
    else:
      print("Elige una función de activación correcta")

  # MÉTODOS PARA EL CAMBIO DE PARÁMETROS
  
  def change_bias(self, bias):
    self.bias = bias
  
  def change_weights(self, weights):
    self.weights = weights
  
  def change_function(self, function):
    self.fuction = function
  
  # FUNCIONES DE ACTIVACIÓN
  @staticmethod
  def __sigmoid_function(x):
    return 1/(1 + np.exp(-x))
  
  @staticmethod
  def __relu_function(x):
    return max(0.0, x)
  
  @staticmethod
  def __tanh_function(x):
    return np.tanh(x)

