# Understanding-and-Implementing-the-Activation-Function

## Activation functions:
Activation functions are a keypoint of neural network design.In artificial neural network,each neuron forms a weighted sum of its inputs and passes a the resulting scalar value through a function reffered to as a activation function or transfer function.
If a neuron has n inputs then the output or activation of neuron is 

y=α (w1x1+w2x2+.....+wnxn+b)

This α  function is reffered as the activation function.

![<img width="424" alt="image" src="https://github.com/khairun123/Understanding-and-Implementing-the-Activation-Function/assets/128392550/6b1bdac7-e02b-49fc-9fa1-6cf61bdd5e62">]

## Why we need activation functions?
The capability and performance of neural network mostly depend onthe choice of activation functions.different activation functions may be used in different parts of the model.Without activation functions, a neural network would essentially behave like a linear regression model but  real life problem  are more complex.So we need to introduce our model with nonlinearity.Activation functions enables neural networks to learn complicated, high dimensional, and non-linear Big Data sets that have an intricate architecture – they contain multiple hidden layers in between the input and output layer and enables nueral network to perform more complex tasks by introducing non-linearity into a neuron's output layers.As a result  neural networks can learn and model other complicated data, including images, speech, videos, audio, etc.



## Structure of nueral network 
A network may have three types of layers: The input layer, the hidden layer, and the output layer.There may be one or more than one hidden layers.Input layers that take raw input from the domain and hidden Layers in a neural network between the input layer and the output layer, where intermediate processing or feature extraction occurs.And output layers that make a prediction .

Feedforward and backpropagation are essential processes that allow neural networks to learn from data and make predictions.In a neural network, we would update the weights and biases of the neurons on the basis of the error at the output. This process is known as back-propagation. Activation functions make the back-propagation possible since the gradients are supplied along with the error to update the weights and biases. All hidden layers generally use the same activation function. The output layer will typically use a different activation function from the hidden layers and is dependent upon the type of prediction required by the model.


## What are the different types of activations functions we have?
Most commonly used activations functions are 

1.Sigmoid Function (Logistic Function)

2.Hyperbolic Tangent Function (tanh)

3.Rectified Linear Unit (ReLU)

4.Leaky ReLU

5.Exponential Linear Unit (ELU)

6.Scaled Exponential Linear Unit (SELU)

7.Softmax Function

8.Swish Function

9.Linear Function

10.Parametric ReLU Function

11.Gaussian Error Linear Unit (GELU) Function**


# 1.Sigmoid Function (Logistic Function)

It is a function which is plotted as ‘S’ shaped graph.

Equation : A = 1/(1 + e-x)
Value Range : 0 to 1

Python code implementations

>import matplotlib.pyplot as plt
>
>import numpy as np
>
>def sigmoid(x):
>
>s=1/(1+np.exp(-x))
>
>ds=s*(1-s)
>
>return s,ds
>
>
![<img width="848" alt="image" src="https://github.com/khairun123/Understanding-and-Implementing-the-Activation-Function/assets/128392550/2984a462-0a8e-4523-a3b9-03cf3b00630e">]


Nature : Non-linear. Notice that X values lies between -2 to 2, Y values are very steep. This means, small changes in x would also bring about large changes in the value of Y.

Uses : Usually used in output layer of a binary classification, where result is either 0 or 1, as value for sigmoid function lies between 0 and 1 only so, result can be predicted easily to be 1 if value is greater than 0.5 and 0 otherwise.

Disadvantages:Suffers from the “vanishing gradients” problem, making it slow to learn.

# 2.Hyperbolic Tangent Function (tanh)
The tanh activation function gives higher values of gradient during training and higher updates in the weights of the network compared to sigmoid activation function. So, if we want strong gradients and big learning steps, we should use the tanh activation function.And also the output of tanh is symmetric around zero leading to faster convergence.

Python code implementations

>import matplotlib.pyplot as plt
>
>import numpy as np
>
>def tanh(x):
>
>s=(2/(1+np.exp(-2*x)))-1
>
>ds=(1-s**2)
>
>return s,ds
>
>
![<img width= "610" alt="image" src="https://github.com/khairun123/Understanding-and-Implementing-the-Activation-Function/assets/128392550/536ed184-1b68-43dd-a3f8-4c55ca54f0ca">]


 Didadvantages:   Still prone to vanishing gradient problems. 
    

    
# 3.Rectified Linear Unit (ReLU)

It Stands for Rectified linear unit. It is the most widely used activation function.Solves the vanishing gradient problem. Chiefly implemented in hidden layers of Neural network.

Equation :- A(x) = max(0,x). It gives an output x if x is positive and 0 otherwise.

Value Range :- [0, inf)

Python code implementations

>import matplotlib.pyplot as plt
>
>import numpy as np
>
>def relu_activation(x):
>
>return max(0,x)
>
>def relu_derivative(x):
>
>return 1 if x>=0  else 0


![image](https://github.com/khairun123/Understanding-and-Implementing-the-Activation-Function/assets/128392550/0c0a2ecb-7a1c-4c7a-a630-a406d16ab0c1)]
Nature :- non-linear, which means we can easily backpropagate the errors and have multiple layers of neurons being activated by the ReLU function.

Uses :- ReLu is less computationally expensive than tanh and sigmoid because it involves simpler mathematical operations. At a time only a few neurons are activated making the network sparse making it efficient and easy for computation. In simple words, RELU learns much faster than sigmoid and Tanh function.

 Disadvantages:
Can lead to “Dead Neurons” due to fragile gradients. Should be used only in hidden layers.


# 7.Softmax Function

The softmax function is also a type of sigmoid function but is handy when we are trying to handle multi- class classification problems.

Nature :- non-linear

Uses :- Usually used when trying to handle multiple classes. the softmax function was commonly found in the output layer of image classification problems.The softmax function would squeeze the outputs for each class between 0 and 1 and would also divide by the sum of the outputs. 

Output:- The softmax function is ideally used in the output layer of the classifier where we are actually trying to attain the probabilities to define the class of each input.



# Characteristics of ideal activations Functions

1.Nonlinearity

2.Differentiable

3.Computationality

4.Zero centered

5.Non-saturating

# Vanishing gradient 

Impact on Vanishing Gradients:

Definition: The vanishing gradient problem occurs when gradients become extremely small as they propagate backward through many layers of a neural network during training. This phenomenon can hinder the learning process, especially in deep networks, as it leads to negligible updates to the weights in early layers.

Saturated Activation Functions like sigmoid and tanh are prone to saturation, particularly for large positive or negative inputs. When inputs are in the saturated regions of these functions, their derivatives approach zero, causing gradients to vanish during backpropagation.

The vanishing gradient problem becomes more pronounced in deeper neural networks due to the cumulative effect of gradients diminishing as they propagate backward through multiple layers. This limits the ability of deep networks to effectively learn hierarchical representations of data.

Mitigation with ReLU and Variants: Activation functions like ReLU and its variants (e.g., Leaky ReLU, ELU) address the vanishing gradient problem by providing non-saturating derivatives for positive inputs. This allows gradients to flow more freely during backpropagation, facilitating more effective weight updates and mitigating the issue of vanishing gradients.

In summary, the choice of activation function profoundly impacts the behavior of gradient descent optimization and directly influences the occurrence of the vanishing gradient problem. Activation functions with non-saturating derivatives, such as ReLU and its variants, are often preferred in practice for deep neural networks due to their ability to mitigate the vanishing gradient problem and facilitate faster convergence.


The vanishing gradient problem arises from the nature of the partial derivative of the activation function used to create the neural network. The problem can bevworse in deep neural networks using Sigmoid activation function. It can be significantly reduced by using activation functions like ReLU and leaky ReLU.

## How to choose the appropriate activation function for neural network?

In nueral network the choice of avtivation functions depends on the types of the problem we are dealing with.
Some activation functions, such as sigmoid and tanh, can suffer from vanishing or exploding gradient problems, making it difficult for the model to learn effectively.
Some activation functions have a limited range of output values, which may constrain the model's ability to learn complex patterns and relationships in the data.
Certain activation functions, such as the exponential functions used in the softmax activation, can be computationally expensive and slow down the training process.
Choosing the right activation function for a neural network can be challenging, as different activation functions may perform better or worse depending on the specific task and data.

The activation function used in hidden layers is typically chosen based on the type of neural network architecture.
  Convolutional Neural Network (CNN)  uses ReLU activation function(Rectified Linear Unit)  which help to solve the vanishing gradient 
  problem. Recurrent Neural Network uses Tanh and/or Sigmoid activation function.
  
If it is a classification problem with binary class we should use Sigmoid activation function  in output layers.If it  is a clssification problem with multilabel class we should use sigmoid activation function too.And incae of multiclass classification we have to use softmax activation function in output layers.
