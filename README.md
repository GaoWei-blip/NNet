## 基于C++的简单神经网络-股价预测为例（包含C的代码）

## 1.主要模块

- `Neuron.h`：用于声明神经元（`Neuron`）类和构造函数
- `Layer.h`：用于声明网络层（`Layer`）类和构造函数
- `NNet.h`：用于声明神经网络（`Neural Network`）类和构造及初始化函数
- `Dataset.h`：用于声明数据集（`Dataset`）类和构造函数
- `Trainer.h`：用于声明神经网络前后向传播（`Forward`和`Backpropagation`）类和相关的函数

## 2. 理解网络结构和待学习参数

![network](.\images\network.png)

上图是一个 输入为2维，输出为1维的3层简单神经网络，w和bias是该网络的学习参数，具体计算流程如下：

**前向传播**

（输入层-隐藏层）

input
$$
A1=X=[x_1,x_2]
$$
Weights
$$
\mathbf{W_1} = \begin{bmatrix} w_{11}^{(1)} & w_{12}^{(1)} \\ w_{21}^{(1)} & w_{22}^{(1)} \\ w_{31}^{(1)} & w_{32}^{(1)} \end{bmatrix}
$$
Biases
$$
\mathbf{B_2} = \begin{bmatrix} b_1^{(2)} \\ b_2^{(2)} \\ b_3^{(2)} \end{bmatrix}
$$
Weighted Sum
$$
\mathbf{Z_2} = \mathbf{W_1} \cdot \mathbf{X}^T + \mathbf{B_2} = \begin{bmatrix} w_{11}^{(1)}x_1 + w_{12}^{(1)}x_2 + b_1^{(2)} \\ w_{21}^{(1)}x_1 + w_{22}^{(1)}x_2 + b_2^{(2)} \\ w_{31}^{(1)}x_1 + w_{32}^{(1)}x_2 + b_3^{(2)} \end{bmatrix}
$$
Activate (ReLU或Sigmoid)
$$
\mathbf{A_2} = \sigma(\mathbf{Z_2}) = \begin{bmatrix} \sigma(z_1) \\ \sigma(z_2) \\ \sigma(z_3) \end{bmatrix} = \begin{bmatrix} a_1^{(2)} \\ a_2^{(2)} \\ a_3^{(2)} \end{bmatrix}
$$
（隐藏层-输出层）

Weights
$$
\mathbf{W_2} = \begin{bmatrix} w_{11}^{(2)} & w_{12}^{(2)} & w_{13}^{(2)} \\ w_{21}^{(2)} & w_{22}^{(2)} & w_{23}^{(2)} \end{bmatrix}
$$
Biases
$$
\mathbf{B_3} = \begin{bmatrix} b_1^{(3)} \\ b_2^{(3)} \end{bmatrix}
$$
Weighted Sum
$$
Z_3 = \mathbf{W_2} \cdot \mathbf{A_2} + \mathbf{B_3} =\begin{bmatrix} w_{11}^{(2)}a_{1}^{(2)} + w_{12}^{(2)}a_{2}^{(2)} + w_{13}^{(2)}a_{3}^{(2)} + b_1^{(3)} \\ w_{21}^{(2)}a_{1}^{(2)} + w_{22}^{(2)}a_{2}^{(2)} + w_{23}^{(2)}a_{3}^{(2)} + b_2^{(3)} \end{bmatrix}
$$


Activate (Sigmoid)
$$
\hat{Y} = A_3 =\sigma(Z_3) =\begin{bmatrix} \hat{y_1} \\ \hat{y_2} \end{bmatrix}
$$
Calculate loss (assuming the Mean Squared Error (MSE) loss function or Cross Entropy Loss function is used)
$$
{MSE}\_\text{Loss} = \frac{1}{2m} \sum_{i=1}^m (\hat{y_i} - y_i)^2 \\
or \\
{CE}\_\text{Loss}= -\frac{1}{m} \sum_{i=1}^m \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$
**后向传播**

（输出层-隐藏层）

Calculate gradients and update weights and biases. Among them, $\alpha$ is the learning rate.
$$
\delta^{(3)} = \frac{\partial \text{Loss}}{\partial Z_3} = (\hat{Y} - Y) = A_3 - Y = \begin{bmatrix} \delta_1^{(3)}  \\ \delta_2^{(3)}  \end{bmatrix} \\

W_2 = W_2 - \alpha \cdot \frac{\partial \text{Loss}}{\partial W_2} = W2 - \alpha \cdot \frac{\partial \text{Loss}}{\partial Z_3} \cdot \frac{\partial Z_3}{\partial W_2}  = W2 - \alpha \cdot \delta^{(3)} \cdot \mathbf{A_2}^T  = \begin{bmatrix} w_{11}^{(2)} - \alpha \delta_1^{(3)}a_1^{(2)} & w_{12}^{(2)} - \alpha \delta_1^{(3)}a_2^{(2)} & w_{13}^{(2)} - \alpha \delta_1^{(3)}a_3^{(2)} \\ w_{21}^{(2)} - \alpha \delta_2^{(3)}a_1^{(2)} & w_{22}^{(2)} - \alpha \delta_2^{(3)}a_2^{(2)} & w_{23}^{(2)} - \alpha \delta_2^{(3)}a_3^{(2)} \end{bmatrix} \\

B_3 = B_3 - \alpha \cdot \frac{\partial \text{Loss}}{\partial B_3}=B_3 - \alpha \cdot \frac{\partial \text{Loss}}{\partial Z_3}\cdot \frac{\partial Z_3}{\partial B_3} = B3 - \alpha \cdot \delta^{(3)} = \begin{bmatrix} b_1^{(3)} - \alpha \delta_1^{(3)} \\ b_2^{(3)} - \alpha \delta_2^{(3)} \end{bmatrix}
$$
> MSE Loss function gradient, usually not use activation function:
> $$
> \frac{\partial \text{L}}{\partial z} = \frac{\partial \text{L}}{\partial \hat{y}_i} = \hat{y_i} - y_i
> $$
> Cross Entropy Loss function gradient (log x=ln x), usually using sigmoid activation function:
> $$
> \frac{\partial L}{\partial \hat{y}_i} = - \left( \frac{y_i}{\hat{y}_i} - \frac{1 - y_i}{1 - \hat{y}_i} \right) \\
> \frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial z} \\
> \frac{\partial L}{\partial z} = - \left( \frac{y_i}{\hat{y}_i} - \frac{1 - y_i}{1 - \hat{y}_i} \right) \cdot \hat{y}_i (1 - \hat{y}_i) \\
> \frac{\partial L}{\partial z} = \hat{y}_i - y_i
> $$
> Sigmoid Activation function gradient:
> $$
> \sigma(x) = \frac{1}{1 + e^{-x}} = (1 + e^{-x})^{-1} \\
> \sigma'(x) = - (1 + e^{-x})^{-2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1 + e^{-x})^2}=\sigma(x) \odot (1 - \sigma(x))
> $$

（隐藏层-输入层）

Calculate gradients and update weights and biases, assuming the use of sigmoid activation function.
$$
\delta^{(2)} = \frac{\partial \text{Loss}}{\partial Z_2} = \frac{\partial \text{Loss}}{\partial Z_3} \cdot \frac{\partial Z_3}{\partial A_2} \cdot \frac{\partial A_2}{\partial Z_2} = (\mathbf{W_2} \cdot \delta_3) \odot \sigma'(\mathbf{Z_2})=\begin{bmatrix} \delta_1^{(2)} \\ \delta_2^{(2)} \\ \delta_3^{(2)} \end{bmatrix} \\

W_1 := W_1 - \alpha \cdot \frac{\partial Loss}{\partial W_1}=W_1 - \alpha \cdot \frac{\partial \text{Loss}}{\partial Z_2} \cdot \frac{\partial Z_2}{\partial W_1} = W_1 - \alpha \cdot \delta^{(2)} \cdot \mathbf{X}=\begin{bmatrix} w_{11}^{(1)}-\alpha\delta_1^{(2)}x1 & w_{12}^{(1)}-\alpha\delta_1^{(2)}x2 \\ w_{21}^{(1)}-\alpha\delta_2^{(2)}x1 & w_{22}^{(1)}-\alpha\delta_2^{(2)}x2 \\ w_{31}^{(1)}-\alpha\delta_3^{(2)}x1 & w_{32}^{(1)}-\alpha\delta_3^{(2)}x2\end{bmatrix} \\

B_2 := B_2 - \alpha \cdot \frac{\partial Loss}{\partial B_2}=B2-\alpha \cdot\delta^{(2)}
$$

> 3.股价预测为例

> - `stock_price_data.txt` 包含100条模拟的股价预测训练数据，每行包括日期、日期时间戳（timestamp）、开盘价 (Open)、最高价 (High)、最低价 (Low)、收盘价 (Close) 和交易量 (Volume)
>
> - 这里将股价预测看成一个纯粹的回归问题，输入是5维（timestamp、Open、High、Low、Volume）、输出1维（Close）

构建一个2层神经网络，第一层神经元数5，第二层神经元数3

训练结果如图：



测试结果如图：

