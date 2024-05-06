# Deep Learning Exam Preparation Questions

Pull requests are welcome :)

## What is this

This a list of questions given by my teacher [Nwoye Chineduinnocent](https://scholar.google.com/citations?user=9lKxBUoAAAAJ&hl=en) to prepare for the deeplearning exam of the deeplearning course of Telecom Strasbourg.

I think these questions give a good overview of deeplearning so my goal is to fill this list with high quality answers. No bs every word matters. Keep the answers concise.

The goal is not to have this be understandable by a total beginner but to be a good reference for someone who already knows the basics of deeplearning.

You can find questions without answers in the file [questions.md](https://github.com/Times0/deeplearning/blob/main/questions.md)

## Questions

1. What is artificial intelligence?

A branch of computer science that aims to create programs that try to solve problems using human-like intelligence.

2. What is an artificial neural network?

It is a computational model inspired by how the human brain works. It has proven to be effective at solving problems in many domains.

3. How does ANN mimic the human brain?

It learns how to produce the correct value to a given question based on a feedback loop. It learns by adjusting its weights and biases on a large amount of data by comparing its output to the expected output.

4. What is machine learning?

Machine learning is an artificial intelligence technique where the machine learns by ingesting large amounts of data.

5. What is deep learning?

Deep learning is the process for a machine to learn by using an artificial neural network and data.

6. How does ANN learn complex patterns and relationships in data?

It can learn complex patterns thanks to its different weights. What makes the neural network powerful is its ability to depict non linear patterns with the activation functions. A network can approximate any function. (Universal approximation theorem)

7. What is an induced field in ANN?

The induced field is the sum of the products of the weights and the inputs of a node.

8. What is an activation function?

   Function that we apply to the input of a node to introduce non-linearity in the network.

9. How is activation function useful in the training of a neural network?

10. Mention at least 10 activation functions used in neural network and their major properties?
    - step
    - sigmoid
    - relu
    - tanh
    - silu
11. Write the mathematical formula of at least 10 activation functions.

    Boring

12. Mention at least 3 activation functions that scale the values of their inputs between 0 and 1?

    sigmoid
    step
    gaussian

13. Why is an activation function so powerful?

    Nonlinearity

14. Mention 5 ways a feedforward neural network is different from a recurrent one?

    Data flows in one direction
    No memory

15. Give 3 examples each of feedforward neural network and recurrent neural network?

16. What is supervised learning?

    We have labels associated to each value in the data

17. What is unsupervised learning?

    We Don't have labels associated to each value in the data

18. Give 3 examples of tasks performed using supervised and unsupervised learning?
19. What is self-supervised learning?
20. What is weakly-supervised learning?
21. What is semi-supervised learning?
22. When would you choose to train your model in an unsupervised manner?
23. Mention the 3 standard splits of a dataset?
24. What are the uses of each of the splits of dataset in deep learning experiment?
25. Differentiate between linear and non-linear classifier?
26. Give 2 examples each for linear and non-linear classifiers?
27. What type of data do you require to use a non-linear classifier?
28. What is a loss function in deep learning?
29. Give 5 examples of loss functions used for classification task?
30. Give 5 examples of loss functions used for regression task?
31. How does a loss function solve the maximum likelihood expectation?
32. How does a loss function minimize the difference between the predicted and actual
    probabilities?
33. What is the difference between binary cross-entropy and categorical cross-entropy?
34. When do you use a binary cross-entropy over a categorical cross-entropy loss function?
35. When do you use a categorical cross-entropy over a binary cross-entropy loss function?
36. What is the main difference between MAE and MSE loss function?
37. What is the mathematical formula for MSE?
38. What is the mathematical formula for Huber loss?
39. What is margin contrastive loss?
40. What is Regularization?
41. Why do you need regularization in deep learning model training?
42. Outline the training loop?
43. What is local minimum?
44. What is global minimum?
45. How can a model get stuck in the local minimum?
46. How can you get your model out of a local minimum?
47. How can you compute a gradient of a function over a computational graph?
48. Give 5 reasons why you need a GPU over a CPU in model training?
49. What is a deep learning framework?
50. Give 4 examples of a standard deep learning framework?
51. Give 4 importance of using a deep learning framework?
52. What does it mean that a deep learning framework's graph is static?
53. What does it mean that a deep learning framework's graph is dynamic?
54. Give 3 examples of dynamic deep learning frameworks?
55. What is a tensor?
56. Mention 4 possible data types of a tensor?
57. What are the properties of a tensor?
58. How can you calculate the dimension and shape of a tensor?
59. Mention 7 groups of tensor operations and give 1 example of each?
60. How do you slice a one-dimensional tensor?
61. What is the use of "axis" in tensor operation?
62. Differentiate between squeeze and reshape operation?
63. Name 4 places we can find tensors in deep learning models?
64. Name 4 properties of images that qualifies them as tensors?
65. Mention 10 different tensor operations that can be performed on images?
66. What is data augmentation?
67. What is the benefit of data augmentation in model training?
68. Give 5 image preprocessing techniques that can form styles of data augmentation?
69. What is a dataset?
70. Name 5 modalities of data in a dataset?
71. What does it mean to feed data in batches?
72. What is the super class of a PyTorch dataset class?
73. Name 3 compulsory functions to implement in a PyTorch dataset class and their functions?
74. What is a dataloader?
75. What are the major considerations when building a dataloader?
76. What is a Convolutional Neural Network and what is it used for?
77. Mention at least 7 layers that can be found in a CNN?
78. What is a convolution?
79. Write the mathematics of a convolution operation?
80. Why is the size of the output of a standard convolution smaller than the input size?
81. How can you keep the size of input and output of a convolution the same?
82. How does convolution change the channel of a feature?
83. What is a convolution filter?
84. How does stride influence the number of parameters in a convolutional layer?
85. What is a receptive field?
86. Mention 10 types of convolution layers and their major characteristics?
87. Which type of convolution downsample an input feature size?
88. Which type of convolution upsample an input feature size?
89. Which type of convolution is mainly targeting the transformation of input feature channel size?
90. How is deformable convolution different from separable convolution?
91. How does MobileNet use fewer parameters than conventional CNNs?
92. When do you use a 1D convolution?
93. When do you use a 2D convolution?
94. When do you use a 3D convolution?
95. What is a pooling layer?
96. How many parameters does the pooling layer have?
97. Name 4 types of pooling operations and their effects on the features?
98. What is the mathematics of a fully-connected layer?
99. When do you use a fully connected layer?
100. How do you determine the input and output size of a dense layer?
101. What is a dropout layer?
102. What is the behavior of a dropout layer during training and during testing?
103. Given an inputs size and convolutional kernel and strides, what is the formula for
     computing the output size?
104. How do you compute the number of parameters of a dense layer?
105. How do you compute the number of parameters of a convolution layer?
106. How many parameters has a batch normalization layer?
107. What is the use of the **init**() in PyTorch model design?
108. What is the use of the forward() in PyTorch model design?
109. What is the super class of a PyTorch NN model?
110. What is a sequential model?
111. What is a functional model?
112. When do you prefer to use a functional model over a sequential one?
113. Where is the order of execution of a functional model determined?
114. Where is the order of model architecture of a sequential model defined?
115. What layers do you consider when you count the number of layers of a CNN model?
116. What is a residual connection?
117. Mention example model using a residual connection?
118. Mention 3 ways an identity input feature can be connected to the output in a residual
     connection?
119. Why do large deep learning models need residual skip connection?
120. How to compute derivatives of a function?
121. How does a deep learning model update its weights?
122. What are the basic deep learning operations that you can find in a forward pass?
123. What are the basic deep learning operations that you can find in a backward pass?
124. Why are the intermediate values of each layer are cached in a memory during a forward
     pass?
125. What types of features are learnt by early-stage layers of a CNN?
126. Differentiate between evaluation metrics and loss function?
127. Explain the following with regards to model evaluation: TP, TN, FP, FN?
128. How do you compute the average precision?
129. What is the formula for precision using TP, FP, FN?
130. What is the formula for recall using TP, FP, FN?
131. What is AUC?
132. What is backpropagation?
133. How are gradients computed during model training?
134. What is gradient descent optimization?
135. List steps for the gradient descent algorithm in proper order?
136. What is stochastic gradient descent?
137. What is batch gradient descent?
138. What is mini-batch gradient descent?
139. Why is mini-batch gradient descent preferred over stochastic one?
140. Why would you use mini-batch gradient descent over batch gradient descent?
141. What is an optimizer?
142. What are the additional features added by the optimizers?
143. Mention the 3 tasks of an optimizer?
144. Give 5 examples of optimizers you know?
145. What does it mean to have a batch size of 8?
146. Differentiate between an epoch and iteration step?
147. What are the impacts of large and small batch sizes on training convergence?
148. What is a learning rate?
149. What are the impacts of large and small learning rates on training convergence?
150. How do you select a learning rate value?
151. Mention 4 ways of performing hyperparameter search?
152. What is generalization?
153. In 5 steps, summarize the training loop?
154. What is the interpretation of a training with oscillating loss?
155. What is the interpretation of a training with diverging loss?
156. What is the interpretation of a training with stagnating loss?
157. What is the interpretation of a training with stable loss?
158. What is the interpretation of a training with decreasing loss?
159. Define overfitting and underfitting?
160. Mention 20 ways of overcoming overfitting in deep learning model training?
161. What is batch normalization?
162. Mention 4 parameters you learn with batch norm?
163. What is early stopping in model training?
164. What do you understand by model convergence?
165. What is transfer learning?
166. What do we train by transfer learning?
167. What are the basic 3 steps in transfer learning?
168. Define unsupervised pretraining?
169. What is finetuning?
170. What are the 4 finetuning configurations you were taught?
171. How is object detection different from object classification?
172. How is object detection different from object localization?
173. Mention 4 possible ways of localizing an object?
174. Interpret the box coordinate values of a localized object?
175. List 10 applications of object detections?
176. In 4 ways, differentiate between one-stage and two-stage detectors?
177. What are anchor boxes in object detection?
178. What is multi-scale in object detection and why is it useful?
179. What is non-maximum suppression?
180. What metrics in used in computing non-maximum suppression?
181. Give 2 examples of single-stage detectors?
182. Give 2 examples of two-stage detectors?
183. Explain the concept of feature pyramid in object detection?
184. What is region proposal in object detection?
185. What is ROI?
186. What is multi-task learning?
187. Explain IoU in the context of object detection?
188. In an object detection problem, what is the metric that defines the "quality" of an
     inference?
189. How is segmentation different from localization?
190. Mention at least 5 types of segmentation?
191. What is the difference between semantic and instance segmentation?
192. When can semantic segmentation be treated as binary segmentation?
193. What is panoptic segmentation?
194. Name 3 common techniques of upsampling in segmentation model?
195. Mention common layers you can find in the encoder layer of a segmentation model?
196. Mention common layers you can find in the decoder layer of a segmentation model?
197. What is unpooling?
198. What is the function of an encoder in a deep learning model?
199. What is the function of a decoder in a deep learning model?
200. Mention 5 examples of segmentation models you know?
201. Explain IoU in the context of segmentation mask evaluation?
202. What type of data requires the use of RNN over FNN?
203. Give 3 examples of a sequential problem?
204. How many states does a GRU have?
205. What is a cell function?
206. Mention two states of an LSTM cell?
207. Explain the unrolling of an RNN layer?
208. What is vanishing gradient?
209. How does an LSTM suffer from vanishing gradient?
210. How can vanishing gradients in LSTM be mitigated?
211. What is exploding gradient?
212. What is bidirectional RNN?
213. Mention and explain the functions of the 3 gates of an LSTM?
214. Name 3 non-RNN temporal models?
215. What is an attention mechanism in deep learning?
216. Why is attention mechanism important?
217. How does attention mechanism overcome the issues in RNN?
218. What is sequence-to-sequence modeling?
219. What is a basic building block of a transformer?
220. Write the equation of self-attention?
221. What are the advantages of multi-head attention?
222. What is a cross attention?
223. What is the use of alignment score in attention process?
224. Differentiate between local and global attention?
225. Differentiate between temporal and spatial attention?
226. Differentiate between positional and channel attention?
227. What is do-product attention?
228. List and explain the different attention operations?
229. What is the use of SoftMax in the attention process?
230. What is the function of positional encoding in a Transformer?
231. What is the use of masked attention in a Transformer?
232. How can an image be treated as a sequence?
233. List at least 5 popular transformer models and their tasks?
234. What are the advantages of a transformer over an RNN?
235. What is the complementary model needed to train a GAN?
236. Explain the concept of min-max in adversarial learning?
237. What is a diffusion process?
238. What is mode collapse?
239. What are the main issues with training a GAN?
240. How can you stabilize the training of a generative model?
