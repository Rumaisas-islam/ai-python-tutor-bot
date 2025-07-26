questions = [

    # Python Basics
    {
        "category": "Python Basics",
        "question": "1. What is Python and why is it popular?",
        "answer": """Python is a high-level, interpreted programming language known for its simplicity and readability.
It supports multiple programming paradigms (procedural, OOP, functional) and has a huge standard library.
Python is popular for web development, data science, automation, AI, and more.

Example:
print('Hello, World!')
"""
    },

    {
        "category": "Python Basics",
        "question": "2. How do you create variables in Python?",
        "answer": """Variables store data values. You create a variable by assigning a value to a name.

Example:
x = 10
name = 'Ali'
print(x, name)
"""
    },

    {
        "category": "Python Basics",
        "question": "3. What are Python data types?",
        "answer": """Python has several built-in data types like int, float, str, list, tuple, dict, set, bool.

Example:
a = 5          # int
b = 3.14       # float
c = 'Hello'    # str
d = [1,2,3]    # list
e = (4,5,6)    # tuple
f = {'a':1}    # dict
g = {1,2,3}    # set
"""
    },

    {
        "category": "Python Basics",
        "question": "4. What is a list and how do you use it?",
        "answer": """A list is an ordered, mutable collection of items that can contain duplicates.

Example:
fruits = ['apple', 'banana', 'cherry']
fruits.append('orange')
print(fruits)
"""
    },

    {
        "category": "Python Basics",
        "question": "5. What is the difference between list and tuple?",
        "answer": """Lists are mutable (can change), tuples are immutable (cannot change).

Example:
my_list = [1, 2, 3]
my_tuple = (1, 2, 3)
my_list[0] = 10  # works
# my_tuple[0] = 10  # error
"""
    },

    {
        "category": "Python Basics",
        "question": "6. What is a dictionary in Python?",
        "answer": """Dictionary stores data in key-value pairs.

Example:
student = {'name': 'Ali', 'age': 22}
print(student['name'])
"""
    },

    {
        "category": "Python Basics",
        "question": "7. How do you write a conditional statement?",
        "answer": """Use if, elif, else for conditional logic.

Example:
x = 10
if x > 0:
    print('Positive')
elif x == 0:
    print('Zero')
else:
    print('Negative')
"""
    },

    {
        "category": "Python Basics",
        "question": "8. How do you write loops in Python?",
        "answer": """Use for and while loops.

Example:
for i in range(5):
    print(i)

count = 0
while count < 5:
    print(count)
    count += 1
"""
    },

    {
        "category": "Python Basics",
        "question": "9. What are functions in Python?",
        "answer": """Functions are reusable blocks of code defined using `def`.

Example:
def greet(name):
    return f'Hello, {name}'

print(greet('Sara'))
"""
    },

    {
        "category": "Python Basics",
        "question": "10. What is the purpose of the return statement?",
        "answer": """`return` sends a value back to the caller.

Example:
def add(a, b):
    return a + b

result = add(3, 4)
print(result)  # 7
"""
    },

    # Intermediate Python
    {
        "category": "Intermediate Python",
        "question": "11. What is list comprehension?",
        "answer": """A concise way to create lists.

Example:
squares = [x*x for x in range(5)]
print(squares)
"""
    },

    {
        "category": "Intermediate Python",
        "question": "12. What are lambda functions?",
        "answer": """Anonymous small functions defined using `lambda`.

Example:
add = lambda a, b: a + b
print(add(3, 5))
"""
    },

    {
        "category": "Intermediate Python",
        "question": "13. What are modules in Python?",
        "answer": """Modules are files containing Python code that can be imported.

Example:
# math_utils.py
def square(x):
    return x * x

# main.py
from math_utils import square
print(square(4))
"""
    },

    {
        "category": "Intermediate Python",
        "question": "14. What is exception handling?",
        "answer": """Handling errors gracefully using try-except.

Example:
try:
    x = 1 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")
"""
    },

    {
        "category": "Intermediate Python",
        "question": "15. What are classes and objects?",
        "answer": """Classes are blueprints, objects are instances.

Example:
class Person:
    def __init__(self, name):
        self.name = name

p = Person('Ali')
print(p.name)
"""
    },

    {
        "category": "Intermediate Python",
        "question": "16. What is inheritance?",
        "answer": """Child class inherits from parent class.

Example:
class Animal:
    def speak(self):
        return "Sound"

class Dog(Animal):
    def speak(self):
        return "Bark"

print(Dog().speak())
"""
    },

    {
        "category": "Intermediate Python",
        "question": "17. How to read and write files?",
        "answer": """Use `open()` function.

Example:
with open('file.txt', 'w') as f:
    f.write('Hello')

with open('file.txt', 'r') as f:
    print(f.read())
"""
    },

    {
        "category": "Intermediate Python",
        "question": "18. What are decorators?",
        "answer": """Functions that modify behavior of other functions.

Example:
def decorator(func):
    def wrapper():
        print('Before call')
        func()
        print('After call')
    return wrapper

@decorator
def say_hello():
    print('Hello')

say_hello()
"""
    },

    {
        "category": "Intermediate Python",
        "question": "19. What are generators?",
        "answer": """Functions that yield values lazily using `yield`.

Example:
def count_up_to(n):
    count = 1
    while count <= n:
        yield count
        count += 1

for num in count_up_to(5):
    print(num)
"""
    },

    {
        "category": "Intermediate Python",
        "question": "20. How to use map(), filter(), reduce()?",
        "answer": """Functional tools to process iterables.

Example:
nums = [1, 2, 3, 4, 5]

# map squares
squares = list(map(lambda x: x*x, nums))

# filter even
evens = list(filter(lambda x: x%2==0, nums))

from functools import reduce
sum_all = reduce(lambda a, b: a+b, nums)

print(squares, evens, sum_all)
"""
    },

    # Advanced Python
    {
        "category": "Advanced Python",
        "question": "21. What is regular expression (regex)?",
        "answer": """Pattern matching strings.

Example:
import re
text = "My number is 1234"
pattern = r'\\d+'
matches = re.findall(pattern, text)
print(matches)
"""
    },

    {
        "category": "Advanced Python",
        "question": "22. What are context managers?",
        "answer": """Manage resources automatically using `with`.

Example:
with open('file.txt', 'r') as f:
    data = f.read()
"""
    },

    {
        "category": "Advanced Python",
        "question": "23. What is multithreading?",
        "answer": """Running multiple threads concurrently.

Example:
import threading

def worker():
    print("Worker running")

thread = threading.Thread(target=worker)
thread.start()
thread.join()
"""
    },

    {
        "category": "Advanced Python",
        "question": "24. What is multiprocessing?",
        "answer": """Running multiple processes in parallel.

Example:
from multiprocessing import Process

def worker():
    print("Worker running")

p = Process(target=worker)
p.start()
p.join()
"""
    },

    {
        "category": "Advanced Python",
        "question": "25. What are Python's built-in data structures?",
        "answer": """Lists, tuples, dictionaries, sets.

Example:
lst = [1,2,3]
tpl = (1,2,3)
dct = {'a':1}
st = {1,2,3}
"""
    },

    # AI Basics
    {
        "category": "AI Basics",
        "question": "26. What is Artificial Intelligence (AI)?",
        "answer": """AI is the simulation of human intelligence processes by machines.

Example: Voice assistants like Siri, Alexa.
"""
    },

    {
        "category": "AI Basics",
        "question": "27. What is Machine Learning?",
        "answer": """A subset of AI where machines learn from data.

Example: Email spam filters.
"""
    },

    {
        "category": "AI Basics",
        "question": "28. What is Deep Learning?",
        "answer": """A subset of ML using neural networks with many layers.

Example: Image recognition.
"""
    },

    {
        "category": "AI Basics",
        "question": "29. What is supervised learning?",
        "answer": """Training models on labeled data.

Example: Classifying emails as spam or not.
"""
    },

    {
        "category": "AI Basics",
        "question": "30. What is unsupervised learning?",
        "answer": """Training models on unlabeled data.

Example: Customer segmentation.
"""
    },

    {
        "category": "AI Basics",
        "question": "31. What is reinforcement learning?",
        "answer": """Learning by trial and error with rewards.

Example: Training a game AI.
"""
    },

    {
        "category": "AI Basics",
        "question": "32. What are neural networks?",
        "answer": """Computational models inspired by the human brain.

Example: Used in image recognition.
"""
    },

    {
        "category": "AI Basics",
        "question": "33. What is NLP?",
        "answer": """Natural Language Processing allows machines to understand human language.

Example: Chatbots.
"""
    },

    {
        "category": "AI Basics",
        "question": "34. What is TensorFlow?",
        "answer": """An open-source library for machine learning.

Example:
import tensorflow as tf
"""
    },

    {
        "category": "AI Basics",
        "question": "35. What is PyTorch?",
        "answer": """An open-source ML library developed by Facebook.

Example:
import torch
"""
    },

    # AI with Python
    {
        "category": "AI with Python",
        "question": "36. How do you install AI libraries in Python?",
        "answer": """Use pip to install libraries like TensorFlow, PyTorch, scikit-learn.

Example:
pip install tensorflow
pip install torch
"""
    },

    {
        "category": "AI with Python",
        "question": "37. How to load datasets in Python?",
        "answer": """Use libraries like pandas or sklearn.

Example:
import pandas as pd
data = pd.read_csv('data.csv')
print(data.head())
"""
    },

    {
        "category": "AI with Python",
        "question": "38. How to preprocess data for ML?",
        "answer": """Clean, normalize, and prepare data.

Example:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
"""
    },

    {
        "category": "AI with Python",
        "question": "39. How to split data into train and test sets?",
        "answer": """Use sklearn's train_test_split.

Example:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
"""
    },

    {
        "category": "AI with Python",
        "question": "40. How to train a model in Python?",
        "answer": """Use sklearn estimators.

Example:
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
"""
    },

    # Continue similarly for rest till 100...
    # For brevity, only first 40 are shown here in full detail.
    # Would you like me to continue and provide full 100 questions with answers & code examples?

]
questions += [
    # AI with Python (continued)
    {
        "category": "AI with Python",
        "question": "41. How do you evaluate a machine learning model?",
        "answer": """Evaluation metrics depend on the task (classification, regression).

Examples:
- Classification: accuracy, precision, recall, F1-score
- Regression: mean squared error (MSE), R² score

Example code (classification accuracy):
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
"""
    },

    {
        "category": "AI with Python",
        "question": "42. What is overfitting and how to prevent it?",
        "answer": """Overfitting happens when a model learns noise in training data and performs poorly on new data.

Prevention:
- Use more data
- Use regularization (L1, L2)
- Use cross-validation
- Prune model complexity

Example (L2 regularization in logistic regression):
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty='l2')
"""
    },

    {
        "category": "AI with Python",
        "question": "43. What is cross-validation?",
        "answer": """Cross-validation splits data into folds to train and test multiple times, improving model reliability.

Example:
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(scores)
"""
    },

    {
        "category": "AI with Python",
        "question": "44. How do you save and load ML models in Python?",
        "answer": """Use joblib or pickle.

Example:
import joblib

joblib.dump(model, 'model.pkl')
loaded_model = joblib.load('model.pkl')
"""
    },

    {
        "category": "AI with Python",
        "question": "45. What is a neural network?",
        "answer": """A neural network is a series of layers with interconnected nodes (neurons) that can learn patterns from data.

Example (simple Keras model):
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dense(1, activation='sigmoid')
])
"""
    },

    {
        "category": "AI with Python",
        "question": "46. How to use activation functions in neural networks?",
        "answer": """Activation functions introduce non-linearity.

Common ones:
- ReLU: `relu`
- Sigmoid: `sigmoid`
- Tanh: `tanh`

Example in Keras:
Dense(64, activation='relu')
"""
    },

    {
        "category": "AI with Python",
        "question": "47. What is backpropagation?",
        "answer": """Backpropagation is the algorithm to update neural network weights by calculating gradients via chain rule.

It helps the network learn from errors.
"""
    },

    {
        "category": "AI with Python",
        "question": "48. What is natural language processing (NLP)?",
        "answer": """NLP enables machines to understand, interpret, and generate human language.

Example tasks: sentiment analysis, translation, chatbots.
"""
    },

    {
        "category": "AI with Python",
        "question": "49. How to use the Hugging Face Transformers library?",
        "answer": """Transformers provide pre-trained state-of-the-art NLP models.

Example:
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
print(classifier('I love Python!'))
"""
    },

    {
        "category": "AI with Python",
        "question": "50. How to build a simple chatbot in Python?",
        "answer": """Basic chatbot uses conditional statements or AI models.

Example (rule-based):
def chatbot(input_text):
    if 'hello' in input_text.lower():
        return 'Hi! How can I help you?'
    return 'Sorry, I didn\'t understand.'

print(chatbot('hello'))
"""
    },

    # Python Advanced Concepts
    {
        "category": "Advanced Python",
        "question": "51. What are metaclasses in Python?",
        "answer": """Metaclasses define how classes behave.

Example:
class Meta(type):
    def __new__(cls, name, bases, dct):
        print('Creating class', name)
        return super().__new__(cls, name, bases, dct)

class MyClass(metaclass=Meta):
    pass

# Output: Creating class MyClass
"""
    },

    {
        "category": "Advanced Python",
        "question": "52. What is the GIL (Global Interpreter Lock)?",
        "answer": """GIL is a mutex that protects access to Python objects, preventing multiple native threads from executing Python bytecodes at once.

This affects multi-threaded CPU-bound programs.

Example: Use multiprocessing to bypass GIL.
"""
    },

    {
        "category": "Advanced Python",
        "question": "53. What are coroutines and async programming?",
        "answer": """Async programming allows concurrent execution without threads.

Example:
import asyncio

async def say_hello():
    await asyncio.sleep(1)
    print('Hello')

asyncio.run(say_hello())
"""
    },

    {
        "category": "Advanced Python",
        "question": "54. How do you use type hinting?",
        "answer": """Type hints improve code readability and help with static analysis.

Example:
def greet(name: str) -> str:
    return 'Hello ' + name
"""
    },

    {
        "category": "Advanced Python",
        "question": "55. What are Python's magic methods?",
        "answer": """Special methods with double underscores that customize class behavior.

Example:
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

v1 = Vector(1, 2)
v2 = Vector(3, 4)
v3 = v1 + v2  # uses __add__
"""
    },

    {
        "category": "Advanced Python",
        "question": "56. What are descriptors?",
        "answer": """Objects that manage the access to attributes via __get__, __set__, and __delete__.

Example:
class Descriptor:
    def __get__(self, instance, owner):
        return 'Getting attribute'

class MyClass:
    attr = Descriptor()

print(MyClass().attr)
"""
    },

    {
        "category": "Advanced Python",
        "question": "57. How to use the `with` statement and context managers?",
        "answer": """`with` ensures resources are properly acquired and released.

Example:
with open('file.txt', 'r') as f:
    data = f.read()
"""
    },

    {
        "category": "Advanced Python",
        "question": "58. How to create custom exceptions?",
        "answer": """Subclass Exception class.

Example:
class MyError(Exception):
    pass

try:
    raise MyError('Error occurred')
except MyError as e:
    print(e)
"""
    },

    {
        "category": "Advanced Python",
        "question": "59. What are Python decorators and how do you create them?",
        "answer": """Functions that modify other functions.

Example:
def decorator(func):
    def wrapper():
        print('Before call')
        func()
        print('After call')
    return wrapper

@decorator
def say_hi():
    print('Hi')

say_hi()
"""
    },

    {
        "category": "Advanced Python",
        "question": "60. How do you profile Python code performance?",
        "answer": """Use `cProfile` module.

Example:
import cProfile

def my_func():
    sum = 0
    for i in range(1000):
        sum += i

cProfile.run('my_func()')
"""
    },

]
questions += [
    # Advanced Python continued
    {
        "category": "Advanced Python",
        "question": "61. What is monkey patching in Python?",
        "answer": """Monkey patching means modifying or extending code at runtime.

Example:
class A:
    def greet(self):
        return 'Hello'

def new_greet(self):
    return 'Hi!'

A.greet = new_greet
print(A().greet())  # Output: Hi!
"""
    },

    {
        "category": "Advanced Python",
        "question": "62. How do Python generators work?",
        "answer": """Generators yield values one at a time, saving memory.

Example:
def gen():
    for i in range(3):
        yield i

for value in gen():
    print(value)
"""
    },

    {
        "category": "Advanced Python",
        "question": "63. What are Python closures?",
        "answer": """Functions that remember values from their enclosing scopes.

Example:
def outer(x):
    def inner(y):
        return x + y
    return inner

add5 = outer(5)
print(add5(3))  # Output: 8
"""
    },

    {
        "category": "Advanced Python",
        "question": "64. Explain Python’s `nonlocal` keyword.",
        "answer": """`nonlocal` allows modifying variables in the nearest enclosing scope.

Example:
def outer():
    x = 5
    def inner():
        nonlocal x
        x = 10
    inner()
    return x

print(outer())  # Output: 10
"""
    },

    {
        "category": "Advanced Python",
        "question": "65. What is the difference between `deepcopy` and `copy`?",
        "answer": """`copy` makes a shallow copy, references nested objects; `deepcopy` copies everything recursively.

Example:
import copy
a = [[1, 2]]
b = copy.copy(a)
c = copy.deepcopy(a)
"""
    },

    {
        "category": "Advanced Python",
        "question": "66. How to use the `collections` module?",
        "answer": """Provides specialized container datatypes.

Example (Counter):
from collections import Counter
cnt = Counter(['a', 'b', 'a'])
print(cnt)  # Output: Counter({'a': 2, 'b': 1})
"""
    },

    {
        "category": "Advanced Python",
        "question": "67. What are Python's built-in data structures?",
        "answer": """List, tuple, set, dict, str.

Example:
my_list = [1,2]
my_dict = {'a': 1}
"""
    },

    {
        "category": "AI with Python",
        "question": "68. How to preprocess data for AI models in Python?",
        "answer": """Common steps: cleaning, normalization, encoding.

Example:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
"""
    },

    {
        "category": "AI with Python",
        "question": "69. What is transfer learning?",
        "answer": """Reusing a pretrained model on a new related task.

Example: Using pretrained ResNet for image classification.
"""
    },

    {
        "category": "AI with Python",
        "question": "70. How to use TensorFlow for building AI models?",
        "answer": """TensorFlow is a library for ML and AI.

Example:
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
"""
    },

    {
        "category": "AI with Python",
        "question": "71. What is reinforcement learning?",
        "answer": """A learning paradigm where agents learn to make decisions by rewards and penalties.

Example: Training an AI to play games.
"""
    },

    {
        "category": "AI with Python",
        "question": "72. How to perform sentiment analysis with Python?",
        "answer": """Using NLP libraries like TextBlob or Hugging Face.

Example (TextBlob):
from textblob import TextBlob
text = 'I love Python!'
blob = TextBlob(text)
print(blob.sentiment)
"""
    },

    {
        "category": "AI with Python",
        "question": "73. What are embeddings in AI?",
        "answer": """Vector representations of text or images capturing semantic information.

Example: Word2Vec embeddings for words.
"""
    },

    {
        "category": "AI with Python",
        "question": "74. How to build a recommendation system in Python?",
        "answer": """Using collaborative or content-based filtering.

Example: Using Surprise library.
"""
    },

    {
        "category": "AI with Python",
        "question": "75. What is overfitting in AI and how to avoid it?",
        "answer": """When a model learns noise instead of patterns.

Solutions: regularization, more data, early stopping.
"""
    },

    {
        "category": "AI with Python",
        "question": "76. How to deploy a machine learning model with Flask?",
        "answer": """Create a Flask app, load the model, and provide prediction routes.

Example:
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data['input']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run()
"""
    },

    {
        "category": "Advanced Python",
        "question": "77. How to manage dependencies in Python projects?",
        "answer": """Use virtual environments and requirements.txt.

Example:
python -m venv venv
pip install -r requirements.txt
"""
    },

    {
        "category": "Advanced Python",
        "question": "78. How to debug Python code?",
        "answer": """Use print statements or pdb module.

Example:
import pdb; pdb.set_trace()
"""
    },

    {
        "category": "Advanced Python",
        "question": "79. What are Python’s iterators and iterables?",
        "answer": """Iterable: object you can loop over.

Iterator: object with __next__ method.

Example:
lst = [1,2,3]
it = iter(lst)
print(next(it))
"""
    },

    {
        "category": "Advanced Python",
        "question": "80. How to handle concurrency in Python?",
        "answer": """Use threading for I/O-bound, multiprocessing for CPU-bound tasks.

Example (threading):
import threading
def task():
    print('Running task')
thread = threading.Thread(target=task)
thread.start()
"""
    },
]
questions += [
    {
        "category": "Advanced Python",
        "question": "81. What is the Global Interpreter Lock (GIL) in Python?",
        "answer": """GIL is a mutex that protects access to Python objects, preventing multiple native threads from executing Python bytecodes simultaneously. It can be a bottleneck in CPU-bound multithreaded programs.

Example:
# GIL limits true parallel execution in threads
import threading

def count():
    x = 0
    for _ in range(1000000):
        x += 1

threads = [threading.Thread(target=count) for _ in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()
"""
    },

    {
        "category": "Advanced Python",
        "question": "82. What are Python metaclasses?",
        "answer": """Metaclasses are 'classes of classes' that define how classes behave.

Example:
class Meta(type):
    def __new__(cls, name, bases, attrs):
        print(f'Creating class {name}')
        return super().__new__(cls, name, bases, attrs)

class MyClass(metaclass=Meta):
    pass

# Output: Creating class MyClass
"""
    },

    {
        "category": "AI with Python",
        "question": "83. What is natural language processing (NLP)?",
        "answer": """NLP is the field of AI that gives machines the ability to read, understand and derive meaning from human language.

Example:
Using NLTK or spaCy for text processing.
"""
    },

    {
        "category": "AI with Python",
        "question": "84. How to use Hugging Face transformers in Python?",
        "answer": """Hugging Face provides pretrained models for NLP tasks.

Example:
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
print(classifier("I love Python!"))
"""
    },

    {
        "category": "AI with Python",
        "question": "85. What is deep learning?",
        "answer": """Deep learning is a subset of machine learning using neural networks with many layers.

Example: CNNs for image recognition.
"""
    },

    {
        "category": "AI with Python",
        "question": "86. How to train a neural network with PyTorch?",
        "answer": """PyTorch is a popular deep learning library.

Example:
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

net = Net()
x = torch.randn(1, 10)
print(net(x))
"""
    },

    {
        "category": "AI with Python",
        "question": "87. What is a confusion matrix?",
        "answer": """A table to evaluate classification models showing TP, FP, TN, FN.

Example:
from sklearn.metrics import confusion_matrix
y_true = [0,1,0,1]
y_pred = [0,0,0,1]
print(confusion_matrix(y_true, y_pred))
"""
    },

    {
        "category": "AI with Python",
        "question": "88. What is data augmentation?",
        "answer": """Techniques to increase training data size by modifying existing data.

Example: flipping images, synonym replacement in text.
"""
    },

    {
        "category": "AI with Python",
        "question": "89. How to evaluate AI model performance?",
        "answer": """Use metrics like accuracy, precision, recall, F1 score.

Example:
from sklearn.metrics import accuracy_score
accuracy_score(y_true, y_pred)
"""
    },

    {
        "category": "AI with Python",
        "question": "90. What is a confusion matrix?",
        "answer": """A matrix showing actual vs predicted classifications for evaluating models.

Example: sklearn's confusion_matrix function.
"""
    },

    {
        "category": "AI with Python",
        "question": "91. What is gradient descent?",
        "answer": """Optimization algorithm to minimize loss by updating weights.

Example:
weights = weights - learning_rate * gradient
"""
    },

    {
        "category": "AI with Python",
        "question": "92. What are activation functions?",
        "answer": """Functions applied on neural network nodes to introduce non-linearity.

Examples: ReLU, Sigmoid, Tanh.
"""
    },

    {
        "category": "AI with Python",
        "question": "93. Explain bias and variance in machine learning.",
        "answer": """Bias: error due to oversimplification.

Variance: error due to model complexity.

Trade-off affects model performance.
"""
    },

    {
        "category": "AI with Python",
        "question": "94. How does dropout work in neural networks?",
        "answer": """Randomly ignores neurons during training to prevent overfitting.

Example: `torch.nn.Dropout` in PyTorch.
"""
    },

    {
        "category": "AI with Python",
        "question": "95. What is batch normalization?",
        "answer": """Technique to stabilize and accelerate training by normalizing layer inputs.

Example: `tf.keras.layers.BatchNormalization`.
"""
    },

    {
        "category": "AI with Python",
        "question": "96. What is reinforcement learning environment?",
        "answer": """The setting or world where an agent learns via actions and rewards.

Example: OpenAI Gym environments.
"""
    },

    {
        "category": "AI with Python",
        "question": "97. How to use OpenAI GPT models via API?",
        "answer": """Send prompt messages, receive generated completions.

Example:
import openai
openai.api_key = 'your_key'
response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[{"role": "user", "content": "Hello"}]
)
print(response['choices'][0]['message']['content'])
"""
    },

    {
        "category": "AI with Python",
        "question": "98. What is the difference between AI, ML, and DL?",
        "answer": """AI: machines performing intelligent tasks.

ML: algorithms learning from data.

DL: deep neural networks, subset of ML.
"""
    },

    {
        "category": "AI with Python",
        "question": "99. How to save and load ML models in Python?",
        "answer": """Use `pickle` or libraries like `joblib`.

Example:
import joblib
joblib.dump(model, 'model.pkl')
model = joblib.load('model.pkl')
"""
    },

    {
        "category": "AI with Python",
        "question": "100. What is the difference between supervised and unsupervised learning?",
        "answer": """Supervised: learns from labeled data.

Unsupervised: finds patterns in unlabeled data.

Example: classification vs clustering.
"""
    },
]
