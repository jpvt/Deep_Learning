# Neural Network for classification Implementation

## Importing Implementation

```python
from tail_scratch_nn import DNN

nn = DNN()

nn.fit(X_train, y_train, X_val, y_val)

nn.train([X.shape[0], y.shape[0]], iterations=100, learning_rate=0.1, adam_optimizer=True)
```


## get code up and running on Pypi

```python
python setup.py sdist
```

```python
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

```python
twine upload dist/*
```
