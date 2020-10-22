## SETUP

* To install all the packages used in this repository:

```shell
> pip install -r requirements.txt
```

* To install only the Neural Network from scratch package:

```shell
> pip install tail-scratch-nn==1.2
```

## Importing Implementation

```python
from tail_scratch_nn import DNN

nn = DNN()

nn.fit(X_train, y_train, X_val, y_val)

nn.train([X.shape[0], y.shape[0]], iterations=100, learning_rate=0.1, adam_optimizer=True)
```
