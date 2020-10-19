## SETUP

* To install all the packages used in this repository:

```shell
> pip install -r requirements.txt
```

* To install only the Neural Network from scratch package:

```shell
> pip install tail-scratch-nn==1.0
```

## Importing Implementation

```python
from tail_scratch_nn import DNN

nn = DNN()

nn.fit(X,y)

nn.train([X.shape[0], y.shape[0]], iterations=100, learning_rate=0.1, adam_optimizer=True)
```
