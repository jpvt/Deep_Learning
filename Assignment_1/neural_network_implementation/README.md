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
