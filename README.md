# saspt

![](doc/_static/logo.png | width=100)

State arrays for single particle tracking

`saspt` is a simple Python tool that identifies subpopulations in noisy single particle tracking (SPT) data. It is **under active development**. See the [Documentation](https://saspt.readthedocs.io/en/latest/) for more info. 

## Install with pip

Currently in testing:
```
python3 -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ saspt-test
```

## Install from source

If using `conda`:
```
# Clone the saspt repo
git clone https://github.com/alecheckert/saspt.git; cd saspt

# Create the saspt_env conda environment
conda env create -f example_env.yaml
   
# Switch to the saspt_env conda environment
conda activate saspt_env

# Install saspt
pip install .
```

## Run tests

We recommend `pytest` or `nose2`. From the top-level repo directory,
run the testing suite with either
```
pytest tests
```

or
```
nose2 tests
```
