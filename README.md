# saspt

State arrays for single particle tracking

`saspt` is a simple Python tool that identifies subpopulations in noisy single particle tracking (SPT) data. It is **under active development**. See the [Documentation](https://saspt.readthedocs.io/en/latest/) or [User Guide](https://github.com/alecheckert/saspt/blob/main/UserGuide.pdf) for more info.

## Install with pip

Currently in testing:
```
    python3 -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ saspt-test==0.1.5
```

## Install from source

If using `conda`:
```
    # Clone the saspt repo
    git clone https://github.com/alecheckert/saspt.git; cd saspt

    # Create the saspt_env conda environment
    conda env create -f example_env.yml
   
    # Switch to the saspt_env conda environment
    conda activate saspt_env

    # Install saspt
    pip install .
```
