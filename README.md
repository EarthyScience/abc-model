# ABC Model
A simple model coupling biosphere and atmosphere made fully differentiable using JAX built up on the [CLASS model](https://github.com/classmodel/modelpy).

## Documentation
Detailed documentation of the model can be accessed [here](https://earthyscience.github.io/abc-model/).

## Installation (MacOS and Linux)
These instructions work on Linux and MacOS and assume that python with pip is installed already. Otherwise, install python and pip with the tool of your choice, such as [miniforge](https://conda-forge.org/download/) or [uv](https://docs.astral.sh/uv/), before you proceed. See below for full instructions to install on Windows.
Install with
```
pip install git@github.com/EarthyScience/abc-model
```

or clone the repo and make an editable install inside your local repo using
```
pip install -e .
```

If you want to use jax on GPUs, change the tag from `[cpu]` to `[gpu]` in an environment with GPUs installed.
This is not necessary to run the examples in this repository.

## Installation (Windows)
On Windows we need some more utilities for jax to work properly, also installing python is not as straigforward. For jax to run, you first need the Microsoft Visual C++ redistributable found [here](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022), which will require a system restart to function. Note that you might want to install uv before the restart, as the PATH update might require a restart as well (see below).

Now clone the repo and cd into it. The following section shows how to set up a python environment with [uv](https://docs.astral.sh/uv/), if you have python with pip running you can skip it.

### UV environment 
First install uv via the terminal with 
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
You can check that uv is available and running by typing `uv` in your terminal, if you receive an error, you will have to add uv to your path manually or [restart your computer](https://github.com/astral-sh/uv/issues/10014). Here, or when executing uv scripts to activate environments windows [execution policiy](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.4#powershell-execution-policies) might stop you, if that is the case you need to change or bypass it.

After uv is installed and running, create a virtual environment in the abc-model directory by running 
```
uv venv --python 3.13.0
```
this will also show you the command needed to activate the venv, which should look similar to
```
.venv\Scripts\activate
```

Lastly, while in the abc-model directory, install the abc-model with uv:
```
uv pip install -e .
```

## See also
For a more advanced model, see [ClimaLand.jl](https://github.com/CliMA/ClimaLand.jl).
