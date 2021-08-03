# %%
#----build distribution---
#source distribution - just python files written. Files must be downloaded and setup script run
#wheel distribution - slightly processed, can be installed without running setup script. smaller in size, preferred

#good practice to include both when uploading to PyPi
python setup.py sdist bdist_wheel

#upload using twine (from terminal)
twine upload dist/*

#or upload in the test PyPi repository
twine upload -r testpypi dist/*
#first have to register an account with PyPi though

#to install your package run pip
pip install --index-url https://test.pypi.org/simple #where the package is downloaded from
    --extra-index-url https://pypi.org/simple #where pip can search for your dependency packages
    mysklearn

# %% Package Templates
#----terminal command---
#cookiecutter <template-url>
cookiecutter https://github.com/audreyr/cookiecutter-pypackage
cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git