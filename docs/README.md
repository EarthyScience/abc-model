# Docs

To generate the API reference for new files, run the following command inside the `docs` directory.

```
sphinx-apidoc -e -t _templates -o source/api ../src
make clean html
```

This also creates a `modules.rst`, but we don't use it... just delete it for now...
