# python3 -m pip install --upgrade build twine

python -m build

# upload to pypi
# twine check dist/*
# twine upload -r testpypi dist/*
# python -m pip install -i https://test.pypi.org/simple realpython-reader
# twine upload dist/*

