#clean up old dists
rm dist/*

python2 setup.py bdist_wheel
python3 setup.py bdist_wheel

twine upload dist/*whl
