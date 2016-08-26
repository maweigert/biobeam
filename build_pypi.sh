rm dist/* build/*

python setup.py sdist bdist_wheel

#python setup.py register -r pypitest
#twine upload -r pypitest dist/biobeam*

python setup.py register
twine upload dist/biobeam*
