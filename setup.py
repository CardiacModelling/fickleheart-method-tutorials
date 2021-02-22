from setuptools import setup, find_packages

setup(
    name='fickleheart-method-tutorials',
    version='0.0.1',
    description='Fickle Heart method tutorials',
    license='BSD 3-clause license',
    maintainer='Chon Lok Lei',
    maintainer_email='chonloklei@gmail.com',
    install_requires=[
        'myokit',
        'pints',
        'joblib',
        'theano',
        'statsmodels',
    ],
)

