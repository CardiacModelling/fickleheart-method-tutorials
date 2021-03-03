from setuptools import setup, find_packages

setup(
    name='fickleheart-method-tutorials',
    version='0.0.1',
    description='Fickle Heart method tutorials',
    license='BSD 3-clause license',
    maintainer='Chon Lok Lei',
    maintainer_email='chonloklei@gmail.com',
    install_requires=[
        'myokit==1.28.9',
        'pints==0.2.2',
        'joblib==0.13.2',
        'theano==1.0.4',
        'statsmodels==0.10.1',
    ],
)

