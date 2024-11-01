from setuptools import setup, find_packages

setup(
    name='GameForecaster',
    version='1.0.0',
    description='A library to predict the success of video games based on various features.',
    long_description=open('README.md').read(),   
    long_description_content_type='text/markdown',   
    author='Si Aziz Bahloul',
    author_email='azizbahloul3@gmail.com',
    url='https://github.com/AzizBahloul/GameForecaster.git',  
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'joblib',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Specify your license
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='game prediction success machine-learning',  # Add relevant keywords
    python_requires='>=3.6',   
)
