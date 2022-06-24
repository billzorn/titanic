import setuptools

with open('README.md', 'rt') as f:
    long_description = f.read()

setuptools.setup(
    name='titanfp',
    version='0.0.0',
    author='Bill Zorn',
    author_email='billzorn@cs.washington.edu',
    url='https://github.com/billzorn/titanic',
    description='number systems research with FPBench, including implementation of posits',
    long_description=long_description,
    license='MIT',
    install_requires=['numpy>=1.23.0', 'gmpy2>=2.1.2', 'antlr4-python3-runtime~=4.9.3'],
    packages=['titanfp', 'titanfp/titanic', 'titanfp/fpbench', 'titanfp/arithmetic'],
    package_data={
        'titanfp/fpbench' : ['antlr/Makefile', 'antlr/*.g4', '*.interp', '*.tokens'],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Operating System :: POSIX :: Linux',
        'License :: OSI Approved :: MIT License',
    ],
)
