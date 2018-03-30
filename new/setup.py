import setuptools

setuptools.setup(
    name='titanfp',
    version='0.1.1',
    description='floating point analysis with fpbench',
    packages=['titanic', 'fpbench'],
    package_data={
        'fpbench' : ['antlr/Makefile', 'antlr/*.g4', '*.interp', '*.tokens'],
    },
)
