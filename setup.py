import setuptools

setuptools.setup(
    name='titanfp',
    version='0.1.1',
    description='floating point analysis with fpbench',
    packages=setuptools.find_packages('.'),
    package_data={
        'titanfp/fpbench' : ['antlr/Makefile', 'antlr/*.g4', '*.interp', '*.tokens'],
    },
)
