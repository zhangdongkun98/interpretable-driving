from setuptools import setup, find_packages

setup(
    name='interpretable-driving',
    packages=find_packages(),
    version='0.0.1',
    author='Zhang Dongkun',
    author_email='zhangdongkun98@gmail.com',
    url='https://github.com/zhangdongkun98/interpretable-driving',
    description='',
    install_requires=[
        'rllib',
        'glvm',
        'carla-utils',
    ],

    include_package_data=True
)
