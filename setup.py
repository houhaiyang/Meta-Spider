from setuptools import setup, find_packages
import datetime

# 获取当前时间
now = datetime.datetime.now()
created=now.strftime('%Y-%m-%d %H:%M:%S')

setup(
    name='metaspider',
    version='0.9.3',
    python_requires='>=3.7',
    author='Haiyang Hou',
    author_email='2868582991@qq.com',
    description=f'{created}The padding value of 0 is the minimum value.',
    keywords='microorganism, metagenomics, network, sample-specificity',
    packages=find_packages(),
    install_requires=['numpy','pandas','joblib','scipy','sklearn','pingouin']
)

# 'scikit-learn'
