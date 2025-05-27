from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ros_visuals'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Required files
        ('share/ament_index/resource_index/packages', ['package.xml']),
        
        # RViz config 
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),

        # Launch files (create launch folder first)
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='djamal',
    maintainer_email='djamal.halim@tum.de',
    description='Package for visualizing cage transforms',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            't11 = ros_visuals.t11:main',
            't12 = ros_visuals.t12:main',
            't13 = ros_visuals.t13:main'
        ],
    },
)