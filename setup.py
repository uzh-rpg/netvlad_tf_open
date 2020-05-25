try:
    ## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

    from catkin_pkg.python_setup import generate_distutils_setup
    from distutils.core import setup

    # fetch values from package.xml
    setup_args = generate_distutils_setup(
        packages=['netvlad_tf'],
        package_dir={'': 'python'},
    )

    setup(**setup_args)
except ImportError:
    from setuptools import setup
    setup(
        name='netvlad_tf',
        packages=['netvlad_tf'],
        package_dir={'': 'python'},
        version='0.0.0'
    )

