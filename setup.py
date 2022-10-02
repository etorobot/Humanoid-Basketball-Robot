from setuptools import setup
setup(
    name='muj_envs',
    version='0.1',
    description="A MuJoCo-Gym Env for Kaleido",
    # packages=setuptools.find_packages(include="drb_module*"),
    install_requires=['gym', 'numpy', 'torch', 'pfrl']  # And any other dependencies foo needs
)
