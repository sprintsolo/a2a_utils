from setuptools import setup, find_packages

setup(
    name='a2a_utils',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'a2a-sdk',
        'langchain-core',
        'langchain-openai',
        'langgraph',
        'composio-langchain',
        'composio-core',
        'pydantic',
        'langchain'
    ],
    description='A common library for A2A projects',
    author='sprintsolo', 
    author_email='firstsolo@sprintsolo.dev',
    url='https://github.com/sprintsolo/a2a_utils', 
) 