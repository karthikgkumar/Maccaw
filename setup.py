from setuptools import setup, find_packages

setup(
    name="maccaw",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.6.1",
        "requests>=2.31.0",
        "mistralai",
        "anthropic",
        "semantic-text-splitter",
        "openai",
        "pydub",
        # List dependencies here
    ],
    author="Karthik G Kumar",
    author_email="karthikgkumar2002@gmail.com",
    description="A very light weight library to create and run agents, with any LLM supported openai spec, Claude, Mistral.",
    long_description="A longer description, possibly with usage examples",
    long_description_content_type="text/markdown",
    license="MIT",
    # url='https://github.com/your_username/your_package_name',
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
