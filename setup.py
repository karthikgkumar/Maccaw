from setuptools import setup, find_packages

setup(
    name="macaw",
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
    author_email="karthikgkumar.pro@gmail.com",
    description="A very light weight library to create and run agents, with any LLM supported openai spec, Claude, Mistral.",
    long_description="A longer description, possibly with usage examples",
    long_description_content_type="text/markdown",
    license="MIT",
    # url='https://github.com/your_username/your_package_name',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers, Coders, Hackers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
