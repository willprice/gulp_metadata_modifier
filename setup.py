import setuptools

setuptools.setup(
    name="gulp_metadata_modifier",  # Replace with your own username
    version="0.0.1",
    author="Will Price",
    author_email="will.price94@gmail.com",
    description="A tool for updating gulp metadata",
    long_description="Update and remove examples from gulp metadata",
    long_description_content_type="text/markdown",
    url="https://github.com/willprice/gulp_metadata_modifier",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["ujson", "pandas"],
    entry_points="""
    [console_scripts]
    gulp_metadata_modifier=gulp_metadata_modifier:main
    """,
)
