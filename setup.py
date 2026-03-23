from setuptools import setup, find_packages

setup(
    name="videoseek-cli",
    version="1.0.0",
    description="CLI tool for extracting key video moments via natural language queries",
    author="NEO",
    packages=find_packages(),
    install_requires=[
        "opencv-python-headless>=4.8.0,<5.0.0",
        "numpy>=1.24.0,<2.0.0",
        "Pillow>=10.0.0,<11.0.0",
        "chromadb>=0.4.18,<0.5.0",
        "fastapi>=0.109.0,<1.0.0",
        "uvicorn>=0.27.0,<1.0.0",
        "httpx>=0.26.0,<1.0.0",
        "click>=8.1.0,<9.0.0",
        "rich>=13.0.0,<14.0.0",
        "python-dotenv>=1.0.0,<2.0.0",
        "pydantic>=2.5.0,<3.0.0",
        "gradio>=4.44.0,<5.0.0",
    ],
    entry_points={
        "console_scripts": [
            "videoseek-cli=videoseek_cli.cli:main",
        ]
    },
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
