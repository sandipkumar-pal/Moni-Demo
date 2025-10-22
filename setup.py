from setuptools import setup, find_packages

setup(
    name="ml_comparator",
    version="0.1.0",
    description="A Streamlit + FastAPI app for comparing machine learning classifiers",
    author="Sandip Pal",
    author_email="sandipkumar.pal@spglobal.com",  # optional
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.25.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.22.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.2.0",
        "requests>=2.31.0",
        "plotly>=5.15.0",
    ],
    python_requires=">=3.9",
)