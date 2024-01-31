# brp-resnet18

# Table of content

- [brp-resnet18](#%62%72%70%2D%72%65%73%6E%65%74%31%38)
- [Table of content](#%74%61%62%6C%65%2D%6F%66%2D%63%6F%6E%74%65%6E%74)
- [Usage guide](#%75%73%61%67%65%2D%67%75%69%64%65)
    - [Dependencies](#%64%65%70%65%6E%64%65%6E%63%69%65%73)
    - [Init environment](#%69%6E%69%74%2D%65%6E%76%69%72%6F%6E%6D%65%6E%74)
    - [Run tests and model evaluation](#%72%75%6E%2D%74%65%73%74%73%2D%61%6E%64%2D%6D%6F%64%65%6C%2D%65%76%61%6C%75%61%74%69%6F%6E)

# Usage guide

All the needed ressources are availables in this repos, including:
- The ResNet18 image descriptors;
- The ground-truth files.

### Dependencies

- Java JDK;
- Python >= 3.10.

### Init environment

1. (Optional) Initalize a virtual environment:
```sh
py -m venv venv
```
2. (Optional) Activate the venv:
```sh
# On Windows
.\venv\Scripts\activate

# On Unix system
source venv/bin/activate
```
3. Install dependencies:
```sh
pip install -r requirements.txt -r requirements-dev.txt 
```

### Run tests and model evaluation

Once you are in your virtual env (if created) with the dependencies installed, run the following command:

```sh
python -m app
```