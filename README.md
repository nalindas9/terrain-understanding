# terrain-understanding
[![License:MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/nalindas9/terrain-understanding/blob/master/LICENSE)

## About
Detects stairs using point cloud data - Kmeans Clustering and Planar Fitting. Flat surface detected based on gradient.

## Output

https://user-images.githubusercontent.com/44141068/155874752-66320811-1e65-4c0e-9387-ee19d74e037e.mp4


## System and library requirements.
 - Python3
 - matplotlib
 - numpy
 - richdem
 - scikit_learn
 - scipy
 
## How to Run
1. Clone this repo. <br>
2. Navigate into the folder `terrain-understanding` <br>
3. Create and activate [Virtual Environment](https://docs.python.org/3/library/venv.html) <br>
4. Install requirements.txt using command `pip install -r requirements.txt`
5. To run the code, from the terminal, run the command `python3 main.py` <br>
6. You should see plots similar to the given examples above. 
7. Voila! Green is steppable, red is not steppable
