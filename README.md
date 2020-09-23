# Adaptive REBAlancing (AREBA)

### Motivation
An enormous and ever-growing volume of data is nowadays becoming available in a sequential fashion in various real-world applications. Learning in nonstationary environments constitutes a major challenge, and this problem becomes orders of magnitude more complex in the presence of class imbalance. We provide new insights into learning from nonstationary and imbalanced data in online learning, a largely unexplored area. We propose the novel **Adaptive REBAlancing (AREBA)** algorithm that selectively includes in the training set a subset of the majority and minority examples that appeared so far, while at its heart lies an adaptive rebalancing mechanism to continually maintain class balance between the selected examples. We compare AREBA to strong baselines and other state-of-the-art algorithms and perform extensive experimental work in scenarios with various class imbalance rates and different concept drift types on both synthetic and real-world data. AREBA significantly outperforms the rest with respect to both learning speed and learning quality. Our code is made publicly available to the scientific community.

### Paper PDF
You can download the AREBA paper from one of these links:
- https://ieeexplore.ieee.org/document/9203853
- https://doi.org/10.1109/TNNLS.2020.3017863

A pre-print (i.e. before the editorial changes) version of the paper is available here:
- TBA (arXiv)
- TBA (Zenodo)

### Citation request
If you have found part of our work useful please cite it as follows:

K. Malialis, C. G. Panayiotou and M. M. Polycarpou, "Online Learning With Adaptive Rebalancing in Nonstationary Environments," in IEEE Transactions on Neural Networks and Learning Systems, doi: 10.1109/TNNLS.2020.3017863.

K. Malialis, C. Panayiotou and M. M. Polycarpou, Queue-based resampling for online class imbalance learning, in Proceedings of the 27th International Conference on Artificial Neural Networks (ICANN), 2018.

### Instructions
Instructions on how to reproduce the experiments in the paper are given in the *main_synthetic.py* file.

### Software
The code has been generated and tested with the following:
- Python 3.7
- tensorflow 1.13.2
- Keras 2.2.4
- numpy 1.17.4
- pandas 0.25.3

### Contact
For any questions or issues please contact [Kleanthis Malialis](https://malialis.bitbucket.io/).
