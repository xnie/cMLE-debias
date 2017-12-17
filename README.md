# cMLE-debias
This is the repository for the cMLE algorithm used in paper "Why Adaptively Collected Data Have Negative Bias, and How to Correct for It" by Xinkun Nie, Xiaoying Tian, Jonathan Taylor, and James Zou.
Details see https://arxiv.org/abs/1708.01977

Code tested on Python 2.7.6, requires Numpy, Scipy, and Matplotlib packages

Note that for potential parallel processing, we divide up all the number of repeated trials into D divisions. Run main.py for each of these divisions, and run process_div.py to tally the output. Example usage see run.sh. 

run python main.py --help for usage
run python process_div.py --help for usage
