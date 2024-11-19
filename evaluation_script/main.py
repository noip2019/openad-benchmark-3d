import random
from evalai_2d import main

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    output = main(test_annotation_file, user_submission_file)
    return output
