"""
Takes a table of values and finds the fraction of them that are in a binary or not
"""
from binary_detection import binary_detection

def binary_fraction(rv_list, error_list):
    assert len(rv_list) == len(error_list)
    detection = 0
    for n in range(len(rv_list)):
        #print(rv_list[n], error_list[n])
        binary = binary_detection(rv_list[n], error_list[n])

        if binary:
            detection += 1
    detection_rate = detection/len(rv_list)
    return detection_rate
