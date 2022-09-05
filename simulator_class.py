# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 20:57:07 2022
Simulator class - to track iteractions, func call and objective function
with a callback
@author: vragu
"""
import os
import time


# Callback for the optimizer
# Next idea - learn how to get the callback to print out the objective function
# Use technique from here: https://stackoverflow.com/questions/16739065/how-to-display-progress-of-scipy-optimize-function
class Simulator:
    def __init__(self, function, log_file=None):
        self.f = function  # actual objective function
        self.num_calls = 0  # how many times f has been called
        self.callback_count = 0  # number of times callback has been called, also measures iteration count
        self.min_obj_sofar = None

        # Set up logging
        self.log_file = log_file
        # Remove old log file, if it exists
        if log_file is not None:
            if os.path.exists(log_file):
                os.remove(log_file)
                time.sleep(1)  # Sleep so that the system has time to delete the file

            self.f_handle = open(log_file, 'a')

    def simulate(self, x, *args):
        """ Evaluate the actual objective and returns the result, while
            updating the best value found so far """

        result = self.f(x, *args)  # the actual evaluation of the function
        if not self.num_calls:  # first call
            self.min_obj_sofar = result
        else:
            if self.min_obj_sofar > result:
                self.min_obj_sofar = result

        self.num_calls += 1

        # Write inputs and function to a log file
        if self.log_file is not None:
            self.f_handle.write(f"{list(x)}, {result}\n")
        return result

    def callback(self, x, *_):
        """Callback function that can be used by optimizers of scipy.optimize.
        The third argument "*_" makes sure that it still works when the
        optimizer calls the callback function with more than one argument. Pass
        to optimizer without arguments or parentheses."""
        self.callback_count += 1
        if self.callback_count % 20 == 0:
            print(f'Iterations: {self.callback_count}, Obj evals: {self.num_calls}, Obj: {self.min_obj_sofar}')
