''' 
This is thhe script for the client side of a Federated Learning application using NVFlare.

What I intend to do here is to have a simple code where the client would receive a number of levels to print for the Fibnoacci Series,
and then it would compute the Fibonacci Series up to that number of levels, and send the result back to the server.

'''

import numpy as np
import nvflare.client as flare # Importing the NVFlare client library for Federated Learning


import numpy as np

def train(input_level):  
    '''
    This is to illustrate the training process in a Federated Learning context.
    The input_level is expected to be a string representing the number of levels for the Fibonacci series.
    The Fibonacci series is computed up to the given level, and the result is returned as a numpy array back to the server.
    
    EXAMPLE:
    Pyramid of 2 Lines:
                    0
                1       1
    
    Pyramid of 4 Lines:
                    0
                1       1
            2       3       5
        8      13      21      34

    Pyramid of 6 Lines:
                    0
                1       1
            2       3       5
        8      13      21      34
    55      89      144     233     377
610     987     1597    2584    4181    6765
    '''

    def fibonacci_pyramid(lines):
        # Generate enough Fibonacci numbers for the pyramid
        total_numbers = lines * (lines + 1) // 2
        fib = [0, 1]
        while len(fib) < total_numbers:
            fib.append(fib[-1] + fib[-2])

        # Print the pyramid
        idx = 0
        max_width = len(str(fib[-1])) + 1
        for i in range(1, lines + 1):
            padding = " " * max_width * (lines - i)
            print(padding, end="")
            for j in range(i):
                print(f"{fib[idx]:>{max_width}}", end=" ")
                idx += 1
            print()
        
        return fib  # Return the list for array conversion

    # Convert input level to integer
    level = int(input_level)

    # Generate Fibonacci series and pyramid
    fib_series = fibonacci_pyramid(level)

    # Convert to numpy array for consistency with ML/DL output format
    output_arr = np.array(fib_series, dtype=np.float32)
    print(f"\nComputed Fibonacci series up to level {level}: {output_arr}")

    return output_arr



def evaluate(input_arr):
    '''
    Simulates the evaluation phase in a federated learning context.
    Instead of evaluating a model, this function returns the sum of the values 
    in the input array (e.g., Fibonacci series from training).
    
    Parameters:
    - input_arr (np.ndarray): The numeric values in the Fibonacci series

    Returns: the sum of all the numbers in the series.
    '''
    
    # This part illustrates what we would have done if we had a model to evaluate.
    # Here, we treat the array as a stand-in for learned parameters or outputs.

    return np.sum(input_arr)

def main():
    
    
    # The main code goes here.
    flare.init() # Initializing the NVFlare client

    sys_info = flare.system_info() # I think this gets the system information for the client.
    print(f"system info is: {sys_info}", flush=True) # Comment out to see the effect.

    while flare.is_running():
        input_model = flare.receive() # Receive the number of lines to print the Fibonacci series.
        
        # print(f"current_round={input_model.current_round}") # Leaving this one to see the current round of the FL process.
        print(f"received input lines: {input_model.params}")  # This should be the number of lines to print.
        # print(f"received weights: {input_model.params}")

        sys_info = flare.system_info()
        print(f"system info is: {sys_info}") # Comment out to see the effect.

        # Extract the Fibonacci level from parameters
        if input_model.params == {}:
            level = 4  # Default
        else:
            level = int(input_model.params.get("fibonacci_level", 4))

        print(f"Using Fibonacci level: {level}")

        # Generate Fibonacci series and pyramid
        fibonacci_array = train(level)

        # Evaluate the generated Fibonacci series
        total_sum = evaluate(fibonacci_array)
        
        print(f"Sum of Fibonacci numbers: {total_sum}")
        print(f"finished round: {input_model.current_round}", flush=True)

        # Send results back to server
        output_model = flare.FLModel(
            params={"numpy_key": fibonacci_array},
            params_type="FULL",
            metrics={"sum": float(total_sum)},
            current_round=input_model.current_round,
        )

        flare.send(output_model)


if __name__ == "__main__":
    main()
