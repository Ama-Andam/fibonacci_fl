# Fibonacci Federated Learning Project with NVIDIA FLARE

I created this project to help me understand how NVIDIA FLARE works by implementing a simple federated learning application. Instead of jumping straight into complex ML models, I decided to start with something more understandable: the Fibonacci sequence.

## Learning Objectives

Through this project, I'm exploring:
- How parameters are passed from server to clients in NVFlare
- The basic federated learning workflow and communication
- How clients process data and return results
- The aggregation of results on the server side

## Project Structure

```
fibonnaci_fl/
├── src/
│   └── fibonacci_client_fl.py      # Client implementation that generates Fibonacci sequences
├── fibonnaci_script_runner.py      # Server-side configuration and job runner
```

## How it Works

The server sends a parameter called `fibonacci_level` to each client. Then:

1. Each client generates a Fibonacci sequence up to the specified level
2. The client displays the sequence in a pyramid pattern (making it easy to visualize)
3. The client calculates the sum of all Fibonacci numbers generated
4. Results are sent back to the server for aggregation

This simple example demonstrates the core concepts of federated learning without the complexity of deep learning models.

## Running the Project

To run the simulation:

```bash
python fibonnaci_script_runner.py
```

This will:
- Launch 5 simulated clients
- Run for 5 rounds
- Generate Fibonacci sequences with 10 levels
- Save results to `/tmp/nvflare/jobs/workdir`

## Implementation Details

In the server script, I'm setting up:
- A model persistor with the Fibonacci level parameter
- The FedAvg controller for coordinating the federated learning process
- A model selector that uses the sum of Fibonacci numbers as the selection metric
- Multiple client executors that run my Fibonacci calculation script

## Next Steps

As I continue learning NVFlare, I plan to:
- Experiment with varying parameters for different clients
- Implement more complex aggregation strategies
- Eventually move to actual machine learning models

## Requirements

You'll need NVIDIA FLARE installed:
```bash
pip install nvflare
```

## Lessons Learned

### What I Expected vs. What Actually Happened

I initially thought I could simply pass a `fibonacci_level` parameter from the server to clients. The reality was quite different:

1. **Parameter Passing**: NVIDIA FLARE doesn't just pass simple variables. Instead, it sends a NumPy array under a specific key (`numpy_key`), and clients must extract values from this array.

2. **Default Values**: I discovered that NPModelPersistor uses a default 3×3 NumPy array when it can't find a model file:
   ```
   [[1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.]]
   ```

3. **Extracting Values**: My clients had to get the Fibonacci level from position [0,0] of this array (which was 1).

4. **Array Shape Changes**: After round 0, the model structure changed from 2D to 1D, causing indexing errors in my client code.

### What I Learned

- NVIDIA FLARE is designed for ML workflows, not simple parameter passing
- The framework has specific expectations about data formats and naming
- Clients must return data using the same key structure the server expects
- Trying to override the initial model is challenging without understanding the framework's internals

This project has been a great first step in understanding federated learning fundamentals through NVIDIA FLARE! That's it! A simple but cool example of federated learning that doesn't involve the usual boring ML stuff. Enjoy!