''' 
This is thhe script for the server side of a Federated Learning application using NVFlare.

What I intend to do here is to have a simple code where the client would receive a number of levels to print for the Fibnoacci Series,
and then it would compute the Fibonacci Series up to that number of levels, and send the result back to the server.
'''

import os
import numpy as np
from nvflare import FedJob
from nvflare.app_common.np.np_model_persistor import NPModelPersistor
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner


if __name__ == "__main__":
    n_clients = 5
    num_rounds = 5
    train_script = "src/fibonacci_client_fl.py"

    job = FedJob(name="fibonacci_fl")

    # Create a directory for the model file
    model_dir = "/tmp/nvflare/jobs/workdir/server/simulate_job/models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Create your custom initial model
    fibonacci_level = 10  # Set your desired level
    custom_model = np.array([[fibonacci_level]], dtype=np.float32)
    
    # Save it to the expected path
    model_file = f"{model_dir}/server.npy"
    np.save(model_file, custom_model)
    print(f"Saved custom initial model to {model_file}")
    
    # Create the persistor and explicitly set the initial model
    persistor = NPModelPersistor()
    persistor.initial_model = {"numpy_key": custom_model}
    
    persistor_id = job.to_server(persistor, "persistor")

    # Define the controller workflow
    controller = FedAvg(
        num_clients=n_clients,
        num_rounds=num_rounds,
        persistor_id=persistor_id,
    )
    job.to(controller, "server")

    job.to(IntimeModelSelector(key_metric="sum"), "server")

    # Add clients
    for i in range(n_clients):
        executor = ScriptRunner(script=train_script, script_args="", framework=FrameworkType.NUMPY)
        job.to(executor, f"site-{i + 1}")

    # Run the simulation
    job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")
