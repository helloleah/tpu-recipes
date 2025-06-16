# Serve Qwen/Qwen2.5-32B with vLLM on Cloud TPU v6e (Trillium) VMs

This guide provides step-by-step instructions to serve the Qwen/Qwen2.5-32B model using vLLM on Cloud TPU v6e (Trillium) virtual machines. By following this recipe, you will deploy the model and be able to send inference requests to it.

The target audience for this guide is ML engineers familiar with Google Cloud, Linux, and Docker.

## Prerequisites

Before you begin, ensure you have the following:

*   **Google Cloud SDK:** You need the `gcloud` command-line tool installed and authenticated. For installation instructions, refer to [Installing the Google Cloud SDK](https://cloud.google.com/sdk/docs/install). After installation, log in to your Google Cloud account by running `gcloud auth login`.
*   **Hugging Face Token:** A Hugging Face token is required to download the model from the Hugging Face Hub. Ensure you have one available. You can find more information on creating tokens on the [Hugging Face website](https://huggingface.co/docs/hub/security-tokens).
*   **Basic Linux and Docker knowledge:** You should be familiar with basic Linux commands and Docker concepts for navigating directories, running scripts, and managing containers.

## Step 1: Create a Cloud TPU v6e instance

This step guides you through creating a Cloud TPU v6e virtual machine (VM). You will set the environment variables `TPU_NAME`, `ZONE`, and `PROJECT` in your local shell where you run `gcloud` commands.

1.  Define environment variables for your Cloud TPU v6e instance configuration:
    ```bash
    export TPU_NAME="your-tpu-name"
    export ZONE="your-tpu-zone"
    export PROJECT="your-gcp-project"
    ```
    Replace `your-tpu-name`, `your-tpu-zone`, and `your-gcp-project` with your desired TPU name, zone, and Google Cloud project ID, respectively.

2.  Execute the command to create the Cloud TPU v6e VM. The following command creates a VM with 4 Trillium chips (using `--accelerator-type=v6e-4` and `--topology=2x2`). If you need a different configuration, adjust these values. For more details on available topologies, refer to [Cloud TPU v6e VM Types](https://cloud.google.com/tpu/docs/v6e#vm-types):
    ```bash
    gcloud alpha compute tpus tpu-vm create ${TPU_NAME} \
        --accelerator-type v6e-4 \
        --project ${PROJECT} \
        --zone ${ZONE} \
        --version v2-alpha-tpuv6e \
        --topology 2x2 # Optional for v6e-4 but good practice
    ```
    This command may take a few minutes to complete. Upon success, `gcloud` will display details of the created Cloud TPU v6e VM.

3.  Once the instance is created, connect to it using SSH. The following command uses the environment variables you defined previously:
    ```bash
    gcloud compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE}
    ```
    You are now connected to your Cloud TPU v6e VM's shell.

## Step 2: Set up the environment and launch vLLM

Execute the following commands on the Cloud TPU v6e VM you connected to in the previous step.

1.  Define the Docker image to use and run the Docker container. You can use `vllm/vllm-tpu:nightly` for the latest vLLM TPU nightly image or specify a pinned image if you require a specific version:
    ```bash
    export DOCKER_URI="vllm/vllm-tpu:nightly" # Or your preferred pinned image, e.g., vllm/vllm-tpu:20240701
    sudo docker run -t --rm --name "${USER}-vllm" --privileged --net=host -v /dev/shm:/dev/shm --shm-size 10gb -p 8000:8000 --entrypoint /bin/bash -it "${DOCKER_URI}"
    ```
    This command starts a Docker container in privileged mode, maps port `8000`, and mounts `/dev/shm` for shared memory. You will now be inside the Docker container's shell.

2.  Inside the container, export your Hugging Face token and set the `HF_HOME` environment variable. The `HF_TOKEN` is necessary for downloading the model.
    ```bash
    export HF_HOME="/dev/shm" # Specifies the cache directory for Hugging Face assets
    export HF_TOKEN="<your HF token>" # Replace <your HF token> with your actual Hugging Face token
    ```

## Step 3: Start the vLLM server

Execute the following commands inside the Docker container on the Cloud TPU v6e VM.

1.  Define environment variables for the vLLM server configuration:
    ```bash
    export MAX_MODEL_LEN=4096 # Maximum sequence length the model can handle
    export TP=4               # Tensor parallelism: number of TPU chips (should match topology, e.g., 4 for 2x2)
    # export RATIO=0.8        # Optional: for benchmark configuration (share of random prompts)
    # export PREFIX_LEN=0     # Optional: for benchmark configuration (fixed prefix length for prompts)
    ```

2.  Start the vLLM server with the Qwen/Qwen2.5-32B model:
    ```bash
    VLLM_USE_V1=1 vllm serve Qwen/Qwen2.5-32B \
        --seed 42 \
        --disable-log-requests \
        --gpu-memory-utilization 0.95 \
        --max-num-batched-tokens 512 \
        --max-num-seqs 512 \
        --tensor-parallel-size ${TP} \
        --max-model-len ${MAX_MODEL_LEN}
    ```
    This command starts the vLLM server. Key parameters include:
    *   `--tensor-parallel-size ${TP}`: Distributes the model across the specified number of TPU chips.
    *   `--max-model-len ${MAX_MODEL_LEN}`: Sets the maximum sequence length for the model.
    *   `--gpu-memory-utilization 0.95`: Instructs vLLM to use 95% of the available TPU memory.

    The server may take a few minutes to download the model and initialize. Keep this terminal open. When the server is ready to accept requests, you will see messages similar to the following in the logs:
    ```text
    INFO:     Started server process [7]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
    ```

## Step 4: Prepare the testing environment

To test the server and run benchmarks, open a new, separate terminal window on your local machine. Do not close the terminal running the vLLM server.

1.  In the new terminal, define the same environment variables for your Cloud TPU v6e instance as in Step 1, if they are not already set in your current shell session:
    ```bash
    export TPU_NAME="your-tpu-name" # Ensure these are the same as in Step 1
    export ZONE="your-tpu-zone"
    export PROJECT="your-gcp-project"
    ```

2.  SSH into the Cloud TPU v6e VM again from this new terminal:
    ```bash
    gcloud compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE}
    ```

3.  Once connected to the Cloud TPU v6e VM in the new terminal, access the running Docker container using `docker exec`:
    ```bash
    sudo docker exec -it "${USER}-vllm" bash
    ```
    You are now inside the same Docker container where the vLLM server is running, but in a new shell session.

## Step 5: Test the server endpoint

Execute this command inside the Docker container (from the new terminal session you opened in Step 4).

Send a completion request to the running vLLM server using `curl`:
```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-32B",
        "prompt": "I love the mornings, because ",
        "max_tokens": 200,
        "temperature": 0
    }'
```
You should receive a JSON response from the model containing the completed text. The `id`, `created` timestamp, and exact `text` will vary:
```json
{
    "id": "cmpl-xxxxxxxxxxxxxxxxxxxxxxxx",
    "object": "text_completion",
    "created": 1678886400,
    "model": "Qwen/Qwen2.5-32B",
    "choices": [
        {
            "index": 0,
            "text": " they are a new beginning for me. I can start fresh and new each day...",
            "logprobs": null,
            "finish_reason": "length"
        }
    ],
    "usage": {
        "prompt_tokens": 7,
        "total_tokens": 207,
        "completion_tokens": 200
    }
}
```

## Step 6: Run the benchmark

Execute these commands inside the Docker container (from the new terminal session opened in Step 4), where you also ran the `curl` test.

1.  Install the `datasets` Python library, which is used by the benchmark script and might not be included in the base vLLM Docker image:
    ```bash
    pip install datasets
    ```

2.  Define environment variables for the benchmark configuration. If your `HF_TOKEN` is not already set in this specific shell session within the container, uncomment and set it:
    ```bash
    export MAX_INPUT_LEN=1800
    export MAX_OUTPUT_LEN=128
    # export HF_TOKEN="<your HF token>" # Uncomment and replace if HF_TOKEN is not set in this session
    ```

3.  Navigate to the vLLM workspace (the path might vary depending on the Docker image structure; `/workspace/vllm` is a common location) and run the benchmark script:
    ```bash
    cd /workspace/vllm # Adjust if your vLLM benchmarks are in a different directory

    python benchmarks/benchmark_serving.py \
        --backend vllm \
        --model "Qwen/Qwen2.5-32B"  \
        --dataset-name random \
        --num-prompts 1000 \
        --random-input-len ${MAX_INPUT_LEN} \
        --random-output-len ${MAX_OUTPUT_LEN} \
        --seed 100
        # --random-range-ratio=${RATIO} # Optional, ensure RATIO is set if used (defined in Step 3.1)
        # --random-prefix-len=${PREFIX_LEN} # Optional, ensure PREFIX_LEN is set if used (defined in Step 3.1)
    ```
The benchmark script will send requests to the server for some time. Upon completion, it will display results showing metrics like throughput (requests per second, tokens per second) and latency (time to first token, time per output token). The exact numbers will vary based on your vLLM version, model size, and specific Cloud TPU v6e instance configuration:
```text
============ Serving Benchmark Result ============
Successful requests:                     1000
Benchmark duration (s):                  123.45
Total input tokens:                      1800000
Total generated tokens:                  128000
Request throughput (req/s):              8.10
Output token throughput (tok/s):         1036.86
Total Token throughput (tok/s):          15616.04
---------------Time to First Token----------------
Mean TTFT (ms):                          100.00
Median TTFT (ms):                        99.00
P99 TTFT (ms):                           110.00
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.90
Median TPOT (ms):                        0.89
P99 TPOT (ms):                           1.00
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.95
Median ITL (ms):                         0.94
P99 ITL (ms):                            1.05
==================================================
```
