# Llama2-7B Inference with JetStream and MaxText

This guide provides instructions to set up and benchmark the Llama2-7B model using JetStream (a multi-host inference engine) and MaxText (a high-performance LLM codebase). These instructions are tailored for users working with Google Cloud TPUs.

## Prerequisites

Before you begin, ensure you have the following:
*   **Google Cloud Project:** A GCP project with billing enabled.
*   **TPU Quota:** Sufficient TPU quota in your project for the desired configuration.
*   **Google Cloud SDK:** The `gcloud` command-line tool installed and configured (including `gsutil`). You can find installation instructions at [Install the Google Cloud SDK](https://cloud.google.com/sdk/docs/install).
*   **Git:** The Git version control system installed.
*   **Python:** Python 3.10 or later, along with `pip` and `venv` for environment management.

## Setup

This section guides you through downloading the necessary repositories, setting up the Python environment, and preparing the Llama 2 model checkpoint.

### Step 1: Download JetStream and MaxText Repositories

You'll need both the [Google JetStream GitHub repository](https://github.com/google/JetStream) (for the inference engine) and the [Google MaxText GitHub repository](https://github.com/google/maxtext) (for the model implementation and utilities).

1.  **Choose a base directory for your projects.** The commands below use `~/code_repos/` as an example. You can use any directory you prefer, such as `~` (your home directory), but organizing projects into a subdirectory can help keep your home directory tidy.

    ```bash
    mkdir -p ~/code_repos
    cd ~/code_repos
    ```

2.  **Clone the repositories:**

    ```bash
    # Clone MaxText
    git clone https://github.com/google/maxtext.git
    cd maxtext
    git checkout main # Consider using a specific release tag for stability in production
    cd .. # Go back to ~/code_repos

    # Clone JetStream
    git clone https://github.com/google/JetStream.git
    cd JetStream
    git checkout main # Consider using a specific release tag for stability in production
    cd .. # Go back to ~/code_repos
    ```

### Step 2: Set Up Python Environment and Install Dependencies

This step creates an isolated Python virtual environment for the project and installs the necessary packages for both JetStream and MaxText.

1.  **Create and activate a virtual environment:**
    This ensures that project dependencies don't conflict with your global Python setup. We'll name the environment `venv-maxtext` and assume it's created in your chosen base directory (e.g., `~/code_repos/`).

    ```bash
    # Navigate to your chosen base directory, e.g., ~/code_repos/
    # If you used a different directory in Step 1, replace ~/code_repos/ accordingly.
    cd ~/code_repos

    # Install python3.10-venv if not already present.
    # This package provides the `venv` module for creating virtual environments.
    sudo apt update && sudo apt install -y python3.10-venv

    # Create the virtual environment
    python3 -m venv venv-maxtext

    # Activate the virtual environment
    source venv-maxtext/bin/activate
    ```
    *Note: You'll need to run `source venv-maxtext/bin/activate` in any new terminal session where you want to work on this project.*

2.  **Install JetStream dependencies:**
    Navigate to the JetStream directory cloned in Step 1 (e.g., `~/code_repos/JetStream/`).

    ```bash
    # Assuming your base directory is ~/code_repos/
    cd ~/code_repos/JetStream

    # Install JetStream in "editable" mode. This means changes to the JetStream source code
    # will be reflected immediately in the environment.
    pip install -e .

    # Install dependencies for running JetStream benchmarks
    cd benchmarks
    pip install -r requirements.in
    cd .. # Return to JetStream directory (e.g., ~/code_repos/JetStream/)
    cd .. # Return to base directory (e.g., ~/code_repos/)
    ```

3.  **Install MaxText dependencies:**
    Navigate to the MaxText directory cloned in Step 1 (e.g., `~/code_repos/maxtext/`). The `setup.sh` script installs MaxText's dependencies, including JAX for running on TPUs.

    ```bash
    # Assuming your base directory is ~/code_repos/
    cd ~/code_repos/maxtext

    # This script installs MaxText's Python dependencies.
    bash setup.sh
    cd .. # Return to base directory (e.g., ~/code_repos/)
    ```

### Step 3: Download Llama 2 Checkpoint and Convert to MaxText Format

This crucial step involves obtaining the official Llama 2 model weights, transferring them to Google Cloud Storage (GCS), and then converting them into the format required by MaxText for TPU execution.

1.  **Download Llama 2 Model Weights from Meta:**

    *   **Request Access:** Go to the official [Meta Llama Downloads page](https://llama.meta.com/llama-downloads/). You will need to fill out a form and agree to Meta's terms of use. Access approval may take some time.
    *   **Clone the Llama repository:** Once you have download access, you'll receive instructions which typically involve cloning a repository from `meta-llama`.

        ```bash
        # Navigate to your base directory (e.g., ~/code_repos/)
        cd ~/code_repos
        git clone https://github.com/meta-llama/llama.git
        # Note: The exact repository URL will be provided by Meta upon approval.
        ```
    *   **Run the download script:** The cloned Llama repository contains a `download.sh` script.

        ```bash
        cd llama # Navigate into the cloned Llama directory (e.g., ~/code_repos/llama/)
        bash download.sh
        ```
        When prompted, choose the `7B` model version. This script will download the Llama 2 7B model weights (approx. 13GB) and tokenizer. It should create a directory named `llama-2-7b` (or similar; verify the exact name from the script's output) inside the `llama` directory, containing the model files.
        ```bash
        cd .. # Return to base directory (e.g., ~/code_repos/)
        ```

2.  **Set Up Google Cloud Storage (GCS) Buckets:**

    You'll need GCS buckets to store the original checkpoint and the converted MaxText checkpoints.
    *   **`CHKPT_BUCKET`**: This bucket will store the original Llama 2 7B model weights you downloaded (e.g., from `~/code_repos/llama/llama-2-7b/`).
    *   **`MAXTEXT_BUCKET_SCANNED`**: This bucket is for the "scanned" version of the MaxText checkpoint. Scanned checkpoints unroll loops in the computational graph, which can be beneficial for some training optimizations but are not typically used for standard inference.
    *   **`MAXTEXT_BUCKET_UNSCANNED`**: This bucket will store the "unscanned" version of the MaxText checkpoint. This is the format generally used for inference with MaxText as it loads faster.

    You must replace the placeholder `gs://your-gcp-project-bucket/...` with your actual GCS bucket paths. If you don't have buckets yet, you can create them using the Google Cloud Console or `gsutil mb gs://your-bucket-name`.

    ```bash
    # Replace with your actual GCS bucket paths!
    export CHKPT_BUCKET=gs://your-gcp-project-bucket/llama2-7b/original_meta_chkpt
    export MAXTEXT_BUCKET_SCANNED=gs://your-gcp-project-bucket/llama2-7b/maxtext_scanned_chkpt
    export MAXTEXT_BUCKET_UNSCANNED=gs://your-gcp-project-bucket/llama2-7b/maxtext_unscanned_chkpt
    ```
    *Tip: You might want to add these `export` commands to your `~/.bashrc` or `~/.zshrc` file if you plan to work with these paths frequently, then `source` the file (e.g., `source ~/.bashrc`).*

3.  **Upload Original Llama 2 Checkpoint to GCS:**

    Copy the downloaded Llama 2 7B model files from your local machine (e.g., `~/code_repos/llama/llama-2-7b/`) to the `CHKPT_BUCKET` you defined.

    ```bash
    # Ensure CHKPT_BUCKET is set from the previous step.
    # Verify the source path '~/code_repos/llama/llama-2-7b/' matches where your weights were downloaded.
    gsutil cp -r ~/code_repos/llama/llama-2-7b/* ${CHKPT_BUCKET}/
    ```

4.  **Convert Checkpoint to MaxText Format:**

    This script converts the original Llama 2 checkpoint into the MaxText format, creating both scanned and unscanned versions in your specified GCS buckets.

    ```bash
    # Navigate to your MaxText directory (e.g., ~/code_repos/maxtext/)
    cd ~/code_repos/maxtext

    # Run the conversion script
    # Usage: model_ckpt_conversion.sh <model_series> <model_size> <meta_ckpt_gcs_path> <maxtext_scanned_ckpt_gcs_path> <maxtext_unscanned_ckpt_gcs_path>
    bash ../JetStream/jetstream/tools/maxtext/model_ckpt_conversion.sh llama2 7b ${CHKPT_BUCKET} ${MAXTEXT_BUCKET_SCANNED} ${MAXTEXT_BUCKET_UNSCANNED}
    ```

    The conversion script will output the exact path to the generated unscanned checkpoint, which you'll need for the `UNSCANNED_CKPT_PATH` environment variable in the benchmarking step. It usually looks something like `gs://your-gcp-project-bucket/llama2-7b/maxtext_unscanned_chkpt/llama2-7b_unscanned_chkpt_YYYY-MM-DD-HH-MM/checkpoints/0/items`.

    **Set `UNSCANNED_CKPT_PATH`:**
    The script *should* automatically export `UNSCANNED_CKPT_PATH`. However, it's good practice to verify or set it explicitly.

    ```bash
    # The script should output the path. If not, you can find it by listing the contents of MAXTEXT_BUCKET_UNSCANNED.
    # For example: gsutil ls gs://${MAXTEXT_BUCKET_UNSCANNED}/
    # Then, find the directory that looks like 'llama2-7b_unscanned_chkpt_YYYY-MM-DD-HH-MM'
    # and append '/checkpoints/0/items' to it.

    # Replace with the actual path output by the script or found via gsutil ls:
    export UNSCANNED_CKPT_PATH=gs://your-gcp-project-bucket/llama2-7b/maxtext_unscanned_chkpt/llama2-7b_unscanned_chkpt_YYYY-MM-DD-HH-MM/checkpoints/0/items
    echo "UNSCANNED_CKPT_PATH is set to: ${UNSCANNED_CKPT_PATH}"
    ```
    *Make sure this `UNSCANNED_CKPT_PATH` variable is correctly set before proceeding to the benchmark, as the server will use this path to load the model weights.*
    ```bash
    cd ~/code_repos # Return to base directory
    ```

## Benchmark

The benchmarking process involves two main steps: starting the MaxText server (which loads the model onto TPUs) and then running the benchmark client script (which sends requests to the server). You'll typically use two separate terminal tabs or sessions for this.

### Terminal Tab 1: Start the MaxText Inference Server

Before running the server, you need to set several environment variables. These configure the server's behavior, model loading paths, and parallelism settings. Ensure your Python virtual environment (`venv-maxtext` from Step 2.1) is active in this terminal.

**Activate Virtual Environment (if not already active):**
```bash
# If you're in a new terminal, navigate to your base project directory (e.g., ~/code_repos/)
cd ~/code_repos
source venv-maxtext/bin/activate
```

**Environment Variables for the Server:**

*   `TOKENIZER_PATH`: Specifies the path to the tokenizer files. For Llama 2, this typically points to `assets/tokenizer.llama2` located within your MaxText repository directory (e.g., `~/code_repos/maxtext/assets/tokenizer.llama2`).
    ```bash
    export TOKENIZER_PATH=~/code_repos/maxtext/assets/tokenizer.llama2
    ```
*   `LOAD_PARAMETERS_PATH`: **Crucial!** This must be the GCS path to your *unscanned* MaxText checkpoint. This is the `UNSCANNED_CKPT_PATH` value you identified and exported in Step 3.4.
    ```bash
    # Ensure UNSCANNED_CKPT_PATH is correctly set in your environment from Step 3.4
    export LOAD_PARAMETERS_PATH=${UNSCANNED_CKPT_PATH}
    echo "Will load model from: ${LOAD_PARAMETERS_PATH}"
    ```
*   `MAX_PREFILL_PREDICT_LENGTH`: Maximum sequence length (number of tokens) the model can take as input for the prefill phase. Default: `1024`.
    ```bash
    export MAX_PREFILL_PREDICT_LENGTH=1024
    ```
*   `MAX_TARGET_LENGTH`: Maximum total sequence length (input tokens + generated tokens) the model can handle. Default: `2048`.
    ```bash
    export MAX_TARGET_LENGTH=2048
    ```
*   `MODEL_NAME`: The name of the model configuration to use from MaxText. For Llama 2 7B, this is `llama2-7b`.
    ```bash
    export MODEL_NAME=llama2-7b
    ```
*   `ICI_FSDP_PARALLELISM`: Inter-Chip Interconnect (ICI) parallelism degree for Fully Sharded Data Parallelism (FSDP). `1` implies no FSDP sharding across the ICI mesh dimension.
    ```bash
    export ICI_FSDP_PARALLELISM=1
    ```
*   `ICI_AUTOREGRESSIVE_PARALLELISM`: ICI parallelism for the autoregressive decoding loop (often related to sequence parallelism). `1` implies no sequence parallelism across the ICI mesh dimension.
    ```bash
    export ICI_AUTOREGRESSIVE_PARALLELISM=1
    ```
*   `ICI_TENSOR_PARALLELISM`: ICI parallelism for tensor operations (sharding model weights and activations). A value of `-1` usually directs MaxText to attempt an optimal value based on the TPU slice topology and other parallelism settings. For a v4-8 slice, with `ICI_FSDP_PARALLELISM` and `ICI_AUTOREGRESSIVE_PARALLELISM` at `1`, this might be `1` or `2`.
    ```bash
    export ICI_TENSOR_PARALLELISM=-1
    ```
*   `SCAN_LAYERS`: If `true`, model layers are processed sequentially using JAX's `scan` operation. This can reduce compilation time but may impact runtime performance for some models. `false` unrolls layers in the graph. Default: `false`.
    ```bash
    export SCAN_LAYERS=false
    ```
*   `WEIGHT_DTYPE`: Data type for the model weights during inference. `bfloat16` is a common choice for TPUs, offering a good balance of numerical precision and performance.
    ```bash
    export WEIGHT_DTYPE=bfloat16
    ```
*   `PER_DEVICE_BATCH_SIZE`: The number of input sequences processed in parallel on each individual TPU device/core. This value can be tuned based on available TPU memory and desired throughput. Default: `11`.
    ```bash
    export PER_DEVICE_BATCH_SIZE=11 # Adjust based on your TPU configuration and memory
    ```

**Start the Server Command:**

Once the environment variables are correctly set, navigate to your MaxText project directory (e.g., `~/code_repos/maxtext/`) and execute the server script:

```bash
# Navigate to your MaxText installation directory (e.g., ~/code_repos/maxtext/)
cd ~/code_repos/maxtext

python MaxText/maxengine_server.py \
  MaxText/configs/base.yml \
  tokenizer_path=${TOKENIZER_PATH} \
  load_parameters_path=${LOAD_PARAMETERS_PATH} \
  max_prefill_predict_length=${MAX_PREFILL_PREDICT_LENGTH} \
  max_target_length=${MAX_TARGET_LENGTH} \
  model_name=${MODEL_NAME} \
  ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM} \
  ici_autoregressive_parallelism=${ICI_AUTOREGRESSIVE_PARALLELISM} \
  ici_tensor_parallelism=${ICI_TENSOR_PARALLELISM} \
  scan_layers=${SCAN_LAYERS} \
  weight_dtype=${WEIGHT_DTYPE} \
  per_device_batch_size=${PER_DEVICE_BATCH_SIZE}
```
This command starts the `maxengine_server.py` script. It uses `MaxText/configs/base.yml` as a foundational configuration, and then applies overrides or sets specific parameters based on the environment variables you've defined (passed as command-line arguments here). The server will proceed to load the Llama 2 7B model onto the available TPUs and await incoming benchmark requests.

### Terminal Tab 2: Run the Benchmark Client

In a new terminal tab or session (distinct from the one running the server), you'll run the benchmark client script. This script sends requests to the MaxText server and measures its performance.

**Activate Virtual Environment (if not already active):**
Make sure your Python virtual environment (`venv-maxtext` from Step 2.1) is active in this terminal as well.
```bash
# If you're in a new terminal, navigate to your base project directory (e.g., ~/code_repos/)
cd ~/code_repos
source venv-maxtext/bin/activate
```

**Run the Benchmark Client Command:**
The main benchmark script, `benchmark_serving.py`, is located in the `benchmarks` subdirectory of your JetStream project (e.g., `~/code_repos/JetStream/benchmarks/`).

```bash
# Navigate to your JetStream project directory (e.g., ~/code_repos/JetStream/)
cd ~/code_repos/JetStream

python benchmarks/benchmark_serving.py \
  --tokenizer ~/code_repos/maxtext/assets/tokenizer.llama2 \
  --num-prompts 1000 \
  --max-output-length 1024 \
  --dataset openorca \
  --warmup-mode sampled \
  --save-result \
  --save-request-outputs \
  --request-outputs-file-path outputs.json
```

**Explanation of Key Client Arguments:**

*   `--tokenizer`: Path to the tokenizer model. This **must** be the same tokenizer used by the server.
    *   Example: `~/code_repos/maxtext/assets/tokenizer.llama2` (adjust if your MaxText path differs).
*   `--num-prompts`: The total number of prompts the client will send to the server for the benchmark.
    *   Example: `1000`. A higher number generally provides more stable performance metrics but extends the benchmark duration.
*   `--max-output-length`: The maximum number of tokens the client will request the server to generate for each prompt.
    *   Example: `1024`.
*   `--dataset`: Specifies the dataset from which prompts are sourced. `openorca` is a commonly used public dataset for this purpose. The script might download this dataset if not found locally.
*   `--warmup-mode sampled`: Defines the strategy for sending warmup requests before the actual measurement phase begins. `sampled` uses a sample of requests for warming up the server.
*   `--save-result`: If this flag is included, the benchmark script will save a summary of the performance results (e.g., throughput, latency) to a file. This file is typically a CSV or text file created in the current working directory (e.g., `~/code_repos/JetStream/` if you run the script from there).
*   `--save-request-outputs`: If this flag is included, the script will save the actual generated text (model outputs) for each prompt.
*   `--request-outputs-file-path outputs.json`: Specifies the filename for storing the model outputs when `--save-request-outputs` is used. The file (e.g., `outputs.json`) will be saved in the current working directory from where the script is executed (e.g., `~/code_repos/JetStream/outputs.json`).

Once launched, the client will attempt to connect to the server you started in Terminal 1. It will then send the specified number of prompts and, upon completion, print the performance metrics to the console and save files if requested.

### Understanding the Benchmark Output

After the benchmark client script completes its run, it will display a summary of performance metrics in your console. Below is an example of such output, followed by explanations of the key metrics:

```text
Successful requests: 995
Benchmark duration: 305.366344 s
Total input tokens: 217011
Total generated tokens: 934964
Request throughput: 3.26 requests/s
Input token throughput: 710.66 tokens/s
Output token throughput: 3061.78 tokens/s
Mean TTFT: 130288.20 ms
Median TTFT: 140039.96 ms
P99 TTFT: 278498.91 ms
Mean TPOT: 5052.76 ms
Median TPOT: 164.01 ms
P99 TPOT: 112171.56 ms
```

**Key Metrics Explained:**

*   **Successful requests:** The total number of prompts that the server processed successfully during the benchmark run.
*   **Benchmark duration:** The total time, in seconds, that the benchmark took to execute (this typically excludes any initial warmup phase).
*   **Total input tokens:** The aggregate count of tokens across all the input prompts sent to the model.
*   **Total generated tokens:** The aggregate count of tokens produced by the model in response to all input prompts.
*   **Request throughput:** The average number of requests (prompts) processed by the server per second. A higher value indicates better request handling capacity.
*   **Input token throughput:** The average number of input tokens processed by the server per second.
*   **Output token throughput:** The average number of output tokens generated by the model per second. This is a critical indicator of the model's generation speed. Higher values are generally better.
*   **TTFT (Time To First Token):** This metric measures the latency from the moment a request is dispatched to the server until the first token of the generated response is received by the client. It's often measured in milliseconds (ms).
    *   **Mean TTFT:** The arithmetic average of the TTFT values for all requests. Lower is better.
    *   **Median TTFT:** The 50th percentile of TTFT values; half of the requests had a TTFT lower than this value. Often more representative than the mean if there are outliers. Lower is better.
    *   **P99 TTFT:** The 99th percentile of TTFT values, meaning 99% of requests achieved a TTFT at or below this value. Useful for understanding worst-case latency for the vast majority of requests. Lower is better.
*   **TPOT (Time Per Output Token):** This metric measures the average time taken to generate each token *after* the first token has been produced for a given request. It reflects the ongoing speed of token generation.
    *   **Mean TPOT:** The arithmetic average of the TPOT values. Lower is better.
    *   **Median TPOT:** The 50th percentile of TPOT values. Lower is better.
    *   **P99 TPOT:** The 99th percentile of TPOT values. Lower is better.

These metrics provide a comprehensive overview of the performance characteristics of the Llama2-7B model running on your specific JetStream and MaxText setup with the given TPU hardware.
