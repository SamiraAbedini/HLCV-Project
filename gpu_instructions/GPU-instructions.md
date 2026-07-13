# HLCV SS26 - GPU Tutorial 
  

### Connect to the VPN of UdS 

The GPU server is only accessible using the UdS network. 

To access from outside the network, you need to connect to the VPN of UdS, following the instructions here: 

https://www.hiz-saarland.de/dienste/vpn 

### Login to the GPU server 

In the Terminal, use the following command to login to the GPU server: 

```
ssh hlcv_teamxxx@conduit.hpc.uni-saarland.de 
```
You can also use `conduit2.hpc.uni-saarland.de`.

Please replace hlcv_teamxxx with your own account. 

You can change your password using: 

```
passwd 
```


### Copy the provided instructions directory to the cluster.
You can do this by running `scp` command *from your local machine*.
```
scp -r [PATH TO THE gpu_instructions dir] hlcv_teamxxx@conduit.hpc.uni-saarland.de:~
```
This will copy the entire directory with sample files (including this READMe file) to the cluster. You can then follow the rest of this README on the cluster.


### Install Miniconda

Install Miniconda on the cluster using the Linux installer from https://docs.conda.io/en/latest/miniconda.html

For example, with PyTorch 3.10, you can do the following:

```
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
chmod +x Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
./Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
```

### Create a conda environment

We have provided an `environment.yml` file with many of the packages you may need (which you might have already used for the assignments). Alternatively, you can create your own environment and install all the packages yourself.

```
conda env create -f environment.yml
conda activate hlcv
```

### Prepare a simple PyTorch example script

The relevant files are `demo2.py`, `demo.py`, `execute.sh`, and `pytorch_docker.sub`.

#### `demo2.py`

This is a simple python script that prints a few lines. This is to be used to debug your submission pipeline and make sure everything is set. Once this is done, try running `demo.py`.

#### `demo.py`

This is a simple script to test that you are able to run a pytorch on the HTCondor cluster. It will make sure your python environment is set up and your code is able to access GPU.

The first line of the Python scripts should be:

```
#!/usr/bin/env python
```


#### `execute.sh`

This is a bash script that will run the Python script under the appropriate conda environment and pass any necessary arguments.

Set the following two variables:

```
# Path to the `.py` file you want to run
PYTHON_SCRIPT_PATH="/home/hlcv_teamxxx/gpu_instructions/"
# Path to the Python binary of the conda environment
CONDA_PYTHON_BINARY_PATH="/home/hlcv_teamxxx/miniconda3/envs/hlcv/bin/python"
```


#### `pytorch_docker.sub`

This describes the job to submit to HTCondor.


Check the following lines:

Specify the Python script to run. This can also include arguments.
```
# Example without arguments
arguments               = demo.py

# Example with arguments
arguments               = demo.py arg1 arg2
```

Specify the paths for the output, error, and log files.
Make sure all the parent directory exist. Otherwise your jobs will be kept on hold by the cluster.
```
output                  = /home/hlcv_teamxxx/condor_logs/demo.$(ClusterId).$(ProcId).out
error                   = /home/hlcv_teamxxx/condor_logs/demo.$(ClusterId).$(ProcId).err
log                     = /home/hlcv_teamxxx/condor_logs/demo.$(ClusterId).log
```

The current `.sub` scripts will use lighter `demo2.py` file that simply prints a few lines. Once you were able to see the printed lines in the output logs, then edit the `.sub` configs to use `demo.py`, to test a real (pytorch) script.

### Run the example


Submit your job 

```
mkdir ~/condor_logs
condor_submit pytorch_docker.sub 
```


Check the state of your job in the condor queue: 

```
condor_q 
```


Analyze how many machines can run your job or if there are problems: 


```
condor_q -analyze 
```
```
condor_q -better 
```

Overview of machines in the cluster: 

```
condor_status 
```

To terminate a job: 

  1. Find the job ID: 

```
condor_q –nobatch 
```

  2.  Terminate the job: 
    
```
condor_rm <job_id> 
```

The output and error logs can be viewed in their respective files at the paths specified in `pytorch_docker.sub`.

To monitor the output file when a job is running:

```
condor_tail -f <job_id>
```


### Using interactive jobs

Interactive jobs can be used for debugging. Note that these jobs are killed automatically after one hour to allow other users to get an interactive slot. So please don’t directly run your code with an interactive job. 


You can set up breakpoints using the Python Debugger library: 

```
import pdb 

pdb.set_trace() 
```

To submit an interactive job: 

```
condor_submit -i pytorch_docker_interactive.sub 
```

To run the code for debugging: 

```
CUDA_VISIBLE_DEVICES=0 python demo.py 
```


### Notes 

  -  There is a restriction on the number of GPUs that can be used at a time. That means each team can only run one job on one GPU in parallel. Additional jobs get queued. If they request more resources, the jobs will be idle indefinitely. There is no restriction on job duration, which means the jobs don’t get killed automatically after a certain time. 

  -  Further readings for the HTCondor (i.e., the system used for the GPU server): https://htcondor.readthedocs.io/en/latest/ 

### Contact 

For further questions, you may contact the TAs of HLCV using this mailing list hlcv-ss26@mpi-inf.mpg.de. 

 

 
