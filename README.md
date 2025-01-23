## Project Overview

`SecureTensor` is a MATLAB project for tensor decomposition and privacy protection. This project provides functionality for generating synthetic tensor data, performing tensor decomposition operations, and executing post-processing steps related to the decomposition results.

## Directory Structure

```
.
├── addLaplaceNoise.m        # Adds Laplace noise
├── computeRatio.m           # Computes ratio
├── ctt_dec.m                # CTT decomposition algorithm
├── ctt_ms.m                 # CTT multi-scale algorithm
├── data/                    # Folder containing synthetic data
├── dectest.m                # Test script
├── end_permute.m            # End dimension permutation
├── generate_synthetic_tensor.m  # Generates synthetic tensor data
├── inc_FCTN_TC.m            # FCTN tensor decomposition implementation
├── main.m                   # Main program entry
├── ms.m                     # Multi-scale computation
├── my_Fold.m                # Tensor fold operation
├── my_Unfold.m              # Tensor unfold operation
├── my_inc_FCTN_TC.m         # Extended FCTN tensor decomposition
├── tensor_contraction.m     # Tensor contraction operation
├── tensor_toolbox-v3.1/     # Tensor toolbox
├── tnprod.m                 # Tensor product operation
├── tnprod_new.m             # New tensor product operation
├── tnprod_rest.m            # Remaining tensor product part
├── tnprod_rest_new.m        # New remaining tensor product part
├── tnreshape.m              # Tensor reshape operation
├── tnreshape_new.m          # New tensor reshape operation
├── tt_reconstruct.m         # TT reconstruction operation
├── ttmlsvdlast.m            # TT last SVD operation
```

## Installation

1. Ensure that you have MATLAB installed on your system.
2. Download the project files.
3. Place all the files in a MATLAB working directory.

## Usage

`main.m` is the entry point of the project. Running it will execute the entire tensor decomposition process:


## Contributing

If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the remote repository (`git push origin feature-branch`).
5. Submit a Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).
