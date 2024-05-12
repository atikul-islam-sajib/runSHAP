# Tree-Based Model Analysis Project

This project is designed to evaluate the performance and importance of features in a custom Random Forest model. The analysis focuses on different settings of the hyperparameter `k` and how they impact feature importance using MDI and SHAP values. The project aims to provide insights without optimizing for specific performance metrics but by documenting the effects across varying settings.

## Prerequisites

Before you begin, make sure you have the following installed:
- Python 3.7 or higher
- Git

## Getting Started

### 1. Clone the Repository

Start by cloning the project repository from GitHub to your local machine:

```bash
git clone https://github.com/atikul-islam-sajib/runSHAP.git
cd runSHAP
```

### 2. Install Python Dependencies

Install the required dependencies specified in the `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Run Setup Script

Execute the `setup.sh` script to finalize setup and prepare your environment:

```bash
./setup.sh
```

Ensure that `setup.sh` has the necessary execution permissions. If not, grant them with:

```bash
chmod +x setup.sh
```

## Usage

After installation, you can run the script as follows:

```bash
python runSHAP.py --n_trees 25 --n_cores 4 --iterations 5
```

### Command Line Arguments

You can customize the execution with several options:
- `--n_trees`: Number of trees in the forest (default: 25)
- `--n_cores`: Number of CPU cores used for parallel processing (default: 1)
- `--iterations`: Number of iterations per value of `k` (default: 5)

### Example Command

To execute the script with specific options, use:

```bash
python runSHAP.py --n_trees 30 --n_cores 4 --iterations 10
```

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests with any enhancements.

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

Thanks to all contributors and third-party libraries that make this project possible.
