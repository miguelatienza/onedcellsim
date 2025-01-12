# Onedcellsim

<img src="https://github.com/miguelatienza/onedcellsim/blob/main/simgui.png" width="700">

**Onedcellsim** is a project designed to simulate one-dimensional cell migration using Julia, based on [a biomechanical model](https://doi.org/10.1016/j.bpj.2023.02.001) as developed by Amiri et al. (2023). Additionally, it uses simulation-based inference in Python via [SBI](https://github.com/mackelab/sbi) to infer biomechanical cell parameters from observed data.

---

## Installation

### Prerequisites
- Python 3
- Linux or Mac OS
### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/miguelatienza/onedcellsim.git
   ```
2. Navigate to the project directory:
   ```bash
   cd onedcellsim
   ```
3. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
4. Install **onedcellsim**:
   ```bash
   pip install git+https://github.com/miguelatienza/onedcellsim@main
   ```
5. Install additional dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Install Julia:
   ```bash
   python install_julia.py
   ```

---

## Running the GUI

Run the following command to start the GUI:
```bash
bash gui.sh
```
---

Feel free to contribute or report issues on the [GitHub repository](https://github.com/miguelatienza/onedcellsim).







