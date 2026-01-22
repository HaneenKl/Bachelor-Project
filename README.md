# Semigroup Explorer

This project is a **web-based tool for computing and visualizing semigroups
arising from regular languages**.  
It was developed as a **Bachelor project**.

The application allows users to input **regular expressions or Arden equations**
and explores the associated **syntactic semigroup** and related automata-theoretic
structures.

---

## Features

The website provides the following functionalities:

- Compute the **syntactic semigroup** of a given regular expression or Arden system
- Visualize the semigroup as an **egg-box diagram**
- Plot the **right Cayley graph** and **left Cayley graph** of the semigroup
- Compute and display the **minimal DFA**
- Check whether **user-provided equations between semigroup elements** hold

---

## Technology Stack

- Python
- Flask
- pyformlang
- Graphviz
- libsemigroups_pybind11 (C++ backend via Python bindings)

**Important:**  
This project requires a **Linux / Unix-like environment**.

---

## System Requirements

### Supported platforms

- Linux (tested on Ubuntu)
- Windows **via WSL (Ubuntu)**
- macOS (not officially tested)


### Why Linux / WSL is required

The library `libsemigroups_pybind11` is implemented in **C++** and compiled during
installation. It requires a POSIX-compatible toolchain and is not available as a
precompiled wheel for native Windows Python.

---

## Recommended Setup (Windows Users)

1. Install WSL:  
   https://learn.microsoft.com/windows/wsl/install

2. Install **Ubuntu** from the Microsoft Store

3. Inside Ubuntu, install Python and required system tools:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv build-essential graphviz
4. Clone this repository and navigate into it:
   ```bash
   git clone https://github.com/HaneenKl/Bachelor-Project.git
   cd <repository_directory>
   ```
5. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
6. Install the required Python dependencies:
   ```bash
   pip install flask pyformlang graphviz libsemigroups_pybind11
---
## Running the Application
- Start the Flask application:
   ```bash
    python app.py
- Open your web browser at:
   ```bash
    http://...
---
## IDE Configuration
If you are using an IDE, make sure that the Python interpreter is set to WSL / Linux

### For PyCharm:
Go to `Settings` > `Python` > `Interpreter` and select the WSL interpreter.