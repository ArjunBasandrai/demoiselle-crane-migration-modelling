import subprocess

def run_random_walk():
    result = subprocess.run(
        ["Rscript", "src/R/rw.R"],
        check=True
    )
