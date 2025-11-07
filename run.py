import subprocess

def run_extinction():
    subprocess.run(["python3", "extinction.py"])

def run_intervention():
    subprocess.run(["python3", "intervention.py"])

if __name__ == "__main__":
    run_extinction()
    run_intervention()