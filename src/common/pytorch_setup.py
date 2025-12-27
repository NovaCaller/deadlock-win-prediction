import subprocess
import sys

import torchruntime


def install_torch():
    subprocess.check_call([sys.executable, "-m", "torchruntime", "install"])

def test_torch():
    subprocess.check_call([sys.executable, "-m", "torchruntime", "test"])
    input("press enter to continue...")

def ensure_torch():
    try:
        import torch
        print("torch already installed, successfully imported")
        torchruntime.configure()
        return
    except ImportError:
        print("torch not installed, prompting for install")
        prompt_for_torch_install()
        torchruntime.configure()
        prompt_for_torch_test()

def prompt_for_torch_install():
    while True:
        install_torch_input = input("pytorch is needed for this script.\nwould you like to install it now? (y/n): ")
        if install_torch_input.lower() == "y":
            install_torch()
            print("installed torch via torchruntime.")
            return
        elif install_torch_input.lower() == "n":
            exit(0)
        else:
            continue

def prompt_for_torch_test():
    while True:
        test_torch_input = input("would you like to test the installed pytorch? (y/n): ")
        if test_torch_input.lower() == "y":
            test_torch()
            return
        elif test_torch_input.lower() == "n":
            return
        else:
            continue