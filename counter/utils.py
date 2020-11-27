import os


def wget_file(filepath: str, file_url: str):
    if not os.path.exists(filepath):
        print(f'{filepath} file not found, downloading from server')
        shell_script = f"wget -O {filepath} {file_url}"

        if os.system(shell_script):
            print("Root permission required, trying with sudo")
            if os.system("sudo " + shell_script):
                raise Exception(f"Unable to download {filepath}, connect a specialist")

    return filepath
