import os
import matplotlib.pyplot as plt
from tensorboardX.proto import event_pb2

import paramiko
import os

from tensorboard.backend.event_processing import event_accumulator
import pandas as pd


def parse_tensorboard(path, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    print(ea.Tags()["scalars"])
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}

# Define the SFTP function
def sftp_transfer(host, port, username, password, root, filename, rootLocal, id):
    try:
        # Create an SSH client
        ssh = paramiko.SSHClient()
        
        # Automatically add the server's SSH key (not recommended for production)
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Connect to the host
        ssh.connect(host, port=port, username=username, password=password)

        # Create an SFTP session
        sftp = ssh.open_sftp()
        
        # List directories in the specified path
        folder_path = root
        folder_names = [f for f in sftp.listdir(folder_path) if sftp.stat(folder_path + f).st_mode & 0o040000]
        
        print("Folders in the specified path:")
        for folder in folder_names:
            if id not in folder:
                continue
            filename_aug = filename.copy()
            print()
            print(folder)
            # Find and copy the file named serial_job.x
            for file_attr in sftp.listdir_attr(os.path.join(folder_path, folder).replace("\\","/")):
                if file_attr.filename.startswith('events.'):
                    filename_aug.append(os.path.join(folder_path, folder, file_attr.filename).replace("\\","/"))

            # Create the rootLocal directory if it doesn't exist
            if not os.path.exists(os.path.join(rootLocal, folder)):
                os.makedirs(os.path.join(rootLocal, folder))

            for file in filename_aug:
                remote_file = os.path.join(root, folder, file).replace("\\","/")
                local_file = os.path.join(rootLocal, folder, os.path.basename(file)).replace("\\","/")
                try:
                    sftp.get(remote_file, local_file)
                    print(f"Successfully downloaded {remote_file} to {local_file}")
                except Exception as e:
                    print(f"Failed to download {remote_file} to {local_file}: {e}")
        
        # Close the SFTP session and SSH connection
        sftp.close()
        ssh.close()
        
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage:
host = '155.69.100.12'
port = 22
username = 'glin0001'
password = 'Hx5dr1hu!!&&'

root = '/home/msai/glin0001/RL/runs/'
filename = [] #, 'all/log.sol']
rootLocal = 'runs_cluster'

ids = ['v_rand_lo']
for id in ids:
    sftp_transfer(host, port, username, password, root, filename, rootLocal, id)

path = 'C:/Cyril/Work/NTU/ai6103-ppo-llm/runs_cluster/FrozenLake-v1__v3__1__1731317298/events.out.tfevents.1731317301.TC2N01.948261.0'
# scalars = 'l'
# parse_tensorboard(path, scalars)
