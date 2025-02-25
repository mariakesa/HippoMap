import os
import subprocess
import scipy.io as sio
import mat73

# Define environment variable for data storage
DATA_DIR = os.getenv("HIPPOMAP_DATA", "./hippomap_data")
os.makedirs(DATA_DIR, exist_ok=True)

# Hardcoded GitHub repository for data
REPO_URL = "https://github.com/winnieyangwannan/Selection-of-experience-for-memory-by-hippocampal-sharp-wave-ripples.git"
REPO_NAME = REPO_URL.split('/')[-1].replace('.git', '')
REPO_PATH = os.path.join(DATA_DIR, REPO_NAME)

# Clone repository if not already available
if not os.path.exists(REPO_PATH):
    subprocess.run(["git", "clone", REPO_URL, REPO_PATH], check=True)
else:
    print("Dataset repository already exists in the data directory.")

class DataLoader:
    """
    Factory class to manage data loading from .mat files.
    """
    def __init__(self, basename, session_id):
        self.basename = os.path.join(REPO_PATH, "Data", basename)
        self.session_id = session_id
    
    def load_data(self):
        """
        Load all relevant data (spike, behavior, session, and cell type) into a dictionary.
        """
        return {
            'spike_time': self._load_mat_file(self.basename + ".spike_time.mat"),
            'behavior': self._load_mat_file(self.basename + ".Behavior.mat"),
            'session': self._load_mat_file(self.basename + ".session.mat"),
            'cell_type': self._load_mat_file(self.basename + ".cell_type.mat")
        }
    
    def _load_mat_file(self, filename):
        """
        Attempts to load a .mat file using scipy.io or mat73 for newer formats.
        """
        try:
            return sio.loadmat(filename)
        except NotImplementedError:
            return mat73.loadmat(filename)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None
