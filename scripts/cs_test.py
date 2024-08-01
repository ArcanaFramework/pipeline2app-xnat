import os
from frametree.core.store import DataStore


os.environ.update(
    {"XNAT_HOST": "localhost:8080", "XNAT_USER": "admin", "XNAT_PASS": "admin"}
)

store = DataStore.load("xnat-cs")
