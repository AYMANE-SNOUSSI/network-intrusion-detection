"""NSL-KDD loading. Downloads the official files once, verifies integrity, returns DataFrames."""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BASE_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
FILES = {
    "KDDTrain+.txt": "1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95",
    "KDDTest+.txt": "fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84",
    "KDDTest-21.txt": "746993ac9e25868827cacf09eab450050a2a1056e1ce48a1ad39f5dc801d531d",
}

FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]

# Standard 4-category taxonomy (Tavallaee et al., 2009). Every label in the files must be mapped.
ATTACK_CATEGORY = {
    "normal": "normal",
    **dict.fromkeys(["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2",
                     "mailbomb", "processtable", "udpstorm"], "DoS"),
    **dict.fromkeys(["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"], "Probe"),
    **dict.fromkeys(["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
                     "warezclient", "warezmaster", "named", "sendmail", "snmpgetattack",
                     "snmpguess", "xlock", "xsnoop", "worm", "httptunnel"], "R2L"),
    **dict.fromkeys(["buffer_overflow", "loadmodule", "perl", "rootkit", "ps",
                     "sqlattack", "xterm"], "U2R"),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for name, expected in FILES.items():
        path = DATA_DIR / name
        if not path.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(BASE_URL + name, path)
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"{name}: sha256 mismatch ({actual}). Delete the file and retry.")


def load(name: str) -> pd.DataFrame:
    """Return features + `label` (attack name) + `category` + `is_attack` (0/1)."""
    download()
    df = pd.read_csv(DATA_DIR / name, header=None, names=FEATURES + ["label", "difficulty"])
    unmapped = set(df["label"]) - set(ATTACK_CATEGORY)
    if unmapped:
        raise ValueError(f"unmapped labels in {name}: {sorted(unmapped)}")
    df["category"] = df["label"].map(ATTACK_CATEGORY)
    df["is_attack"] = (df["label"] != "normal").astype(int)
    return df.drop(columns="difficulty")


if __name__ == "__main__":
    train, test = load("KDDTrain+.txt"), load("KDDTest+.txt")
    novel = sorted(set(test["label"]) - set(train["label"]))
    print(f"train {train.shape}, test {test.shape}")
    print("train categories:", train["category"].value_counts().to_dict())
    print("test categories: ", test["category"].value_counts().to_dict())
    print(f"{len(novel)} attack types absent from train, "
          f"{test['label'].isin(novel).sum()} test rows: {novel}")
