import os
from pathlib import Path

import requests
from datetime import datetime, timezone, timedelta
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://files.deadlock-api.com/buckets/db-snapshot/public/"
DOMAIN = "https://files.deadlock-api.com"
LOCAL_DIR = "db-dump"
FILE_SKIP_LIST = ["active_matches.parquet", "active_matches.sql", "steam_profiles_old.parquet", "steam_profiles_old.sql"]

def download_files_recursively(url: str, output_path: Path) -> None:
    output_path.mkdir(exist_ok=True)

    # get html and parse into soup
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.body.find("table", attrs={"class": "striped"})
    assert table is not None
    assert table.tbody is not None
    table_rows = table.tbody.find_all("tr")

    for tr in table_rows:
        tds = tr.find_all("td")
        assert len(tds) == 5
        if tds[0].get_text(strip=True).startswith("folder"): # if current table element is a directory instead of a file
            tds[0].i.decompose() # remove icon tag
            subdir_name = tds[0].get_text(strip=True)
            download_files_recursively(urljoin(url, subdir_name + "/"), output_path / subdir_name)
        else: # if current table element is a file
            tds[0].i.decompose()  # remove icon tag
            file_name = tds[0].get_text(strip=True)

            # skip if file is not needed
            if file_name in FILE_SKIP_LIST:
                continue

            target_download_path = output_path / file_name
            if target_download_path.exists():
                assert target_download_path.is_file()
                local_last_modified_utc: datetime = datetime.fromtimestamp(target_download_path.stat().st_mtime, tz=timezone.utc)
                remote_last_modified_utc: datetime = datetime.strptime(tds[3].get_text(strip=True).removesuffix(" UTC"), "%Y-%m-%d %H:%M:%S.%f %z").astimezone(timezone.utc)
                # skip download if local is newer or both files were modified within a minute of each other
                if remote_last_modified_utc <= local_last_modified_utc or (remote_last_modified_utc - local_last_modified_utc < timedelta(minutes=1)):
                    print(f"Skipping download of {file_name} (local file is up to date)")
                    continue
            download_url = urljoin(DOMAIN, tds[4].find_all("li")[0].a["href"])
            download_file(download_url, target_download_path, int(tds[1].get_text(strip=True).split(" ")[0]))


def download_file(url: str, output_path: Path, total_size: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, headers={"Accept-Encoding": "*/*"}) as r:
        r.raise_for_status()

        with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):  # 1 MB
                    f.write(chunk)
                    pbar.update(len(chunk))


if __name__ == "__main__":
    download_files_recursively(BASE_URL, Path(LOCAL_DIR))
    exit(0)