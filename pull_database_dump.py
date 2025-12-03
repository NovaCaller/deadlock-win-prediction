from pathlib import Path

import requests
from datetime import datetime, timezone, timedelta
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://files.deadlock-api.com/buckets/db-snapshot/public/"
DOMAIN = "https://files.deadlock-api.com"
LOCAL_DIR = "db_dump"
TOP_DIR_FILE_INCLUDE_LIST = ["heroes.parquet"]
MATCH_ID_RANGE = range(45, 48)  # only pull parquets with ids in this range (45 to 47)
MAX_DOWNLOAD_RETRIES = 10


def download_files_recursively(url: str, output_path: Path, is_metadata_dir: bool) -> None:
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
        if tds[0].get_text(strip=True).startswith(
                "folder"):  # if current table element is a directory instead of a file
            tds[0].i.decompose()  # remove icon tag
            subdir_name = tds[0].get_text(strip=True)
            if subdir_name != "match_metadata":
                print(f"Skipping download of unexpected directory {subdir_name}")
                continue
            download_files_recursively(urljoin(url, subdir_name + "/"), output_path / subdir_name, True)
        else:  # if current table element is a file
            tds[0].i.decompose()  # remove icon tag
            file_name = tds[0].get_text(strip=True)

            # skip if file is not needed
            if not is_metadata_dir and file_name not in TOP_DIR_FILE_INCLUDE_LIST:
                continue
            elif is_metadata_dir:
                file_match_num = int(file_name.split(".")[0].split("_")[-1])
                if file_match_num not in MATCH_ID_RANGE:
                    continue

            target_download_path = output_path / file_name
            if target_download_path.exists():
                assert target_download_path.is_file()
                local_last_modified_utc: datetime = datetime.fromtimestamp(target_download_path.stat().st_mtime,
                                                                           tz=timezone.utc)
                remote_last_modified_utc: datetime = datetime.strptime(tds[3].get_text(strip=True).removesuffix(" UTC"),
                                                                       "%Y-%m-%d %H:%M:%S.%f %z").astimezone(
                    timezone.utc)
                # skip download if local is newer or both files were modified within a minute of each other
                if remote_last_modified_utc <= local_last_modified_utc or (
                        remote_last_modified_utc - local_last_modified_utc < timedelta(minutes=1)):
                    print(f"Skipping download of {file_name} (local file is up to date)")
                    continue
            download_url = urljoin(DOMAIN, tds[4].find_all("li")[0].a["href"])
            download_file(download_url, target_download_path, int(tds[1].get_text(strip=True).split(" ")[0]))


def download_file(url: str, output_path: Path, total_size: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for retry in range(MAX_DOWNLOAD_RETRIES):
        try:
            with requests.get(
                    url,
                    stream=True,
                    headers={"Accept-Encoding": "*/*", "Connection": "close"},
                    timeout=(5, 10),
            ) as r:
                r.raise_for_status()

                with open(output_path, "wb") as f, tqdm(total=total_size, unit='B', unit_scale=True,
                                                        desc=output_path.name) as pbar:
                    for chunk in r.iter_content(chunk_size=64 * 1024):  # 64 KiB
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
        except Exception as e:
            print(f"Failed to download {url}: {e} (attempt {retry + 1}/{MAX_DOWNLOAD_RETRIES})")
        else:
            break


if __name__ == "__main__":
    download_files_recursively(BASE_URL, Path(LOCAL_DIR), False)
    exit(0)
