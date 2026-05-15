"""
UMA dataset downloader.

Logs into the protected WordPress page that hosts the UMA dataset, scans for
pack files (metadata / video / fgseg / neus2), filters by the SUBJECTS and
MODALITIES selected at the top of this file, and downloads each match with a
progress bar. Resumable: existing files are skipped, partial *.tmp files are
removed on failure.

Pack filename convention used to filter:
    Subject_<N>_metadata.zip
    Subject_<N>_<training|testing>_<modality>.<ext>[.part_<NN>]
where  N         ∈ 0..4
       modality  ∈ {metadata, fgseg, neus2, video}
       ext       ∈ {zip, tar, tar.gz, tar.part_NN}
"""

import os
import re
import sys
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# ============================================================
# 🔧 CONFIGURATION — edit these before running
# ============================================================
BASE_URL = "https://gvv-assets.mpi-inf.mpg.de/uma"   # UMA dataset page
USERNAME = "login"                                    # WordPress login
PASSWORD = "pw"                                       # WordPress password
OUTPUT_FOLDER = "./uma_downloads"                     # where to save files

# --- Selection (set to [] to mean "all") ---
SUBJECTS   = [0, 1, 2, 3, 4]                          # which subjects (0..4)
MODALITIES = ["metadata", "fgseg", "neus2", "video"]  # which data types

# --- File-extension matching (rarely needs changing) ---
FILE_TYPES = "part_*,tar.gz,tar,zip,txt,md"
RECURSIVE  = False
TIMEOUT    = 30
CHUNK      = 1024 * 256          # 256 KB per read
# ============================================================


def find_login_form(soup):
    for form in soup.find_all("form"):
        text = form.get_text(" ").lower()
        if any(k in text for k in ["login", "sign in", "password"]):
            return form
    return None


def wordpress_login(session, base_url, username, password):
    print("[+] Accessing protected page...")
    resp = session.get(base_url, allow_redirects=True, timeout=TIMEOUT)
    if resp.url == base_url and "logout" in resp.text.lower():
        print("[+] Already logged in or public page.")
        return True
    login_url = resp.url
    print(f"[+] Redirected to login page: {login_url}")
    soup = BeautifulSoup(resp.text, "html.parser")
    form = find_login_form(soup)
    if not form:
        print("[-] Could not find login form.")
        return False
    action = form.get("action") or login_url
    login_action = urljoin(login_url, action)
    payload = {i.get("name"): i.get("value", "") for i in form.find_all("input") if i.get("name")}
    user_field = next((n for n in payload if "user" in n or "login" in n or "email" in n), None)
    pass_field = next((n for n in payload if "pass" in n or "pwd" in n), None)
    if not user_field or not pass_field:
        print("[-] Could not detect username/password fields.")
        return False
    payload[user_field] = username
    payload[pass_field] = password
    print(f"[+] Logging in via {login_action}")
    r = session.post(login_action, data=payload, allow_redirects=True, timeout=TIMEOUT)
    if "logout" in r.text.lower() or "dashboard" in r.text.lower() or r.url.startswith(base_url):
        print("[+] Login successful.")
        return True
    print("[-] Login failed.")
    return False


def build_ext_pattern(file_types):
    """Convert comma-separated extensions (with * wildcard) to a regex."""
    parts = []
    for raw in file_types.split(","):
        ext = raw.strip().lstrip(".")
        if not ext:
            continue
        regex = re.escape(ext).replace(r"\*", r"[A-Za-z0-9_]*")
        parts.append(regex)
    return re.compile(r"\.(?:" + "|".join(parts) + r")$", re.IGNORECASE)


def matches_uma_filter(url, subjects, modalities):
    """True if the filename matches a Subject_<N>_..._<modality>.* pattern
    and (N, modality) are both in the allowed sets. Empty set = allow all."""
    filename = os.path.basename(url)
    m = re.match(r"^Subject_(\d+)_", filename)
    if not m:
        return False
    subj = int(m.group(1))
    if subjects and subj not in subjects:
        return False
    if not modalities:
        return True
    for mod in modalities:
        if re.match(rf"^Subject_{subj}_(?:training_|testing_)?{re.escape(mod)}[._]", filename):
            return True
    return False


def find_files(session, base_url, ext_re, visited=None):
    if visited is None:
        visited = set()
    visited.add(base_url)
    print(f"[+] Scanning: {base_url}")
    resp = session.get(base_url, timeout=TIMEOUT)
    soup = BeautifulSoup(resp.text, "html.parser")

    files_set = set()
    for link in soup.find_all("a", href=True):
        href = link["href"]
        full = urljoin(base_url, href)
        if ext_re.search(href):
            files_set.add(full)
        elif RECURSIVE and full not in visited and full.startswith(base_url) and not re.search(r"\.\w+$", href):
            files_set.update(find_files(session, full, ext_re, visited))
    return list(files_set)


def download_file(session, url, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.basename(url)
    local_path = os.path.join(output_folder, filename)
    tmp_path = local_path + ".tmp"

    if os.path.exists(local_path):
        print(f"[+] File already exists, skipping: {filename}")
        return

    headers = {"Accept-Encoding": "identity", "Accept": "application/octet-stream"}

    try:
        with session.get(url, headers=headers, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            total = r.headers.get("Content-Length")
            total = int(total) if total else None

            with open(tmp_path, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=filename,
                leave=True,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=CHUNK):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        os.rename(tmp_path, local_path)
        print(f"[+] Saved: {filename} ({os.path.getsize(local_path)} bytes)")

    except Exception as e:
        print(f"[!] Error downloading {filename}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    print("=== UMA Dataset Downloader ===")
    print(f"    subjects   = {SUBJECTS if SUBJECTS else 'ALL'}")
    print(f"    modalities = {MODALITIES if MODALITIES else 'ALL'}")
    print(f"    output     = {OUTPUT_FOLDER}")

    ext_re = build_ext_pattern(FILE_TYPES)

    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0 (UMADownloader/1.0)", "Accept-Encoding": "identity"})

        if not wordpress_login(session, BASE_URL, USERNAME, PASSWORD):
            sys.exit(1)

        all_files = find_files(session, BASE_URL, ext_re)
        files = [u for u in all_files if matches_uma_filter(u, SUBJECTS, MODALITIES)]
        files.sort(key=lambda x: os.path.basename(x).lower())

        print(f"[+] Found {len(all_files)} candidate files, "
              f"{len(files)} match the subject+modality filter.")
        if not files:
            sys.exit(0)

        for url in files:
            download_file(session, url, OUTPUT_FOLDER)

    print("\n[+] All downloads completed.")


if __name__ == "__main__":
    main()
