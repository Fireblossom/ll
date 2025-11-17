import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import time
import sys
import os
import json

URL = "https://catalog.ldc.upenn.edu/byyear"

YEARS_TO_PROCESS = None  
MAX_DATASETS = None      
DELAY_BETWEEN_REQUESTS = 1  

PROGRESS_FILE = "/mlde/ll/ldc_progress.json"
METADATA_FILE = "/mlde/ll/ldc_datasets_metadata.csv"
ENABLE_RESUME = True  

def save_progress(all_datasets_info, processed_indices, all_metadata):
    progress_data = {
        'total_datasets': len(all_datasets_info),
        'processed_indices': processed_indices,
        'processed_count': len(processed_indices),
        'metadata': all_metadata,
        'timestamp': time.time()
    }

    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, ensure_ascii=False, indent=2)

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load progress file: {e}")
    return None

def extract_dataset_metadata(dataset_url, max_retries=3):
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(2 ** attempt) 

            page = requests.get(dataset_url, timeout=30)
            page.raise_for_status()
            soup = BeautifulSoup(page.content, "html.parser")

            metadata = {}

            for row in soup.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) >= 2:
                    field_name = cells[0].get_text(strip=True).rstrip(':')
                    field_value = cells[1].get_text(strip=True)

                    if field_name == "Item Name":
                        title_span = cells[1].find("span")
                        if title_span:
                            metadata["Title"] = title_span.get_text(strip=True)
                    elif field_name in [
                        "Author(s)", "LDC Catalog No.", "ISLRN", "DOI", "Release Date",
                        "Member Year(s)", "DCMI Type(s)", "Data Source(s)", "Application(s)",
                        "Language(s)", "Language ID(s)", "License(s)", "Online Documentation",
                        "Licensing Instructions", "Citation",
                        "ISBN", "Sample Type", "Sample Rate", "Project(s)", "Related Works"
                    ]:
                        metadata[field_name] = field_value

            related_works_element = soup.find("td", string="Related Works:")
            if related_works_element:
                next_td = related_works_element.find_next_sibling("td")
                if next_td:
                    related_links = next_td.find_all("a")
                    if related_links:
                        related_works = []
                        for link in related_links:
                            if link.get_text(strip=True) not in ["View", "Hide"]:
                                related_works.append(link.get_text(strip=True))
                        if related_works:
                            metadata["Related Works"] = "; ".join(related_works)

            return metadata

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"Failed to fetch {dataset_url} after {max_retries} attempts: {e}")
                return None
            else:
                print(f"Attempt {attempt + 1} failed for {dataset_url}, retrying...")
                continue
        except Exception as e:
            print(f"Error extracting metadata from {dataset_url}: {e}")
            return None

    return None

try:
    start_index = 0
    existing_metadata = []

    if ENABLE_RESUME:
        existing_progress = load_progress()
        if existing_progress:
            print("Existing progress file found.")
            print(f"Processed: {existing_progress['processed_count']}/{existing_progress['total_datasets']}")
            print(f"Last saved at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(existing_progress['timestamp']))}")

            choice = input("Continue from last progress? (y/n): ").lower().strip()
            if choice == 'y':
                print("Continuing from last progress...")
                start_index = existing_progress['processed_count']
                existing_metadata = existing_progress.get('metadata', [])
                print(f"Starting from dataset index {start_index + 1}...")
            else:
                print("Starting over...")
                if os.path.exists(PROGRESS_FILE):
                    os.remove(PROGRESS_FILE)

    page = requests.get(URL)
    page.raise_for_status()

    soup = BeautifulSoup(page.content, "html.parser")

    year_elements = soup.find_all("h2")

    all_datasets_info = []
    total_datasets = 0

    for year_element in year_elements:
        year = year_element.get_text(strip=True)

        if YEARS_TO_PROCESS and year not in YEARS_TO_PROCESS:
            continue

        dataset_table = year_element.find_next_sibling("table")

        if dataset_table:
            for row in dataset_table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) >= 3:
                    link_element = cells[0].find("a")
                    if link_element and 'href' in link_element.attrs:
                        dataset_url = urljoin(URL, link_element['href'])
                        catalog_id = link_element.get_text(strip=True)

                        title_span = cells[2].find("span")
                        title = title_span.get_text(strip=True) if title_span else "Unknown"

                        dataset_info = {
                            'Year': year,
                            'Catalog_ID': catalog_id,
                            'Title': title,
                            'URL': dataset_url
                        }

                        all_datasets_info.append(dataset_info)
                        total_datasets += 1

                        if MAX_DATASETS and total_datasets >= MAX_DATASETS:
                            print(f"Reached maximum dataset limit: {MAX_DATASETS}")
                            break

            if MAX_DATASETS and total_datasets >= MAX_DATASETS:
                break

    print(f"Found {total_datasets} datasets, starting metadata collection...")
    print("This may take some time, please wait...")


    all_metadata = existing_metadata.copy()
    success_count = len([m for m in all_metadata if 'Title' in m and m.get('Title') != 'Unknown'])
    error_count = len(all_metadata) - success_count
    processed_indices = list(range(start_index))

    for i in range(start_index, total_datasets):
        dataset_info = all_datasets_info[i]

        percent = ((i + 1) / total_datasets) * 100
        print(f"\rProgress: [{i+1:4d}/{total_datasets:4d}] ({percent:5.1f}%) - Processing: {dataset_info['Title'][:50]}...", end="")

        metadata = extract_dataset_metadata(dataset_info['URL'])

        if metadata:
            complete_info = {**dataset_info, **metadata}
            if i < len(all_metadata):
                all_metadata[i] = complete_info
            else:
                all_metadata.append(complete_info)
            success_count += 1
        else:
            if i < len(all_metadata):
                all_metadata[i] = dataset_info
            else:
                all_metadata.append(dataset_info)
            error_count += 1

        processed_indices.append(i)

        if (i + 1) % 10 == 0:
            save_progress(all_datasets_info, processed_indices, all_metadata)
            print(f"\rProgress: [{i+1:4d}/{total_datasets:4d}] ({percent:5.1f}%) - Progress saved")

        if i < total_datasets - 1 and DELAY_BETWEEN_REQUESTS > 0:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\rProgress: [{total_datasets:4d}/{total_datasets:4d}] (100.0%)")
    print(f"\nMetadata collection completed: success {success_count}, failed {error_count}")

    if ENABLE_RESUME:
        save_progress(all_datasets_info, list(range(total_datasets)), all_metadata)

    if all_metadata:
        df = pd.DataFrame(all_metadata)

        columns_order = ['Year', 'Catalog_ID', 'Title', 'Author(s)', 'LDC Catalog No.', 'ISBN',
                        'ISLRN', 'DOI', 'Release Date', 'Member Year(s)', 'DCMI Type(s)',
                        'Sample Type', 'Sample Rate', 'Data Source(s)', 'Project(s)',
                        'Application(s)', 'Language(s)', 'Language ID(s)', 'License(s)',
                        'Online Documentation', 'Licensing Instructions', 'Citation',
                        'Related Works', 'URL']

        existing_columns = [col for col in columns_order if col in df.columns]
        df = df[existing_columns]

        output_file = "/mlde/ll/ldc_datasets_metadata.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nMetadata CSV saved to: {output_file}")
        print(f"Collected metadata for {len(all_metadata)} datasets")

        print("\n--- Data preview ---")
        print(df.head())

        print("\n--- Statistics ---")
        print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
        print(f"Number of unique languages: {df['Language(s)'].nunique()}")
        print(f"Number of unique projects: {df['Project(s)'].nunique()}")

    else:
        print("No metadata could be collected")


except requests.exceptions.RequestException as e:
    print(f"Error: unable to access URL. Please check your network connection. ({e})")
except Exception as e:
    print(f"Unexpected error during processing: {e}")