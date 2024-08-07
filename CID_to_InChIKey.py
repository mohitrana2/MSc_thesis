import requests
import logging
import time
import csv

# Configure logging
logging.basicConfig(filename='pubchem_inchi_download.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_inchi_key(cid):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/InChIKey/TXT"
    response = requests.get(url)

    if response.status_code == 200:
        inchi_key = response.text.strip()
        logging.info(f"Successfully retrieved InChI Key for CID {cid}: {inchi_key}")
        return inchi_key
    else:
        logging.error(f"Failed to retrieve InChI Key for CID {cid}. Status code: {response.status_code}")
        return None

def download_inchi_keys(cid_list, output_file):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["CID", "InChIKey"])
        for cid in cid_list:
            inchi_key = get_inchi_key(cid)
            if inchi_key:
                writer.writerow([cid, inchi_key])
            # Add a delay to prevent overloading the server
            time.sleep(0.5)  # Sleep time to prevent overloading the PubChem server

def read_cid_list(csv_file):
    cid_list = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and len(row) > 0:  # Check if row is not empty
                cid_list.append(row[0])
    return cid_list

if __name__ == "__main__":
    # Read CIDs from the input CSV file
    input_csv = "/data4/msc23104470/TTD/P1-03-TTD_CIDs.csv"
    cid_list = read_cid_list(input_csv)

    output_file = "/data4/msc23104470/TTD/P1-03-TTD_CID_to_InChIKey.csv"

    logging.info("Starting download of InChI Keys for provided CIDs")
    download_inchi_keys(cid_list, output_file)
    logging.info("Finished downloading InChI Keys")
