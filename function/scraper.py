
import os
import re
import unicodedata
import numpy as np
from urllib.parse import urlparse
import pandas as pd

import requests
from cleanco import basename
from polyfuzz import PolyFuzz
from polyfuzz.models import TFIDF
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options


def clean(company_name):
    ### Clean company name

    # Remove all capital letter
    company_name = company_name.lower()

    # Replace non-ASCII characters
    company_name = unicodedata.normalize('NFKD', company_name).encode('ASCII', 'ignore').decode()

    # Remove punctuation
    company_name = re.sub(r'[^\w\s]','',company_name)

    # Remove common legal business suffixes
    company_name = basename(company_name)

    # Remove the most common words using regular expressions
    suffix = ['holding', 'holdings', 'co', 'se', 'ua', 'corporation', 'international', 'group', 'groep',\
              'investments', 'acquisition']
    pattern = '|'.join(map(re.escape, suffix))
    company_name = re.sub(r'\b(?:{})\b'.format(pattern), '', company_name)
    return company_name

def name_match(company, string):
    ### Match company name
    model = PolyFuzz('TF-IDF').match([clean(company)], [clean(string)])
    score = model.get_matches()['Similarity'][0]
    return score

def fuzzy_merge(df_1, df_2, key1, key2, threshold=80, limit=1):
    """
    :param df_1: the left table to join
    :param df_2: the right table to join
    :param key1: key column of the left table
    :param key2: key column of the right table
    :param threshold: how close the matches should be to return a match, based on Levenshtein distance
    :param limit: the amount of matches that will get returned, these are sorted high to low
    :return: dataframe with boths keys and matches
    """
    s = df_2[key2].tolist()
    
    m = df_1[key1].apply(lambda x: process.extract(x, s, limit=limit))    
    df_1['matches'] = m
    
    m2 = df_1['matches'].apply(lambda x: ', '.join([i[0] for i in x if i[1] >= threshold]))
    df_1['matches'] = m2
    
    return df_1

def download_pdf(link_href, folder_name, downloads):
    ### Download pdf from the url
    # Get company and pdf file name
    filename = os.path.basename(link_href)
    folder_path = os.path.join('./data/googlesearch/', folder_name)

    # Check if folder exists. If not, create it.
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    try:
        header={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}
        response = requests.get(link_href, verify=False, headers=header)
        with open(os.path.join(folder_path, filename), 'wb') as f:
            f.write(response.content)
        print('Download pdf from: ' + link_href)
        downloads.append(link_href)
    except:
        pass

def load_html(url):
    ### Use selenium to fully load html
    # Opens the browser up in background
    chrome_options = Options()  
    chrome_options.add_argument("--headless") 

    with Chrome(options=chrome_options) as browser:
        browser.get(url)
        html = browser.page_source
    
    return html

def google_search(search_term, api_key, cse_id, num_results=5, **kwargs):
    #create google search query
    query = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cse_id}&q={search_term}&num={num_results}"
    response = requests.get(query).json()
    return response

def extract_domain(url):
    # Function to extract domain from report url
    parsed_url = urlparse(url)
    # only to get the domain
    domain = '{uri.scheme}://{uri.netloc}'.format(uri=parsed_url)
    return domain

def calculate_percentile_thresholds(data, percentiles=[25, 50, 75, 80, 90, 95]):
    calculated_percentiles = np.percentile(data, percentiles)
    percentile_table = {
        percentile: value
        for percentile, value in zip(percentiles, calculated_percentiles)
    }
    return percentile_table


def extract_year(url, current_year):
    # Find all four-digit numbers that could represent a year
    potential_years = re.findall(r'(\d{4})', url)
    # Filter out numbers that are not plausible years (e.g., not between 1900 and the current year)
    plausible_years = [int(year) for year in potential_years if 1900 <= int(year) <= current_year]
    # Return the most plausible year 
    return max(plausible_years, default=None)


def search_and_save_results(
        search_companies,
        search_year,
        api_key,
        cse_id,
        search_query = "sustainability report pdf",
        num_results = 5,
        data_directory = "data/WS_ISA",
        save_to = "searchresult/isa_report_url.csv",
        load_from = "searchresult/isa_report_url.csv",
):
    # Load existing data if available
    if os.path.isfile(load_from):
        existing_data = pd.read_csv(load_from, index_col=0)
    else:
        existing_data = pd.DataFrame(columns=['company', 'url'])

    # Loop through search companies
    for search_company in search_companies:
        if "/" in search_company:
            search_company = search_company.replace("/", "-")
        search_query = f'{search_company} {search_query} {search_year}'
        results = google_search(search_query, api_key, cse_id, num_results=num_results)
        if not results:
            print(f"no result found")
        # create directory if does not yet exist
        try:
            os.mkdir(f"{data_directory}/{search_company}")
        except:
            pass
        for item in results.get('items', []):
            link = item.get('link')
            print(f"found link: {link}")
            if link.endswith('.pdf'):
                print(f"PDF link found: {link}")
                # Download the PDF file
                filename = link.split('/')[-1]
                if not os.path.isfile(f"{data_directory}/{search_company}/{filename}"):
                    try:
                        response = requests.get(link, timeout=30)
                    except Exception as e:
                        print(f"error downloading file: {e}")
                        print(f"continuing with the rest...")
                    with open(f"{data_directory}/{search_company}/{filename}", 'wb') as file:
                        file.write(response.content)
                    print(f'Downloaded to {data_directory}/{search_company}/{filename}')
                else:
                    print(f"{filename} already exists. Skipping download.")
                company = pd.DataFrame([[search_company, link]], columns=['company', 'url'])
                existing_data = pd.concat([existing_data, company])
    
    # Save the updated DataFrame
    existing_data.to_csv(save_to)
    return existing_data
