import json
import os
import os.path
import re
import zipfile
from datetime import datetime

import pandas as pd
from adobe.pdfservices.operation.auth.service_principal_credentials import (
    ServicePrincipalCredentials,
)
from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
from adobe.pdfservices.operation.io.stream_asset import StreamAsset
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import (
    ExtractElementType,
)
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import (
    ExtractPDFParams,
)
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_renditions_element_type import (
    ExtractRenditionsElementType,
)
from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import (
    ExtractPDFResult,
)
from pydantic import SecretStr


def get_dict_xlsx(outputzipextract, xlsx_file):
    """
    Function to read excel output from adobe API
    """
    # Read excel
    df = pd.read_excel(
        os.path.join(outputzipextract, xlsx_file),
        sheet_name="Sheet1",
        engine="openpyxl",
    )

    # Clean df
    df.columns = [re.sub(r"_x([0-9a-fA-F]{4})_", "", col) for col in df.columns]
    df = df.replace({r"_x([0-9a-fA-F]{4})_": ""}, regex=True)

    # Convert df to string
    data_dict = df.to_dict(orient="records")

    return data_dict


class Secrets:
    def __getattribute__(self, name: str) -> SecretStr:
        secret = os.getenv(name)
        if secret is None:
            raise ValueError(f"Secret {name} not found in environment variables")
        return SecretStr(secret)


secrets = Secrets()


# adopted from: https://github.com/adobe/pdfservices-python-sdk-samples/blob/main/src/extractpdf/extract_text_table_info_with_figures_tables_renditions_from_pdf.py
def adobeLoader(input_pdf, output_zip_path):
    """
    Function to run adobe API and create output zip file
    """
    client_id = os.getenv("ADOBE_CLIENT_ID")
    client_secret = os.getenv("ADOBE_CLIENT_SECRET")
    # Initial setup, create credentials instance.
    with open(input_pdf, "rb") as file:
        input_stream = file.read()

    # Initial setup, create credentials instance
    credentials = ServicePrincipalCredentials(
        client_id=client_id,
        client_secret=client_secret,
    )

    # Creates a PDF Services instance
    pdf_services = PDFServices(credentials=credentials)

    # Creates an asset(s) from source file(s) and upload
    input_asset = pdf_services.upload(
        input_stream=input_stream, mime_type=PDFServicesMediaType.PDF
    )

    # Create parameters for the job
    extract_pdf_params = ExtractPDFParams(
        elements_to_extract=[ExtractElementType.TEXT, ExtractElementType.TABLES],
        elements_to_extract_renditions=[
            ExtractRenditionsElementType.TABLES,
            ExtractRenditionsElementType.FIGURES,
        ],
    )

    # Creates a new job instance
    extract_pdf_job = ExtractPDFJob(
        input_asset=input_asset, extract_pdf_params=extract_pdf_params
    )

    # Submit the job and gets the job result
    location = pdf_services.submit(extract_pdf_job)
    pdf_services_response = pdf_services.get_job_result(location, ExtractPDFResult)

    # Get content from the resulting asset(s)
    result_asset: CloudAsset = pdf_services_response.get_result().get_resource()
    stream_asset: StreamAsset = pdf_services.get_content(result_asset)

    # Creates an output stream and copy stream asset's content to it
    with open(output_zip_path, "wb") as file:
        file.write(stream_asset.get_input_stream())


def extract_text_from_file_adobe(output_zip_path, output_zipextract_folder):
    """
    Function to extract text and table from adobe output zip file
    """
    json_file_path = os.path.join(output_zipextract_folder, "structuredData.json")
    # check if json file exist:
    if os.path.exists(json_file_path):
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} JSON file already exists. Skipping extraction."
        )
    else:
        try:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} unzip file")
            # Open the ZIP file
            with zipfile.ZipFile(output_zip_path, "r") as zip_ref:
                # Extract all the contents of the ZIP file to the current working directory
                zip_ref.extractall(path=output_zipextract_folder)
        except Exception as e:
            print("----Error: cannot unzip file:")
            print(e)

    try:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} open json file")
        # Opening JSON file
        with open(
            os.path.join(output_zipextract_folder, "structuredData.json")
        ) as json_file:
            data = json.load(json_file)
    except Exception as e:
        print("----Error: cannot open json file")
        print(e)
        return pd.DataFrame()

    # try:
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} extract text")
    dfs = pd.DataFrame()
    page = ""
    try:  # Loop through elements in the document
        for ele in data["elements"]:
            df = pd.DataFrame()

            # Get element page
            if "Page" in ele.keys():
                page = ele["Page"]

            # Append table
            if any(x in ele["Path"] for x in ["Table"]):
                if "filePaths" in ele:
                    if [s for s in ele["filePaths"] if "xlsx" in s]:
                        # Read excel table
                        data_dict = get_dict_xlsx(
                            output_zipextract_folder, ele["filePaths"][0]
                        )
                        json_string = json.dumps(data_dict)
                        df = pd.DataFrame({"text": json_string}, index=[0])

            # Append text
            elif ("Text" in ele.keys()) and ("Figure" not in ele["Path"]):
                df = pd.DataFrame({"text": ele["Text"]}, index=[0])

            # print(page)
            df["page_number"] = page
            dfs = pd.concat([dfs, df], axis=0)

    except Exception as e:
        print("----Error: processing elements in JSON")
        print(e)

    dfs = dfs.reset_index(drop=True)
    # Groupby page
    dfs = dfs.dropna()
    dfs = dfs.groupby("page_number")["text"].apply(lambda x: "\n".join(x)).reset_index()
    text_content = "\n".join(dfs["text"].values)
    # text_content = dfs["text"].apply(lambda x: "\n".join(x)).reset_index()
    return text_content
