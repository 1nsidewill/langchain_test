import os
import json
from unstructured_ingest.v2.pipeline.pipeline import Pipeline
from unstructured_ingest.v2.interfaces import ProcessorConfig
from unstructured_ingest.v2.processes.connectors.local import (
    LocalIndexerConfig,
    LocalDownloaderConfig,
    LocalConnectionConfig,
    LocalUploaderConfig
)
from unstructured_ingest.v2.processes.partitioner import PartitionerConfig

# Define input and output folders
INPUT_FOLDER = "input_folder"
OUTPUT_FOLDER = "unstructured_folder"
PARSED_FOLDER = "parsed_folder"

# Create output folders if they don't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

if not os.path.exists(PARSED_FOLDER):
    os.makedirs(PARSED_FOLDER)

# Function to parse JSON files and save texts to parsed_folder
def parse_json_file(input_file_path: str, output_file_path: str):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        file_elements = json.load(file)

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for element in file_elements:
            text = element.get("text", "")
            output_file.write(text + "\n")

if __name__ == "__main__":
    # # Run the pipeline to process files
    # Pipeline.from_configs(
    #     context=ProcessorConfig(),
    #     indexer_config=LocalIndexerConfig(input_path=INPUT_FOLDER),
    #     downloader_config=LocalDownloaderConfig(),
    #     source_connection_config=LocalConnectionConfig(),
    #     partitioner_config=PartitionerConfig(
    #         partition_by_api=True,
    #         api_key=os.getenv("UNSTRUCTURED_API_KEY"),
    #         partition_endpoint=os.getenv("UNSTRUCTURED_API_URL"),
    #         strategy="hi_res",
    #         additional_partition_args={
    #             "split_pdf_page": True,
    #             "split_pdf_allow_failed": True,
    #             "split_pdf_concurrency_level": 15
    #         }
    #     ),
    #     uploader_config=LocalUploaderConfig(output_dir=OUTPUT_FOLDER)
    # ).run()

    # Parse the processed JSON files and save texts to parsed_folder
    for file_name in os.listdir(OUTPUT_FOLDER):
        if file_name.endswith('.json'):
            input_file_path = os.path.join(OUTPUT_FOLDER, file_name)
            output_file_path = os.path.join(PARSED_FOLDER, f"parsed_{file_name}")
            parse_json_file(input_file_path, output_file_path)

print("Processing and parsing complete.")