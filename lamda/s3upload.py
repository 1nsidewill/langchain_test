import json
import requests
import os

FASTAPI_URL = "http://<your-fastapi-server>/process_s3_files/"
NOTION_API_URL = "https://api.notion.com/v1/pages"
NOTION_API_KEY= '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'# Add your Notion API key here
NOTION_PAGE_ID = '12063685a48180febdbfd5217c50e00a'


# Notion notification helper function
def send_notion_notification(title, message):
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }

    payload = {
        "parent": {"page_id": NOTION_PAGE_ID},
        "properties": {
            "title": [
                {
                    "text": {
                        "content": title
                    }
                }
            ]
        },
        "children": [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "text": [
                        {
                            "type": "text",
                            "text": {
                                "content": message
                            }
                        }
                    ]
                }
            }
        ]
    }

    response = requests.post(NOTION_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"Failed to send notification to Notion: {response.status_code}, {response.text}")
    else:
        print("Notion notification sent successfully.")

def lambda_handler(event, context):
    try:
        # Lists to store successful and failed file names
        successful_files = []
        failed_files = []

        # List to store file names
        file_names = []

        # Iterate over S3 event records
        for record in event['Records']:
            s3_bucket = record['s3']['bucket']['name']
            s3_object_key = record['s3']['object']['key']
            file_names.append(s3_object_key)

        # Payload to send to FastAPI
        payload = {
            "file_names": file_names,
            "s3_bucket": s3_bucket
        }

        # Call the FastAPI server to process files
        response = requests.post(FASTAPI_URL, json=payload)

        if response.status_code == 200:
            # Assuming the FastAPI response contains a list of successfully processed files
            result = response.json()
            successful_files = [file['file_name'] for file in result.get('processed_files', [])]
        else:
            failed_files = file_names

        # After processing all files, send a summary notification to Notion
        message = f"Successfully processed files: {successful_files}\nFailed files: {failed_files}"
        send_notion_notification("S3 File Processing Summary", message)

        return {
            'statusCode': 200,
            'body': json.dumps('Files processed successfully!')
        }

    except Exception as e:
        # Send failure notification to Notion
        send_notion_notification("S3 File Processing Failure", f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error processing files: {str(e)}")
        }
