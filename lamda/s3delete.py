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
        successful_files = []
        failed_files = []

        for record in event['Records']:
            s3_bucket = record['s3']['bucket']['name']
            s3_object_key = record['s3']['object']['key']

            # Prepare payload to call FastAPI endpoint for deletion
            payload = {
                "file_name": s3_object_key,
                "collection_name": "document_collection"
            }

            # Call FastAPI to delete file record from Milvus
            response = requests.post(FASTAPI_URL, json=payload)

            if response.status_code == 200:
                successful_files.append(s3_object_key)
            else:
                failed_files.append(s3_object_key)

        # Send summary notification to Notion
        message = f"Successfully deleted files: {successful_files}\nFailed deletions: {failed_files}"
        send_notion_notification("S3 File Deletion Summary", message)

        return {
            'statusCode': 200,
            'body': json.dumps(f"File deletions processed. Success: {successful_files}, Failed: {failed_files}")
        }

    except Exception as e:
        send_notion_notification("S3 File Deletion Failure", f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error deleting files: {str(e)}")
        }
