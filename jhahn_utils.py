import os.path
import io

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload


def download_file(cred_folder, folder_name_in_drive, file_name, save_dir):
    

    def get_id_from_path(service, path):
        """
        Translates a folder path into a Google Drive folder ID.
        Returns the ID of the final folder in the path, or None if not found.
        """
        # Start searching from the root directory
        parent_id = 'root'
        
        # Split the path into individual folder names
        folders = path.split('/')
        
        for folder_name in folders:
            # Query for the current folder in the path
            query = (f"name = '{folder_name}' and "
                    f"mimeType = 'application/vnd.google-apps.folder' and "
                    f"'{parent_id}' in parents and "
                    f"trashed = false")
            
            response = service.files().list(q=query,
                                            spaces='drive',
                                            fields='files(id, name)').execute()
            results = response.get('files', [])

            if not results:
                print(f"❌ Folder '{folder_name}' not found in the path.")
                return None # Path is invalid
            elif len(results) > 1:
                print(f"⚠️ Multiple folders named '{folder_name}' found. The path is ambiguous.")
                return None # Path is ambiguous

            # The path is valid so far, move to the next level
            parent_id = results[0].get('id')
            print(f"✅ Found folder '{folder_name}' (ID: {parent_id})")

        # Return the ID of the last folder in the path
        return parent_id



    # This scope allows the script to read all files in your Google Drive.
    # For more granular control, you could use 'https'://www.googleapis.com/auth/drive.readonly'
    # if you only need to read files.
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

    """
    Authenticates with Google Drive API and downloads a file.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(cred_folder+"/"+'token.json'):
        creds = Credentials.from_authorized_user_file(cred_folder+"/"+'token.json', SCOPES)
        
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # This will trigger the OAuth 2.0 flow. It will open a browser
            # window for you to log in and authorize the script.
            flow = InstalledAppFlow.from_client_secrets_file(
                cred_folder+"/"+'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            
        # Save the credentials for the next run
        with open(cred_folder+"/"+'token.json', 'w') as token:
            token.write(creds.to_json())
            

    try:
        # Build the service object for interacting with the API
        service = build('drive', 'v3', credentials=creds)


        #folder_name  = 'jupyterhub/2024_skin'
        folder_id = get_id_from_path(service, folder_name_in_drive)


        # --- YOU NEED TO CHANGE THIS ---
        # Get the ID of the file you want to download.
        # You can get this from the URL of the file in Google Drive.
        # e.g., https://drive.google.com/file/d/THIS_IS_THE_FILE_ID/view
        
        

        # 2. Search for the file within that folder
        print(f"📄 Searching for file '{file_name}' inside '{folder_name_in_drive}'...")
        file_query = (f"name = '{file_name}' and "
                        f"'{folder_id}' in parents and "
                        f"trashed = false")

        response = service.files().list(q=file_query,
                                            spaces='drive',
                                            fields='files(id, name)').execute()
        files = response.get('files', [])

        if not files:
            print(f"❌ No file found with the name '{file_name}' in folder '{folder_name_in_drive}'")
        if len(files) > 1:
            print(f"⚠️ Multiple files found with this name in the folder. Please use a unique name.")
        file_id = files[0]['id']



        # 1. Get file metadata to find the original name
        file_metadata = service.files().get(fileId=file_id).execute()
        file_name = file_metadata.get('name')
        print(f"Found file: '{file_name}'. Starting download...")

        # 2. Download the file's content
        request = service.files().get_media(fileId=file_id)
        
        # Use an in-memory binary stream to hold the downloaded data
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")

        os.makedirs(save_dir,exist_ok=True)
        # 3. Save the downloaded content to a local file
        fh.seek(0) # Move the cursor to the beginning of the stream
        with open(save_dir+"/"+file_name, 'wb') as f:
            f.write(fh.read())
            
        print(f"\nFile '{file_name}' downloaded successfully!")

    except HttpError as error:
        print(f"An error occurred: {error}")
    except FileNotFoundError:
        print("The file with the specified ID was not found.")

if __name__ == '__main__':
    download_file('jupyterhub/2024_skin','skin.csv','data')

