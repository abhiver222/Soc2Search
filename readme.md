# Soc2Search
Search through AWS's SOC2 and view comprehensive answers with references by asking it natural language questions through a chat interface. 


## How to run chat interface
The app can be run with a web app or cli for debugging

### web app
1. export OPENAI_API_KEY, can be acquired from https://platform.openai.com/account/api-keys
2. run `python app.py`
3. Can we views on replits web view

### cli
1. export OPENAI_API_KEY, can be acquired from https://platform.openai.com/account/api-keys
2. run `python soc2search.py`

## Create embeddings
1. This  app is hardcoded to process AWS's SOC2 and will require changes to run for other documents.
It can be run as a one off to create new document embeddings.
2. Update the file path in the `soc2search.getEmbeddings` function to <file_to_process_embeddings_path>
3. Update the main fn for `soc2search.py` to uncomment the `getEmbeddings` call
4. run `python soc2search.py`