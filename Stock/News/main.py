import requests

# Define the CoreNLP server URL
url = 'http://localhost:9000'

# Read the input text file
with open('headlines.txt', 'r') as file:
    text = file.read()

# Define the properties for CoreNLP
properties = {
    'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,dcoref',
    'outputFormat': 'json'
}

# Make the request to the CoreNLP server
response = requests.post(url, params={'properties': str(properties)}, data=text.encode('utf-8'))

# Check if the request was successful
if response.status_code == 200:
    # Save the output to a file
    with open('output.json', 'w') as output_file:
        output_file.write(response.text)
    print('Processing completed successfully. Output saved to output.json.')
else:
    print(f'Error: {response.status_code}')
