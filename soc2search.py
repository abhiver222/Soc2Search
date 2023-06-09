from PyPDF4 import PdfFileReader
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
import openai
import os
import math

import textract as textract
#read the content of pdf as text
import re


class Soc2Search():  # this is hardcoded for AWS Soc2

  def __init__(self):
    self.df = pd.read_csv("aiembedding_splitkandji.csv")
    self.df["embeddings"] = self.df.embeddings.apply(eval).apply(np.array)

  def test(self):
    print("testing")

  def process_text(self, text):
    replace_strings = [
      'Amazon.com, Inc. or its affiliates', 'Trade Secret',
      'Description of the Amazon Web Services System', '©20 22', '©2022',
      'Proprietary and Confidential Information',
      '8bbb8a4b-068c-4ae1-8aac-1848a8cddba0',
      '8bbb8a4b-068c-4ae1-8aac-1848a8cddba0'
    ]
    for rep in replace_strings:
      text = text.replace(rep, ' ')
    return " ".join(text.split())

  def process_text_new(self, text):
    replace_strings = [
      'Amazon.com, Inc. or its affiliates', 'Trade Secret',
      'Description of the Amazon Web Services System', '©20 22', '©2022',
      'Proprietary and Confidential Information',
      '8bbb8a4b-068c-4ae1-8aac-1848a8cddba0',
      '8bbb8a4b-068c-4ae1-8aac-1848a8cddba0'
    ]
    for rep in replace_strings:
      text = text.replace(rep, ' ')
    return text

  # def get_parts(self, text):
  #  num_parts = 4
  #  start = 0
  #  i = 0
  #  parts = []
  #  text_lengths = math.ceil(len(text) / num_parts)
  #  while start < len(text):
  #    next_end = text.find('.', (i + 1) * text_lengths, len(text))
  #    if next_end == -1:
  #      next_end = len(text)
  #    print(start)
  #    print(next_end)
  #    print(len(text))
  #    parts.append(text[start:next_end])  # include period
  #    start = start + next_end + 1  # start after period
  #  return parts

  def parse_doc(self, pdf):
    print("Parsing document")
    number_of_pages = len(pdf.pages)
    print(f"Total number of pages: {number_of_pages}")
    doc_text = []
    # for aws
    # if number_of_pages < 24:
    #   return

    start_aws = 24
    table_start_aws = 86
    start_dashlane = 10
    table_start_dashlane = 23
    start_kandji = 6
    table_start_kandji = 15
    print(pdf.pages[0].cropBox.upperLeft)
    print(pdf.pages[0].cropBox.lowerRight)
    for i in range(start_kandji, number_of_pages):
      page = pdf.pages[i]
      page.mediaBox.upperLeft = (0, float(page.mediaBox.upperRight[1]) * .9)
      page.mediaBox.lowerLeft = (0, float(page.mediaBox.upperRight[1]) * .1)
      page.mediaBox.upperRight = (page.mediaBox.upperRight[1],
                                  float(page.mediaBox.upperRight[1]) * .9)
      page.mediaBox.lowerRight = (float(page.mediaBox.upperRight[1]),
                                  float(page.mediaBox.upperRight[1]) * .1)

      ptext = self.process_text(page.extractText())
      parts = 4
      n = math.ceil(len(ptext) / parts)
      if n == 0:
        continue
      parts = [ptext[i:i + n] for i in range(0, len(ptext), n)]
      for p in parts:
        doc_text.append({'text': p, 'page': i + 1})
        # add an object of 2 cols, one page num, one text

    print("Done parsing document")
    # print(doc_text)
    return doc_text

  def doc_df(self, pdf):
    print('Creating dataframe')
    filtered_pdf = []
    for row in pdf:
      if len(row['text']) < 30:
        continue
      filtered_pdf.append(row)
    df = pd.DataFrame(filtered_pdf)
    df = df.drop_duplicates(subset=['text', 'page'], keep='first')
    df['length'] = df['text'].apply(lambda x: len(x))
    print('Done creating dataframe')
    return df

  def calculate_embeddings(self, df):
    print('Calculating embeddings')
    openai.api_key = os.getenv('OPENAI_API_KEY')
    embedding_model = "text-embedding-ada-002"
    embeddings = df.text.apply(
      [lambda x: get_embedding(x, engine=embedding_model)])
    df["embeddings"] = embeddings
    print('Done calculating embeddings')
    return df

  def search_embeddings(self, df, query, n=3, pprint=True):
    print("Searching embeddings")
    print(query)
    query_embedding = get_embedding(query, engine="text-embedding-ada-002")
    df["similarity"] = df.embeddings.apply(
      lambda x: cosine_similarity(x, query_embedding))

    results = df.sort_values("similarity", ascending=False, ignore_index=True)
    results = results.head(n)
    # print("search results")
    # print(results)
    global sources
    sources = []
    for i in range(n):
      sources.append({
        'Page ' + str(results.iloc[i]['page']):
        '... ' + results.iloc[i]['text'][:150] + ' ...'
      })
    # print(sources)
    return results

  """
  Create prompt for GPT3 model
  """

  def create_prompt(self, user_input, context):
    prompt = """
      You are a large language model whose expertise is reading and   
      summarizing soc 2 compliance documents and parts of soc 2 compliance 
      documents.
      
      You are given a question and some context from AWS SOC2 to answer from.

      Question: """ + user_input + """

      Answer the question as truthfully as possible using the provided text, 
      and if the answer is not contained within the text below, say "I don't 
      know". Be strict about this rule. Answer in a detailed manner.
            
      Annotate the page number from the context for sentences in the answer.
      Page number is marked as ### page: <page_number> in the context. 
      Remove the "###" string from the page number in the anotation.
      Be strict about this rule.
            
      Context:     
            
      """ + context + """
      
      Return a detailed answer only using the above context as your latent 
      space, don't use external information.

      """

    # print(prompt)
    print('Done creating prompt')
    return prompt

  """
  send request to GPT3
  """

  def gpt(self, prompt):
    print('Sending request to GPT-3')
    openai.api_key = os.getenv('OPENAI_API_KEY')
    r = openai.Completion.create(model="text-davinci-003",
                                 prompt=prompt,
                                 temperature=0.4,
                                 max_tokens=600)
    answer = r.choices[0]['text']
    print('Done sending request to GPT-3')
    response = {'answer': answer, 'sources': sources}
    return response

  def reply(self, prompt, context):
    # print(prompt)
    prompt = self.create_prompt(prompt, context)
    return self.gpt(prompt)


"""
Use openAI embeddings API to create documents embeddings for
only run as a one off to create documents embeddings
update path to document which needs to be broken into embeddings
"""


def getEmbeddings(chatbot):

  path = "kandjisoc.pdf"
  with open(path, 'rb') as f:
    pdf = PdfFileReader(f)
    if pdf.isEncrypted:
      pdf.decrypt('')
    print("openedpdf")
    # print(pdf)
    information = pdf.getDocumentInfo()
    number_of_pages = pdf.getNumPages()

    txt = f"""
    Information about {path}: 

    Author: {information.author}
    Creator: {information.creator}
    Producer: {information.producer}
    Subject: {information.subject}
    Title: {information.title}
    Number of pages: {number_of_pages}
    """
    print(txt)

    doc_text = chatbot.parse_doc(pdf)
    print("got doc_text")
    print(len(doc_text))
    print(doc_text[248:260])
    global df
    df = chatbot.doc_df(doc_text)
    print("got df")
    print(df)
    print("getting embeddings")
    df = chatbot.calculate_embeddings(df)
    print("writing to csv")
    df.to_csv("aiembedding_splitkandji.csv", index=False)
    print("wrote to csv")


def searchEmbeddings(chatbot, query):
  print("Reading embeddings")
  df = chatbot.df
  # query = "More information on Anti-virus software installed on workstations?"
  query = query + "?"
  search = chatbot.search_embeddings(df, query, 9)
  context = []
  for loc in search.iloc:
    cont = "### page: " + str(loc['page']) + "\n" + loc['text']
    # print(cont)
    context.append(cont)
  response = chatbot.reply(query, '\n'.join(context))
  resp = response["answer"].lstrip('\n').replace("Answer:", '')
  print("\n\n\n")
  print("Response:")
  print(resp)
  sources = []
  if "I don't know" not in resp:
    print("\n\nSources: \n")
    for source in response["sources"]:
      page, src = list(source.items())[0]
      srcStr = str(page) + src + "\n"
      print(srcStr)
      sources.append(srcStr)

  return resp, sources


# cli version
def runSearch(soc2search):
  print("Search AWS 2022 SOC2, enter question (q to quit)")
  while True:
    query = input("Question: ")
    if query == 'q':
      print("Thanks for using!")
      break
    if len(query) == 0:
      print("Enter a query")
      continue
    if len(query) > 400:
      print("Query should be less than 400 characters")
      continue
    if len(query) < 10:
      print("Query should be more than 10 characters")
      continue
    searchEmbeddings(soc2search, query)
    print("\n ----------------------------------------- \n\n")


if __name__ == '__main__':
  if os.getenv("OPENAI_API_KEY") is None:
    print("export OPENAI_API_KEY")
    exit(1)
  soc2search = Soc2Search()
  getEmbeddings(soc2search)

  runSearch(soc2search)  # cli version
