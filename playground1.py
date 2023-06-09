import click
from pdfminer.layout import LAParams
from pdfminer.converter import TextConverter

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from io import StringIO

import pandas as pd
import numpy as np
import re
from openai.embeddings_utils import get_embedding, cosine_similarity
import openai
import os


class SecurityAssessmentQA():

  def __init__(self, report):
    self.report = report

  """
  Use openAI embeddings API to create documents embeddings for
  only run as a one off to create documents embeddings
  update path to document which needs to be broken into embeddings
  """

  def generateEmbeddings(self, force):
    print("Initializing embeddings")
    reportPath = f'{self.report}.pdf'
    embeddingsPath = f'{self.report}.csv'

    # Load from our embeddings file if it's already there
    if not force and os.path.exists(embeddingsPath):
      self.df = pd.read_csv(embeddingsPath)
      self.df["embeddings"] = self.df.embeddings.apply(eval).apply(np.array)
      return

    # Parse the PDF to get some text out of it
    doc_text = self.parse_doc(reportPath)
    print("Parsed the doc to get text")

    # Create the dataframe and then embeddings from the parsed text
    dataframe = self.parse_dataframe(doc_text)
    self.df = self.calculate_embeddings(dataframe)
    self.df.to_csv(embeddingsPath, index=False)
    print("Wrote embeddings file")

  def parse_doc(self, reportPath):
    doc_chunks = []

    with open(reportPath, 'rb') as f:
      resource_manager = PDFResourceManager()
      output = StringIO()
      laparams = LAParams(word_margin=0.75)
      device = TextConverter(resource_manager,
                             output,
                             codec='utf-8',
                             laparams=laparams)
      interpreter = PDFPageInterpreter(resource_manager, device)

      # Loop through the pages of the PDF document
      pageIndex = 0
      for page in PDFPage.get_pages(f):
        pageIndex += 1

        (x0, y0, x1, y1) = page.mediabox
        page.mediabox = (x0, float(y1) * .1, x1, float(y1) * .9)
        interpreter.process_page(page)
        textOnPage = output.getvalue()
        textOnPage = re.sub('\s+', ' ', re.sub('\n', ' ', textOnPage)).strip()

        output.truncate(0)
        output.seek(0)

        CHARACTERS_PER_CHUNK = 1000

        for i in range(0, len(textOnPage), CHARACTERS_PER_CHUNK):
          doc_chunks.append({
            'text': textOnPage[i:i + CHARACTERS_PER_CHUNK],
            'page': pageIndex
          })

        for i in range(int(CHARACTERS_PER_CHUNK / 2), len(textOnPage),
                       CHARACTERS_PER_CHUNK):
          doc_chunks.append({
            'text': textOnPage[i:i + CHARACTERS_PER_CHUNK],
            'page': pageIndex
          })

    return doc_chunks

  def parse_dataframe(self, pdf):
    df = pd.DataFrame(pdf)
    df = df.drop_duplicates(subset=['text', 'page'], keep='first')
    df['length'] = df['text'].apply(lambda x: len(x))
    return df

  def calculate_embeddings(self, df):
    openai.api_key = os.getenv('OPENAI_API_KEY')
    embedding_model = "text-embedding-ada-002"
    embeddings = df.text.apply(
      [lambda x: get_embedding(x, engine=embedding_model)])
    df["embeddings"] = embeddings
    return df

  def search_embeddings(self, df, query, n, pprint=True):
    query_embedding = get_embedding(query, engine="text-embedding-ada-002")
    df["similarity"] = df.embeddings.apply(
      lambda x: cosine_similarity(x, query_embedding))

    results = df.sort_values("similarity", ascending=False, ignore_index=True)
    results = results.head(n)
    return results

  def answer_question(self, query):
    df = self.df
    search = self.search_embeddings(df, query, 10)
    print("searching embeddings...")
    context = []
    for loc in search.iloc:
      text = loc['text']
      pageNumber = loc['page']
      context.append("Page " + str(pageNumber) + ":\n" + text)
    contextString = '\n'.join(context)

    raw_response = self.get_gpt4_reply(query, contextString)
    response = raw_response["answer"].lstrip('\n').replace("Answer:", '')
    return response

  def get_system_prompt(self):
    return """
      You are responsible for summarizing information about a software vendor's security practices. We will give you some context about the vendor and a question. Answer the question as truthfully as possible using only information from the context. Also, include in the answer, all page numbers from which you got the information from. If the information is not available, answer "Vanta could not answer the question from the files. Format this answer as a JSON with response as a key and pages as another key."

    Example:
    Question: Were there any deviations noted in this report?
    Answer: {response: "No, there weren't any deviations noted in this report", pages: "1,2,4,5"}
    """

  def create_prompt(self, user_input, context):
    system_prompt = self.get_system_prompt()
    prompt = system_prompt + """ Context: """ + context + """
      Question: """ + user_input

    return prompt

  def create_chat_prompt(self, prompt, context):
    system_prompt = self.get_system_prompt()
    chat_prompt = """
      Context: """ + context + """
      Answer below question based on above context. 
      Question: """ + prompt

    return [{
      "role": "system",
      "content": system_prompt
    }, {
      "role": "user",
      "content": chat_prompt
    }]

  """
  Send request to GPT3
  """

  def get_reply(self, prompt, context):
    prompt = self.create_prompt(prompt, context)

    print('Sending request to GPT-3')
    openai.api_key = os.getenv('OPENAI_API_KEY')
    r = openai.Completion.create(model="text-davinci-003",
                                 prompt=prompt,
                                 temperature=0.2,
                                 max_tokens=1000)
    answer = r.choices[0]['text']
    response = {'answer': answer}
    return response

  def get_gpt4_reply(self, prompt, context):
    chat_messages = self.create_chat_prompt(prompt, context)
    openai.api_key = os.getenv('OPENAI_API_KEY')
    print('Sending request to GPT-4')
    r = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                     messages=chat_messages,
                                     temperature=0.2,
                                     max_tokens=1000)

    answer = r.choices[0]["message"]["content"].strip()
    response = {'answer': answer}
    return response


# cli version
def runQAndA(chatbot):
  print("Ask any question about this security report:")
  while True:
    query = input("Question: ")
    if query == 'q':
      print("Thanks for using!")
      break
    if len(query) <= 10 or len(query) >= 400:
      print("Enter a question between 10 and 400 characters.")
      continue
    try:
      print(chatbot.answer_question(query))
    except Exception as e:
      print(e)
      print("An error occurred, try again")
    print("\n -------------------------------------------\n\n")


@click.command()
@click.option('--report')
@click.option('--force/--no-force', default=False)
def main(report, force=False):
  chatBot = SecurityAssessmentQA(report)
  chatBot.generateEmbeddings(force)
  runQAndA(chatBot)


if __name__ == '__main__':
  if os.getenv("OPENAI_API_KEY") is None:
    print("export OPENAI_API_KEY")
    exit(1)
  main()
