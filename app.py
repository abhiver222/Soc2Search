from flask import Flask, render_template, request, url_for, redirect
from flask_cors import CORS
import os
from soc2search import Soc2Search, searchEmbeddings
import time

app = Flask(__name__)
CORS(app)

global chatbot


def get_answer_and_references(question):
  answer = "Sample answer"
  references = ["Reference 1", "Reference 2"]
  print("searching embeddings")
  chatbot.test()
  # time.sleep(15)
  question = question.lstrip().rstrip()
  print(question)
  answer, references = searchEmbeddings(chatbot, question)
  print(answer)
  print("refs")
  print(references)
  return answer, references


# Define the index route
@app.route("/", methods=["GET", "POST"])
def index():
  # Check if the form has been submitted
  if request.method == "POST":
    # Get the user's question from the form
    question = request.form["question"]
    # Call your function to get the answer and references
    answer, references = get_answer_and_references(question)
    # Render the answer page with the answer and references as parameters
    return render_template("answer.html",
                           answer=answer,
                           references=references,
                           question=question)
  # If the form has not been submitted, render the index page
  texts = ["hello", "worls"]
  return render_template("index.html", texts=texts)


# Define the answer route
@app.route('/answer', methods=['GET', 'POST'])
def answer():
  question = request.form['question']
  print(question)
  answer, references = get_answer_and_references(question)
  return render_template('answer.html',
                         answer=answer,
                         references=references,
                         question=question)


if __name__ == "__main__":
  if os.getenv("OPENAI_API_KEY") is None:
    print("export OPENAI_API_KEY")
    exit(1)
  global chatbot
  chatbot = Soc2Search()
  app.run(host="0.0.0.0")
