from webbrowser import MacOSX
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
# from sentence_similarity import sentence_similarity
import pandas as pd

# from appium import an
import time

import numpy as np
# import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


class sentence_similarity: # source code: https://drive.google.com/file/d/1pYmRhZtz2ae4MPhbu63BG14mvBcENgFk/view?usp=sharing
    def __init__(self):
        self.module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
        self.threshold = 0.5

    # get cosine similarity matrix
    def cos_sim(self, input_vectors):
        similarity = cosine_similarity(input_vectors)
        return similarity

    # get topN similar sentences
    def get_top_similar(self, s, sentence_list, similarity_matrix, N):
         # find the index of sentence in list
        index = sentence_list.index(s)
        # get the corresponding row in similarity matrix
        similarity_row = np.array(similarity_matrix[index, :])
        # get the indices of top similar
        sorted_arr = np. sort(similarity_row[:-1])
        print(sorted_arr)
        indices = similarity_row[:-1].argsort()
        return [sentence_list[i] for i in indices], sorted_arr

    def _run(self, sentences_list, sentence):
       # Import the Universal Sentence Encoder's TF Hub module
        embed = hub.Module(self.module_url)
        # Reduce logging output.
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        if sentence not in sentences_list:
            sentences_list.append(sentence)
        with tf.compat.v1.Session() as session:
            session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
            sentences_embeddings = session.run(embed(sentences_list))

        similarity_matrix = self.cos_sim(np.array(sentences_embeddings))
        top_similar, val = self.get_top_similar(sentence, sentences_list, similarity_matrix, 3)
        print(top_similar)
        if val[-1] > self.threshold:
            return True
        else:
            return False


class Test():
    def setup_method(self):
        # Log in the first time for the profile (The test will fail the first time)
        # Once logged in the test will be able to retain the cookie session
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("user-data-dir=/Users/thinhnguyen/Library/Application Support/Google/Chrome/Profile 1")
        self.driver = webdriver.Chrome(options=chrome_options) # Need to download chromedriver
        self.driver.get("https://chat.kuki.ai/chat")
        self.vars = {}
  
    def teardown_method(self):
        self.driver.quit()

    def SendMessage(self, text):
        try:
            time.sleep(2)
            self.driver.find_element(By.CSS_SELECTOR, "input").send_keys(text)
            self.driver.find_element(By.CSS_SELECTOR, "input").send_keys(Keys.ENTER)
        except:
            print("err with sending msg")
            pass

    def GetResponse(self):
        try:
            time.sleep(6) # adjust timing to wait for the response
            responses = self.driver.find_elements(by=By.CLASS_NAME, value="pb-chat-bubble__bot")
            return responses[len(responses) - 1]
        except:
            print("err with retrieving the messages")
            pass

def main():
    #Question sets
    questions = [
     "Who is my favorite artist?",
     "What is my favorite game?",
     "What is my favorite food?",
     "What did we talk about the other day",
     "What fruit is named after a color?"
     ]

    #List of possible responses for each questions
    responses = [
        ["Doja Cat"],
        ["Monster Hunter: Rise"],
        ["Cheese Burger"],
        ["Movie", "Artist", "Monster Hunter", "Game", "Eat", "San Jose", "Tangled", "5f11"],
        ["Orange", "Blueberry"]
    
    ]
    ss = sentence_similarity()
    t = Test()
    
    t.setup_method()
    passCount = 0
    for i in range(len(questions)):
        print("Questions: ", questions[i])
        t.SendMessage(questions[i])
        message = t.GetResponse().text
        print("Bot's response: ", message) #get the last message
        result = ss._run(responses[i], message)
        if result:
            passCount += 1
    
    print("====================SUMMARY=========================")
    print("Number of test cases that passed = ", passCount)
    print("Number of test cases that failed  = ", len(questions) - passCount)
    print("====================================================")

    t.teardown_method()

if __name__ == '__main__':
    main()