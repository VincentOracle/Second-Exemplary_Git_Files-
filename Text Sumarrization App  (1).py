#!/usr/bin/env python
# coding: utf-8

# In[7]:


# !pip install transformers


# In[8]:


# install transformers with sentencepiece
# !pip install transformers[sentencepiece]


# In[9]:



# open and read the file from google drive
file = open("C:\\Users\\n\\Desktop\\marine.txt", "r")
FileContent = file.read().strip()
print("File read successfully!")


# In[10]:


# display file content
FileContent 


# In[11]:


# total characters in the file
len(FileContent) 


# In[12]:


from transformers import pipeline


# In[ ]:


summarizer = pipeline("summarization")


# In[ ]:


summarizer(FileContent, max_length=300, min_length=30, do_sample=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Load the Model and Tokenizer

# In[ ]:


# !pip install transformers

  import nltk
  nltk.download('punkt')


# In[36]:


# import and initialize the tokenizer and model from the checkpoint
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# checkpoint = "sshleifer/distilbart-cnn-12-6"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)


# # Some model statistics

# In[37]:


# max tokens including the special tokens
tokenizer.model_max_length 


# In[38]:



# max tokens excluding the special tokens
tokenizer.max_len_single_sentence 


# In[39]:


# number of special tokens
tokenizer.num_special_tokens_to_add() 


# # Convert file content to sentences

# In[40]:



# extract the sentences from the document
import nltk
nltk.download('punkt')
sentences = nltk.tokenize.sent_tokenize(FileContent)


# In[ ]:


# find the max tokens in the longest sentence
max([len(tokenizer.tokenize(sentence)) for sentence in sentences])


# # Create the chunks

# In[ ]:


# initialize
length = 0
chunk = ""
chunks = []
count = -1
for sentence in sentences:
  count += 1
  combined_length = len(tokenizer.tokenize(sentence)) + length # add the no. of sentence tokens to the length counter

  if combined_length  <= tokenizer.max_len_single_sentence: # if it doesn't exceed
    chunk += sentence + " " # add the sentence to the chunk
    length = combined_length # update the length counter

    # if it is the last sentence
    if count == len(sentences) - 1:
      chunks.append(chunk.strip()) # save the chunk
    
  else: 
    chunks.append(chunk.strip()) # save the chunk
    
    # reset 
    length = 0 
    chunk = ""

    # take care of the overflow sentence
    chunk += sentence + " "
    length = len(tokenizer.tokenize(sentence))
len(chunks)


# # Some checks

# In[ ]:



[len(tokenizer.tokenize(c)) for c in chunks]


# In[ ]:


[len(tokenizer(c).input_ids) for c in chunks]
     


# # With special tokens added

# In[ ]:


sum([len(tokenizer(c).input_ids) for c in chunks])


# In[ ]:



len(tokenizer(FileContent).input_ids)


# # Without special tokens added

# In[ ]:


sum([len(tokenizer.tokenize(c)) for c in chunks])


# In[ ]:


len(tokenizer.tokenize(FileContent))
     


# # Get the inputs

# In[ ]:


# inputs to the model
inputs = [tokenizer(chunk, return_tensors="pt") for chunk in chunks]


# # Output

# In[ ]:



for input in inputs:
  output = model.generate(**input)
  print(tokenizer.decode(*output, skip_special_tokens=True))


# In[ ]:





# In[ ]:





# In[ ]:




