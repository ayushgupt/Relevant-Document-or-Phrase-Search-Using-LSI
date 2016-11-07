#!/usr/bin/python
import re
import sys


word_input = sys.argv[12]
word_output = sys.argv[14] 
concepts_number = int(sys.argv[2])
num_results_reqd = int(sys.argv[4])
directory_location = sys.argv[6]
document_input = sys.argv[8]
query_input = sys.argv[16]
document_output = sys.argv[10]
query_output = sys.argv[18]


# num_results_reqd=10
# concepts_number = 200


import os
current_directory = os.getcwd()
doc_location = current_directory+'/'+directory_location
os.chdir(doc_location)




pattern = re.compile(r'\W+')

from collections import Counter


# curr_file_words.remove('')
dictionary_list = [];
set_of_words = set()
Title_index={}
index_Title={}


file_up_limit=5000;

for file_num in range(1,file_up_limit+1):
	file_str = str(file_num) + ".txt"
	reading_file = open(file_str,'r')
	title=reading_file.readline().rstrip()
	curr_file_str=reading_file.read()
	reading_file.close()
	Title_index[title]=file_num
	index_Title[file_num]=title
	curr_file_words = pattern.split(curr_file_str.lower().strip(' ').rstrip('\n'))
	set_of_words=set_of_words.union(set(curr_file_words))
	dictionary_list.append(dict(Counter(curr_file_words) ))
	# print file_num


# TODO->convert to lower case

# print(len(dictionary_list))

os.chdir(current_directory)

list_of_words = list(set_of_words);

unique_words_num=len(list_of_words)

words_index={};

for i in range(0,unique_words_num):
	words_index[list_of_words[i]]=i


list_doc_col=[];
list_word_row=[];
list_count_data=[];

# print("Started making the 3 lists")

for doc_num in range (0,file_up_limit):
	for current_word in dictionary_list[doc_num]:
		list_doc_col.append(doc_num)
		list_word_row.append(words_index[current_word])
		list_count_data.append(dictionary_list[doc_num][current_word])

# print("Made the 3 lists")

import numpy as np
import scipy.sparse as sp
from scipy import spatial
sparse_doc_word_matrix=sp.csc_matrix((np.array(list_count_data), (np.array(list_word_row), np.array(list_doc_col) ) ), shape = (unique_words_num, file_up_limit)  ) 

# sp.coo_matrix.asfptype(sparse_doc_word_matrix)

from scipy.sparse.linalg import svds
term_row_matrix , strength_matrix , doc_col_matrix = svds(sparse_doc_word_matrix.astype(float), concepts_number , which = 'LM')


doc_row_matrix= np.dot(np.transpose(doc_col_matrix),np.diag(strength_matrix))
# Now a query is given and we need to find out the Documents!!

# print("Started Query IO")

multiply1= np.dot( np.array(np.array(term_row_matrix)),np.linalg.inv(np.diag(strength_matrix))  ) 

query_input_list = [(line.rstrip('\n')).strip(' ') for line in open(query_input)]

write_query_results = open(query_output,"a")

for query_input_index in range(0,len(query_input_list)):
	query_string = query_input_list[query_input_index]
	query_words=pattern.split(query_string.lower())
	query_dictionary=dict(Counter(query_words) );
	query_list_doc_col=[];
	query_list_word_row=[];
	query_list_count_data=[];
	for current_word in query_dictionary:
		query_list_doc_col.append(0)
		query_list_word_row.append(words_index[current_word])
		query_list_count_data.append(query_dictionary[current_word])
	query_sparse_doc_word_matrix=sp.csc_matrix((np.array(query_list_count_data), (np.array(query_list_word_row), np.array(query_list_doc_col) ) ), shape = (unique_words_num, 1)  )
	query_sparse_doc_word_matrix=query_sparse_doc_word_matrix.todense();
	Final_query_vector=np.dot(  np.transpose(query_sparse_doc_word_matrix),multiply1)
	# doc_row_matrix= doc_col_matrix.transpose()
	# doc_query=1
	# doc_query_index=doc_query-1
	doc_dot_product_list=np.array([])
	doc_query_row=Final_query_vector.transpose();
	for other_doc_row in doc_row_matrix:
		doc_dot_product_list=np.append(doc_dot_product_list,1 - spatial.distance.cosine(doc_query_row,other_doc_row))
	unsorted_answer_indices = np.argpartition(  doc_dot_product_list , -1*(num_results_reqd))[-1*(num_results_reqd):]
	sorted_answer_indices= unsorted_answer_indices[np.argsort(doc_dot_product_list[unsorted_answer_indices])]
	final_doc_answer_list=[]
	for i in range(0,num_results_reqd):
		write_query_results.write(index_Title[sorted_answer_indices[num_results_reqd-i-1]+1]+";"+"\t")
		# final_words_answer_list.append(sorted_answer_indices[num_results_reqd-i-1]+1)
		# print (sorted_answer_indices[num_results_reqd-i-1]+1)
	write_query_results.write('\n')

write_query_results.close()

# print("Ended Query IO")



# Given a word, then finding its best k words
# print("Started Word IO")

compare_term_matrix=np.dot(term_row_matrix,np.diag(strength_matrix))

word_input_list = [(line.rstrip('\n')).strip(' ') for line in open(word_input)]


write_word_results= open(word_output,"a")

# print "size is ",len(word_input_list)

for word_input_index in range(0,len(word_input_list)):
	# print "Doing for index",word_input_index
	word_query=word_input_list[word_input_index]
	word_query_index=words_index[word_query]
	word_dot_product_list=[0 for x in xrange(unique_words_num)]
	word_query_row=compare_term_matrix[word_query_index];
	# print "step1"
	for i in xrange(unique_words_num):
		word_dot_product_list[i]=(1 - spatial.distance.cosine(word_query_row,compare_term_matrix[i]))
	# print "step2"
	# print len(word_dot_product_list)
	# print len(word_dot_product_list[0])
	unsorted_answer_indices = np.argpartition(  word_dot_product_list , -1*(num_results_reqd))[-1*(num_results_reqd):]
	# print "WORD DOT",word_dot_product_list
	# print "UNSORTED INDICES",unsorted_answer_indices
	# print "step3"
	# sorted_answer_indices= unsorted_answer_indices[np.argsort(word_dot_product_list[unsorted_answer_indices])]
	# final_words_answer_list=[]
	# print "step4"
	for i in range(0,num_results_reqd):
		write_word_results.write(list_of_words[unsorted_answer_indices[num_results_reqd-1-i]]+";"+"\t")
		# final_words_answer_list.append(list_of_words[sorted_answer_indices[num_results_reqd-1-i]])
		# print list_of_words[sorted_answer_indices[num_results_reqd-1-i]]
	# print "step5"
	write_word_results.write('\n')

write_word_results.close()


# print("Ended Word IO")




# Given a doc then finding similar documents

# print("Started DOC IO")



doc_input_list = [(line.rstrip('\n')).strip(' ') for line in open(document_input)]

write_doc_results= open(document_output,"a")

# print "Size is",len(doc_input_list)

for doc_input_index in range(0,len(doc_input_list)):
	# print(doc_input_index)
	doc_query=Title_index[doc_input_list[doc_input_index]]
	doc_query_index=doc_query-1
	doc_dot_product_list=np.array([])
	doc_query_row=doc_row_matrix[doc_query_index];
	for other_doc_row in doc_row_matrix:
		doc_dot_product_list=np.append(doc_dot_product_list,1 - spatial.distance.cosine(doc_query_row,other_doc_row))
	unsorted_answer_indices = np.argpartition(  doc_dot_product_list , -1*(num_results_reqd))[-1*(num_results_reqd):]
	sorted_answer_indices= unsorted_answer_indices[np.argsort(doc_dot_product_list[unsorted_answer_indices])]
	final_doc_answer_list=[]
	for i in range(0,num_results_reqd):
		write_doc_results.write(index_Title[sorted_answer_indices[num_results_reqd-i-1]+1]+";"+"\t")
		# final_words_answer_list.append(sorted_answer_indices[num_results_reqd-i-1]+1)
		# print (sorted_answer_indices[num_results_reqd-i-1]+1)
	write_doc_results.write('\n')

write_doc_results.close()

# print("Ended DOC IO")


