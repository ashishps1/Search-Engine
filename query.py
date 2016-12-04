import nltk
import os
import pickle
from nltk.stem.porter import *
import math
import re, collections
import webbrowser
import Tkinter as tk
from Tkinter import *
import speech_recognition as sr


################################# TRIE ##########################################


lis=map(chr,range(97,123))
lis=list(lis)
lis.append("'")

word=''

class TrieNode:     #A node for trie
    def __init__(self):
        self.val = None     #value at current node
        self.pointers={}    #list of all the valid pointers
        self.end=0          #marking end of a word


class Trie:     #A trie class which implements all the required functions

    def __init__(self):     #Constructor making the root node for trie
        self.root = TrieNode()

    def insert(self, word):     #inserting a word into trie
        self.recInsert(word, self.root)
        return

    def recInsert(self, word, node):    #helper recursive function for inserting the word in trie
        if len(word[:1])==0:
            node.end=1
            return        
        if word[:1] not in node.pointers:
            newNode=TrieNode()
            newNode.val=word[:1]
            node.pointers[word[:1]]=newNode
            self.recInsert(word[1:], node)
        else:            
            nextNode = node.pointers[word[:1]]
            self.recInsert(word[1:], nextNode)

    def search(self, word):     #searching a word in trie
        if len(word)==0:
            return False
        return self.recSearch(word,self.root)

    def recSearch(self, word, node):    #helper recursive function for searching in trie
        if len(word[:1])==0:
            if node.end == 1:
                return True
            else:
                return False        
        elif word[:1] not in node.pointers:
            return False
        else:
            nextNode = node.pointers[word[:1]]
            return self.recSearch(word[1:],nextNode)

    def startsWith(self, prefix):   #searching if there is a word begining with a prefix
        if len(prefix)==0:
            return True
        return self.recSearchPrefix(prefix,self.root)

    def recSearchPrefix(self, word, node):  #helper function for prefix boolean search
        if len(word[:1])==0:
            return True        
        elif word[:1] not in node.pointers:
            return False
        else:
            nextNode = node.pointers[word[:1]]
            return self.recSearchPrefix(word[1:],nextNode)
            
    
    def findAll(self,node,word,sugg):   #storing all the words resulting from prefix search
        for c in lis:
            if c in node.pointers:
                if node.pointers[c].end==1:
                    sugg.append(word+str(c))
                self.findAll(node.pointers[c],word+str(c),sugg)
        return


    def didUMean(self,word,sugg):   #searching for all the words begining with a prefix
        if self.startsWith(word):
            top=self.root
            for c in word:
                top=top.pointers[c]
            self.findAll(top,word,sugg)
        else:
            return



trie=Trie()     #creating a trie object

words = pickle.load(open("words.p","rb"))   #loading all the distinct words from corpus

for word in words:
    trie.insert(word.lower())   #inserting each word in trie


########################### TRIE END ###########################################


############################# EDIT DISTANCE ####################################


def train(features):    #training a model to suggest a word on the basis of its occurence in the real world
    model = collections.defaultdict(lambda: 1)
    for f in features:
        if model[f]>1 or trie.search(f):
            model[f]+=1
    return model

def words(text): 
    return re.findall('[a-z]+', text.lower())

NWORDS = train(words(open('big.txt','r').read()))


class EditDist:     #Edit distance class
    def __init__(self):
        pass

    alphabet='abcdefghijklmnopqrstuvwxyz'

    def edits1(self,word):  #function for calculating all the words at edit distance 1
       splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
       deletes    = [a + b[1:] for a, b in splits if b]
       transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
       replaces   = [a + c + b[1:] for a, b in splits for c in self.alphabet if b]
       inserts    = [a + c + b  for a, b in splits for c in self.alphabet]
       return set(deletes + transposes + replaces + inserts)

    def knownEdits2(self,word): #function for calculating all the words at edit distance 2
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if trie.search(e2))

    def known(self,words):
        return set(w for w in words if w in NWORDS)

    def correct(self,word):     #Returns all the words which are closer to the given word
        candidates = self.known([word]) or self.known(self.edits1(word)) or self.knownEdits2(word) or [word]
        sugg=list(candidates)
        
        sugg.sort(key = lambda s: nltk.edit_distance(word,s))

        return sugg[:min(len(sugg),10)]

############################# EDIT DISTANCE END ################################


lengths = {}

stemmer = PorterStemmer()

#Loading all the data structures created before
dict = pickle.load(open("dict.p","rb"))
invertedIndex = pickle.load(open("invertedIndex.p","rb"))
tf_idf = pickle.load(open("tf_idf.p","rb"))


N = len(dict)

for key in tf_idf:  #Finding the length of each vector(doc represented as a vector)
    temp = 0.0
    for word in tf_idf[key]:
        temp = temp + tf_idf[key][word] * tf_idf[key][word]
    lengths[key] = math.sqrt(temp)


def pageRank(query):    #function to implement page ranking
    query_dic = {}
    q_list = []
    
    for word in query.split():  #Representoing query as a vector
        word=word.lower()
        word = stemmer.stem(word)
        if query_dic.has_key(word):
            k=query_dic[word]
            query_dic[word] = k+1
        else:
            query_dic[word] = 1

    for key in query_dic:
        q_list.append(key)
        print key

    score = {}

    # print len(dict)

    for word in q_list:     #Calculating the cosine similarity of the query vector with the docs
        wtq = 0
        if invertedIndex.has_key(word):
            df = len(invertedIndex[word])
            idf = math.log( N/( df * 1.0 ), 10.0 )
            wtq = idf * ( 1.0 + math.log( query_dic[word] , 10.0))
            
            for doc in invertedIndex[word]:
                if score.has_key(doc):
                    temp = score[doc]
                    wtd = tf_idf[doc][word]
                    score[doc] = temp + wtq * wtd
                else:
                    wtd = tf_idf[doc][word]
                    score[doc] = wtq * wtd

    ranking = []

    for key in score:   #Length Normalization of the cosine similarity
        score[key] = score[key]/(1.0 * lengths[key])
        ranking.append((key, score[key]))
        #print key, score[key], lengths[key]

    # sorted(ranking,key=itemgetter(1))
    ranking = sorted(ranking , key=lambda x: x[1], reverse = True)  #sorting all the docs on the basis of their cosine similarity

    print ranking[:20]

    text = '\n'.join(chunk[0] for chunk in ranking[:min(len(ranking),20)])  #Returning the top 20 search results
    return text


############################### GUI ############################################
def util(word):     #A function for auto suggestion
    word=word.lower()
    ed=EditDist()
    sugg=[]
    trie.didUMean(word,sugg)    #calling the prefix search function and storing the results
    
    global words
    
    if len(sugg)!=0:
            sugg.sort(key = lambda s: len(s))
    else:
            sugg=ed.correct(word)

    sugg = [s for s in sugg if s in words]
				

    text = '\n'.join(chunk for chunk in sugg[:min(len(sugg),10)])   #returning the top 10 suggestions
    return text

class AutocompleteEntry(Entry): #class to implement autocomplete combobox GUI
    def __init__(self,*args, **kwargs):
        Entry.__init__(self, *args, **kwargs)
        self.var = self["textvariable"]
        if self.var == '':
            self.var = self["textvariable"] = StringVar()

        self.var.trace('w', self.changed)
        self.bind("<Right>", self.selection)
        self.bind("<Up>", self.up)
        self.bind("<Down>", self.down)

        self.lb_up = False

    def changed(self, name, index, mode):
        if self.var.get() == '':
            self.lb.destroy()
            self.lb_up = False
        else:
            words = self.comparison()
            if words:
                if not self.lb_up:
                    self.lb = Listbox()
                    self.lb.bind("<Double-Button-1>", self.selection)
                    self.lb.bind("<Right>", self.selection)
                    self.lb.place(x=self.winfo_x()+self.winfo_width()-35, y=self.winfo_y()+self.winfo_height())
                    self.lb_up = True

                self.lb.delete(0, END)
                for w in words:
                    self.lb.insert(END,w)
            else:
                if self.lb_up:
                    self.lb.destroy()
                    self.lb_up = False
        
    def selection(self, event):     #function to make a selection in the combobox
        if self.lb_up:
            self.var.set(self.lb.get(ACTIVE))
            self.lb.destroy()
            self.lb_up = False
            self.icursor(END)

    def up(self, event):    #function to move up the combobox
        if self.lb_up:
            if self.lb.curselection() == ():
                index = '0'
            else:
                index = self.lb.curselection()[0]
            if index != '0':
                self.lb.selection_clear(first=index)
                index = str(int(index)-1)
                self.lb.selection_set(first=index)
                self.lb.activate(index)

    def down(self, event):  #function to move down the combobox
        if self.lb_up:
            if self.lb.curselection() == ():
                index = '0'
            else:
                index = self.lb.curselection()[0]
            if index != END:
                self.lb.selection_clear(first=index)
                index = str(int(index)+1)
                self.lb.selection_set(first=index)
                self.lb.activate(index)

    def comparison(self):
        word=self.var.get()
        word=word.lower()
        ed=EditDist()
        sugg=[]
        trie.didUMean(word,sugg)
        if len(sugg)!=0:
            sugg.sort(key = lambda s: len(s))
        else:
            sugg=ed.correct(word)
            
        if trie.search(word):
            sugg.insert(0,word)
        
        res=[chunk for chunk in sugg[:min(len(sugg),10)]]

        return res

def callback(event):    #function which gets called when a doc link is clicked
    s = event.widget.cget("text")
    webbrowser.get("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s").open(s)
    

def showSearchResults():    #function to show all the relevant documents inside the GUI
    key=entry.get()
    word=key
    text=pageRank(key)
    print text.split()
    for words in text.split():
        lbl = tk.Label(content_text, text=r"brown/"+words, fg="blue", cursor="hand2")
        lbl.pack()
        lbl.bind("<Button-1>", callback)

def speakNow():     #function to implement speach to text conversion
    r = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)

      
    try:
        #str1 = ''
        print("Google thinks you said " + r.recognize_google(audio))
        key = r.recognize_google(audio)
        #entry = StringVar()
        #entry.set(str1)
        #showSearchResults()
        entry.delete(0, END)
        entry.insert(0,key)
        word=key
        text=pageRank(key)
        content_text.delete('1.0', END)
        content_text.insert('0.0',text)
        
    except sr.UnknownValueError:
        print("Google could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google service; {0}".format(e))



PROGRAM_NAME="DASH"

#GUI Window creation

root = Tk()     
root.geometry('500x500')
root.title(PROGRAM_NAME)

frame1 = Frame(root)
frame1.pack()

entry = AutocompleteEntry(frame1)
entry.pack(side=LEFT)
button = Button(frame1, text='Search', width=25, command=showSearchResults)
button.pack(side=LEFT)
button = Button(frame1, text='Speak Now', width=10, command=speakNow)
button.pack(side=LEFT)
entry.focus()


content_text = Text(root, wrap='word')
content_text.pack(expand='yes', fill='both')
scroll_bar = Scrollbar(content_text)
content_text.configure(yscrollcommand=scroll_bar.set)
scroll_bar.config(command=content_text.yview)
scroll_bar.pack(side='right', fill='y')


root.mainloop()

################################ GUI END #######################################

