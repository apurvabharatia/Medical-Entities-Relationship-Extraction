import nltk
from nltk.chunk import conlltags2tree, tree2conlltags
import glob
import os
import numpy as np

completeWordFeature = []
fullDoc = []
file_list = []
merged_file = "merged_file.txt"
OneEHR = []
OneEHRLabels = []
OneEHRRel = []

class Word:
    def __init__(self,txt,pos):
        self.txt = txt
        self.pos = pos
        self.biotag = 'UNKNOWN'
        self.source = 'False'
        self.target = 'False'
        self.relType = 'NoRel'
        self.relSuperType = 'NoSuperRel'

    def printword(self):
        w = self.txt+","+self.pos+","+self.biotag+","+str(self.source)+","+str(self.target)
        print(w)

    def makeList(self):
        wordList = [self.txt,self.pos,self.biotag,str(self.source),str(self.target)]
        return wordList

def MatchConText(featureList, singleFileWordFeature):
    if featureList[1]==featureList[3]:
        lengthOfPhrase=int(featureList[4])-int(featureList[2])+1
        for i in range(0,lengthOfPhrase):
            currentWord = singleFileWordFeature[int(featureList[1]) - 1][int(featureList[2])+i]
            if i==0:
                currentWord.biotag = "B-" + featureList[5]
            elif i==lengthOfPhrase-1:
                currentWord.biotag = "E-" + featureList[5]
            else:
                currentWord.biotag = "M-" + featureList[5]
    return singleFileWordFeature

def TextFileReader(txtfile):
    singleFileWordFeature = []
    txtfile_handle=open(txtfile,"r")
    no_of_lines = 0
    f = open(merged_file, "a")
    for txtline in txtfile_handle:
        l1 = []
        token = nltk.word_tokenize(txtline)
        k = nltk.pos_tag(token)
        for i in range(0, len(k)):
            txt = k[i][0]
            pos = k[i][1]
            w = Word(txt, pos)
            l1.append(w)
        if len(k) != 0:
            f.write(txtline)
        if len(k) == 0:
            print("empty line")
        singleFileWordFeature.append(l1)
        no_of_lines += 1
    f.close()
    file_list.append([txtfile,no_of_lines])

    return singleFileWordFeature

def ConceptFileReader(confile, singleFileWordFeature):
    confile = open(confile, "r")
    for line in confile:
        line = line.split('"')
        entityList = line[1]
        entityList = entityList.replace("c=", "")
        indicesList = line[2]
        bioTag = line[3]
        indicesList = indicesList.replace("||t=", "")
        indicesList = indicesList.replace(" ", ":")
        indicesList = indicesList.split(":")
        if entityList!="":
            featureList = [entityList, indicesList[1], indicesList[2], indicesList[3], indicesList[4], bioTag]
            singleFileWordFeature = MatchConText(featureList, singleFileWordFeature)

    completeWordFeature.append(singleFileWordFeature)

def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]

def NoRels(textLine):
    NotSourceOrTargetVector = [1, 0, 0]
    textLine = textLine.split(" ")
    TrainFile = []
    LabelFile = []
    RelLabel = []

    for j in range(0, len(textLine)):
        TrainFile.append(textLine[j])
        LabelFile.append(NotSourceOrTargetVector)
    OneEHR.append(TrainFile)
    OneEHRLabels.append(LabelFile)
    OneEHRRel.append([1, 0, 0, 0, 0, 0, 0, 0, 0])

def HasRel(textLine,relations):
    ''''[relType, source, target, SourceStartLine, SourceStartInd, SourceEndInd, TargetStartInd, TargetEndInd]'''
    nrel = len(relations)
    sourceVector=[0,0,1]
    targetVector=[0,1,0]
    NotSourceOrTargetVector=[1,0,0]

    textLine = textLine.split(" ")

    for i in range(0, len(relations)):
        TrainFile = []
        LabelFile = []
        RelLabel = []

        relType=relations[i][0]
        startSource=relations[i][4]
        endSource=relations[i][5]
        startTarget=relations[i][6]
        endTarget=relations[i][7]

        for j in range(0, len(textLine)):
            TrainFile.append(textLine[j])
            if startSource<=j<=endSource:
                LabelFile.append(sourceVector)
            elif startTarget<=j<=endTarget:
                LabelFile.append(targetVector)
            else:
                LabelFile.append(NotSourceOrTargetVector)

        #Assign labels for each type of relation
        if relType=='TrIP':
            RelLabel=[0,1,0,0,0,0,0,0,0]
        if relType=='TrWP':
            RelLabel=[0, 0, 1, 0, 0, 0, 0, 0, 0]
        if relType=='TrCP':
            RelLabel=[0, 0, 0, 1, 0, 0, 0, 0, 0]
        if relType=='TrAP':
            RelLabel=[0, 0, 0, 0, 1, 0, 0, 0, 0]
        if relType=='TrNAP':
            RelLabel=[0, 0, 0, 0, 0, 1, 0, 0, 0]
        if relType=='TeRP':
            RelLabel=[0, 0, 0, 0, 0, 0, 1, 0, 0]
        if relType == 'TeCP':
            RelLabel=[0, 0, 0, 0, 0, 0, 0, 1, 0]
        if relType == 'PIP':
            RelLabel=[0, 0, 0, 0, 0, 0, 0, 0, 1]

        OneEHR.append(TrainFile)
        OneEHRLabels.append(LabelFile)
        OneEHRRel.append(RelLabel)

def CreateTrainingData(listWithRels, txtfilename):

    OneEHR=[]
    OneEHRLabels=[]
    textfile=open(txtfilename,'r')

    linenum = 1
    #For each line in current EHR, do:
    """
    The annotations file has labels associated with lines which have some relation present in them.
    E. g. 
    Annotations file : c="levofloxacin" (Word 1 ) 22:4 22:4 (line and character number) ||r="TrCP" (Relationship between them) ||c="hypotensive" 22:9 22:9 (Word 2)
    Corresponding line 22 in text file : She was started on Levofloxacin but the patient became hypotensive at that point with 
    Each line in the EHR either has some corresponding line in the annotations file or does not.
    A default value is assigned to each word if there is no corresponding value in the annotations file for a word.
    """
    for line in textfile :
        if linenum in listWithRels:
            HasRel(line,listWithRels[linenum])
        else:
            NoRels(line)
        linenum+=1

def RelationFileReader(relfile):

    """
    Annotations file : c="levofloxacin" (Word 1 ) 22:4 22:4 (line and character number) ||r="TrCP" (Relationship between them) ||c="hypotensive" 22:9 22:9 (Word 2)
    This file splits the annotation into source, target and relation using '||' as a splitter
    :param relfile:
    :return:
    """

    listWithRels = dict()
    relfile = open(relfile, "r")
    for line in relfile:

        line=line.split('"')
        source=line[1]
        target=line[5]
        relType=line[3]
        sourceInd=line[2]
        targetInd=line[6]

        sourceInd=sourceInd.replace(" ",":")
        sourceInd = sourceInd.split("||")
        sourceIndList=sourceInd[0].split(":")


        targetInd=targetInd.replace(" ",":")
        targetInd = targetInd.split("||")
        targetIndList=targetInd[0].split(":")

        SourceStartLine=int(sourceIndList[1])
        SourceStartInd=int(sourceIndList[2])

        SourceEndLine=int(sourceIndList[3])
        SourceEndInd=int(sourceIndList[4])

        TargetStartLine=int(targetIndList[1])
        TargetStartInd=int(targetIndList[2])

        TargetEndLine=int(targetIndList[3])
        TargetEndInd=int(targetIndList[4])

        if SourceStartLine == TargetStartLine:
            values = [relType, source, target, SourceStartLine , SourceStartInd, SourceEndInd, TargetStartInd, TargetEndInd]

            if SourceStartLine not in  listWithRels:
                listWithRels[SourceStartLine] = []

            listWithRels[SourceStartLine].append(values)

    return listWithRels

#if __name__ == "__main__":
def parser():
    """
    Reads all the files in a given dorectory
    :return:
    """
    contextfileNameList = glob.glob(
        "/Users/apurvabharatia/Desktop/i2b2 challenges n dataset/2010 Relations Challenge/concept_assertion_relation_training_data/beth/concept/*.con")
    textFileNameList = glob.glob(
        "/Users/apurvabharatia/Desktop/i2b2 challenges n dataset/2010 Relations Challenge/concept_assertion_relation_training_data/beth/txt/*.txt")
    cnt=0

    try:
        os.remove(merged_file)
    except OSError:
        pass


    for i in range(0,len(contextfileNameList)):
        """
        Similar analysis was done for context files and the following resuses naming of context folder structure, please ignore. 
        """
        conFileName=contextfileNameList[i]
        textFileName = conFileName.replace(".con", ".txt")
        textFileName = textFileName.replace("concept/", "txt/")

        relFileName = conFileName.replace(".con", ".rel")
        relFileName = relFileName.replace("concept/", "rel/")
        correspondingTextObject = TextFileReader(textFileName)
        ConceptFileReader(conFileName,correspondingTextObject)
        listWithRels = RelationFileReader(relFileName)
        CreateTrainingData(listWithRels,  textFileName)

        if i != len(contextfileNameList)-1:
            f = open(merged_file, "a")
            f.write("\n")
            f.close()

        for sentence in completeWordFeature[-1]:
            for word in sentence:
                fullDoc.append(word.makeList())
    print("Parsing Completed")

def getFullDoc():
    parser()
    return fullDoc

def getCompleteWordFeature ():
    parser()
    return OneEHR, OneEHRLabels, OneEHRRel

if __name__ == '__main__':
    getCompleteWordFeature()

