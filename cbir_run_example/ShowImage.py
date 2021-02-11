import cv2
def change_numberL_to_namesL(A=[], parisnames=''):
    X=[]#list A:contains numbers
    P=[]#list p : should contain paris_names
    file = open(parisnames,'r')
    P=file.readlines()
    file.close()
    for i in range(0,len(A)):
        #print(A[i])
        if(A[i]<6424):
            X.append(P[A[i]])
    return X
def change_name_to_number(q,parisnames=''):
    X=[]#list A:contains numbers
    P=[]#list p : should contain paris_names
    file = open(parisnames,'r')
    P=file.readlines()
    file.close()
    return P[q]

def images_functuion(q, ranked_list=[]  , gnd_t=[] , parisnames='' , number_of_images_to_show=50,number_of_the_first=0 ,number_of_the_last=49 ):#qeuery number
    q=q
    #parisnames1='D:/CBIR code/learnedcode/effinetB1Rmac/paris_names.txt'
    parisnames1='D:/CBIR code/learnedcode/effinetb1a/paris_names.txt'
    #parisQuery="D:/CBIR code/learnedcode/effinetB1Rmac/paris_queries_names.txt"
    parisQuery="D:/CBIR code/learnedcode/effinetb1a/paris_queries_names.txt"
    qurey_name=change_name_to_number(q,parisnames=parisQuery)
    qurey_path="D:\\CBIR code\\paris6K\\"+ qurey_name
    print(qurey_path)
    img = cv2.imread(qurey_path[:-1],1)
    imag = cv2.resize(img,(166,166))
    cv2.imshow('qurey',imag)
    retrive_images_numbers=ranked_list[q]
    retrived_images_names=[]
    retrived_images_names=change_numberL_to_namesL(retrive_images_numbers,parisnames=parisnames1)
    images=[]
    for i in range(number_of_the_first,number_of_the_last):    #(0,number_of_images_to_show):
        image="D:\\CBIR code\\paris6K\\" + retrived_images_names[i] 
        #print(image)
        img = cv2.imread(image[:-1],1)
        imag = cv2.resize(img,(166,166))
        images.append(imag)
    import numpy as np
    numpy_vertical1 = np.hstack(images[0:8])
    p=number_of_images_to_show
    cv2.imshow('retrived1',numpy_vertical1)
    if(p>8):
        numpy_vertical2 = np.hstack(images[9:17])
        cv2.imshow('retrived2',numpy_vertical2)
    if(p>17):
        numpy_vertical3 = np.hstack(images[18:26])
        cv2.imshow('retrived3',numpy_vertical3)
    if(p>26):
        numpy_vertical4 = np.hstack(images[27:35])
        cv2.imshow('retrived4',numpy_vertical4)
    if(p>35):
        numpy_vertical5 = np.hstack(images[36:44])
        cv2.imshow('retrived4',numpy_vertical5)
    if(p>44):
        numpy_vertical6 = np.hstack(images[45:53])
        cv2.imshow('retrived5',numpy_vertical6)
    
    cv2.waitKey(0)
    

