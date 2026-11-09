# -*- coding: UTF-8 -*-
import numpy as np
import cv2


## http://stackoverflow.com/questions/1697334/algorithm-for-solving-sudoku
def findNextCellToFill(grid, i, j):
    for x in range(i,9):
        for y in range(j,9):
            if grid[x][y] == 0:
                return x,y
    for x in range(0,9):
        for y in range(0,9):
            if grid[x][y] == 0:
                return x,y
    return -1,-1

def isValid(grid, i, j, e):
    rowOk = all([e != grid[i][x] for x in range(9)])
    if rowOk:
        columnOk = all([e != grid[x][j] for x in range(9)])
        if columnOk:
            # finding the top left x,y co-ordinates of the section containing the i,j cell
            secTopX, secTopY = 3 *int(i/3), 3 *int(j/3)
            for x in range(secTopX, secTopX+3):
                for y in range(secTopY, secTopY+3):
                    if grid[x][y] == e:
                        return False
                return True
    return False

def solveSudoku(grid, i=0, j=0):
    i,j = findNextCellToFill(grid, i, j)
    if i == -1:
        return True
    for e in range(1,10):
        if isValid(grid,i,j,e):
            grid[i][j] = e
            if solveSudoku(grid, i, j):
                return True
            # Undo the current cell for backtracking
            grid[i][j] = 0
    return False

## 训练knn模型
samples = np.load('samples.npy')
labels = np.load('label.npy')

k = 80
train_label = labels[:k]
train_input = samples[:k]
test_input = samples[k:]
test_label = labels[k:]

model = cv2.ml.KNearest_create()
model.train(train_input,cv2.ml.ROW_SAMPLE,train_label)

#retval, results, neigh_resp, dists = model.findNearest(test_input, 1)
#string = results.ravel()
#print(string)
#print(test_label.reshape(1,len(test_label))[0])

img = cv2.imread('./images/001.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,200,255,1)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5, 5))     
dilated = cv2.dilate(thresh,kernel)
 

image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

boxes = []
for i in range(len(hierarchy[0])):
    if hierarchy[0][i][3] == 0:
        boxes.append(hierarchy[0][i])
        
height,width = img.shape[:2]
box_h = height/9
box_w = width/9
number_boxes = []
soduko = np.zeros((9, 9),np.int32)

for j in range(len(boxes)):
    if boxes[j][2] != -1:
        #number_boxes.append(boxes[j])
        x,y,w,h = cv2.boundingRect(contours[boxes[j][2]])
        number_boxes.append([x,y,w,h])
        #img = cv2.rectangle(img,(x-1,y-1),(x+w+1,y+h+1),(0,0,255),2)
        #img = cv2.drawContours(img, contours, boxes[j][2], (0,255,0), 1)
        
        number_roi = gray[y:y+h, x:x+w]
        resized_roi=cv2.resize(number_roi,(20,40))
        thresh1 = cv2.adaptiveThreshold(resized_roi,255,1,1,11,2)
        normalized_roi = thresh1/255.  
        
        sample1 = normalized_roi.reshape((1,800))
        sample1 = np.array(sample1,np.float32)
        
        
        retval, results, neigh_resp, dists = model.findNearest(sample1, 1)        
        number = int(results.ravel()[0])
        
        cv2.putText(img,str(number),(x+w+1,y+h-20), 3, 2., (255, 0, 0), 2, cv2.LINE_AA)
        
        soduko[int(y/box_h)][int(x/box_w)] = number
               
        #print(number)
        cv2.namedWindow("img", cv2.WINDOW_NORMAL); 
        cv2.imshow("img", img)
        cv2.waitKey(30)
print("\Generate sudoku\n")
print(soduko)
print("\Solved Sudoku\n")

solveSudoku(soduko)

print(soduko)
print("\n Verification ：Calculate the sum of each row and column\n")
row_sum = map(sum,soduko)
col_sum = map(sum,zip(*soduko))
print(list(row_sum))
print(list(col_sum))

#print(sum(soduko.transpose))  
for i in range(9):
    for j in range(9):
        x = int((i+0.25)*box_w)
        y = int((j+0.5)*box_h)
        cv2.putText(img,str(soduko[j][i]),(x,y), 3, 2.5, (0, 0, 255), 2, cv2.LINE_AA)
#print(number_boxes)
cv2.namedWindow("img", cv2.WINDOW_NORMAL);
cv2.imshow("img", img)
cv2.waitKey(0)


#retval, results, neigh_resp, dists = model.findNearest(test_input, 1)
#string = results.ravel()
#print(string)
#print(test_label.reshape(1,len(test_label))[0])

'''
C = 5
gamma = 0.5
model = cv2.ml.SVM_create()
model.setGamma(gamma)
model.setC(C)
model.setKernel(cv2.ml.SVM_LINEAR)
model.setType(cv2.ml.SVM_C_SVC)
model.train(train_input,cv2.ml.ROW_SAMPLE,train_label)
predict_label = model.predict(test_input)[1].ravel()
print(predict_label)
print(test_label.reshape(1,len(test_label))[0])


class MLP():
    class_n = 10
    def __init__(self):
        self.model = cv2.ml.ANN_MLP_create()

    def unroll_responses(self, responses):
        sample_n = len(responses)
        labels = []
        for i in range(len(responses)):
            label_b = np.zeros(self.class_n, np.int32)
            label = responses[i]
            label_b[int(label)] = 1
            labels.append(label_b)
        #print(labels)
            
        return labels
    
    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        new_responses = self.unroll_responses(responses)
        layer_sizes = np.int32([var_n, 200, 50, self.class_n])

        self.model.setLayerSizes(layer_sizes)
        self.model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
        self.model.setBackpropMomentumScale(0.1)
        self.model.setBackpropWeightScale(0.001)
        #self.model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 20, 0.1))
        self.model.setTermCriteria((cv2.TERM_CRITERIA_EPS, 50, 0.001))
        self.model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)

        self.model.train(samples, cv2.ml.ROW_SAMPLE, np.float32(new_responses))

    def predict(self, samples):
        ret, resp = self.model.predict(samples)
        return resp.argmax(-1)

model = MLP()
model.train(train_input,train_label)
predict_label = model.predict(test_input)
print(predict_label)
print(test_label.reshape(1,len(test_label))[0])
 
''' 
